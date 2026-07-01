"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses auxiliary model (cheap/fast) to summarize middle turns while
protecting head and tail context.

Improvements over v2:
  - Structured summary template with Resolved/Pending question tracking
  - Filter-safe summarizer preamble that treats prior turns as source material
  - Historical (reference-only) section headings replace "Next Steps"/"Remaining Work" to avoid reading as active instructions
  - Clear separator when summary merges into tail message
  - Iterative summary updates (preserves info across multiple compactions)
  - Token-budget tail protection instead of fixed message count
  - Tool output pruning before LLM summarization (cheap pre-pass)
  - Scaled summary budget (proportional to compressed content)
  - Richer tool call/result detail in summarizer input
"""

import hashlib
import json
import logging
import sqlite3
import re
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, _is_connection_error, aux_interrupt_protection
from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
)
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

HISTORICAL_TASK_HEADING = "## Historical Task Snapshot"
HISTORICAL_IN_PROGRESS_HEADING = "## Historical In-Progress State"
HISTORICAL_PENDING_ASKS_HEADING = "## Historical Pending User Asks"
HISTORICAL_REMAINING_WORK_HEADING = "## Historical Remaining Work"


SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "Topic overlap with the summary does NOT mean you should resume its "
    "task: even on similar topics, the latest user message WINS. Treat ONLY "
    "the latest message as the active task and discard stale items from "
    f"'{HISTORICAL_TASK_HEADING}' / '{HISTORICAL_IN_PROGRESS_HEADING}' / "
    f"'{HISTORICAL_PENDING_ASKS_HEADING}' / "
    f"'{HISTORICAL_REMAINING_WORK_HEADING}' entirely — do not 'wrap up' or "
    "'finish' work described there unless the latest message explicitly "
    "asks for it. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# Metadata key added to context compression summary messages so that frontends
# (CLI, Desktop, gateway, TUI) can distinguish them from real assistant/user
# messages and filter or render them appropriately without content-prefix
# heuristics. See https://github.com/NousResearch/hermes-agent/issues/38389
#
# Underscore-prefixed ON PURPOSE: the wire sanitizers
# (agent/transports/chat_completions.py convert_messages and the summary-path
# mirror in agent/chat_completion_helpers.py) strip every top-level message
# key starting with "_" before the request leaves the process. Strict
# OpenAI-compatible gateways (Fireworks, Mistral, Moonshot/Kimi, opencode-go)
# reject payloads carrying unknown keys with "Extra inputs are not permitted",
# poisoning every subsequent request in the session — a bare key like
# "is_compressed_summary" would reach the wire and trip exactly that.
COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"

# Appended to every standalone summary message (and to the merged-into-tail
# prefix) so the model has an unambiguous "summary ends here" boundary.
# Without it, weak models read the verbatim "## Active Task" quote as fresh
# user input (#11475, #14521) or regurgitate an assistant-role summary as
# their own output (#33256).
_SUMMARY_END_MARKER = (
    "--- END OF CONTEXT SUMMARY — "
    "respond to the message below, not the summary above ---"
)

# Handoff prefixes that shipped in earlier releases. A summary persisted under
# one of these can be inherited into a resumed lineage (#35344); when it is
# re-normalized on re-compaction we must strip the OLD prefix too, otherwise the
# stale directive it carried (e.g. "resume exactly from Active Task") survives
# embedded in the body and keeps hijacking replies. Keep newest-first; entries
# are matched literally. Add a frozen copy here whenever SUMMARY_PREFIX changes.
_HISTORICAL_SUMMARY_PREFIXES = (
    # Carveout era (#41607/#38364/#42812): "consistent → use as background"
    # licensed stale-task resumption on topic overlap.
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "If the latest user message is consistent with the '## Active Task' "
    "section, you may use the summary as background. If the latest user "
    "message contradicts, supersedes, changes topic from, or in any way "
    "diverges from '## Active Task' / '## In Progress' / '## Pending User "
    "Asks' / '## Remaining Work', the latest message WINS — discard those "
    "stale items entirely and do not 'wrap up the old task first'. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:",
    # Pre-#35344: contained the self-contradicting "resume exactly" directive.
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:",
)

# Minimum tokens for the summary output
_MIN_SUMMARY_TOKENS = 2000
# Proportion of compressed content to allocate for summary
_SUMMARY_RATIO = 0.20
# Absolute ceiling for summary tokens (even on very large context windows)
_SUMMARY_TOKENS_CEILING = 12_000

# Placeholder used when pruning old tool results
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Chars per token rough estimate
_CHARS_PER_TOKEN = 4
# Flat token cost per attached image part.  Real cost varies by provider and
# dimensions (Anthropic ≈ width×height/750, GPT-4o up to ~1700 for
# high-detail 2048×2048, Gemini 258/tile), but 1600 is a realistic ceiling
# that keeps compression budgeting honest for multi-image conversations.
# Matches Claude Code's IMAGE_TOKEN_ESTIMATE constant.
_IMAGE_TOKEN_ESTIMATE = 1600
# Same figure expressed in the char-budget currency the rest of the
# compressor speaks in.  Used when accumulating message "content length"
# for tail-cut decisions.
_IMAGE_CHAR_EQUIVALENT = _IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600

# Hard ceiling for the deterministic summary-failure handoff.  The fallback is
# only meant to preserve continuity anchors from the dropped window, not to
# become another unbounded transcript copy after the LLM summarizer failed.
_FALLBACK_SUMMARY_MAX_CHARS = 8_000
_FALLBACK_TURN_MAX_CHARS = 700
_AUTO_FOCUS_MAX_TURNS = 3
_AUTO_FOCUS_TURN_MAX_CHARS = 260
_AUTO_FOCUS_MAX_CHARS = 700
# Keep a short run of recent messages verbatim even when the token budget is
# already exhausted.  The public ``protect_last_n`` default is intentionally
# high for small/light tails, but using all 20 as a hard floor here would bring
# back the old large-tool-output case where nothing can be compacted.
_MAX_TAIL_MESSAGE_FLOOR = 8


_PATH_MENTION_RE = re.compile(r"(?:/|~/?|[A-Za-z]:\\)[^\s`'\")\]}<>]+")

# MEDIA delivery directives must not reach the summarizer — if one leaks into
# the summary, the downstream model may re-emit it as an active directive on
# the next turn, triggering bogus attachment sends (#14665).
_MEDIA_DIRECTIVE_RE = re.compile(r"MEDIA:\S+")


def _dedupe_append(items: list[str], value: str, *, limit: int) -> None:
    value = value.strip()
    if value and value not in items and len(items) < limit:
        items.append(value)


def _extract_tool_call_name_and_args(tool_call: Any) -> tuple[str, str]:
    """Return a best-effort ``(name, arguments)`` pair for dict/object tool calls."""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        return str(fn.get("name") or "unknown"), str(fn.get("arguments") or "")

    fn = getattr(tool_call, "function", None)
    if fn is None:
        return "unknown", ""
    return str(getattr(fn, "name", None) or "unknown"), str(getattr(fn, "arguments", None) or "")


def _extract_tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "")
    return str(getattr(tool_call, "id", "") or "")


def _collect_path_mentions(text: str, relevant_files: list[str], *, limit: int = 12) -> None:
    for match in _PATH_MENTION_RE.findall(text):
        _dedupe_append(relevant_files, match.rstrip(".,:;"), limit=limit)


def _content_length_for_budget(raw_content: Any) -> int:
    """Return the effective char-length of a message's content for token budgeting.

    Plain strings: ``len(content)``. Multimodal lists: sum of text-part
    ``len(text)`` plus a flat ``_IMAGE_CHAR_EQUIVALENT`` per image part
    (``image_url`` / ``input_image`` / Anthropic-style ``image``). This
    keeps the compressor from treating a turn with 5 attached images as
    near-zero tokens just because the text part is empty.
    """
    if isinstance(raw_content, str):
        return len(raw_content)
    if not isinstance(raw_content, list):
        return len(str(raw_content or ""))

    total = 0
    for p in raw_content:
        if isinstance(p, str):
            total += len(p)
            continue
        if not isinstance(p, dict):
            total += len(str(p))
            continue
        ptype = p.get("type")
        if ptype in {"image_url", "input_image", "image"}:
            total += _IMAGE_CHAR_EQUIVALENT
        else:
            # text / input_text / tool_result-with-text / anything else with
            # a text field.  Ignore the raw base64 payload inside image_url
            # dicts — dimensions don't matter, only whether it's an image.
            total += len(p.get("text", "") or "")
    return total


def _estimate_msg_budget_tokens(msg: dict) -> int:
    """Token estimate for one message in the tail-protection budget walks.

    Counts the message content plus the **full** ``tool_call`` envelope —
    ``id``, ``type``, ``function.name`` and JSON structure — not just
    ``function.arguments``.  Counting only the arguments string undercounted
    assistant turns that fan out into parallel tool calls by 2-15x (a
    4-tool-call turn measures ~73 vs ~1,090 real tokens), so the protected
    tail overshot ``tail_token_budget`` and compression became ineffective.
    See issue #28053.
    """
    content_len = _content_length_for_budget(msg.get("content") or "")
    tokens = content_len // _CHARS_PER_TOKEN + 10  # +10 for role/key overhead
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            tokens += len(str(tc)) // _CHARS_PER_TOKEN
    return tokens


def _content_text_for_contains(content: Any) -> str:
    """Return a best-effort text view of message content.

    Used only for substring checks when we need to know whether we've already
    appended a note to a message. Keeps multimodal lists intact elsewhere.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _append_text_to_content(content: Any, text: str, *, prepend: bool = False) -> Any:
    """Append or prepend plain text to message content safely.

    Compression sometimes needs to add a note or merge a summary into an
    existing message. Message content may be plain text or a multimodal list of
    blocks, so direct string concatenation is not always safe.
    """
    if content is None:
        return text
    if isinstance(content, str):
        return text + content if prepend else content + text
    if isinstance(content, list):
        text_block = {"type": "text", "text": text}
        return [text_block, *content] if prepend else [*content, text_block]
    rendered = str(content)
    return text + rendered if prepend else rendered + text


def _strip_image_parts_from_parts(parts: Any) -> Any:
    """Strip image parts from an OpenAI-style content-parts list.

    Returns a new list with image_url / image / input_image parts replaced
    by a text placeholder, or None if the list had no images (callers
    skip the replacement in that case). Used by the compressor to prune
    old computer_use screenshots.
    """
    if not isinstance(parts, list):
        return None
    had_image = False
    out = []
    for part in parts:
        if not isinstance(part, dict):
            out.append(part)
            continue
        ptype = part.get("type")
        if ptype in {"image", "image_url", "input_image"}:
            had_image = True
            out.append({"type": "text", "text": "[screenshot removed to save context]"})
        else:
            out.append(part)
    return out if had_image else None


def _truncate_tool_call_args_json(args: str, head_chars: int = 200) -> str:
    """Shrink long string values inside a tool-call arguments JSON blob while
    preserving JSON validity.

    The ``function.arguments`` field on a tool call is a JSON-encoded string
    passed through to the LLM provider; downstream providers strictly
    validate it and return a non-retryable 400 when it is not well-formed.
    An earlier implementation sliced the raw JSON at a fixed byte offset and
    appended ``...[truncated]`` — which routinely produced strings like::

        {"path": "/foo/bar", "content": "# long markdown
        ...[truncated]

    i.e. an unterminated string and a missing closing brace. MiniMax, for
    example, rejects this with ``invalid function arguments json string``
    and the session gets stuck re-sending the same broken history on every
    turn. See issue #11762 for the observed loop.

    This helper parses the arguments, shrinks long string leaves inside the
    parsed structure, and re-serialises. Non-string values (paths, ints,
    booleans) are preserved intact. If the arguments are not valid JSON
    to begin with — some model backends use non-JSON tool arguments — the
    original string is returned unchanged rather than replaced with
    something neither we nor the backend can parse.
    """
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    shrunken = _shrink(parsed)
    # ensure_ascii=False preserves CJK/emoji instead of bloating with \uXXXX
    return json.dumps(shrunken, ensure_ascii=False)


_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _is_image_part(part: Any) -> bool:
    """True if ``part`` is a multimodal image content block.

    Recognizes all three shapes the agent handles:
      - OpenAI chat.completions: ``{"type": "image_url", "image_url": ...}``
      - OpenAI Responses API:    ``{"type": "input_image", "image_url": "..."}``
      - Anthropic native:        ``{"type": "image", "source": {...}}``
    """
    if not isinstance(part, dict):
        return False
    return part.get("type") in _IMAGE_PART_TYPES


def _content_has_images(content: Any) -> bool:
    """True if a message's ``content`` is a multimodal list with image parts."""
    if not isinstance(content, list):
        return False
    return any(_is_image_part(p) for p in content)


def _strip_images_from_content(content: Any) -> Any:
    """Return a copy of ``content`` with every image part replaced by a
    short text placeholder.

    - String content is returned unchanged.
    - Non-list, non-string content is returned unchanged.
    - List content: image parts become ``{"type": "text", "text": "[Attached
      image — stripped after compression]"}``; other parts are preserved as-is.

    Input is never mutated.
    """
    if not isinstance(content, list):
        return content
    if not any(_is_image_part(p) for p in content):
        return content

    new_parts: List[Any] = []
    for p in content:
        if _is_image_part(p):
            new_parts.append({
                "type": "text",
                "text": "[Attached image — stripped after compression]",
            })
        else:
            new_parts.append(p)
    return new_parts


def _strip_historical_media(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace image parts in older messages with placeholder text.

    The anchor is the *last* user message that has any image content. Every
    message before that anchor gets its image parts replaced with a short
    placeholder so the outgoing request stops re-shipping the same multi-MB
    base-64 image blobs on every turn.

    If no user message carries images, the list is returned unchanged.
    If the only user message with images is the very first one (nothing
    earlier to strip), the list is returned unchanged.

    Shallow copies of touched messages only; input is never mutated.
    Port of Kilo-Org/kilocode#9434 (adapted for the OpenAI-style message
    shape the hermes compressor emits).
    """
    if not messages:
        return messages

    # Find the newest user message that carries at least one image part.
    # We anchor on image-bearing user messages (not all user messages) so
    # a plain text follow-up after a big-image turn still strips the old
    # image — matching the problem kilocode#9434 set out to solve.
    anchor = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        if _content_has_images(msg.get("content")):
            anchor = i
            break

    if anchor <= 0:
        # No image-bearing user message, or it's the very first message —
        # nothing before it to strip.
        return messages

    changed = False
    result: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= anchor or not isinstance(msg, dict):
            result.append(msg)
            continue
        content = msg.get("content")
        if not _content_has_images(content):
            result.append(msg)
            continue
        new_msg = msg.copy()
        new_msg["content"] = _strip_images_from_content(content)
        result.append(new_msg)
        changed = True

    return result if changed else messages


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Create an informative 1-line summary of a tool call + result.

    Used during the pre-compression pruning pass to replace large tool
    outputs with a short but useful description of what the tool did,
    rather than a generic placeholder that carries zero information.

    Returns strings like::

        [terminal] ran `npm test` -> exit 0, 47 lines output
        [read_file] read config.py from line 1 (1,200 chars)
        [search_files] content search for 'compress' in agent/ -> 12 matches
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    if tool_name in {"browser_navigate", "browser_click", "browser_snapshot",
                     "browser_type", "browser_scroll", "browser_vision"}:
        url = args.get("url", "")
        ref = args.get("ref", "")
        detail = f" {url}" if url else (f" ref={ref}" if ref else "")
        return f"[{tool_name}]{detail} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({content_len:,} chars result)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"

    if tool_name == "execute_code":
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    if tool_name in {"skill_view", "skills_list", "skill_manage"}:
        name = args.get("name", "?")
        return f"[{tool_name}] name={name} ({content_len:,} chars)"

    if tool_name == "vision_analyze":
        question = args.get("question", "")[:50]
        return f"[vision_analyze] '{question}' ({content_len:,} chars)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    if tool_name == "clarify":
        return "[clarify] asked user a question"

    if tool_name == "text_to_speech":
        return f"[text_to_speech] generated audio ({content_len:,} chars)"

    if tool_name == "cronjob":
        action = args.get("action", "?")
        return f"[cronjob] {action}"

    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "?")
        return f"[process] {action} session={sid}"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


class ContextCompressor(ContextEngine):
    """Default context engine — compresses conversation context via lossy summarization.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Protect tail messages by token budget (most recent ~20K tokens)
      4. Summarize middle turns with structured LLM prompt
      5. On subsequent compactions, iteratively update the previous summary
    """

    @property
    def name(self) -> str:
        return "compressor"

    def on_session_reset(self) -> None:
        """Reset all per-session state for /new or /reset."""
        super().on_session_reset()
        self._context_probed = False
        self._context_probe_persistable = False
        self._previous_summary = None
        self._last_summary_error = None
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._summary_failure_cooldown_until = 0.0  # transient errors must not block a fresh session
        self._last_summary_error = None
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Clear per-session compaction state at a real session boundary.

        ``_previous_summary`` is per-session iterative-summary state. It is
        cleared on ``on_session_reset()`` (/new, /reset), but session *end*
        (CLI exit, gateway expiry, session-id rotation) goes through
        ``on_session_end()`` instead — which inherited a no-op from
        ``ContextEngine``. Without clearing here, a cron/background session's
        summary could survive on a reused compressor instance and leak into the
        next live session via the ``_generate_summary()`` iterative-update path
        (#38788). ``compress()`` already guards the leak at the point of use;
        this is defense-in-depth that drops the stale summary the moment the
        owning session ends.
        """
        self._previous_summary = None

    def bind_session_state(self, session_db: Any = None, session_id: str = "") -> None:
        """Bind the current session row so durable cooldowns can round-trip."""
        self._session_db = session_db
        self._session_id = session_id or ""
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error = None
        self.get_active_compression_failure_cooldown()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Bind session-scoped compression state for a new or resumed session."""
        super().on_session_start(session_id, **kwargs)
        self.bind_session_state(kwargs.get("session_db", getattr(self, "_session_db", None)), session_id)

    def get_active_compression_failure_cooldown(self) -> Optional[Dict[str, Any]]:
        """Return the live compression-failure cooldown for the bound session."""
        now_mono = time.monotonic()
        if self._summary_failure_cooldown_until > now_mono:
            return {
                "cooldown_until": time.time() + (
                    self._summary_failure_cooldown_until - now_mono
                ),
                "remaining_seconds": self._summary_failure_cooldown_until - now_mono,
                "error": self._last_summary_error,
            }

        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return None

        getter = getattr(session_db, "get_compression_failure_cooldown", None)
        if getter is None:
            return None
        try:
            state = getter(session_id)
        except sqlite3.Error as exc:
            logger.debug("compression failure cooldown lookup failed: %s", exc)
            return None
        except Exception:
            return None
        if not state:
            return None

        remaining_seconds = float(state.get("remaining_seconds") or 0.0)
        if remaining_seconds <= 0:
            return None

        self._summary_failure_cooldown_until = now_mono + remaining_seconds
        self._last_summary_error = state.get("error")
        return {
            "cooldown_until": float(state.get("cooldown_until") or 0.0),
            "remaining_seconds": remaining_seconds,
            "error": self._last_summary_error,
        }

    def _record_compression_failure_cooldown(
        self,
        cooldown_seconds: float,
        error: Optional[str],
    ) -> None:
        cooldown_until = time.time() + cooldown_seconds
        self._summary_failure_cooldown_until = time.monotonic() + cooldown_seconds
        self._last_summary_error = error

        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return

        recorder = getattr(session_db, "record_compression_failure_cooldown", None)
        if recorder is None:
            return
        try:
            recorder(session_id, cooldown_until, error)
        except sqlite3.Error as exc:
            logger.debug("compression failure cooldown persist failed: %s", exc)
        except Exception as exc:
            logger.debug("compression failure cooldown persist failed (non-sqlite): %s", exc)

    def _clear_compression_failure_cooldown(self) -> None:
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error = None

        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return

        clearer = getattr(session_db, "clear_compression_failure_cooldown", None)
        if clearer is None:
            return
        try:
            clearer(session_id)
        except sqlite3.Error as exc:
            logger.debug("compression failure cooldown clear failed: %s", exc)
        except Exception as exc:
            logger.debug("compression failure cooldown clear failed (non-sqlite): %s", exc)

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: Any = "",
        provider: str = "",
        api_mode: str = "",
        max_tokens: int | None = None,
    ) -> None:
        """Update model info after a model switch or fallback activation."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        # max_tokens=None here means "caller didn't specify" → keep the existing
        # output reservation. A switch that genuinely changes the output budget
        # passes the new value explicitly. (#43547)
        if max_tokens is not None:
            self.max_tokens = self._coerce_max_tokens(max_tokens)
        self.threshold_tokens = self._compute_threshold_tokens(
            context_length, self.threshold_percent, self.max_tokens,
        )
        # Recalculate token budgets for the new context length so the
        # compressor stays calibrated after a model switch (e.g. 200K → 32K).
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        # Reset cross-call calibration state captured under the PREVIOUS model.
        # These fields encode "the provider proved this prompt fit" / "preflight
        # can be deferred" decisions that are only valid for the model that
        # produced them. Carrying them across a switch to a smaller-context
        # model would let should_defer_preflight_to_real_usage() suppress a
        # preflight compression the new model actually needs — the exact
        # oversized-send-after-switch failure in #23767. The new model's first
        # response repopulates them via update_from_response(). Setting
        # last_prompt_tokens to 0 (NOT -1) is deliberate: 0 is the documented
        # "no real usage yet -> use the rough estimate" state, so the post-
        # response should_compress path falls back to estimate_request_tokens_rough
        # rather than skipping compression. -1 is a different sentinel
        # (#36718, "compression just ran, await real usage") and must not be set here.
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_real_prompt_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.last_compression_rough_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._ineffective_compression_count = 0

    # When the MINIMUM_CONTEXT_LENGTH floor meets/exceeds a small context
    # window, compacting at the percentage (50% → 32K of a 64K window) wastes
    # half the usable context. Trigger near the top of the window instead so a
    # minimum-context model uses most of its budget before compacting — same
    # rationale as the gpt-5.5/Codex 85% autoraise.
    _MIN_CTX_TRIGGER_RATIO = 0.85

    @staticmethod
    def _coerce_max_tokens(value: Any) -> int | None:
        """Normalize a max_tokens value to a positive int or None.

        Only a positive integer is a real output reservation. None (provider
        default), non-numeric values, or <= 0 all mean "no reservation" — this
        keeps the threshold arithmetic safe from non-int inputs (e.g. a test
        MagicMock reaching ContextCompressor via a mocked parent agent).
        """
        if value is None:
            return None
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    @staticmethod
    def _compute_threshold_tokens(
        context_length: int, threshold_percent: float, max_tokens: int | None = None,
    ) -> int:
        """Compute the compaction trigger threshold in tokens.

        The base value is ``effective_input_budget * threshold_percent``, floored
        at ``MINIMUM_CONTEXT_LENGTH`` so large-context models don't compress
        prematurely at 50%. BUT that floor degenerates at small windows: for a
        model whose ``context_length`` is at/below the minimum (e.g. a 64K
        local model), ``max(0.5*64000, 64000) == 64000`` makes the threshold
        equal the ENTIRE window — auto-compression can never fire because the
        provider rejects the request before usage reaches 100% (#14690).

        When the floor would meet or exceed the context window, trigger at
        ``_MIN_CTX_TRIGGER_RATIO`` (85%) of the window — high enough that a
        small model uses most of its context before compacting, but below
        100% so compaction fires before the provider rejects the request.

        The provider reserves ``max_tokens`` of output space out of the same
        window, so the usable INPUT budget is ``context_length - max_tokens``.
        With a large ``max_tokens`` (e.g. 65536 on a custom provider) the input
        budget is materially smaller than the raw window, and a threshold based
        on the full window lets the session hit a provider 400 before compaction
        fires (#43547). The percentage and the degenerate-window check below both
        operate on the effective input budget. ``max_tokens=None`` (provider
        default) conservatively assumes no reservation (full window).
        """
        effective_window = context_length - (max_tokens or 0)
        if effective_window <= 0:
            effective_window = context_length
        pct_value = int(effective_window * threshold_percent)
        floored = max(pct_value, MINIMUM_CONTEXT_LENGTH)
        # If flooring pushed the threshold to/over the effective window it can
        # never be reached. Trigger at 85% of the effective input budget so a
        # minimum-context model rides most of its budget before compacting
        # instead of wasting half.
        if effective_window > 0 and floored >= effective_window:
            return max(1, min(int(effective_window * ContextCompressor._MIN_CTX_TRIGGER_RATIO),
                              effective_window - 1))
        return floored

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
        abort_on_summary_failure: bool = False,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.quiet_mode = quiet_mode
        # Output-token reservation: the provider carves max_tokens out of the
        # context window, so the usable input budget is context_length -
        # max_tokens. None = provider default => assume no reservation. (#43547)
        # Coerce defensively: only a positive int is a real reservation; any
        # other value (None, non-numeric, <=0) means "no reservation" so the
        # threshold arithmetic never sees a non-int (e.g. a test MagicMock).
        self.max_tokens = self._coerce_max_tokens(max_tokens)
        # When True, summary-generation failure aborts compression entirely
        # (returns messages unchanged, sets _last_compress_aborted=True).
        # When False (default = historical behavior), insert a
        # deterministic "summary unavailable" handoff and drop the middle window.
        self.abort_on_summary_failure = abort_on_summary_failure

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        # Floor: never compress below MINIMUM_CONTEXT_LENGTH tokens even if
        # the percentage would suggest a lower value.  This prevents premature
        # compression on large-context models at 50% while keeping the % sane
        # for models right at the minimum. _compute_threshold_tokens also
        # guards the degenerate case where the floor would equal/exceed the
        # window (small models), so auto-compression can still fire (#14690).
        self.threshold_tokens = self._compute_threshold_tokens(
            self.context_length, threshold_percent, self.max_tokens,
        )
        self.compression_count = 0

        # Derive token budgets: ratio is relative to the threshold, not total context
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: model=%s context_length=%d "
                "threshold=%d (%.0f%%) target_ratio=%.0f%% tail_budget=%d "
                "provider=%s base_url=%s",
                model, self.context_length, self.threshold_tokens,
                threshold_percent * 100, self.summary_target_ratio * 100,
                self.tail_token_budget,
                provider or "none", base_url or "none",
            )
        self._context_probed = False  # True after a step-down from context error

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

        self.summary_model = summary_model_override or ""
        self._session_db: Any = None
        self._session_id: str = ""

        # Stores the previous compaction summary for iterative updates
        self._previous_summary: Optional[str] = None
        # Anti-thrashing: track whether last compression was effective
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        self._summary_failure_cooldown_until: float = 0.0
        self._last_summary_error: Optional[str] = None
        # When summary generation fails and a static fallback is inserted,
        # record how many turns were unrecoverably dropped so callers
        # (gateway hygiene, /compress) can surface a visible warning.
        self._last_summary_dropped_count: int = 0
        self._last_summary_fallback_used: bool = False
        # When summary generation fails we now ABORT compression entirely
        # and return the original messages unchanged instead of dropping
        # the middle window with a static placeholder.  Callers inspect
        # this flag to know "compression was attempted but aborted, freeze
        # the chat until the user manually retries via /compress".
        self._last_compress_aborted: bool = False
        # Set True when the summary call failed with an authentication /
        # permission error (HTTP 401/403). Auth failures are non-recoverable
        # at the request level — the credential or endpoint is broken — so
        # compress() must ABORT (preserve the session unchanged) rather than
        # rotate into a degraded child session with a placeholder summary.
        # This is independent of the abort_on_summary_failure config flag:
        # rotating on a broken credential is never the right behavior.
        self._last_summary_auth_failure: bool = False
        # Set when summary generation ultimately fails due to a transient
        # network/connection error (httpx/httpcore connection drop, premature
        # stream close, etc.) — distinct from auth failures but treated the
        # same way by compress(): ABORT and preserve the session unchanged
        # rather than destroy the middle window for a deterministic
        # "summary unavailable" marker. Retrying once the network recovers is
        # strictly better than discarding context for a transient blip
        # (#29559, #25585). Independent of abort_on_summary_failure.
        self._last_summary_network_failure: bool = False
        # retrying on the main model, record the failure so gateway /
        # CLI callers can still warn the user even though compression
        # succeeded.  Silent recovery would hide the broken config.
        self._last_aux_model_failure_error: Optional[str] = None
        self._last_aux_model_failure_model: Optional[str] = None

    def update_from_response(self, usage: Dict[str, Any]):
        """Update tracked token usage from API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens)
        if self.last_prompt_tokens > 0:
            self.last_real_prompt_tokens = self.last_prompt_tokens
            if self.last_prompt_tokens < self.threshold_tokens:
                if self.awaiting_real_usage_after_compression and self.last_compression_rough_tokens > 0:
                    self.last_rough_tokens_when_real_prompt_fit = self.last_compression_rough_tokens
            else:
                self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """Return True when a high rough preflight estimate is known-noisy.

        ``estimate_request_tokens_rough(..., tools=...)`` intentionally
        overestimates schema-heavy requests so Hermes compresses before a
        provider rejects the payload. After a successful compressed API call,
        though, provider ``prompt_tokens`` are a better signal than repeating
        compaction from the same rough schema overhead. Defer only while the
        rough estimate has grown modestly since a request the provider proved
        fit under the threshold.
        """
        if rough_tokens < self.threshold_tokens:
            return False
        # Immediately after a compaction the post-compression path sets
        # ``awaiting_real_usage_after_compression`` and parks
        # ``last_prompt_tokens = -1``, but ``last_real_prompt_tokens`` still
        # holds the STALE pre-compression value (above threshold — that's why
        # compaction fired).  Without this guard that stale value defeats the
        # ``last_real_prompt_tokens >= threshold_tokens`` check below, so
        # preflight fires a SECOND compaction before the provider has reported
        # real token usage for the now-shorter conversation.  Defer for exactly
        # one turn; update_from_response() clears the flag when real usage
        # arrives.  (#36718)
        if self.awaiting_real_usage_after_compression:
            return True
        if self.last_real_prompt_tokens <= 0:
            return False
        if self.last_real_prompt_tokens >= self.threshold_tokens:
            return False

        baseline = self.last_rough_tokens_when_real_prompt_fit or self.last_compression_rough_tokens
        if baseline <= 0:
            return False

        growth = max(0, rough_tokens - baseline)
        tolerated_growth = max(4096, int(self.threshold_tokens * 0.05))
        if growth > tolerated_growth:
            return False

        self.last_rough_tokens_when_real_prompt_fit = max(baseline, rough_tokens)
        return True

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Check if context exceeds the compression threshold.

        Includes anti-thrashing protection: if the last two compressions
        each saved less than 10%, skip compression to avoid infinite loops
        where each pass removes only 1-2 messages.
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        # Do not trigger compression while the summary LLM is in cooldown.
        # On a 429/transient failure _generate_summary() sets a cooldown and
        # returns None; compress() then inserts a static fallback marker and
        # returns. Tokens stay above threshold, so without this guard every
        # subsequent turn re-fires _compress_context() — re-inserting the
        # marker and re-entering the loop, making the CLI appear frozen until
        # the cooldown expires (issue #11529). Manual /compress passes
        # force=True, which clears this cooldown in compress() before running,
        # so it still retries immediately.
        _cooldown_remaining = self._summary_failure_cooldown_until - time.monotonic()
        if _cooldown_remaining > 0:
            if not self.quiet_mode:
                logger.debug(
                    "Compression deferred — summary LLM in cooldown for %.0fs more",
                    _cooldown_remaining,
                )
            return False
        # Anti-thrashing: back off if recent compressions were ineffective
        if self._ineffective_compression_count >= 2:
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped — last %d compressions saved <10%% each. "
                    "Consider /new to start a fresh session, or /compress <topic> "
                    "for focused compression.",
                    self._ineffective_compression_count,
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Tool output pruning (cheap pre-pass, no LLM call)
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Replace old tool result contents with informative 1-line summaries.

        Instead of a generic placeholder, generates a summary like::

            [terminal] ran `npm test` -> exit 0, 47 lines output
            [read_file] read config.py from line 1 (3,400 chars)

        Also deduplicates identical tool results (e.g. reading the same file
        5x keeps only the newest full copy) and truncates large tool_call
        arguments in assistant messages outside the protected tail.

        Walks backward from the end, protecting the most recent messages that
        fall within ``protect_tail_tokens`` (when provided) OR the last
        ``protect_tail_count`` messages (backward-compatible default).
        When both are given, the token budget takes priority and the message
        count acts as a hard minimum floor.

        Returns (pruned_messages, pruned_count).
        """
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0

        # Build index: tool_call_id -> (tool_name, arguments_json)
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                    else:
                        cid = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "unknown") if fn else "unknown"
                        args_str = getattr(fn, "arguments", "") if fn else ""
                        call_id_to_tool[cid] = (name, args_str)

        # Determine the prune boundary
        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            # Token-budget approach: walk backward accumulating tokens
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result))
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                msg_tokens = _estimate_msg_budget_tokens(msg)
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            # Translate the budget walk into a "protected count", apply the
            # floor in count-space (where `max` reads naturally: protect at
            # least `min_protect` messages or whatever the budget reserved,
            # whichever is more), then convert back to a prune boundary.
            # Doing this in index-space with `max` would invert the direction
            # (smaller index = MORE protected), so a generous budget would
            # silently get truncated back down to `min_protect`.
            budget_protect_count = len(result) - boundary
            protected_count = max(budget_protect_count, min_protect)
            prune_boundary = len(result) - protected_count
        else:
            prune_boundary = len(result) - protect_tail_count

        # Pass 1: Deduplicate identical tool results.
        # When the same file is read multiple times, keep only the most recent
        # full copy and replace older duplicates with a back-reference.
        content_hashes: dict = {}  # hash -> (index, tool_call_id)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            # Multimodal content — dedupe by the text summary if available.
            if isinstance(content, list):
                continue
            if not isinstance(content, str):
                # Multimodal dict envelopes ({_multimodal: True, content: [...]}) and
                # other non-string tool-result shapes can't be hashed/deduped by text.
                continue
            if len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                # This is an older duplicate — replace with back-reference
                result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
                pruned += 1
            else:
                content_hashes[h] = (i, msg.get("tool_call_id", "?"))

        # Pass 2: Replace old tool results with informative summaries
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Multimodal content (base64 screenshots etc.): strip the image
            # payload — keep a lightweight text placeholder in its place.
            # Without this, an old computer_use screenshot (~1MB base64 +
            # ~1500 real tokens) survives every compression pass forever.
            if isinstance(content, list):
                stripped = _strip_image_parts_from_parts(content)
                if stripped is not None:
                    result[i] = {**msg, "content": stripped}
                    pruned += 1
                continue
            if isinstance(content, dict) and content.get("_multimodal"):
                summary = content.get("text_summary") or "[screenshot removed to save context]"
                result[i] = {**msg, "content": f"[screenshot removed] {summary[:200]}"}
                pruned += 1
                continue
            if not isinstance(content, str):
                continue
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            # Skip already-deduplicated or previously-summarized results
            if content.startswith("[Duplicate tool output"):
                continue
            # Only prune if the content is substantial (>200 chars)
            if len(content) > 200:
                call_id = msg.get("tool_call_id", "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                summary = _summarize_tool_result(tool_name, tool_args, content)
                result[i] = {**msg, "content": summary}
                pruned += 1

        # Pass 3: Truncate large tool_call arguments in assistant messages
        # outside the protected tail. write_file with 50KB content, for
        # example, survives pruning entirely without this.
        #
        # The shrinking is done inside the parsed JSON structure so the
        # result remains valid JSON — otherwise downstream providers 400
        # on every subsequent turn until the broken call falls out of
        # the window. See ``_truncate_tool_call_args_json`` docstring.
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        new_args = _truncate_tool_call_args_json(args)
                        if new_args != args:
                            tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                            modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """Scale summary token budget with the amount of content being compressed.

        The maximum scales with the model's context window (5% of context,
        capped at ``_SUMMARY_TOKENS_CEILING``) so large-context models get
        richer summaries instead of being hard-capped at 8K tokens.
        """
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    # Truncation limits for the summarizer input.  These bound how much of
    # each message the summary model sees — the budget is the *summary*
    # model's context window, not the main model's.
    _CONTENT_MAX = 6000       # total chars per message body
    _CONTENT_HEAD = 4000      # chars kept from the start
    _CONTENT_TAIL = 1500      # chars kept from the end
    _TOOL_ARGS_MAX = 1500     # tool call argument chars
    _TOOL_ARGS_HEAD = 1200    # kept from the start of tool args

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Serialize conversation turns into labeled text for the summarizer.

        Includes tool call arguments and result content (up to
        ``_CONTENT_MAX`` chars per message) so the summarizer can preserve
        specific details like file paths, commands, and outputs.

        All content is redacted before serialization to prevent secrets
        (API keys, tokens, passwords) from leaking into the summary that
        gets sent to the auxiliary model and persisted across compactions.
        """
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = redact_sensitive_text(msg.get("content") or "")
            content = _MEDIA_DIRECTIVE_RE.sub("[media attachment]", content)

            # Tool results: keep enough content for the summarizer
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            # Assistant messages: include tool call names AND arguments
            if role == "assistant":
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = redact_sensitive_text(fn.get("arguments", ""))
                            # Truncate long arguments but keep enough for context
                            if len(args) > self._TOOL_ARGS_MAX:
                                args = args[:self._TOOL_ARGS_HEAD] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            # User and other roles
            if len(content) > self._CONTENT_MAX:
                content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    def _build_static_fallback_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        reason: str | None = None,
    ) -> str:
        """Build a deterministic handoff when the LLM summarizer is unavailable.

        This is intentionally much less rich than an LLM-written summary, but it
        is still better than a bare "N messages were removed" marker.  It keeps
        the most useful continuity anchors that can be extracted locally:
        recent user asks, assistant/tool actions, files/commands mentioned in
        tool calls, and any error text.  The result uses the normal summary
        structure so downstream prompts can recover gracefully after a provider
        outage or summary-model failure.
        """
        user_asks: list[str] = []
        assistant_actions: list[str] = []
        tool_actions: list[str] = []
        relevant_files: list[str] = []
        blockers: list[str] = []
        last_dropped_turns: list[str] = []

        def _compact_fallback_turn(value: Any) -> str:
            text = redact_sensitive_text(_content_text_for_contains(value))
            text = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b", "[REDACTED]", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > _FALLBACK_TURN_MAX_CHARS:
                text = text[: _FALLBACK_TURN_MAX_CHARS - 15].rstrip() + " ...[truncated]"
            return re.sub(r"\bgh[pousr]_[A-Za-z0-9_.-]+", "[REDACTED]", text)

        def _remember_dropped_turn(label: str, text: str, *, limit: int = 8) -> None:
            text = text.strip()
            if not text:
                return
            last_dropped_turns.append(f"{label}: {text}")
            if len(last_dropped_turns) > limit:
                del last_dropped_turns[0]

        def _collect_paths_from_jsonish(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key in {"path", "workdir", "file_path", "output_path"} and isinstance(val, str):
                        _dedupe_append(relevant_files, val, limit=12)
                    _collect_paths_from_jsonish(val)
            elif isinstance(obj, list):
                for val in obj:
                    _collect_paths_from_jsonish(val)
            elif isinstance(obj, str):
                _collect_path_mentions(obj, relevant_files)

        call_id_to_tool: dict[str, tuple[str, str]] = {}
        for msg in turns_to_summarize:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    name, raw_args = _extract_tool_call_name_and_args(tc)
                    args = redact_sensitive_text(raw_args)
                    call_id = _extract_tool_call_id(tc)
                    if call_id:
                        call_id_to_tool[call_id] = (name, args)
                    if args:
                        try:
                            parsed = json.loads(args)
                        except Exception:
                            parsed = args
                        _collect_paths_from_jsonish(parsed)

        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            text = _compact_fallback_turn(msg.get("content"))
            _collect_path_mentions(text, relevant_files)

            turn_text = text
            turn_tool_names: list[str] = []
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    name, _args = _extract_tool_call_name_and_args(tc)
                    turn_tool_names.append(name)
                if turn_tool_names:
                    prefix = "tool calls: " + ", ".join(turn_tool_names[:6])
                    turn_text = f"{prefix}; {turn_text}" if turn_text else prefix
            _remember_dropped_turn(str(role).upper(), turn_text)

            if len(text) > 600:
                text = text[:420].rstrip() + " ... " + text[-160:].lstrip()

            if role == "user" and text:
                user_asks.append(text)
            elif role == "assistant":
                tool_names: list[str] = []
                for tc in msg.get("tool_calls") or []:
                    name, _args = _extract_tool_call_name_and_args(tc)
                    tool_names.append(name)
                if tool_names:
                    assistant_actions.append(
                        "Called tool(s): " + ", ".join(tool_names[:6])
                    )
                elif text:
                    assistant_actions.append(text)
            elif role == "tool":
                call_id = str(msg.get("tool_call_id") or "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                tool_actions.append(
                    _summarize_tool_result(tool_name, tool_args, text or "")
                )
                if re.search(
                    r"\b(error|failed|exception|traceback|timeout|timed out|fatal)\b",
                    text,
                    re.I,
                ):
                    blockers.append(text[:500])

        def _bullets(items: list[str], limit: int = 8) -> str:
            unique: list[str] = []
            seen: set[str] = set()
            for item in items:
                item = item.strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                unique.append(item)
                if len(unique) >= limit:
                    break
            return "\n".join(f"- {item}" for item in unique) if unique else "None."

        completed: list[str] = []
        for idx, item in enumerate((assistant_actions + tool_actions)[:12], start=1):
            completed.append(f"{idx}. {item}")

        active_task = (
            f"User asked: {user_asks[-1]!r}"
            if user_asks
            else "Unknown from deterministic fallback."
        )
        previous_summary_note = ""
        if self._previous_summary:
            previous_summary_note = (
                "\n\nPrevious compaction summary was present and should still be treated as "
                "background continuity context, but the latest LLM summary update failed."
            )

        reason_text = f" Summary failure reason: {reason}." if reason else ""
        body = f"""{HISTORICAL_TASK_HEADING}
{active_task}

## Goal
Recovered from a deterministic fallback because the LLM context summarizer was unavailable. Continue from the protected recent messages after this summary and use current file/system state for exact details.{previous_summary_note}

## Constraints & Preferences
- This fallback was generated locally without an LLM summary call.
- Secrets and credentials were redacted before preservation.
- The summary may be incomplete; prefer verifying current files, git state, processes, and test results instead of assuming omitted details.

## Completed Actions
{chr(10).join(completed) if completed else "None recoverable from compacted turns."}

## Active State
Unknown from deterministic fallback. Inspect current repository/session state if needed.

{HISTORICAL_IN_PROGRESS_HEADING}
Unknown from deterministic fallback — the latest user ask is recorded once under
"{HISTORICAL_TASK_HEADING}" above as historical context only. Do NOT treat it as an
unfulfilled instruction to re-answer; verify current state and continue from the
protected recent messages after this summary.

## Blocked
{_bullets(blockers, limit=5)}

## Key Decisions
None recoverable from deterministic fallback.

## Resolved Questions
None recoverable from deterministic fallback.

{HISTORICAL_PENDING_ASKS_HEADING}
None recoverable from deterministic fallback. (The latest user ask is preserved once
under "{HISTORICAL_TASK_HEADING}" as historical context — it is NOT necessarily
outstanding.)

## Relevant Files
{_bullets(relevant_files, limit=12)}

{HISTORICAL_REMAINING_WORK_HEADING}
Continue from the most recent unfulfilled user ask and protected tail messages. Verify state with tools before making claims.

## Last Dropped Turns
{_bullets(last_dropped_turns, limit=8)}

## Critical Context
Summary generation was unavailable, so this is a best-effort deterministic fallback for {len(turns_to_summarize)} compacted message(s).{reason_text}"""
        summary = self._with_summary_prefix(redact_sensitive_text(body.strip()))
        if len(summary) > _FALLBACK_SUMMARY_MAX_CHARS:
            summary = summary[: _FALLBACK_SUMMARY_MAX_CHARS - 42].rstrip() + "\n...[fallback summary truncated]"
        return summary

    def _fallback_to_main_for_compression(self, e: Exception, reason: str) -> None:
        """Switch from a separate ``summary_model`` back to the main model.

        Centralises the bookkeeping shared by every fallback branch in
        :meth:`_generate_summary` (model-not-found, timeout, JSON decode,
        unknown error): record the aux-model failure for ``/usage``-style
        callers, clear the summary model so the next call uses the main one,
        and clear the cooldown so the immediate retry can run.

        ``reason`` is a short human-readable phrase ("unavailable",
        "timed out", "returned invalid JSON", "failed") that is interpolated
        into the warning log.
        """
        self._summary_model_fallen_back = True
        logger.warning(
            "Summary model '%s' %s (%s). "
            "Falling back to main model '%s' for compression.",
            self.summary_model, reason, e, self.model,
        )
        _err_text = str(e).strip() or e.__class__.__name__
        if len(_err_text) > 220:
            _err_text = _err_text[:217].rstrip() + "..."
        self._last_aux_model_failure_error = _err_text
        self._last_aux_model_failure_model = self.summary_model
        self.summary_model = ""  # empty = use main model
        self._clear_compression_failure_cooldown()  # no cooldown — retry immediately

    def _generate_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a structured summary of conversation turns.

        Uses a structured template (Goal, Progress, Decisions, Resolved/Pending
        Questions, Files, Remaining Work) with explicit preamble telling the
        summarizer not to answer questions.  When a previous summary exists,
        generates an iterative update instead of summarizing from scratch.

        Args:
            focus_topic: Optional focus string for guided compression.  When
                provided, the summariser prioritises preserving information
                related to this topic and is more aggressive about compressing
                everything else.  Inspired by Claude Code's ``/compact``.

        Returns None if all attempts fail — the caller should drop
        the middle turns without a summary rather than inject a useless
        placeholder.
        """
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug(
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - now,
            )
            return None

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)

        # Current date for temporal anchoring (see ## Temporal Anchoring below).
        # Date-only granularity matches system_prompt.py:337 (PR #20451) and the
        # user's configured timezone via hermes_time.now(). The compaction summary
        # is a mid-conversation message that is NOT part of the cached prefix, so a
        # date here never affects prompt-cache stability. Resolved defensively —
        # a clock failure must never block compaction.
        try:
            from hermes_time import now as _hermes_now

            _today_str = _hermes_now().strftime("%Y-%m-%d")
        except Exception:  # pragma: no cover - clock resolution is best-effort
            _today_str = ""

        # Preamble shared by both first-compaction and iterative-update prompts.
        # Keep the wording deliberately plain: Azure/OpenAI-compatible content
        # filters have flagged stronger "injection" / "do not respond" framing.
        _summarizer_preamble = (
            "You are a summarization agent creating a context checkpoint. "
            "Treat the conversation turns below as source material for a "
            "compact record of prior work. "
            "Produce only the structured summary; do not add a greeting, "
            "preamble, or prefix. "
            "Write the summary in the same language the user was using in the "
            "conversation — do not translate or switch to English. "
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary — replace any that appear "
            "with [REDACTED]. Note that the user had credentials present, but "
            "do not preserve their values."
        )

        # Temporal anchoring directive. Rewrites relative / still-pending-sounding
        # references into absolute, dated, past-tense facts so a resumed
        # conversation does not re-issue completed actions. Only emitted when the
        # current date resolved successfully; otherwise the rule is omitted so the
        # summarizer is never handed an empty date placeholder.
        if _today_str:
            _temporal_anchoring_rule = (
                f"\nTEMPORAL ANCHORING: The current date is {_today_str}. When an "
                "action has already been carried out, phrase it as a completed, "
                "dated, past-tense fact rather than an open instruction. For "
                'example, rewrite "email John about the proposal" as "Sent the '
                f'proposal email to John on {_today_str}." Never leave a finished '
                "action worded as if it still needs doing, and never invent a date "
                "for work that has not happened yet.\n"
            )
        else:
            _temporal_anchoring_rule = ""

        # Shared structured template (used by both paths).
        _template_sections = f"""{HISTORICAL_TASK_HEADING}
[THE SINGLE MOST IMPORTANT FIELD. Capture the user's most recent unfulfilled
input verbatim — the exact words they used. This includes:
- Explicit task assignments ("refactor the auth module")
- Questions awaiting an answer ("waarom staat X op Y?", "wat zijn de volgende stappen?")
- Decisions awaiting input ("optie A of B?")
- Ongoing discussions where the assistant owes the next substantive reply
A conversation where the user just asked a question IS an active task — the
task is "answer that question with full context". Do NOT write "None" merely
because the user did not issue an imperative command; reserve "None" for the
rare case where the last exchange was fully resolved and the user said
something like "thanks, that's all".
If multiple items are outstanding, list only the ones NOT yet completed.
Continuation should pick up exactly here. Examples:
"User asked: 'Now refactor the auth module to use JWT instead of sessions'"
"User asked: 'Waarom stond provider ineens op openrouter?' — needs investigation + answer"
"User chose option A; awaiting implementation of step 2"
If the user's most recent message was a reverse signal (stop, undo, roll
back, never mind, just verify, change of topic) that supersedes earlier
work, write the reverse signal verbatim and DO NOT carry forward the
cancelled task. Example: "User asked: 'Stop the i18n refactor and just
verify the current diff' — earlier i18n in-flight work is cancelled."
If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

{HISTORICAL_IN_PROGRESS_HEADING}
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so it is not repeated]

{HISTORICAL_PENDING_ASKS_HEADING}
[Questions or requests from the user that have NOT yet been answered or fulfilled. These are STALE — they were from the compacted turns. Write them here for reference only. The agent must NOT act on them unless the latest user message explicitly requests it. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

{HISTORICAL_REMAINING_WORK_HEADING}
[What remains to be done — framed as STALE context for reference only. The agent must NOT resume this work unless the latest user message explicitly asks for it.]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.
{_temporal_anchoring_rule}
Write only the summary body. Do not include any preamble or prefix."""

        if self._previous_summary:
            # Iterative update: preserve existing info, add new progress
            prompt = f"""{_summarizer_preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete. CRITICAL: Update "## Active Task" to reflect the user's most recent unfulfilled input — this includes any question, decision request, or discussion turn that the assistant has not yet answered. Only write "None" if the last exchange was fully resolved.

{_template_sections}"""
        else:
            # First compaction: summarize from scratch
            prompt = f"""{_summarizer_preamble}

Create a structured checkpoint summary for the conversation after earlier turns are compacted. The summary should preserve enough detail for continuity without re-reading the original turns.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{_template_sections}"""

        # Inject focus topic guidance when the user provides one via /compress <focus>.
        # This goes at the end of the prompt so it takes precedence.
        if focus_topic:
            prompt += f"""

FOCUS TOPIC: "{focus_topic}"
This compaction should PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED]."""

        try:
            call_kwargs = {
                "task": "compression",
                "main_runtime": {
                    "model": self.model,
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "api_mode": self.api_mode,
                },
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(summary_budget * 1.3),
                # timeout resolved from auxiliary.compression.timeout config by call_llm
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            # Compression is atomic: protect the in-flight summary call from a
            # mid-turn gateway interrupt. Without this, an incoming user message
            # aborts the summary and compression falls back to a degraded static
            # marker, losing the real handoff (#23975). Re-entrant: a main-model
            # retry (_generate_summary recursion) re-enters harmlessly.
            with aux_interrupt_protection():
                response = call_llm(**call_kwargs)
            # ``_validate_llm_response`` only guarantees ``choices[0].message``
            # exists, not that it's an object with ``.content``. Some
            # OpenAI-compatible proxies / local backends return a dict- or
            # str-shaped message; coerce defensively instead of crashing.
            message = response.choices[0].message
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", message)
            # Handle cases where content is not a string (e.g., dict from llama.cpp)
            if not isinstance(content, str):
                content = str(content) if content else ""
            # Some OpenAI-compatible proxies (e.g. cmkey.cn, one-api channels)
            # return a well-formed HTTP 200 with an empty or whitespace-only
            # ``content`` instead of an error or empty ``choices``. That payload
            # passes ``_validate_llm_response`` (a ``message`` exists), so it
            # reaches here and would otherwise be stored as a prefix-only
            # summary with no body — silently wiping the compacted turns and
            # making the model forget the in-progress task (#11978, #11914).
            # Treat empty content as a failure so it routes through the same
            # main-model fallback + cooldown machinery as a transport error,
            # rather than replacing real context with an empty summary.
            if not content.strip():
                raise RuntimeError(
                    "Context compression LLM returned empty content "
                    f"(provider={self.provider or 'auto'} "
                    f"model={self.summary_model or self.model})"
                )
            # Redact the summary output as well — the summarizer LLM may
            # ignore prompt instructions and echo back secrets verbatim.
            summary = redact_sensitive_text(content.strip())
            # Store for iterative updates on next compaction
            self._previous_summary = summary
            self._clear_compression_failure_cooldown()
            self._summary_model_fallen_back = False
            self._last_summary_error = None
            self._last_summary_auth_failure = False
            self._last_summary_network_failure = False
            return self._with_summary_prefix(summary)
        except Exception as e:
            # ``call_llm`` raises ``RuntimeError`` for two very different cases:
            #   1. No provider configured ("No LLM provider configured ...") —
            #      a permanent misconfiguration, long cooldown is correct.
            #   2. An empty/invalid response from a configured provider
            #      (``_validate_llm_response`` empty-``choices``/``None``, or our
            #      empty-``content`` guard above) — a transient/proxy fault that
            #      should fall back to the main model first, exactly like the
            #      transport errors handled below.
            # Only (1) belongs in the long no-provider cooldown; (2) and every
            # other exception flow into the generic fallback logic so they get
            # a main-model retry before any cooldown. (#11978, #11914)
            if isinstance(e, RuntimeError) and "no llm provider configured" in str(e).lower():
                # No provider configured — long cooldown, unlikely to self-resolve
                self._record_compression_failure_cooldown(
                    _SUMMARY_FAILURE_COOLDOWN_SECONDS,
                    "no auxiliary LLM provider configured",
                )
                self._last_summary_error = "no auxiliary LLM provider configured"
                logger.warning("Context compression: no provider available for "
                                "summary. Middle turns will be dropped without summary "
                                "for %d seconds.",
                                _SUMMARY_FAILURE_COOLDOWN_SECONDS)
                return None
            # If the summary model is different from the main model and the
            # error looks permanent (model not found, 503, 404), fall back to
            # using the main model instead of entering cooldown that leaves
            # context growing unbounded.  (#8620 sub-issue 4)
            _status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            _err_str = str(e).lower()
            _is_model_not_found = (
                _status in {404, 503}
                or "model_not_found" in _err_str
                or "does not exist" in _err_str
                or "no available channel" in _err_str
            )
            _is_timeout = (
                _status in {408, 429, 502, 504}
                or "timeout" in _err_str
            )
            # Non-JSON / malformed-body responses from misconfigured providers
            # or proxies (e.g. an HTML 502 page returned with
            # ``Content-Type: application/json``) bubble up as
            # ``json.JSONDecodeError`` from the OpenAI SDK's ``response.json()``,
            # or as a wrapping ``APIResponseValidationError`` whose message
            # carries the substring "expecting value".  Treat these like a
            # transient provider failure: one retry on the main model, then a
            # short cooldown.  Issue #22244.
            _is_json_decode = (
                isinstance(e, json.JSONDecodeError)
                or "expecting value" in _err_str
            )
            # httpcore / httpx streaming premature-close errors surface as
            # ConnectionError subclasses or plain Exception with characteristic
            # substrings ("incomplete chunked read", "peer closed connection",
            # "response ended prematurely", "unexpected eof").  These are
            # transient network events; treat them like a timeout so we fall
            # back to the main model instead of entering a 60-second cooldown.
            # See issue #18458.
            _is_streaming_closed = _is_connection_error(e)
            # Authentication / permission failures (401/403) are NOT transient
            # and NOT fixable by retrying the same request: the credential is
            # invalid/blocked/expired or the endpoint is wrong (e.g. a prod
            # token sent to a staging inference URL). Flag them so compress()
            # aborts and preserves the session instead of rotating into a
            # degraded child with a placeholder summary. We still allow the
            # one-shot fallback to the MAIN model below when the failure came
            # from a distinct auxiliary summary_model (its dedicated creds may
            # be the only broken thing); only a failure on the main model — or
            # a fallback that also auth-fails — makes the abort stick.
            _is_auth_error = (
                _status in {401, 403}
                or "invalid api key" in _err_str
                or "invalid x-api-key" in _err_str
                or ("api key" in _err_str and ("invalid" in _err_str or "blocked" in _err_str))
                or "unauthorized" in _err_str
                or "authentication" in _err_str
            )
            if _is_auth_error:
                self._last_summary_auth_failure = True
            if _is_json_decode and not _is_model_not_found and not _is_timeout:
                logger.error(
                    "Context compression failed: auxiliary LLM returned a "
                    "non-JSON response. provider=%s summary_model=%s "
                    "main_model=%s base_url=%s err=%s",
                    self.provider or "auto",
                    self.summary_model or "(main)",
                    self.model,
                    self.base_url or "default",
                    e,
                )
            if (
                (_is_model_not_found or _is_timeout or _is_json_decode or _is_streaming_closed)
                and self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                if _is_json_decode:
                    _reason = "returned invalid JSON"
                elif _is_model_not_found:
                    _reason = "unavailable"
                elif _is_streaming_closed:
                    _reason = "closed stream prematurely"
                else:
                    _reason = "timed out"
                self._fallback_to_main_for_compression(e, _reason)
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)  # retry immediately

            # Unknown-error best-effort retry on main model.  Losing N turns of
            # context is almost always worse than one extra summary attempt, so
            # if we haven't already fallen back and the summary model differs
            # from the main model, try once more on main before entering
            # cooldown.  Errors that DID match _is_model_not_found above are
            # already handled by the fast-path retry; this branch catches
            # everything else (400s, provider-specific "no route" strings,
            # aggregator rejections, etc.) where auto-retry is still safer
            # than dropping the turns.
            if (
                self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                self._fallback_to_main_for_compression(e, "failed")
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

            # Transient errors (timeout, rate limit, network, JSON decode,
            # streaming premature-close) — shorter cooldown for JSON decode and
            # streaming-closed since those conditions can self-resolve quickly.
            _transient_cooldown = 30 if (_is_json_decode or _is_streaming_closed) else 60
            err_text = str(e).strip() or e.__class__.__name__
            if len(err_text) > 220:
                err_text = err_text[:217].rstrip() + "..."
            self._record_compression_failure_cooldown(_transient_cooldown, err_text)
            self._last_summary_error = err_text
            # A terminal connection/network failure (we reach this branch only
            # after any main-model fallback has already been tried or is
            # unavailable). Flag it so compress() ABORTS and preserves the
            # session unchanged instead of destroying the middle window for a
            # placeholder marker — retrying once the network recovers is
            # strictly better than dropping context (#29559, #25585). Mirrors
            # the auth-failure carve-out; independent of abort_on_summary_failure.
            if _is_streaming_closed:
                self._last_summary_network_failure = True
            logger.warning(
                "Failed to generate context summary: %s. "
                "Further summary attempts paused for %d seconds.",
                e,
                _transient_cooldown,
            )
            return None

    @staticmethod
    def _strip_summary_prefix(summary: str) -> str:
        """Return summary body without the current, legacy, or any historical
        handoff prefix.

        Historical prefixes must be stripped too: a handoff persisted under an
        older prefix can be inherited into a resumed lineage (#35344), and if we
        only re-prepend the current prefix without removing the old one, the
        stale directive it carried stays embedded in the body.
        """
        text = (summary or "").strip()
        for prefix in (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX, *_HISTORICAL_SUMMARY_PREFIXES):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        # Strip the trailing end marker too — a rehydrated handoff body that
        # keeps it would leak the boundary directive into the iterative-update
        # summarizer prompt (and the marker is re-appended on insertion anyway).
        if text.endswith(_SUMMARY_END_MARKER):
            text = text[: -len(_SUMMARY_END_MARKER)].rstrip()
        return text

    @classmethod
    def _with_summary_prefix(cls, summary: str) -> str:
        """Normalize summary text to the current compaction handoff format."""
        text = cls._strip_summary_prefix(summary)
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        text = _content_text_for_contains(content).lstrip()
        if text.startswith(SUMMARY_PREFIX) or text.startswith(LEGACY_SUMMARY_PREFIX):
            return True
        return any(text.startswith(p) for p in _HISTORICAL_SUMMARY_PREFIXES)

    @staticmethod
    def _has_compressed_summary_metadata(message: Any) -> bool:
        """Return True if *message* carries the compressed-summary flag.

        Callers (frontends, CLI, gateway) can use this to distinguish context
        compaction summaries from real assistant or user messages without
        relying on content-prefix heuristics.  The flag is in-process only —
        the wire sanitizers strip underscore-prefixed keys before API calls.
        """
        if not isinstance(message, dict):
            return False
        return bool(message.get(COMPRESSED_SUMMARY_METADATA_KEY))

    @classmethod
    def _derive_auto_focus_topic(
        cls,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Infer a compact focus hint from the most recent real user turns."""
        candidates: list[str] = []
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if cls._is_context_summary_content(content):
                continue
            text = redact_sensitive_text(_content_text_for_contains(content).strip())
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) > _AUTO_FOCUS_TURN_MAX_CHARS:
                text = text[: _AUTO_FOCUS_TURN_MAX_CHARS - 1].rstrip() + "…"
            candidates.append(text)
            if len(candidates) >= _AUTO_FOCUS_MAX_TURNS:
                break

        if not candidates:
            return None

        candidates.reverse()
        focus = "Recent user focus:\n" + "\n".join(f"- {item}" for item in candidates)
        if len(focus) > _AUTO_FOCUS_MAX_CHARS:
            focus = focus[: _AUTO_FOCUS_MAX_CHARS - 1].rstrip() + "…"
        return focus

    @classmethod
    def _find_latest_context_summary(
        cls,
        messages: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> tuple[Optional[int], str]:
        """Find the newest handoff summary inside a compression window."""
        for idx in range(end - 1, start - 1, -1):
            content = messages[idx].get("content")
            if cls._is_context_summary_content(content):
                return idx, cls._strip_summary_prefix(_content_text_for_contains(content))
        return None, ""

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("call_id", "") or tc.get("id", "") or ""
        return getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs after compression.

        Two failure modes:
        1. A tool *result* references a call_id whose assistant tool_call was
           removed (summarized/truncated).  The API rejects this with
           "No tool call found for function call output with call_id ...".
        2. An assistant message has tool_calls whose results were dropped.
           The API rejects this because every tool_call must be followed by
           a tool result with the matching call_id.

        This method removes orphaned results and strips orphaned tool_calls
        from assistant messages so the message list is always well-formed.

        Previous approach inserted stub ``role="tool"`` results for orphaned
        tool_calls.  That caused a secondary failure: the pre-API
        ``repair_message_sequence()`` uses ``tc.get("id")`` to track known
        call IDs while this sanitizer uses ``call_id || id``.  When the two
        disagree (Codex Responses API format: ``id != call_id``), stubs get
        silently dropped by the repair pass, re-exposing the original orphans.
        Stripping at the source avoids this entire class of mismatch.
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Remove tool results whose call_id has no matching assistant tool_call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

        # 2. Strip orphaned tool_calls from assistant messages whose results
        #    were dropped.  Stripping is preferred over inserting stub results
        #    because stubs can be dropped by downstream repair_message_sequence
        #    when call_id != id (Codex Responses API format), re-exposing orphans.
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                tcs = msg.get("tool_calls")
                if not tcs:
                    continue
                kept = [tc for tc in tcs if self._get_tool_call_id(tc) not in missing_results]
                if len(kept) != len(tcs):
                    if kept:
                        msg["tool_calls"] = kept
                    else:
                        msg.pop("tool_calls", None)
                        # Ensure the assistant message still has visible
                        # content so the API does not reject an empty turn.
                        content = msg.get("content")
                        if not content or (isinstance(content, str) and not content.strip()):
                            msg["content"] = "(tool call removed)"
            if not self.quiet_mode:
                logger.info(
                    "Compression sanitizer: stripped %d orphaned tool_call(s) from assistant messages",
                    len(missing_results),
                )

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Push a compress-start boundary forward past any orphan tool results.

        If ``messages[idx]`` is a tool result, slide forward until we hit a
        non-tool message so we don't start the summarised region mid-group.
        """
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _effective_protect_first_n(self) -> int:
        """``protect_first_n`` decayed across compression cycles.

        ``protect_first_n`` keeps the first N non-system messages verbatim so
        the original task framing survives the FIRST compaction. But applying
        it on every subsequent pass fossilizes those early turns — they're
        re-copied into each child session and never summarized away, so old
        user messages become immortal and grow the head unboundedly across a
        long session (#11996). Once the session has been compressed at least
        once, the early turns are already captured in the handoff summary, so
        there's no need to keep re-protecting them: decay to 0 (the system
        prompt is still always protected separately by _protect_head_size).
        """
        if self.compression_count >= 1 or self._previous_summary:
            return 0
        return self.protect_first_n

    def _protect_head_size(self, messages: List[Dict[str, Any]]) -> int:
        """Total count of head messages to protect.

        ``protect_first_n`` is defined as *additional* messages protected
        beyond the system prompt.  The system prompt (if present at index 0)
        is always implicitly protected — it's load-bearing context that
        must never be summarised away.  This keeps semantics stable across
        call paths where the system prompt may or may not be included in
        the ``messages`` list (e.g. the gateway ``/compress`` handler
        strips it before calling compress()).

        The ``protect_first_n`` portion DECAYS after the first compression
        (see _effective_protect_first_n) so early user turns don't fossilize
        across repeated compactions (#11996).

        Examples (first compaction):
          protect_first_n=0 → system prompt only (or nothing if no system msg)
          protect_first_n=3 → system + first 3 non-system messages
        After the first compaction: system prompt only.
        """
        head = 0
        if messages and messages[0].get("role") == "system":
            head = 1
        return head + self._effective_protect_first_n()

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Pull a compress-end boundary backward to avoid splitting a
        tool_call / result group.

        If the boundary falls in the middle of a tool-result group (i.e.
        there are consecutive tool messages before ``idx``), walk backward
        past all of them to find the parent assistant message.  If found,
        move the boundary before the assistant so the entire
        assistant + tool_results group is included in the summarised region
        rather than being split (which causes silent data loss when
        ``_sanitize_tool_pairs`` removes the orphaned tail results).
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        # Walk backward past consecutive tool results
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        # If we landed on the parent assistant with tool_calls, pull the
        # boundary before it so the whole group gets summarised together.
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            idx = check
        return idx

    # ------------------------------------------------------------------
    # Tail protection by token budget
    # ------------------------------------------------------------------

    def _find_last_user_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Return the index of the last user-role message at or after *head_end*, or -1.

        A context-compaction handoff banner can be inserted as a ``role="user"``
        message (see the summary-role selection in ``compress``). It is internal
        continuity state, not a real user turn, so it must not be picked as the
        tail anchor — otherwise ``_ensure_last_user_message_in_tail`` protects
        the summary and rolls the genuine last user message into the next
        compaction, re-triggering the active-task loss the anchor exists to
        prevent.
        """
        for i in range(len(messages) - 1, head_end - 1, -1):
            msg = messages[i]
            if msg.get("role") == "user" and not self._is_context_summary_content(
                msg.get("content")
            ):
                return i
        return -1

    def _find_last_assistant_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Return the index of the last user-visible assistant reply at or
        after *head_end*, or -1.

        A "user-visible reply" is an assistant message with non-empty
        textual content — i.e. one that the WebUI / TUI / SessionsPage
        rendered as a bubble the operator could read. We deliberately
        skip assistant messages that contain only ``tool_calls`` (and
        no text), because those render as small "calling tool X"
        indicators and aren't what the reporter means by "the output
        of the last message you sent" (#29824).

        Falling back to the most recent assistant message of ANY kind
        only kicks in when no content-bearing assistant message exists
        in the compressible region — typically a fresh session that
        just started a multi-step tool sequence with no prior reply
        to anchor. In that case the agent fix is a no-op and the
        existing user-message anchor carries the load.
        """
        last_any = -1
        for i in range(len(messages) - 1, head_end - 1, -1):
            msg = messages[i]
            if msg.get("role") != "assistant":
                continue
            if last_any < 0:
                last_any = i
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return i
            if isinstance(content, list):
                # Multimodal / Anthropic-style content: look for any
                # text block with non-empty text.
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content")
                        if isinstance(text, str) and text.strip():
                            return i
        return last_any

    def _ensure_last_assistant_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent assistant message is in the protected tail.

        WebUI / TUI / SessionsPage bug (#29824). Without this anchor,
        ``_find_tail_cut_by_tokens`` can leave the user's most recent
        visible assistant response inside the compressed middle region —
        especially when the conversation has a single oversized tool
        result or a long stretch of tool-call/result pairs after the
        last assistant reply. The summariser then rolls that reply up
        into the single ``[CONTEXT COMPACTION — REFERENCE ONLY]`` block
        persisted as ``role="user"`` or ``role="assistant"``. From the
        operator's perspective the WebUI session viewer
        (``web/src/pages/SessionsPage.tsx``) and the TUI chat panel
        both suddenly show the opaque "Context compaction" block in the
        slot where they were just reading the assistant's actual reply:

            User:       "i cant see the output of the last message you
                         sent, i did see it previously, however now see
                         'context compaction'"

        Mirror of ``_ensure_last_user_message_in_tail`` but anchors on
        the last assistant-role message. Re-runs the tool-group
        alignment so we don't split a ``tool_call`` / ``tool_result``
        group that immediately precedes the anchored message — orphaned
        tool messages would otherwise be removed by
        ``_sanitize_tool_pairs`` and trigger the same data-loss symptom
        we're trying to prevent.
        """
        last_asst_idx = self._find_last_assistant_message_idx(messages, head_end)
        if last_asst_idx < 0:
            # No assistant message in the compressible region — nothing
            # to anchor (single-turn pre-reply state, etc.).
            return cut_idx
        if last_asst_idx >= cut_idx:
            # Already in the tail — the token-budget walk did the right
            # thing on its own.
            return cut_idx
        # Pull cut_idx back to the assistant message, then re-align so
        # we don't split a tool group that immediately precedes it
        # (e.g. an ``assistant(tool_calls)`` → ``tool(result)`` →
        # ``assistant(final reply)`` sequence would otherwise leave the
        # ``tool`` orphan when cut lands at the final reply).
        new_cut = self._align_boundary_backward(messages, last_asst_idx)
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last assistant message at index %d "
                "(was %d, aligned to %d) to keep the previously-visible "
                "reply out of the compaction summary (#29824)",
                last_asst_idx, cut_idx, new_cut,
            )
        # Safety: never go back into the head region.
        return max(new_cut, head_end + 1)

    def _ensure_last_user_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent user message is in the protected tail.

        Context compressor bug (#10896): ``_align_boundary_backward`` can pull
        ``cut_idx`` past a user message when it tries to keep tool_call/result
        groups together.  If the last user message ends up in the *compressed*
        middle region the LLM summariser writes it into "Historical Pending User Asks",
        but ``SUMMARY_PREFIX`` tells the next model to respond only to user
        messages *after* the summary — so the task effectively disappears from
        the active context, causing the agent to stall, repeat completed work,
        or silently drop the user's latest request.

        Fix: if the last user-role message is not already in the tail
        (``messages[cut_idx:]``), walk ``cut_idx`` back to include it.  We
        then re-align backward one more time to avoid splitting any
        tool_call/result group that immediately precedes the user message.

        Causal Coupling guard (#22523): the final ``max(last_user_idx,
        head_end + 1)`` clamp can push the cut *past* the user message when
        the user sits at ``head_end`` (the first compressible index) — the
        only case where ``head_end + 1 > last_user_idx``.  That splits the
        turn-pair: the user lands in the compressed region without its
        assistant reply, so the summariser records it as a pending ask and
        the next session re-executes the already-completed task.  When this
        split is unavoidable, push the cut *forward* to ``pair_end`` so the
        full pair (user + reply + tool results) is summarised together and
        correctly marked as completed.
        """
        last_user_idx = self._find_last_user_message_idx(messages, head_end)
        if last_user_idx < 0:
            # No user message found beyond head — nothing to anchor.
            return cut_idx

        if last_user_idx >= cut_idx:
            # Already in the tail; nothing to do.
            return cut_idx

        # The last user message is in the middle (compressed) region.
        # Pull cut_idx back to it directly — a user message is already a
        # clean boundary (no tool_call/result splitting risk), so there is no
        # need to call _align_boundary_backward here; doing so would
        # unnecessarily pull the cut further back into the preceding
        # assistant + tool_calls group.
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last user message at index %d "
                "(was %d) to prevent active-task loss after compression",
                last_user_idx,
                cut_idx,
            )
        # Safety: never go back into the head region.
        adjusted = max(last_user_idx, head_end + 1)
        if adjusted > last_user_idx:
            # The clamp would leave the user in the compressed region without
            # its reply.  Keep the pair intact by pushing the cut forward past
            # the whole (user + assistant + tool results) turn-pair so it is
            # summarised as a completed unit rather than a dangling ask.
            pair_end = self._find_turn_pair_end(messages, last_user_idx)
            if not self.quiet_mode:
                logger.debug(
                    "Causal Coupling: cut would split turn-pair at user %d; "
                    "pushing cut forward to pair_end %d so the completed pair "
                    "is summarised together (#22523)",
                    last_user_idx,
                    pair_end,
                )
            return max(pair_end, head_end + 1)
        return adjusted

    def _find_turn_pair_end(
        self,
        messages: List[Dict[str, Any]],
        user_idx: int,
    ) -> int:
        """Return the index *after* the complete turn-pair starting at *user_idx*.

        A turn-pair is: ``user`` -> ``assistant`` [-> zero-or-more ``tool``
        results].  Returns the index of the first message that does *not*
        belong to the pair, i.e. the natural cut point that keeps the pair
        intact on one side of the boundary.

        If *user_idx* is the last message (no assistant reply yet), returns
        ``user_idx + 1`` so the user message itself is minimally covered.
        """
        n = len(messages)
        idx = user_idx + 1
        if idx >= n:
            return idx  # user is the very last message — no reply yet
        if messages[idx].get("role") != "assistant":
            return idx  # no assistant reply immediately following
        idx += 1
        # Include any tool results that belong to this assistant turn.
        while idx < n and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """Walk backward from the end of messages, accumulating tokens until
        the budget is reached. Returns the index where the tail starts.

        ``token_budget`` defaults to ``self.tail_token_budget`` which is
        derived from ``summary_target_ratio * context_length``, so it
        scales automatically with the model's context window.

        Token budget is the primary criterion.  A bounded message-count floor
        keeps a short run of recent turns verbatim even when the budget is
        exhausted, but the budget is allowed to exceed by up to 1.5x to avoid
        cutting inside an oversized message (tool output, file read, etc.). If
        even that floor exceeds 1.5x the budget, the cut is placed right after
        the head so compression still runs.

        Never cuts inside a tool_call/result group.  Always ensures the most
        recent user message is in the tail (see ``_ensure_last_user_message_in_tail``).
        """
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        # Hard minimum: always keep a bounded recent-message floor in the tail.
        # ``protect_last_n`` remains a minimum up to the cap; the cap avoids
        # preserving a whole run of bulky tool outputs on every compaction.
        available_tail = max(0, n - head_end - 1)
        min_tail_floor = max(3, min(self.protect_last_n, _MAX_TAIL_MESSAGE_FLOOR))
        # Leave at least two non-head messages available to summarize on short
        # transcripts; otherwise compression can replace a tiny middle with a
        # summary and save no messages at all.
        compressible_tail_cap = max(3, available_tail - 2)
        min_tail = (
            min(min_tail_floor, compressible_tail_cap, available_tail)
            if available_tail > 1 else 0
        )
        soft_ceiling = int(token_budget * 1.5)
        accumulated = 0
        cut_idx = n  # start from beyond the end

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            msg_tokens = _estimate_msg_budget_tokens(msg)
            # Stop once we exceed the soft ceiling (unless we haven't hit min_tail yet)
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        # If the backward walk never broke early because the entire transcript
        # fits within soft_ceiling, accumulated now holds the total transcript
        # size.  Without intervention _ensure_last_user_message_in_tail pushes
        # cut_idx forward to include the last user message, and the caller's
        # compress_start >= compress_end guard either returns unchanged (no-op)
        # or compresses a single message — both of which trigger the infinite
        # compaction loop described in #40803.
        #
        # Fix: when the whole transcript fits in soft_ceiling, compute a
        # meaningful cut point using the raw (non-inflated) budget so that
        # compression actually summarizes a worthwhile middle section.
        if cut_idx <= head_end and accumulated <= soft_ceiling and accumulated > 0:
            # The entire compressable region fits in the soft ceiling.
            # Re-walk with the raw budget (no 1.5x multiplier) to find a
            # split that gives the summarizer something useful.
            raw_budget = token_budget
            raw_accumulated = 0
            for j in range(n - 1, head_end - 1, -1):
                raw_msg = messages[j]
                raw_tok = _estimate_msg_budget_tokens(raw_msg)
                if raw_accumulated + raw_tok > raw_budget and (n - j) >= min_tail:
                    cut_idx = j
                    break
                raw_accumulated += raw_tok
                cut_idx = j
            # If the raw-budget walk also consumed everything (very small
            # transcript), fall through — the existing fallback logic below
            # will still force a minimal cut after head_end.

        # Ensure we protect at least min_tail messages
        fallback_cut = n - min_tail
        cut_idx = min(cut_idx, fallback_cut)

        # If the token budget would protect everything (small conversations),
        # force a cut after the head so compression can still remove middle turns.
        if cut_idx <= head_end:
            cut_idx = max(fallback_cut, head_end + 1)

        # Align to avoid splitting tool groups
        cut_idx = self._align_boundary_backward(messages, cut_idx)

        # Ensure the most recent user message is always in the tail so the
        # active task is never lost to compression (fixes #10896).
        cut_idx = self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)

        # Ensure the most recent assistant message is always in the tail
        # so the previously-visible reply isn't silently rolled into the
        # ``[CONTEXT COMPACTION — REFERENCE ONLY]`` block (fixes #29824).
        # Each anchor only walks ``cut_idx`` backward, so chaining them is
        # monotonic — the tail can only grow, never shrink.
        cut_idx = self._ensure_last_assistant_message_in_tail(messages, cut_idx, head_end)

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # ContextEngine: manual /compress preflight
    # ------------------------------------------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Return True if there is a non-empty middle region to compact.

        Overrides the ABC default so the gateway ``/compress`` guard can
        skip the LLM call when the transcript is still entirely inside
        the protected head/tail.
        """
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        return compress_start < compress_end

    # ------------------------------------------------------------------
    # Main compression entry point
    # ------------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None, force: bool = False) -> List[Dict[str, Any]]:
        """Compress conversation messages by summarizing middle turns.

        Algorithm:
          1. Prune old tool results (cheap pre-pass, no LLM call)
          2. Protect head messages (system prompt + first exchange)
          3. Find tail boundary by token budget (~20K tokens of recent context)
          4. Summarize middle turns with structured LLM prompt
          5. On re-compression, iteratively update the previous summary

        After compression, orphaned tool_call / tool_result pairs are cleaned
        up so the API never receives mismatched IDs.

        Args:
            focus_topic: Optional focus string for guided compression.  When
                provided, the summariser will prioritise preserving information
                related to this topic and be more aggressive about compressing
                everything else.  Inspired by Claude Code's ``/compact``.
            force: If True, clear any active summary-failure cooldown before
                running so a manual ``/compress`` can retry immediately after
                an auto-compression abort.  Auto-compress callers pass False.
        """
        # Reset per-call summary failure state — callers inspect these fields
        # after compress() returns to decide whether to surface a warning.
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error = None
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_compress_aborted = False
        self._last_summary_auth_failure = False
        self._last_summary_network_failure = False

        # Manual /compress (force=True) bypasses the failure cooldown so the
        # user can retry immediately after an auto-compress abort.  Without
        # this, /compress would silently no-op for 30-60s after a failure.
        if force:
            self._clear_compression_failure_cooldown()
        n_messages = len(messages)
        # Only need head + 3 tail messages minimum (token budget decides the real tail size)
        _min_for_compress = self._protect_head_size(messages) + 3 + 1
        if n_messages <= _min_for_compress:
            if not self.quiet_mode:
                logger.warning(
                    "Cannot compress: only %d messages (need > %d)",
                    n_messages, _min_for_compress,
                )
            return messages

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: Prune old tool results (cheap, no LLM call)
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2: Determine boundaries
        compress_start = self._protect_head_size(messages)
        compress_start = self._align_boundary_forward(messages, compress_start)

        # Use token-budget tail protection instead of fixed message count
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            # No compressable window — the entire transcript fits within
            # the tail budget (soft_ceiling).  Without recording this as
            # an ineffective compression the anti-thrashing guard in
            # should_compress() never fires and every subsequent turn
            # re-triggers a no-op compression loop.  (#40803)
            self._ineffective_compression_count += 1
            self._last_compression_savings_pct = 0.0
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped: compress_start (%d) >= compress_end (%d) "
                    "— transcript fits within tail budget, nothing to compress. "
                    "ineffective_compression_count=%d",
                    compress_start, compress_end,
                    self._ineffective_compression_count,
                )
            return messages

        turns_to_summarize = messages[compress_start:compress_end]
        # A persisted handoff summary can sit in the protected head after a
        # resume (commonly immediately after the system prompt). Search from
        # the first non-system message through the compression window so we can
        # rehydrate iterative-summary state without serializing that handoff as
        # a new turn. Protected messages after the handoff remain live context,
        # so only summarize messages that are both after the handoff and inside
        # the current compression window.
        summary_search_start = 1 if messages and messages[0].get("role") == "system" else 0
        summary_idx, summary_body = self._find_latest_context_summary(
            messages,
            summary_search_start,
            compress_end,
        )
        if summary_idx is not None:
            if summary_body and not self._previous_summary:
                self._previous_summary = summary_body
            turns_to_summarize = messages[max(compress_start, summary_idx + 1):compress_end]
        elif self._previous_summary:
            # No handoff summary found in the current messages, but
            # _previous_summary is non-empty — it was set by a different
            # (now-ended) session (e.g., a cron job, a prior /new).  Discard
            # it so _generate_summary() does not inject cross-session content
            # into the summarizer prompt via the iterative-update path.
            self._previous_summary = None

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered (%d tokens >= %d threshold)",
                display_tokens,
                self.threshold_tokens,
            )
            logger.info(
                "Model context limit: %d tokens (%.0f%% = %d)",
                self.context_length,
                self.threshold_percent * 100,
                self.threshold_tokens,
            )
            tail_msgs = n_messages - compress_end
            logger.info(
                "Summarizing turns %d-%d (%d turns), protecting %d head + %d tail messages",
                compress_start + 1,
                compress_end,
                len(turns_to_summarize),
                compress_start,
                tail_msgs,
            )

        # Phase 3: Generate structured summary
        summary_focus_topic = focus_topic or self._derive_auto_focus_topic(messages)
        summary = self._generate_summary(turns_to_summarize, focus_topic=summary_focus_topic)

        # If summary generation failed, behavior splits on
        # ``abort_on_summary_failure`` (config: compression.abort_on_summary_failure):
        #   True  → ABORT compression entirely. Return messages unchanged
        #           and set _last_compress_aborted=True so callers can warn
        #           the user and stop the auto-compress retry loop.
        #   False → Fall through to the default fallback path below: insert
        #           a deterministic "summary unavailable" handoff and drop
        #           the middle window.  Records _last_summary_fallback_used /
        #           _last_summary_dropped_count for gateway hygiene to
        #           surface a warning.
        # Default is False (historical behavior).
        #
        # EXCEPTION — auth AND transient network failures always abort. A
        # 401/403 from the summary call means the credential or endpoint is
        # broken (invalid/blocked key, or a token pointed at the wrong
        # inference host). A connection/stream-close error means the network
        # blipped at the compaction moment (#29559). In BOTH cases rotating into
        # a child session with a placeholder summary on a broken credential
        # strands the user on a degraded session for zero benefit — every
        # subsequent call fails the same way. So when the failure was an auth
        # error we abort regardless of abort_on_summary_failure, preserving
        # the conversation unchanged until the credential is fixed.
        if not summary and (
            self.abort_on_summary_failure
            or self._last_summary_auth_failure
            or self._last_summary_network_failure
        ):
            n_skipped = compress_end - compress_start
            self._last_summary_dropped_count = 0  # nothing actually dropped
            self._last_summary_fallback_used = False
            self._last_compress_aborted = True
            if not self.quiet_mode:
                if self._last_summary_auth_failure:
                    logger.warning(
                        "Summary generation failed with an authentication "
                        "error — aborting compression. %d message(s) preserved "
                        "unchanged; the session was NOT rotated. Check your "
                        "provider credential / inference endpoint, then retry "
                        "with /compress or start fresh with /new.",
                        n_skipped,
                    )
                elif self._last_summary_network_failure:
                    logger.warning(
                        "Summary generation failed with a network/connection "
                        "error — aborting compression. %d message(s) preserved "
                        "unchanged; the session was NOT rotated. This is "
                        "transient: retry with /compress once connectivity "
                        "recovers, or continue the conversation as-is.",
                        n_skipped,
                    )
                else:
                    logger.warning(
                        "Summary generation failed — aborting compression "
                        "(compression.abort_on_summary_failure=true). "
                        "%d message(s) preserved unchanged. Conversation is "
                        "frozen until the next /compress or /new.",
                        n_skipped,
                    )
            return messages

        # Phase 4: Assemble compressed message list
        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content")
                _compression_note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work. Your persistent memory (MEMORY.md, USER.md) remains fully authoritative regardless of compaction.]"
                if _compression_note not in _content_text_for_contains(existing):
                    msg["content"] = _append_text_to_content(
                        existing,
                        "\n\n" + _compression_note if isinstance(existing, str) and existing else _compression_note,
                    )
            compressed.append(msg)

        # If LLM summary failed, insert a deterministic fallback so the model
        # gets at least locally recoverable continuity anchors instead of a
        # content-free "N messages were removed" marker.
        if not summary:
            if not self.quiet_mode:
                logger.warning("Summary generation failed — inserting deterministic fallback context summary")
            n_dropped = compress_end - compress_start
            self._last_summary_dropped_count = n_dropped
            self._last_summary_fallback_used = True
            summary = self._build_static_fallback_summary(
                turns_to_summarize,
                reason=self._last_summary_error,
            )

        _merge_summary_into_tail = False
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
        # Pick a role that avoids consecutive same-role with both neighbors.
        # Priority: avoid colliding with head (already committed), then tail.
        if last_head_role in {"assistant", "tool"}:
            summary_role = "user"
        else:
            summary_role = "assistant"
        # If the chosen role collides with the tail AND flipping wouldn't
        # collide with the head, flip it.
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            else:
                # Both roles would create consecutive same-role messages
                # (e.g. head=assistant, tail=user — neither role works).
                # Merge the summary into the first tail message instead
                # of inserting a standalone message that breaks alternation.
                _merge_summary_into_tail = True

        # When the summary lands as a standalone role="user" message,
        # weak models read the verbatim "## Active Task" quote of a past
        # user request as fresh input (#11475, #14521).
        # When it lands as role="assistant", models may regurgitate the
        # summary text as their own output (#33256). In both cases, append
        # the explicit end marker so the model has a clear "summary ends
        # here, respond to the message below" signal.
        if not _merge_summary_into_tail:
            summary = summary + "\n\n" + _SUMMARY_END_MARKER

        if not _merge_summary_into_tail:
            compressed.append({
                "role": summary_role,
                "content": summary,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
            })

        for i in range(compress_end, n_messages):
            msg = messages[i].copy()
            if _merge_summary_into_tail and i == compress_end:
                merged_prefix = summary + "\n\n" + _SUMMARY_END_MARKER + "\n\n"
                msg["content"] = _append_text_to_content(
                    msg.get("content"),
                    merged_prefix,
                    prepend=True,
                )
                # Mark the merged message so frontends can identify it as
                # containing a compression summary prefix.
                msg[COMPRESSED_SUMMARY_METADATA_KEY] = True
                _merge_summary_into_tail = False
            compressed.append(msg)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        # Replace image parts in all compressed messages before the newest
        # image-bearing user turn with a short text placeholder. Without
        # this, tail messages keep their original multi-MB base-64 image
        # payloads forever, which can push every subsequent API request
        # past the provider's body-size limit and wedge the session.
        # Port of Kilo-Org/kilocode#9434.
        compressed = _strip_historical_media(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)
        saved_estimate = display_tokens - new_estimate

        # Anti-thrashing: track compression effectiveness
        savings_pct = (saved_estimate / display_tokens * 100) if display_tokens > 0 else 0
        self._last_compression_savings_pct = savings_pct
        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        if not self.quiet_mode:
            logger.info(
                "Compressed: %d -> %d messages (~%d tokens saved, %.0f%%)",
                n_messages,
                len(compressed),
                saved_estimate,
                savings_pct,
            )
            logger.info("Compression #%d complete", self.compression_count)

        return compressed
