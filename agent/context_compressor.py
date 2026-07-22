import hashlib
import json
import logging
import sqlite3
import re
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, _is_connection_error, aux_interrupt_protection
from agent.context_engine import ContextEngine, sanitize_memory_context
from agent.error_classifier import FailoverReason, classify_api_error
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
    estimate_tokens_rough,
)
from agent.redact import redact_sensitive_text
from agent.turn_context import drop_stale_api_content
from tools.todo_tool import TODO_INJECTION_HEADER

logger = logging.getLogger(__name__)


_SUMMARY_PERMANENT_QUOTA_MARKERS: tuple[str, ...] = (
    "insufficient_quota",
    "quota exceeded",
    "quota_exceeded",
    "out of funds",
    "out of credits",
    "out of credit",
    "out of extra usage",
)

_SUMMARY_MISSING_CREDENTIAL_MARKERS: tuple[str, ...] = (
    "no api key was found",
    "no api key found",
)


def _is_summary_access_or_quota_error(exc: Exception) -> bool:
    """Return True for non-retryable summary auth, permission, or quota errors."""

    classified = classify_api_error(exc)
    if classified.reason is FailoverReason.rate_limit:
        return False
    if classified.reason in {FailoverReason.auth, FailoverReason.auth_permanent}:
        return True

    err_text = str(exc).lower()
    if any(marker in err_text for marker in _SUMMARY_MISSING_CREDENTIAL_MARKERS):
        return True

    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status in {401, 402, 403}:
        return True

    if classified.reason is FailoverReason.billing:
        return any(marker in err_text for marker in _SUMMARY_PERMANENT_QUOTA_MARKERS)
    return any(marker in err_text for marker in _SUMMARY_PERMANENT_QUOTA_MARKERS)


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
    "None of the above restricts HOW you work: your tools remain fully "
    "active — keep calling them normally for the active task (edit files, "
    "run commands, search) instead of merely narrating what you would do. "
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
COMPRESSED_SUMMARY_HAS_USER_TURN_KEY = "_compressed_summary_has_user_turn"
_DB_PERSISTED_MARKER = "_db_persisted"

_NO_USER_TASK_SENTINEL = "None. This session contains no user-authored turns."
COMPRESSION_CONTINUATION_USER_CONTENT = (
    "Continue from the compressed conversation context above. "
    "This marker exists because no human user turn was available."
)
_LEGACY_COMPRESSION_CONTINUATION_USER_CONTENT = (
    "Continue from the compressed conversation context above. "
    "This marker exists because the compacted transcript contained "
    "no preserved user turn."
)


def _fresh_compaction_message_copy(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a message for compaction assembly without persistence markers.

    Live cached-gateway transcripts stamp ``_db_persisted`` during incremental
    flushes.  Shallow ``.copy()`` propagates that marker into the post-rotation
    compressed list, so ``_flush_messages_to_session_db`` skips every row when
    writing to the new child session (#57491).

    This strips at the copy site (clearest intent, and cheap), but the
    authoritative guarantee is the single terminal sweep in ``compress()``
    (``_strip_persistence_markers``): no message may leave ``compress()``
    carrying ``_db_persisted`` regardless of how many intermediate copy sites
    a future refactor adds.
    """
    fresh = msg.copy()
    fresh.pop(_DB_PERSISTED_MARKER, None)
    return fresh


def _strip_persistence_markers(messages: List[Dict[str, Any]]) -> None:
    """Enforce the compaction invariant: no assembled message carries a
    session-store persistence marker.

    ``compress()`` copies protected head/tail messages out of the live
    cached-gateway transcript, which stamps ``_db_persisted`` on every message
    over the life of the session.  If any copied dict keeps that marker, the
    rotation flush to the child session skips it and the compacted transcript is
    lost from ``state.db`` (#57491).  Stripping at each copy site is necessary
    but *positional* — a copy site added after the assembly loops would re-leak.
    This single terminal sweep makes the guarantee structural instead: run it
    once on the fully-assembled list so the invariant holds no matter where the
    copies happened.  Mutates in place (the dicts are compaction-local copies).
    """
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop(_DB_PERSISTED_MARKER, None)


# Appended to every standalone summary message (and to the merged-into-tail
# prefix) so the model has an unambiguous \"summary ends here\" boundary.
# Without it, weak models read the verbatim \"## Active Task\" quote as fresh
# user input (#11475, #14521) or regurgitate an assistant-role summary as
# their own output (#33256).
_SUMMARY_END_MARKER = (
    "--- END OF CONTEXT SUMMARY — "
    "respond to the message below, not the summary above ---"
)

# When the summary must be merged into the first tail message (the alternation
# corner case where a standalone summary role would collide with both head and
# tail), the tail message's own prior content is preserved BEFORE the summary,
# wrapped in these delimiters so the model doesn't read it as a fresh message.
# The summary prefix therefore lands AFTER _MERGED_SUMMARY_DELIMITER rather than
# at the start of the message, so _is_context_summary_content must look past it.
_MERGED_PRIOR_CONTEXT_HEADER = "[PRIOR CONTEXT — for reference only; not a new message]"
_MERGED_SUMMARY_DELIMITER = "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]"

# Handoff prefixes that shipped in earlier releases. A summary persisted under
# one of these can be inherited into a resumed lineage (#35344); when it is
# re-normalized on re-compaction we must strip the OLD prefix too, otherwise the
# stale directive it carried (e.g. \"resume exactly from Active Task\") survives
# embedded in the body and keeps hijacking replies. Keep newest-first; entries
# are matched literally. Add a frozen copy here whenever SUMMARY_PREFIX changes.
_HISTORICAL_SUMMARY_PREFIXES = (
    # Jul 2026 (#65848 class): identical to the current prefix except it
    # lacked the explicit \"tools remain fully active\" clause — the strong
    # REFERENCE ONLY framing bled into general tool-use suppression
    # (observed: 7 consecutive narration-only turns immediately after a
    # compression event on a production deployment).
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
    "None of the above restricts HOW you work: your tools remain fully "
    "active — keep calling them normally for the active task (edit files, "
    "run commands, search) instead of merely narrating what you would do. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:",
    # Carveout era (#41607/#38364/#42812): \"consistent → use as background\"
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
    # Pre-#35344: contained the self-contradicting \"resume exactly\" directive.
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
# Absolute ceiling for summary tokens (even on very large context windows).
# Summaries must stay within a 1K-10K token envelope — anything larger is
# itself a context-pressure source and slows every compaction.
_SUMMARY_TOKENS_CEILING = 10_000

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
# compressor speaks in.  Used when accumulating message \"content length\"
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
_ACTIVE_TASK_MAX_CHARS = 1400
# Keep a short run of recent messages verbatim even when the token budget is
# already exhausted.  The public ``protect_last_n`` default is intentionally
# high for small/light tails, but using all 20 as a hard floor here would bring
# back the old large-tool-output case where nothing can be compacted.
_MAX_TAIL_MESSAGE_FLOOR = 8

# Models with context windows below this get their compression threshold
# floored at ``_SMALL_CTX_THRESHOLD_PERCENT`` (raise-only — an explicitly
# higher user/model threshold always wins).  At the default 50% trigger a
# 128K-262K model compacts with only ~64-131K consumed; the incompressible
# floor (system prompt + tool schemas + protected tail + rolling summary)
# eats most of the reclaimed headroom, so compaction re-fires every 1-2
# turns and the session spends most of its wall-clock summarizing.
_SMALL_CTX_WINDOW_LIMIT = 512_000
_SMALL_CTX_THRESHOLD_PERCENT = 0.75


_PATH_MENTION_RE = re.compile(r"(?:/|~/?|[A-Za-z]:\\)[^\s`'\")\]}<>]+")

# MEDIA delivery directives must not reach the summarizer — if one leaks into
# the summary, the downstream model may re-emit it as an active directive on
# the next turn, triggering bogus attachment sends (#14665).
_MEDIA_DIRECTIVE_RE = re.compile(r"MEDIA:\S+")
_HISTORICAL_TASK_SECTION_RE = re.compile(
    rf"(?ms)^{re.escape(HISTORICAL_TASK_HEADING)}\s*\n.*?(?=^## |\Z)"
)


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


def _serialized_length_for_budget(value: Any) -> int:
    """Return a stable char-length for non-content replay/metadata fields."""
    if value is None or value == "":
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return len(str(value))


# Provider replay/metadata fields that ride the wire on every request but are
# invisible to ``msg["content"]``/``msg["tool_calls"]`` accounting.  Codex
# Responses sessions in particular carry ``codex_reasoning_items`` blobs of
# ``encrypted_content`` that can dominate the serialized session (a measured
# 214-turn session held ~115K tokens / 27% of its payload there — #55572).
_REPLAY_BUDGET_KEYS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


def _estimate_msg_budget_tokens(msg: dict) -> int:
    """Token estimate for one message in the tail-protection budget walks.

    Counts the message content plus the **full** ``tool_call`` envelope —
    ``id``, ``type``, ``function.name`` and JSON structure — not just
    ``function.arguments``.  Counting only the arguments string undercounted
    assistant turns that fan out into parallel tool calls by 2-15x (a
    4-tool-call turn measures ~73 vs ~1,090 real tokens), so the protected
    tail overshot ``tail_token_budget`` and compression became ineffective.
    See issue #28053.

    Also counts provider replay fields (``codex_reasoning_items`` etc. —
    see ``_REPLAY_BUDGET_KEYS``).  The preflight \"should I compress?\"
    estimator sees the full message shape, so the tail walk must use the
    same size class; otherwise an assistant message with tiny visible
    content but large hidden replay blobs is protected as if it were small,
    the post-compression session stays near the context limit, and
    compaction re-fires continuously (#55572).  Accounting-only: replay
    fields are never mutated or pruned here.
    """
    content = msg.get("content") or ""
    if isinstance(content, str):
        tokens = estimate_tokens_rough(content) + 10  # +10 for role/key overhead
    else:
        content_len = _content_length_for_budget(content)
        tokens = content_len // _CHARS_PER_TOKEN + 10
    
    # #69280: Omit empty tool_calls from token estimation to match wire behavior.
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                tokens += estimate_tokens_rough(str(tc))
                
    for key in _REPLAY_BUDGET_KEYS:
        tokens += _serialized_length_for_budget(msg.get(key)) // _CHARS_PER_TOKEN
    return tokens


def _content_text_for_contains(content: Any) -> str:
    """Return a best-effort text view of message content.

    Used only for substring checks when we need to know whether we've already
    appended a note to a message. Keeps multimodal lists intact elsewhere."""
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
    blocks, so direct string concatenation is not always safe."""
    if content is None:
        return text
    if isinstance(content, str):
        return text + content if prepend else content + text
    if isinstance(content, list):
        text_block = {"type": "text", "text": text}
        return [text_block, *content] if prepend else [*content, text_block]
    rendered = str(content)
    return text + rendered if prepend else rendered + text
