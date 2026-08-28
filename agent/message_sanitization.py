"""Message and tool-payload sanitization helpers.

Pure functions extracted from ``run_agent.py`` so the AIAgent module can
stay focused on the conversation loop.  These walk OpenAI-format message
lists and structured payloads, repairing or stripping problematic
characters that would otherwise crash ``json.dumps`` inside the OpenAI
SDK or be rejected by upstream APIs.

All helpers are stateless and side-effect-free except for in-place
mutation of their input (where documented).  Backward-compatible
re-exports from ``run_agent`` remain in place so existing imports
``from run_agent import _sanitize_surrogates`` keep working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Lone surrogate code points are invalid in UTF-8 and crash json.dumps
# inside the OpenAI SDK.  Used by every surrogate-sanitization helper
# below as well as by run_agent and the CLI for paste-from-clipboard
# scrubbing.
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD (replacement character).

    Surrogates are invalid in UTF-8 and will crash ``json.dumps()`` inside the
    OpenAI SDK.  This is a fast no-op when the text contains no surrogates.
    """
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _sanitize_structure_surrogates(payload: Any) -> bool:
    """Replace surrogate code points in nested dict/list payloads in-place.

    Mirror of ``_sanitize_structure_non_ascii`` but for surrogate recovery.
    Used to scrub nested structured fields (e.g. ``reasoning_details`` — an
    array of dicts with ``summary``/``text`` strings) that flat per-field
    checks don't reach.  Returns True if any surrogates were replaced.
    """
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[key] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[idx] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found


def _sanitize_messages_surrogates(messages: list) -> bool:
    """Sanitize surrogate characters from all string content in a messages list.

    Walks message dicts in-place. Returns True if any surrogates were found
    and replaced, False otherwise. Covers content/text, name, tool call
    metadata/arguments, AND any additional string or nested structured fields
    (``reasoning``, ``reasoning_content``, ``reasoning_details``, etc.) so
    retries don't fail on a non-content field.  Byte-level reasoning models
    (xiaomi/mimo, kimi, glm) can emit lone surrogates in reasoning output
    that flow through to ``api_messages["reasoning_content"]`` on the next
    turn and crash json.dumps inside the OpenAI SDK.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        name = msg.get("name")
        if isinstance(name, str) and _SURROGATE_RE.search(name):
            msg["name"] = _SURROGATE_RE.sub('\ufffd', name)
            found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and _SURROGATE_RE.search(tc_id):
                    tc["id"] = _SURROGATE_RE.sub('\ufffd', tc_id)
                    found = True
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and _SURROGATE_RE.search(fn_name):
                        fn["name"] = _SURROGATE_RE.sub('\ufffd', fn_name)
                        found = True
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str) and _SURROGATE_RE.search(fn_args):
                        fn["arguments"] = _SURROGATE_RE.sub('\ufffd', fn_args)
                        found = True
        # Walk any additional string / nested fields (reasoning,
        # reasoning_content, reasoning_details, etc.) — surrogates from
        # byte-level reasoning models (xiaomi/mimo, kimi, glm) can lurk
        # in these fields and aren't covered by the per-field checks above.
        # Matches _sanitize_messages_non_ascii's coverage (PR #10537).
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                if _SURROGATE_RE.search(value):
                    msg[key] = _SURROGATE_RE.sub('\ufffd', value)
                    found = True
            elif isinstance(value, (dict, list)):
                if _sanitize_structure_surrogates(value):
                    found = True
    return found


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    """Escape unescaped control chars inside JSON string values.

    Walks the raw JSON character-by-character, tracking whether we are
    inside a double-quoted string. Inside strings, replaces literal
    control characters (0x00-0x1F) that aren't already part of an escape
    sequence with their ``\\uXXXX`` equivalents. Pass-through for everything
    else.

    Ported from #12093 — complements the other repair passes in
    ``_repair_tool_call_arguments`` when ``json.loads(strict=False)`` is
    not enough (e.g. llama.cpp backends that emit literal apostrophes or
    tabs alongside other malformations).
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                # Already-escaped char — pass through as-is
                out.append(ch)
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
        i += 1
    return "".join(out)


# When a repair is about to destroy the only copy of a tool call's original
# argument bytes (rewriting them to "{}"), the WARNING log is the last
# surviving copy of content that can hold real user data (#80498). Bound the
# logged string at this size instead of a short preview so it stays
# recoverable from agent.log without letting a pathological payload flood
# the log.
_FULL_ARGS_LOG_BOUND = 100_000


def _repair_tool_call_arguments(raw_args: str, tool_name: str = "?") -> str:
    """Attempt to repair malformed tool_call argument JSON.

    Models like GLM-5.1 via Ollama can produce truncated JSON, trailing
    commas, Python ``None``, etc.  The API proxy rejects these with HTTP 400
    "invalid tool call arguments".  This function applies common repairs;
    if all fail it returns ``"{}"`` so the request succeeds (better than
    crashing the session).  All repairs are logged at WARNING level.
    """
    raw_stripped = raw_args.strip() if isinstance(raw_args, str) else ""

    # Fast-path: empty / whitespace-only -> empty object
    if not raw_stripped:
        logger.warning("Sanitized empty tool_call arguments for %s", tool_name)
        return "{}"

    # Python-literal None -> normalise to {}
    if raw_stripped == "None":
        logger.warning("Sanitized Python-None tool_call arguments for %s", tool_name)
        return "{}"

    # Repair pass 0: llama.cpp backends sometimes emit literal control
    # characters (tabs, newlines) inside JSON string values. json.loads
    # with strict=False accepts these and lets us re-serialise the
    # result into wire-valid JSON without any string surgery. This is
    # the most common local-model repair case (#12068).
    try:
        parsed = json.loads(raw_stripped, strict=False)
        reserialised = json.dumps(parsed, separators=(",", ":"))
        if reserialised != raw_stripped:
            logger.warning(
                "Repaired unescaped control chars in tool_call arguments for %s",
                tool_name,
            )
        return reserialised
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Attempt common JSON repairs
    fixed = raw_stripped
    # 1. Strip trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # 2. Close unclosed structures
    open_curly = fixed.count('{') - fixed.count('}')
    open_bracket = fixed.count('[') - fixed.count(']')
    if open_curly > 0:
        fixed += '}' * open_curly
    if open_bracket > 0:
        fixed += ']' * open_bracket
    # 3. Remove excess closing braces/brackets (bounded to 50 iterations)
    for _ in range(50):
        try:
            json.loads(fixed)
            break
        except json.JSONDecodeError:
            if fixed.endswith('}') and fixed.count('}') > fixed.count('{'):
                fixed = fixed[:-1]
            elif fixed.endswith(']') and fixed.count(']') > fixed.count('['):
                fixed = fixed[:-1]
            else:
                break

    try:
        json.loads(fixed)
        logger.warning(
            "Repaired malformed tool_call arguments for %s: %s → %s",
            tool_name, raw_stripped[:80], fixed[:80],
        )
        return fixed
    except json.JSONDecodeError:
        pass

    # Repair pass 4: escape unescaped control chars inside JSON strings,
    # then retry. Catches cases where strict=False alone fails because
    # other malformations are present too.
    try:
        escaped = _escape_invalid_chars_in_json_strings(fixed)
        if escaped != fixed:
            json.loads(escaped)
            logger.warning(
                "Repaired control-char-laced tool_call arguments for %s: %s → %s",
                tool_name, raw_stripped[:80], escaped[:80],
            )
            return escaped
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Last resort: replace with empty object so the API request doesn't
    # crash the entire session. Log the FULL original string (bounded) —
    # for callers that discard the original (e.g. the pre-send transcript
    # sanitizer), this WARNING is the last surviving copy of bytes that can
    # contain real user content (#80498: a truncated write_file call's
    # streamed file content).
    logger.warning(
        "Unrepairable tool_call arguments for %s — "
        "replaced with empty object (was: %s)",
        tool_name, raw_stripped[:_FULL_ARGS_LOG_BOUND],
    )
    return "{}"


def close_interrupted_tool_sequence(messages: list, final_response: Any = None) -> bool:
    """Append a synthetic assistant turn when an interrupted tail is a tool result.

    A turn cut short by ``/stop`` can leave the transcript ending on a raw
    ``tool`` message (a tool finished, or its execution was cancelled, but the
    model never streamed a closing assistant turn). Persisting that tail means
    the next user message lands as ``… tool → user`` — a role-alternation
    violation that strict providers (Gemini, Claude) react to by hallucinating
    a continuation of the user's message and ignoring prior context, which
    reads to the user as "lost context" (#48879).

    ``finalize_turn`` closes this on the happy interrupt path, but the
    retry/backoff/error interrupt aborts in ``conversation_loop`` ``return``
    early and never reach it — this shared helper closes the sequence on all of
    them. ``final_response`` is usually empty on an interrupt, so an explicit
    placeholder is used rather than an empty-content assistant turn.

    Mutates ``messages`` in place. Returns True if a closing turn was appended.
    """
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "tool":
        return False
    text = final_response if isinstance(final_response, str) else ""
    from agent.message_metadata import append_message

    append_message(messages, {
        "role": "assistant",
        "content": text.strip() or "Operation interrupted.",
    })
    return True


def _strip_non_ascii(text: str) -> str:
    """Remove non-ASCII characters, replacing with closest ASCII equivalent or removing.

    Used as a last resort when the system encoding is ASCII and can't handle
    any non-ASCII characters (e.g. LANG=C on Chromebooks).
    """
    return text.encode('ascii', errors='ignore').decode('ascii')


def _sanitize_messages_non_ascii(messages: list) -> bool:
    """Strip non-ASCII characters from all string content in a messages list.

    This is a last-resort recovery for systems with ASCII-only encoding
    (LANG=C, Chromebooks, minimal containers).  Returns True if any
    non-ASCII content was found and sanitized.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # Sanitize content (string)
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _strip_non_ascii(content)
            if sanitized != content:
                msg["content"] = sanitized
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        sanitized = _strip_non_ascii(text)
                        if sanitized != text:
                            part["text"] = sanitized
                            found = True
        # Sanitize name field (can contain non-ASCII in tool results)
        name = msg.get("name")
        if isinstance(name, str):
            sanitized = _strip_non_ascii(name)
            if sanitized != name:
                msg["name"] = sanitized
                found = True
        # Sanitize tool_calls
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            sanitized = _strip_non_ascii(fn_args)
                            if sanitized != fn_args:
                                fn["arguments"] = sanitized
                                found = True
        # Sanitize any additional top-level string fields (e.g. reasoning_content)
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                sanitized = _strip_non_ascii(value)
                if sanitized != value:
                    msg[key] = sanitized
                    found = True
    return found


def _sanitize_tools_non_ascii(tools: list) -> bool:
    """Strip non-ASCII characters from tool payloads in-place."""
    return _sanitize_structure_non_ascii(tools)


def serialized_messages_bytes(messages: list) -> int:
    """Exact serialized size, in bytes, of the ``messages`` request payload.

    Recovery path for HTTP 413 (payload too large).  A 413 is a *byte*-size
    error, but Hermes' context estimator deliberately prices an image at a
    flat per-image token cost so that a screenshot does not trigger premature
    compaction (see ``estimate_messages_tokens_rough``).  That makes the
    token estimate structurally unable to *score* recovery from an
    image-dominated 413: compaction can free megabytes of base64 while the
    estimate barely moves, so a token-scored progress check reports
    "no progress" and the turn dies permanently.

    This measures the thing the provider actually rejected — serialized
    bytes — exactly and for free.  It is a faithful proxy for the request
    body's ``messages`` field (the only part recovery can shrink) and is
    measured identically before and after each compression pass, so the
    before/after ratio is exact.  It is NOT an estimate.

    Non-serializable values fall back to ``str()`` so a malformed message
    can never crash the 413 recovery path.
    """
    if not isinstance(messages, list) or not messages:
        return 0
    try:
        return len(
            json.dumps(
                messages, ensure_ascii=False, separators=(",", ":"), default=str
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        # Extremely defensive — ``default=str`` already covers exotic
        # values.  Never let byte accounting take down error recovery.
        return sum(len(str(m)) for m in messages)


def _strip_images_from_messages(messages: list) -> bool:
    """Remove image_url content parts from all messages in-place.

    Called when a server signals it does not support images (e.g.
    "Only 'text' content type is supported.").  Mutates messages so the
    next API call sends text only.

    Preserves message alternation invariants:
      * ``tool``-role messages whose content was entirely images are replaced
        with a plaintext placeholder, NOT deleted — deleting them would leave
        the paired ``tool_call_id`` on the prior assistant message unmatched,
        which providers reject with HTTP 400.
      * Assistant messages carrying ``tool_calls`` are likewise replaced, not
        deleted — dropping them would orphan their tool responses.
      * Other messages whose content becomes empty are dropped.  In practice
        this only hits synthetic image-only user messages appended for
        attachment delivery; real user turns always include text.

    This runs on the persistent history as well as the per-call copy, so any
    message it rewrites must also lose its ``api_content`` sidecar: the sidecar
    carries the exact bytes previously sent — here, the images this strip
    exists to remove — and the next turn substitutes it back into ``content``,
    undoing the strip on the wire.

    Returns True if any image parts were removed.
    """
    from agent.turn_context import drop_stale_api_content

    found = False
    to_delete = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "image", "input_image"}:
                found = True
            else:
                new_parts.append(part)
        if len(new_parts) < len(content):
            if new_parts:
                msg["content"] = new_parts
            elif msg.get("role") == "tool" or msg.get("tool_calls"):
                # Preserve message linkage — providers require every assistant
                # tool_call to have a matching tool response, and an assistant
                # message carrying tool_calls must survive even if its content
                # was entirely images.
                msg["content"] = "[image content removed — server does not support images]"
            else:
                # Synthetic image-only user/assistant message with no text and
                # no tool_calls; safe to drop.
                to_delete.append(i)
            # Content was rewritten — the pre-strip sidecar is now stale.
            drop_stale_api_content(msg)
    for i in reversed(to_delete):
        del messages[i]
    return found


_IMAGE_REJECTION_PHRASES = (
    "only 'text' content type is supported",
    "only text content type is supported",
    "image_url is not supported",
    "image content is not supported",
    "multimodal is not supported",
    "multimodal content is not supported",
    "multimodal input is not supported",
    "vision is not supported",
    "vision input is not supported",
    "does not support images",
    "does not support image input",
    "does not support multimodal",
    "does not support vision",
    "model does not support image",
    # Some OpenAI-compatible endpoints (e.g. Alibaba/DashScope-style
    # gateways) reject non-text content blocks with this generic body
    # instead of naming image_url or vision support explicitly.
    # (issue #57948)
    "unexpected item type in content",
    # ChatGPT-account Codex backend
    # (https://chatgpt.com/backend-api/codex) rejects
    # data:image/...base64 URLs in input_image fields
    # with HTTP 400 "Invalid 'input[N].content[K].image_url'.
    # Expected a valid URL, but got a value with an
    # invalid format." The OpenAI Responses API on the
    # public endpoint accepts data URLs, but the
    # ChatGPT-account variant does not. Without this
    # phrase the agent cascaded into compression /
    # context-too-large recovery instead of just
    # stripping the images. Match is narrow on
    # purpose — keyed on the field-path apostrophe so
    # we don't false-trip on other URL validation
    # errors. (issue #23570)
    "image_url'. expected",
    # ChatGPT-account Codex can also reject corrupt/unsupported
    # native image payloads with this wording. Treat it like a
    # provider image rejection so the loop strips images and
    # retries text-only instead of aborting the session.
    "image data you provided does not represent a valid image",
    # DeepSeek's OpenAI-compatible API reports text-only
    # request-body variants as:
    # "unknown variant `image_url`, expected `text`".
    "unknown variant `image_url`, expected `text`",
    "unknown variant image_url, expected text",
    # OpenRouter routes a request to upstream endpoints and,
    # when none of the candidate endpoints for the model accept
    # image input, returns HTTP 404 "No endpoints found that
    # support image input". Without this phrase the agent never
    # strips the images, the retry loop re-sends the same
    # rejected request until exhaustion, and the gateway leaves
    # every subsequent message queued behind the stuck turn —
    # the P1 in issue #21160. The 404 passes the 4xx gate in the
    # conversation loop.
    "no endpoints found that support image input",
    # Kimi / Moonshot / other OpenAI-compatible Chinese
    # providers reject truncated or corrupt image bytes with
    # HTTP 400 "Invalid request: prepare image failed ...
    # failed to decode image: invalid or unsupported image
    # format". Like the Codex case above, the bad bytes are
    # baked into immutable conversation history and re-sent on
    # every retry, wedging the session. Strip the images so the
    # turn recovers instead of exhausting retries. (issue
    # #76884; complements the proactive full-decode validation
    # in tools/vision_tools._normalize_to_supported_image)
    "failed to decode image",
)


def _looks_like_image_content_rejection(error_body: str) -> bool:
    """Return True when a provider error says image/multimodal input is unsupported."""
    body = str(error_body or "").lower()
    return any(phrase in body for phrase in _IMAGE_REJECTION_PHRASES)


def _sanitize_structure_non_ascii(payload: Any) -> bool:
    """Strip non-ASCII characters from nested dict/list payloads in-place."""
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[key] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[idx] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found


__all__ = [
    "_SURROGATE_RE",
    "close_interrupted_tool_sequence",
    "_sanitize_surrogates",
    "_sanitize_structure_surrogates",
    "_sanitize_messages_surrogates",
    "_escape_invalid_chars_in_json_strings",
    "_repair_tool_call_arguments",
    "_strip_non_ascii",
    "_sanitize_messages_non_ascii",
    "_sanitize_tools_non_ascii",
    "_strip_images_from_messages",
    "_sanitize_structure_non_ascii",
    # call_id policy owners (F4 consolidation)
    "deterministic_call_id",
    "coalesce_tool_call_id",
    "tool_call_id_variants",
    "tool_result_id_variants",
    "uniquify_tool_call_ids",
    # reasoning_content policy owners (F4 consolidation)
    "reasoning_echo_family",
    "matches_reasoning_echo_family",
    "needs_reasoning_echo",
    "apply_reasoning_content_policy",
    "reapply_reasoning_echo",
]


# ---------------------------------------------------------------------------
# call_id policy — single owner (audit F4, incident chain I4)
# ---------------------------------------------------------------------------
#
# Three forked policy sites converged here:
#   * agent/codex_responses_adapter.py `_deterministic_call_id` — hash
#     synthesis when a provider omits call_id (fa3ab2ffd0 → e45f2b39e2).
#   * run_agent.AIAgent._get_tool_call_id_static — `call_id or id`
#     coalescing for dicts and SDK objects.
#   * run_agent.AIAgent._uniquify_tool_call_ids — duplicate-id repair with
#     deterministic `_d<n>` suffixes (#58327 loss class).
#
# NOT consolidated (different scheme on purpose):
#   agent/transports/codex_event_projector._deterministic_call_id maps codex
#   app-server ITEM ids (`codex_<type>_<item_id>`), not chat tool-call
#   content; merging the two would change ids and invalidate prompt caches.
#
# HARD INVARIANT: everything here must stay deterministic (never uuid4) and
# byte-identical for existing inputs — these ids feed prompt-cache prefixes.


def deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Generate a deterministic call_id from tool call content.

    Used as a fallback when the API doesn't provide a call_id.
    Deterministic IDs prevent cache invalidation — random UUIDs would
    make every API call's prefix unique, breaking OpenAI's prompt cache.
    """
    seed = f"{fn_name}:{arguments}:{index}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"call_{digest}"


def _expand_tool_id_variants(values: tuple[Any, ...]) -> frozenset[str]:
    """Return every wire spelling of one or more tool-call identifiers.

    Responses bridges may expose the pairing id and response-item id
    separately, or encode both as ``call_id|response_item_id``.  The values
    are aliases for one call, not distinct calls.  Keeping the expansion in
    the shared policy module prevents the repair and pre-send paths from
    drifting apart again.
    """
    variants: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if not value:
            continue
        variants.add(value)
        if "|" in value:
            for part in value.split("|"):
                part = part.strip()
                if part:
                    variants.add(part)
    return frozenset(variants)


def tool_call_id_variants(tc: Any) -> frozenset[str]:
    """Return all pairing-id variants carried by a tool-call entry."""
    if isinstance(tc, dict):
        values = (
            tc.get("call_id"),
            tc.get("id"),
            tc.get("response_item_id"),
        )
    else:
        values = (
            getattr(tc, "call_id", None),
            getattr(tc, "id", None),
            getattr(tc, "response_item_id", None),
        )
    return _expand_tool_id_variants(values)


def tool_result_id_variants(tool_call_id: Any) -> frozenset[str]:
    """Return all matching variants for a role=tool ``tool_call_id``."""
    return _expand_tool_id_variants((tool_call_id,))


def coalesce_tool_call_id(tc: Any) -> str:
    """Extract the effective call ID from a tool_call entry (dict or object).

    Single owner for the canonical pairing rule: Codex Responses tool calls
    carry ``call_id`` (authoritative pairing key), Chat Completions ones carry
    ``id`` only, and bridge ids may encode ``call_id|response_item_id``.
    Returns ``""`` when neither pairing field is set.
    """
    if isinstance(tc, dict):
        values = (tc.get("call_id"), tc.get("id"))
    else:
        values = (getattr(tc, "call_id", None), getattr(tc, "id", None))
    for raw in values:
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if value:
            return value.split("|", 1)[0].strip() or value
    return ""


def uniquify_tool_call_ids(tool_calls: list) -> list:
    """Ensure every tool call in a single assistant turn has a distinct id.

    Some models/providers reuse one call id across different calls in a
    single batch (observed with native Kimi Responses replays, Ollama-
    compatible endpoints, and degraded models at long context; same bug
    class as openclaw/openclaw#110518 / #110956). Duplicate ids are lossy
    downstream: the pre-API sanitizer keeps only the first call/result
    pair per id (#58327), so the later call's result silently vanishes
    from every replayed payload, and strict providers (Anthropic
    tool_use, DeepSeek) reject duplicate ids outright.

    The first occurrence keeps its id; later collisions get a
    deterministic ``<id>_d<n>`` suffix — never a random UUID, which would
    break prompt-cache prefix stability across replays. Mutates the
    entries in place (SDK models / SimpleNamespace / dicts) and returns
    the same list. Blank/missing ids are left for the deterministic
    fallback in ``build_assistant_message``.
    """
    seen: set = set()
    for tc in tool_calls or []:
        # Same coalescing rule as ``coalesce_tool_call_id`` but tolerant of
        # non-string ids (degraded models can emit ints/None here).
        if isinstance(tc, dict):
            raw = tc.get("call_id") or tc.get("id") or ""
        else:
            raw = getattr(tc, "call_id", None) or getattr(tc, "id", None) or ""
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            continue
        # Composite Responses ids ("call_x|fc_y") collide on the call
        # half — that's the pairing key providers enforce per turn.
        cid = raw.split("|", 1)[0]
        if not cid:
            continue
        if cid not in seen:
            seen.add(cid)
            continue
        n = 2
        new_id = f"{cid}_d{n}"
        while new_id in seen:
            n += 1
            new_id = f"{cid}_d{n}"
        seen.add(new_id)

        def _renamed(value):
            # Preserve a composite id's response-item half so the
            # provider's real fc_/item id survives the rename.
            if isinstance(value, str) and "|" in value:
                return f"{new_id}|{value.split('|', 1)[1]}"
            return new_id

        try:
            if isinstance(tc, dict):
                if tc.get("id"):
                    tc["id"] = _renamed(tc["id"])
                else:
                    tc["id"] = new_id
                if tc.get("call_id"):
                    tc["call_id"] = new_id
            else:
                tc.id = _renamed(getattr(tc, "id", None))
                if getattr(tc, "call_id", None):
                    tc.call_id = new_id
        except Exception:
            logger.warning(
                "Could not uniquify duplicate tool call id %s", cid
            )
            continue
        _fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
        _fn_name = (_fn.get("name") if isinstance(_fn, dict) else getattr(_fn, "name", None)) or "?"
        logger.warning(
            "Model reused tool call id %s within one turn; renamed the "
            "duplicate to %s (tool=%s) to keep call/result pairing "
            "lossless.", cid, new_id, _fn_name,
        )
    return tool_calls


# ---------------------------------------------------------------------------
# reasoning_content policy — single owner (audit F4)
# ---------------------------------------------------------------------------
#
# The strip-vs-repad decision was previously forked across the wire files in
# separate incident commits (2b3a4f0af8 strip for strict providers,
# b5495db701 re-pad for require-side, 94b3131be7/9a9f8a6d99 kimi pad).  The
# POLICY — which provider direction gets which treatment — lives here as one
# rule table + apply functions; adapters keep only SYNTAX mapping (e.g.
# anthropic_adapter turning reasoning_content into a thinking block).
#
# Direction table:
#   require-side (echo-back enforced; replays 400 without the field):
#     kimi     — provider kimi-coding/kimi-coding-cn, or host api.kimi.com /
#                moonshot.ai / moonshot.cn.  Host-driven on purpose:
#                aggregators re-exporting kimi models reject the echo.
#     deepseek — provider "deepseek", model contains "deepseek", or host
#                api.deepseek.com (#15250; V4 rejects empty-string pads,
#                hence the " " single-space pad, #17341).
#     mimo     — provider "xiaomi", model contains "mimo", or host
#                *.xiaomimimo.com.
#   strict side (field rejected with 400/422 "Extra inputs are not
#     permitted"): everyone else — Mistral, Cerebras, Groq, SambaNova, …
#     (#45655). Strip the key entirely, even a single-space pad.

_REASONING_ECHO_RULES: tuple = (
    # (family, exact providers (raw), exact providers (lowered),
    #  model substrings (lowered), base_url hosts)
    ("kimi", frozenset({"kimi-coding", "kimi-coding-cn"}), frozenset(), (),
     ("api.kimi.com", "moonshot.ai", "moonshot.cn")),
    ("deepseek", frozenset(), frozenset({"deepseek"}), ("deepseek",),
     ("api.deepseek.com",)),
    ("mimo", frozenset(), frozenset({"xiaomi"}), ("mimo",),
     ("api.xiaomimimo.com", "xiaomimimo.com")),
)


def _family_rule(family: str) -> tuple:
    for rule in _REASONING_ECHO_RULES:
        if rule[0] == family:
            return rule
    raise KeyError(family)


def matches_reasoning_echo_family(
    family: str, provider: Any, model: Any, base_url: Any
) -> bool:
    """True when (provider, model, base_url) matches one echo-back family.

    Families can overlap (e.g. a deepseek-named model pointed at a kimi
    host); this membership test is independent per family so per-family
    predicates keep their original semantics.
    """
    from utils import base_url_host_matches

    _, raw_providers, lowered_providers, model_subs, hosts = _family_rule(family)
    provider_lower = (provider or "").lower()
    model_lower = (model or "").lower()
    if provider in raw_providers or provider_lower in lowered_providers:
        return True
    if any(sub in model_lower for sub in model_subs):
        return True
    return any(base_url_host_matches(base_url, host) for host in hosts)


def reasoning_echo_family(provider: Any, model: Any, base_url: Any) -> "str | None":
    """Classify the provider direction for the reasoning_content echo policy.

    Returns ``"kimi"``, ``"deepseek"``, or ``"mimo"`` (first match in table
    order) when the target endpoint enforces reasoning_content echo-back on
    assistant turns, else ``None`` (strict/indifferent side — the field must
    be stripped).
    """
    for rule in _REASONING_ECHO_RULES:
        if matches_reasoning_echo_family(rule[0], provider, model, base_url):
            return rule[0]
    return None


def needs_reasoning_echo(provider: Any, model: Any, base_url: Any) -> bool:
    """True when the endpoint requires reasoning_content echo-back."""
    return reasoning_echo_family(provider, model, base_url) is not None


def apply_reasoning_content_policy(
    source_msg: dict, api_msg: dict, needs_thinking_pad: bool
) -> None:
    """Copy provider-facing reasoning fields onto an API replay message.

    ``needs_thinking_pad`` is the require-side flag (see
    ``needs_reasoning_echo`` / the agent's cached
    ``_needs_thinking_reasoning_pad``). Mutates ``api_msg`` in place.
    """
    if source_msg.get("role") != "assistant":
        return

    # 1. Explicit reasoning_content already set.
    #
    # When the active provider enforces the thinking-mode echo-back
    # (DeepSeek / Kimi / MiMo), preserve it verbatim — that includes their
    # own space-placeholder written at creation time and any valid reasoning
    # from the same provider. Sessions persisted BEFORE #17341 have
    # empty-string placeholders pinned at creation time; DeepSeek V4 Pro
    # rejects those with HTTP 400, so upgrade "" → " " on replay.
    #
    # When the active provider does NOT enforce echo-back, strip the field
    # entirely. Strict OpenAI-compatible providers (Mistral, Cerebras, Groq,
    # SambaNova, …) reject ANY reasoning_content key in input messages with
    # HTTP 400/422 ("Extra inputs are not permitted"), even an empty string
    # or a single-space pad. This is the cross-provider fallback case: a
    # reasoning primary (DeepSeek/Kimi/MiMo) pads history with " ", then a
    # fallback to a strict provider replays that pad and 422s. Stripping
    # here covers the rebuild path; ``reapply_reasoning_echo`` covers the
    # already-built api_messages path. Refs #45655.
    existing = source_msg.get("reasoning_content")
    if isinstance(existing, str):
        if not needs_thinking_pad:
            api_msg.pop("reasoning_content", None)
        elif existing == "":
            api_msg["reasoning_content"] = " "
        else:
            api_msg["reasoning_content"] = existing
        return

    # 2. Cross-provider poisoned history (#15748): on DeepSeek/Kimi,
    # if the source turn has tool_calls AND a 'reasoning' field but no
    # 'reasoning_content' key, the 'reasoning' text was written by a
    # prior provider (e.g. MiniMax) — DeepSeek's own _build_assistant_message
    # pins reasoning_content at creation time for tool-call turns, so the
    # shape (reasoning set, reasoning_content absent, tool_calls present)
    # is unreachable from same-provider DeepSeek history after this fix.
    # Inject a single space to satisfy the API without leaking another
    # provider's chain of thought to DeepSeek/Kimi. Space (not "")
    # because DeepSeek V4 Pro rejects empty-string reasoning_content
    # in thinking mode (refs #17341).
    normalized_reasoning = source_msg.get("reasoning")
    if (
        needs_thinking_pad
        and source_msg.get("tool_calls")
        and isinstance(normalized_reasoning, str)
        and normalized_reasoning
    ):
        api_msg["reasoning_content"] = " "
        return

    # 3. Healthy session: promote 'reasoning' field to 'reasoning_content'
    # for providers that use the internal 'reasoning' key.
    # This must happen before the unconditional empty-string fallback so
    # genuine reasoning content is not overwritten (#15812 regression in
    # PR #15478). Only promote for providers that enforce echo-back —
    # strict providers reject the field (refs #45655).
    if isinstance(normalized_reasoning, str) and normalized_reasoning:
        if needs_thinking_pad:
            api_msg["reasoning_content"] = normalized_reasoning
        else:
            api_msg.pop("reasoning_content", None)
        return

    # 4. DeepSeek / Kimi thinking mode: all assistant messages need
    # reasoning_content. Inject a single space to satisfy the provider's
    # requirement when no explicit reasoning content is present. Covers
    # both tool-call turns (already-poisoned history with no reasoning
    # at all) and plain text turns. Space (not "") because DeepSeek V4
    # Pro tightened validation and rejects empty string with HTTP 400
    # ("The reasoning content in the thinking mode must be passed back
    # to the API"). Refs #17341.
    if needs_thinking_pad:
        api_msg["reasoning_content"] = " "
        return

    # 5. reasoning_content was present but not a string (e.g. None after
    # context compaction).  Don't pass null to the API.
    api_msg.pop("reasoning_content", None)


def reapply_reasoning_echo(api_messages: list, needs_thinking_pad: bool) -> int:
    """Re-pad (or strip) assistant turns' reasoning_content for the active provider.

    ``api_messages`` is built once, before the retry loop, while the *primary*
    provider is active.  A mid-conversation fallback can then switch providers,
    so the reasoning fields baked into ``api_messages`` are shaped for the
    *prior* provider and must be reconciled against the *current* one:

    * Switching TO a require-side provider (DeepSeek / Kimi / MiMo thinking
      mode): assistant turns built when the prior provider did NOT need the
      echo-back go out without ``reasoning_content`` and the new provider
      rejects them with HTTP 400 ("The reasoning_content in the thinking mode
      must be passed back").  Re-apply the pad.

    * Switching TO a strict provider that rejects the field (Mistral,
      Cerebras, Groq, SambaNova, …): assistant turns built under a reasoning
      primary carry a ``reasoning_content`` pad (often a single space ``" "``),
      and the strict provider rejects it with HTTP 400/422 ("Extra inputs are
      not permitted").  Strip the field.  This is the exact cross-provider
      fallback bug from #45655 — a DeepSeek primary pads history with ``" "``,
      the request falls back to Mistral, and Mistral 422s on the stale pad.

    Calling this immediately before building the request kwargs reconciles the
    fields against the *current* provider.  It is idempotent and safe to call
    every iteration; it covers every fallback path.

    Returns the number of assistant turns whose reasoning_content was added or
    removed.
    """
    changed = 0
    for api_msg in api_messages:
        if api_msg.get("role") != "assistant":
            continue
        if needs_thinking_pad:
            if api_msg.get("reasoning_content"):
                continue
            apply_reasoning_content_policy(api_msg, api_msg, needs_thinking_pad)
            if api_msg.get("reasoning_content"):
                changed += 1
        else:
            # Strict provider — strip any stale reasoning_content pad left
            # over from a reasoning primary so the fallback request doesn't
            # 400/422 on it.
            if "reasoning_content" in api_msg:
                api_msg.pop("reasoning_content", None)
                changed += 1
    return changed


# ---------------------------------------------------------------------------
# Image / multimodal parts — evaluated, NOT consolidated (verdict: syntax)
# ---------------------------------------------------------------------------
#
# The per-adapter image handling is format-specific SYNTAX, not shared policy:
#   * anthropic_adapter (~1817): data-URL → Anthropic `source: {type: base64}`
#     block mapping — Anthropic wire shape only.
#   * codex_responses_adapter (~113/165/812): chat `image_url` parts →
#     Responses `input_image` items and image counting for log summaries —
#     Responses wire shape only.
#   * transports/chat_completions: pass-through (native format).
# The one genuinely shared image POLICY — removing images when a server
# rejects them while preserving tool_call_id pairing — already has a single
# owner here: ``_strip_images_from_messages`` above.
