"""Message and tool-payload sanitization helpers (pure; documented in-place mutation).

Walk OpenAI-format message lists and structured payloads, repairing or stripping
characters that would crash ``json.dumps`` in the OpenAI SDK or be rejected upstream.
``run_agent`` re-exports them for old imports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import partial
from typing import Any, Callable

from agent.vision_message_prep import _provider_model_key

logger = logging.getLogger(__name__)

# Lone surrogates are invalid UTF-8 and crash json.dumps in the OpenAI SDK; also used for
# CLI paste scrubbing.
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

# Keys handled explicitly by _sanitize_messages; every OTHER key is swept generically.
_MESSAGE_CORE_KEYS = frozenset({"content", "name", "tool_calls", "role"})

# Provider-native replay state is opaque, may contain hidden or signed reasoning,
# and must never survive a route/provider boundary without matching provenance.
REASONING_REPLAY_SIDECAR_FIELDS = (
    "anthropic_content_blocks",
    "bedrock_content_blocks",
    "codex_reasoning_items",
    "codex_message_items",
)
REASONING_REPLAY_SIDECAR_API_MODES = {
    "anthropic_content_blocks": "anthropic_messages",
    "bedrock_content_blocks": "bedrock_converse",
    "codex_reasoning_items": "codex_responses",
    "codex_message_items": "codex_responses",
}
REASONING_REPLAY_HIDDEN_FIELDS = (
    "_reasoning_route",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    *REASONING_REPLAY_SIDECAR_FIELDS,
)


def strip_non_assistant_reasoning_replay_fields(message: dict) -> None:
    """Remove assistant-only hidden replay state from a non-assistant message."""
    if message.get("role") == "assistant":
        return
    for field in REASONING_REPLAY_HIDDEN_FIELDS:
        message.pop(field, None)


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD; no-op when none present."""
    # ``str.isascii`` is an O(1) flag check; surrogates are never ASCII, so the
    # regex scan only runs for the (rare) non-ASCII leaf.
    if text.isascii():
        return text
    return _SURROGATE_RE.sub('\ufffd', text)


# OpenAI / Anthropic / Responses all bound ``function.name`` to this; one poisoned stored name
# (``multi_tool_use.parallel``, a shell command a weak model put in ``name``) 400s every later
# request on a strict endpoint (#51944).
_VALID_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def coerce_tool_name(name: Any, fallback: str = "invalid_tool_call") -> str:
    """Coerce a *replayed* tool/function name to ``^[A-Za-z0-9_-]{1,64}$``. Valid names are returned
    as-is (identity — prompt-cache safe); invalid runs collapse to ``_`` and the result is cut at 64;
    empty/all-invalid → ``fallback``. Deterministic, so the same stored name always renders the same
    bytes. Never apply to live tool definitions (schema names must match the dispatch registry)."""
    if not isinstance(name, str):
        return fallback
    if _VALID_TOOL_NAME_RE.fullmatch(name):
        return name
    coerced = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "_", name.strip())).strip("_")
    return coerced[:64] or fallback


def _strip_non_ascii(text: str) -> str:
    """Drop non-ASCII characters — last resort for ASCII-only system encodings (LANG=C)."""
    if text.isascii():
        return text
    return text.encode('ascii', errors='ignore').decode('ascii')


def _fix_str_field(container: Any, key: Any, fix: Callable[[str], str]) -> bool:
    """Apply ``fix`` to ``container[key]`` if it is a str; True if it changed."""
    value = container.get(key) if isinstance(container, dict) else container[key]
    fixed = fix(value) if isinstance(value, str) else value
    if fixed == value:
        return False
    container[key] = fixed
    return True


def _sanitize_structure(payload: Any, fix: Callable[[str], str]) -> bool:
    """Apply ``fix`` to every str inside nested dict/list ``payload`` in-place."""
    found = False
    stack = [payload]
    while stack:
        node = stack.pop()
        items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for key, value in list(items):
            if isinstance(value, str):
                found |= _fix_str_field(node, key, fix)
            elif isinstance(value, (dict, list)):
                stack.append(value)
    return found


def _sanitize_messages(messages: list, fix: Callable[[str], str], *, deep: bool) -> bool:
    """Apply ``fix`` to the string fields of every message dict in-place (content / part text,
    name, tool_call arguments, non-core top-level str fields). ``deep=True`` adds tool_call ids,
    function names, and NESTED non-core fields (``reasoning_details`` from byte-level models)."""
    from agent.context_compressor import _DB_PERSISTED_MARKER

    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_found = False
        content = msg.get("content")
        parts = [(p, "text") for p in content if isinstance(p, dict)] if isinstance(content, list) else None
        fields = parts if parts is not None else [(msg, "content")]
        fields.append((msg, "name"))
        tool_calls = msg.get("tool_calls")
        for tc in tool_calls if isinstance(tool_calls, list) else ():
            fn = tc.get("function") if isinstance(tc, dict) else None
            fields += [(tc, "id")] if deep and isinstance(tc, dict) else []
            fields += ([(fn, "name")] if deep else []) + [(fn, "arguments")] if isinstance(fn, dict) else []
        for container, key in fields:
            msg_found |= _fix_str_field(container, key, fix)
        for key, value in [kv for kv in msg.items() if kv[0] not in _MESSAGE_CORE_KEYS]:
            if isinstance(value, str):
                msg_found |= _fix_str_field(msg, key, fix)
            elif deep and isinstance(value, (dict, list)):
                msg_found |= _sanitize_structure(value, fix)
        if msg_found:
            # In-place repair of a live dict stales its persisted row; pop the marker so the
            # flush rewrites it (no-op on api_messages wire copies).
            msg.pop(_DB_PERSISTED_MARKER, None)
            found = True
    return found


# In-place sanitizers; each returns True when anything changed. Surrogate repair is deep
# (tool_call ids, nested reasoning_details); the ASCII-only-locale strip is shallow.
_sanitize_structure_surrogates = partial(_sanitize_structure, fix=_sanitize_surrogates)
_sanitize_messages_surrogates = partial(_sanitize_messages, fix=_sanitize_surrogates, deep=True)
_sanitize_structure_non_ascii = partial(_sanitize_structure, fix=_strip_non_ascii)
_sanitize_messages_non_ascii = partial(_sanitize_messages, fix=_strip_non_ascii, deep=False)
_sanitize_tools_non_ascii = _sanitize_structure_non_ascii


def sanitize_outbound_kwargs(agent: Any, api_kwargs: dict) -> None:
    """Outbound-request chokepoint for every built kwargs dict (main loop and iteration summary).

    Tool descriptions, extra_body and kwargs strings can carry invalid code points that
    providers reject with a non-retryable 400 (#50959); one in-place walk makes the whole
    payload json.dumps()-safe. The ASCII strip is opt-in via the recovery flag set after an
    ASCII-codec rejection.
    """
    _sanitize_structure_surrogates(api_kwargs)
    if agent._force_ascii_payload:
        # ``tools`` is built from ``agent.tools`` per attempt and usually aliases it; detach
        # before the in-place strip so the retry never rewrites the canonical tool schemas.
        # A structural clone suffices: ``_sanitize_structure`` only rebinds str leaves
        # inside dict/list containers.
        if api_kwargs.get("tools") is not None and api_kwargs["tools"] is getattr(agent, "tools", None):
            # Lazy: conversation_loop imports this module (cycle).
            from agent.conversation_loop import _clone_message_for_send

            api_kwargs["tools"] = _clone_message_for_send(api_kwargs["tools"])
        _sanitize_structure_non_ascii(api_kwargs)


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    """Escape literal control chars (0x00-0x1F) inside JSON string values as ``\\uXXXX``
    (for llama.cpp-style output mixing control chars with other malformations)."""
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_string and ch == "\\" and i + 1 < len(raw):
            out.append(raw[i:i + 2])
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        out.append(f"\\u{ord(ch):04x}" if in_string and ord(ch) < 0x20 else ch)
        i += 1
    return "".join(out)


# When a repair rewrites arguments to "{}", the WARNING log is the last surviving copy of
# content that can hold real user data (a truncated write_file), so bound it generously.
_FULL_ARGS_LOG_BOUND = 100_000


def _loads_ok(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


_JSON_CLOSERS = {"{": "}", "[": "]"}


def _rebalance_json_closers(raw: str) -> str | None:
    """Close a JSON prefix's open braces/brackets in stack order, ignoring delimiters
    inside string values (``{"code": "}"}`` keeps one open brace, not a balanced
    document). A closer that does not match the stack top but does match a deeper opener
    gets the missing inner closers inserted BEFORE it: ``{"a": [{"b": 1}}`` → the model
    dropped the ``]`` and let the neighbouring ``}`` close in its place, so the counts
    balance and nothing can be appended. ``None`` when the text ends inside an
    unterminated string — that content is unrecoverable and must not be guessed.
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if in_string:
            if ch == "\\":
                out.append(raw[i:i + 2])
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in _JSON_CLOSERS:
            stack.append(ch)
        elif ch in "}]" and ch in (_JSON_CLOSERS[o] for o in stack):
            while _JSON_CLOSERS[stack[-1]] != ch:
                out.append(_JSON_CLOSERS[stack.pop()])
            stack.pop()
        out.append(ch)
        i += 1
    if in_string:
        return None
    return "".join(out) + "".join(_JSON_CLOSERS[ch] for ch in reversed(stack))


def _repair_tool_call_arguments(raw_args: str, tool_name: str = "?") -> str:
    """Repair malformed tool_call argument JSON (truncation, trailing commas, Python ``None``,
    control chars); ``"{}"`` if unrepairable so the request succeeds. Repairs log at WARNING."""
    raw_stripped = raw_args.strip() if isinstance(raw_args, str) else ""

    if not raw_stripped:
        logger.warning("Sanitized empty tool_call arguments for %s", tool_name)
        return "{}"

    if raw_stripped == "None":
        logger.warning("Sanitized Python-None tool_call arguments for %s", tool_name)
        return "{}"

    # Pass 0: strict=False accepts literal control chars inside strings (the most common
    # local-model case) and re-serialises to wire-valid JSON.
    try:
        reserialised = json.dumps(json.loads(raw_stripped, strict=False), separators=(",", ":"))
        if reserialised != raw_stripped:
            logger.warning("Repaired unescaped control chars in tool_call arguments for %s", tool_name)
        return reserialised
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Passes 2-4: strip trailing commas, close unclosed structures, trim excess closers
    # (bounded). Bracket counting is string-aware: delimiters inside string values
    # ({"code": "}"}) are not structure, and the closers land in stack order — a truncated
    # {"items": [{"n": 1}, {"n": 2 needs "}]}" appended, and a misnested
    # {"a": [{"b": 1}, {"c": 2}} needs "]" inserted before the misplaced "}".
    fixed = re.sub(r",\s*([}\]])", r"\1", raw_stripped)
    fixed = _rebalance_json_closers(fixed) or fixed
    for _ in range(50):
        if _loads_ok(fixed) or not (
            (fixed.endswith('}') and fixed.count('}') > fixed.count('{'))
            or (fixed.endswith(']') and fixed.count(']') > fixed.count('['))
        ):
            break
        fixed = fixed[:-1]

    if _loads_ok(fixed):
        logger.warning("Repaired malformed tool_call arguments for %s: %s → %s", tool_name, raw_stripped[:80], fixed[:80])
        return fixed

    # Pass 5: escape control chars inside strings (strict=False alone fails when other
    # malformations are present too), then retry.
    escaped = _escape_invalid_chars_in_json_strings(fixed)
    if escaped != fixed and _loads_ok(escaped):
        logger.warning(
            "Repaired control-char-laced tool_call arguments for %s: %s → %s", tool_name, raw_stripped[:80], escaped[:80],
        )
        return escaped

    logger.warning(
        "Unrepairable tool_call arguments for %s — replaced with empty object (was: %s)",
        tool_name, raw_stripped[:_FULL_ARGS_LOG_BOUND],
    )
    return "{}"


def close_interrupted_tool_sequence(messages: list, final_response: Any = None) -> bool:
    """Append a synthetic assistant turn when an interrupted tail is a tool result: a transcript
    ending on a raw ``tool`` message makes the next user message land as ``tool → user``, an
    alternation violation strict providers (Gemini, Claude) answer by hallucinating a
    continuation. Mutates in place; True if a closing turn was appended."""
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "tool":
        return False
    text = final_response if isinstance(final_response, str) else ""
    from agent.message_metadata import append_message

    append_message(messages, {"role": "assistant", "content": text.strip() or "Operation interrupted."})
    return True


# finish_reason wire normalization. Some OpenAI-compatible gateways fronting
# Gemini backends emit the native uppercase reasons (STOP, MAX_TOKENS); every
# downstream comparison uses the lowercase OpenAI literals, so an uppercase
# reason silently skips stop handling and length recovery. Single owner —
# call at wire intake (transport normalize_response, stream chunk capture),
# never re-fold at comparison sites.
_FINISH_REASON_ALIASES = {
    "max_tokens": "length",  # Gemini-native / Anthropic-style cap reason
    "end": "stop",  # some gateways' clean-completion spelling
    "function_call": "tool_calls",  # OpenAI legacy pre-tools spelling
}


def normalize_finish_reason(raw: Any) -> Any:
    """Fold a wire ``finish_reason`` to the lowercase OpenAI contract value.

    Non-string and empty values pass through unchanged (callers keep their
    ``or "stop"`` defaults and the Poolside int-reason path); contract values
    are returned byte-identical.
    """
    if not isinstance(raw, str) or not raw:
        return raw
    lowered = raw.lower()
    return _FINISH_REASON_ALIASES.get(lowered, lowered)


def serialized_messages_bytes(messages: list) -> int:
    """Exact serialized byte size of ``messages`` (HTTP 413 is a BYTE-size error the token
    estimator, pricing images flat, cannot score). Non-serializable values fall back to
    ``str()`` so a malformed message can never crash recovery."""
    if not isinstance(messages, list) or not messages:
        return 0
    try:
        return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return sum(len(str(m)) for m in messages)


_IMAGE_PART_TYPES = {"image_url", "image", "input_image"}


def _strip_images_from_messages(messages: list) -> bool:
    """Remove image content parts from all messages in-place (server rejected images).

    ``tool`` / ``tool_calls`` messages left empty get a placeholder, NOT deleted (deleting
    orphans the paired ``tool_call_id`` → HTTP 400); other now-empty messages are dropped.
    Rewritten messages lose their ``api_content`` sidecar (it carries the removed images):
    a caller rewriting a persisted row must not leave bytes that replay them next turn. The
    current callers pass per-call clones, where this is a no-op.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.turn_context import drop_stale_api_content

    found = False
    to_delete = []
    for i, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        new_parts = [p for p in content if not (isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES)]
        if len(new_parts) < len(content):
            found = True
            if new_parts:
                msg["content"] = new_parts
                # Rewriting a stamped live dict stales its persisted row; pop the marker.
                msg.pop(_DB_PERSISTED_MARKER, None)
            elif msg.get("role") == "tool" or msg.get("tool_calls"):
                msg["content"] = "[image content removed — server does not support images]"
                msg.pop(_DB_PERSISTED_MARKER, None)
            else:
                to_delete.append(i)
            drop_stale_api_content(msg)
    for i in reversed(to_delete):
        del messages[i]
    return found


# Provider error bodies (lowercased substring match) meaning "image/multimodal input
# unsupported" — the loop then strips images and retries text-only instead of cascading
# into compression / context-too-large recovery or wedging on retries.
_IMAGE_REJECTION_PHRASES = (
    "only 'text' content type is supported", "only text content type is supported",
    "image_url is not supported", "image content is not supported",
    "multimodal is not supported", "multimodal content is not supported", "multimodal input is not supported",
    "vision is not supported", "vision input is not supported",
    "does not support images", "does not support image input", "does not support multimodal",
    "does not support vision", "model does not support image",
    # DashScope-style gateways reject non-text blocks with this generic body.
    # Some OpenAI-compatible endpoints (e.g. (issue #57948)
    "unexpected item type in content",
    # ChatGPT-account Codex backend rejects data:image URLs in input_image; keyed on the
    # field-path apostrophe so other URL errors don't false-trip.
    "image_url'. expected",
    # DeepSeek's text-only request-body variant error.
    "unknown variant `image_url`, expected `text`", "unknown variant image_url, expected text",
    # OpenRouter HTTP 404 when no upstream endpoint accepts image input (passes the 4xx
    # gate; without this the gateway queue wedges behind the stuck turn).
    # Without this phrase the agent never strips the images, the retry loop re-sends the same rejected
    # request until exhaustion, and the gateway leaves every subsequent message queued behind the stuck turn
    # — the P1 in issue #21160.
    "no endpoints found that support image input",
)

# Provider error bodies meaning "this particular image payload is bad" — the model CAN see, it
# just could not decode what it was sent. Disjoint from ``_IMAGE_REJECTION_PHRASES``: the turn
# recovers the same way (strip and retry) but must NOT remember the model as image-rejecting,
# or the next request with a good image would be needlessly stripped for the rest of the session.
_IMAGE_CORRUPT_PHRASES = (
    # ChatGPT-account Codex backend's wording for corrupt/unsupported native image payloads.
    "image data you provided does not represent a valid image",
    # Kimi/Moonshot et al. reject truncated/corrupt image bytes baked into history.
    # Kimi / Moonshot / other OpenAI-compatible Chinese providers reject truncated or corrupt image bytes
    # with HTTP 400 "Invalid request: prepare image failed ... failed to decode image: invalid or
    # unsupported image format". Like the Codex case above, the bad bytes are baked into immutable
    # conversation history and re-sent on every retry, wedging the session. Strip the images so the turn
    # recovers instead of exhausting retries. (issue #76884; complements the proactive full-decode
    # validation in tools/vision_tools._normalize_to_supported_image)
    "failed to decode image",
)

def strip_images_for_rejecting_model(agent: Any, api_messages: Any) -> bool:
    """Send-path image strip for a model that rejected image content (see turn_recovery).

    Runs on the per-call ``api_messages`` copy in Hermes's own message format, BEFORE the
    provider-specific conversion: the part types this stripper knows are that format's, and a
    converted payload (Bedrock Converse ``{"image": ...}`` blocks carry no ``type``) would slip
    past it. History is never touched. Keyed on each rejecting (provider, model), so a model
    that accepts images gets them again.
    """
    if _provider_model_key(agent) not in agent._image_rejecting_models:
        return False
    return isinstance(api_messages, list) and _strip_images_from_messages(api_messages)


def _looks_like_image_content_rejection(error_body: str) -> bool:
    """Return True when a provider error says image/multimodal input is unsupported."""
    body = str(error_body or "").lower()
    return any(phrase in body for phrase in _IMAGE_REJECTION_PHRASES)


def _looks_like_corrupt_image_rejection(error_body: str) -> bool:
    """Return True when the rejection is about a bad image payload, not the model's capability."""
    body = str(error_body or "").lower()
    return any(phrase in body for phrase in _IMAGE_CORRUPT_PHRASES)


__all__ = [
    "_SURROGATE_RE", "close_interrupted_tool_sequence",
    "_sanitize_surrogates", "_sanitize_structure_surrogates", "_sanitize_messages_surrogates",
    "coerce_tool_name",
    "_escape_invalid_chars_in_json_strings", "_repair_tool_call_arguments",
    "_strip_non_ascii", "_sanitize_messages_non_ascii", "_sanitize_tools_non_ascii",
    "_strip_images_from_messages", "_sanitize_structure_non_ascii", "sanitize_outbound_kwargs",
    "strip_images_for_rejecting_model",
    # call_id policy owners
    "deterministic_call_id", "coalesce_tool_call_id", "tool_call_id_variants",
    "tool_result_id_variants", "uniquify_tool_call_ids",
    # reasoning_content policy owners
    "reasoning_echo_family", "matches_reasoning_echo_family", "needs_reasoning_echo",
    "stale_thinking_reaches_wire", "apply_reasoning_content_policy", "reapply_reasoning_echo",
]


# -- call_id policy: hash synthesis, ``call_id or id`` coalescing, duplicate-id repair ----
# NOT merged with codex_event_projector._deterministic_call_id (maps app-server ITEM ids,
# not chat tool-call content; merging would change ids and invalidate caches).
# HARD INVARIANT: deterministic (never uuid4) and byte-identical for existing inputs —
# these ids feed prompt-cache prefixes.


def _tc_field(tc: Any, key: str) -> Any:
    """Read ``key`` from a tool-call entry that may be a dict or an SDK object."""
    return tc.get(key) if isinstance(tc, dict) else getattr(tc, key, None)


def _tc_set(tc: Any, key: str, value: Any) -> None:
    tc.__setitem__(key, value) if isinstance(tc, dict) else setattr(tc, key, value)


# --------------------------------------------------------------------------- call_id policy — single owner
# (audit F4, incident chain I4) ---------------------------------------------------------------------------
# Three forked policy sites converged here: * agent/codex_responses_adapter.py `_deterministic_call_id` —
# hash synthesis when a provider omits call_id (fa3ab2ffd0 → e45f2b39e2). *
# run_agent.AIAgent._get_tool_call_id_static — `call_id or id` coalescing for dicts and SDK objects. *
# run_agent.AIAgent._uniquify_tool_call_ids — duplicate-id repair with deterministic `_d<n>` suffixes
# (#58327 loss class). NOT consolidated (different scheme on purpose):
# agent/transports/codex_event_projector._deterministic_call_id maps codex app-server ITEM ids
# (`codex_<type>_<item_id>`), not chat tool-call content; merging the two would change ids and invalidate
# prompt caches. HARD INVARIANT: everything here must stay deterministic (never uuid4) and byte-identical
# for existing inputs — these ids feed prompt-cache prefixes.
def deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Deterministic call_id fallback when the API omits one (random ids would break caching)."""
    seed = f"{fn_name}:{arguments}:{index}"
    return f"call_{hashlib.sha256(seed.encode('utf-8', errors='replace')).hexdigest()[:12]}"


def _expand_tool_id_variants(values: tuple[Any, ...]) -> frozenset[str]:
    """Every wire spelling of one tool-call identifier: Responses bridges may expose the pairing
    id and response-item id separately or as ``call_id|response_item_id``; all alias ONE call."""
    variants: set[str] = set()
    for raw in values:
        value = raw.strip() if isinstance(raw, str) else ""
        if value:
            variants.add(value)
            variants.update(p for p in (part.strip() for part in value.split("|")) if p)
    return frozenset(variants)


def tool_call_id_variants(tc: Any) -> frozenset[str]:
    """Return all pairing-id variants carried by a tool-call entry."""
    return _expand_tool_id_variants(tuple(_tc_field(tc, k) for k in ("call_id", "id", "response_item_id")))


def tool_result_id_variants(tool_call_id: Any) -> frozenset[str]:
    """Return all matching variants for a role=tool ``tool_call_id``."""
    return _expand_tool_id_variants((tool_call_id,))


def coalesce_tool_call_id(tc: Any) -> str:
    """Effective call id of a tool_call entry (dict or object); ``""`` when none. Codex Responses
    carry ``call_id`` (authoritative pairing key), Chat Completions ``id`` only, and bridge ids
    may be ``call_id|response_item_id``."""
    for raw in (_tc_field(tc, "call_id"), _tc_field(tc, "id")):
        value = raw.strip() if isinstance(raw, str) else ""
        if value:
            return value.split("|", 1)[0].strip() or value
    return ""


def uniquify_tool_call_ids(tool_calls: list) -> list:
    """Ensure every tool call in one assistant turn has a distinct id.

    Some providers reuse one id across a batch; the pre-API sanitizer then keeps only the
    first call/result pair per id and strict providers reject duplicates. Later collisions
    get a deterministic ``<id>_d<n>`` suffix (never uuid4 — cache-prefix stability). Mutates
    entries (SDK models / SimpleNamespace / dicts) in place. Blank ids are left for the
    deterministic fallback in ``build_assistant_message``.
    """
    seen: set = set()
    for tc in tool_calls or []:
        # Same coalescing rule as coalesce_tool_call_id, tolerant of non-string ids.
        raw = _tc_field(tc, "call_id") or _tc_field(tc, "id") or ""
        raw = raw.strip() if isinstance(raw, str) else ""
        # Composite Responses ids ("call_x|fc_y") collide on the call half — the pairing key.
        cid = raw.split("|", 1)[0]
        if not cid:
            continue
        if cid not in seen:
            seen.add(cid)
            continue
        # range is bounded: at most len(seen) suffixes can already be taken.
        new_id = next(f"{cid}_d{n}" for n in range(2, len(seen) + 3) if f"{cid}_d{n}" not in seen)
        seen.add(new_id)

        try:
            # Keep a composite id's response-item half so the provider's fc_/item id survives.
            old = _tc_field(tc, "id")
            _tc_set(tc, "id", f"{new_id}|{old.split('|', 1)[1]}" if isinstance(old, str) and "|" in old else new_id)
            if _tc_field(tc, "call_id"):
                _tc_set(tc, "call_id", new_id)
        except Exception:
            logger.warning("Could not uniquify duplicate tool call id %s", cid)
            continue
        _fn_name = _tc_field(_tc_field(tc, "function"), "name") or "?"
        logger.warning(
            "Model reused tool call id %s within one turn; renamed the duplicate to %s (tool=%s) to keep "
            "call/result pairing lossless.", cid, new_id, _fn_name,
        )
    return tool_calls


# -- reasoning_content policy: single owner of strip-vs-re-pad; adapters keep only SYNTAX --
# Require side (echo-back enforced; replays 400 without the field): the families below. Kimi
# is host-driven on purpose (aggregators re-exporting kimi reject it); DeepSeek V4 rejects
# empty-string pads → " ". Strict side (400/422 "Extra inputs are not permitted"): everyone
# else — Mistral, Cerebras, Groq, SambaNova, … Strip the key entirely, even a one-space pad.

# --------------------------------------------------------------------------- reasoning_content policy —
# single owner (audit F4) --------------------------------------------------------------------------- The
# strip-vs-repad decision was previously forked across the wire files in separate incident commits
# (2b3a4f0af8 strip for strict providers, b5495db701 re-pad for require-side, 94b3131be7/9a9f8a6d99 kimi
# pad). The POLICY — which provider direction gets which treatment — lives here as one rule table + apply
# functions; adapters keep only SYNTAX mapping (e.g. anthropic_adapter turning reasoning_content into a
# thinking block). Direction table: require-side (echo-back enforced; replays 400 without the field): kimi
# — provider kimi-coding/kimi-coding-cn, or host api.kimi.com / moonshot.ai / moonshot.cn. Host-driven on
# purpose: aggregators re-exporting kimi models reject the echo. deepseek — provider "deepseek", model
# contains "deepseek", or host api.deepseek.com (#15250; V4 rejects empty-string pads, hence the " "
# single-space pad, #17341). mimo     — provider "xiaomi", model contains "mimo", or host *.xiaomimimo.com.
# strict side (field rejected with 400/422 "Extra inputs are not permitted"): everyone else — Mistral,
# Cerebras, Groq, SambaNova, … (#45655). Strip the key entirely, even a single-space pad.
_REASONING_ECHO_RULES: tuple = (
    # (family, exact providers (raw), exact providers (lowered), model substrings (lowered), hosts)
    ("kimi", frozenset({"kimi-coding", "kimi-coding-cn"}), frozenset(), (), ("api.kimi.com", "moonshot.ai", "moonshot.cn")),
    ("deepseek", frozenset(), frozenset({"deepseek"}), ("deepseek",), ("api.deepseek.com",)),
    ("mimo", frozenset(), frozenset({"xiaomi"}), ("mimo",), ("api.xiaomimimo.com", "xiaomimimo.com")),
)
_REASONING_ECHO_RULE_BY_FAMILY = {rule[0]: rule for rule in _REASONING_ECHO_RULES}


def matches_reasoning_echo_family(family: str, provider: Any, model: Any, base_url: Any) -> bool:
    """True when (provider, model, base_url) matches one echo-back family (families can overlap;
    membership is tested independently). Raises KeyError for an unknown family."""
    from utils import base_url_host_matches

    _, raw_providers, lowered_providers, model_subs, hosts = _REASONING_ECHO_RULE_BY_FAMILY[family]
    model_lower = (model or "").lower()
    return (
        provider in raw_providers or (provider or "").lower() in lowered_providers
        or any(sub in model_lower for sub in model_subs) or any(base_url_host_matches(base_url, host) for host in hosts)
    )


def reasoning_echo_family(provider: Any, model: Any, base_url: Any) -> "str | None":
    """``"kimi"`` / ``"deepseek"`` / ``"mimo"`` (first match in table order) when the
    endpoint enforces reasoning_content echo-back, else ``None`` (strip side)."""
    families = (rule[0] for rule in _REASONING_ECHO_RULES)
    return next((f for f in families if matches_reasoning_echo_family(f, provider, model, base_url)), None)


def needs_reasoning_echo(provider: Any, model: Any, base_url: Any) -> bool:
    """True when the endpoint requires reasoning_content echo-back."""
    return reasoning_echo_family(provider, model, base_url) is not None


# Named local/self-hosted providers that may opt into soft historical replay.
# ``custom`` / ``custom:<key>`` are handled separately by prefix.
_SELF_HOSTED_REASONING_REPLAY_PROVIDERS = frozenset(
    {"local", "llamacpp", "lm-studio", "ollama", "vllm"}
)


def _is_self_hosted_reasoning_replay_provider(provider: Any) -> bool:
    provider_lower = str(provider or "").strip().lower()
    return (
        provider_lower == "custom"
        or provider_lower.startswith("custom:")
        or provider_lower in _SELF_HOSTED_REASONING_REPLAY_PROVIDERS
    )


# Loopback classification and its edge-case policy are adapted from PR #87123
# by glitchbunny0.  Automatic carrier selection remains narrower here: server
# identity alone is not enough except for llama.cpp's demonstrated
# --reasoning-preserve / reasoning_content contract.
def _is_loopback_reasoning_route(base_url: Any) -> bool:
    import ipaddress

    from utils import base_url_hostname

    host = base_url_hostname(base_url)
    if not host:
        return False
    host = host.strip().lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(
        address.is_loopback
        or address.is_unspecified
        or (mapped is not None and (mapped.is_loopback or mapped.is_unspecified))
    )


def resolve_reasoning_replay_field(
    configured_mode: Any,
    *,
    provider: Any,
    model: Any,
    base_url: Any,
    api_mode: Any,
    detected_server_type: Any = None,
) -> "str | None":
    """Resolve the soft-replay wire carrier for one provider route.

    Explicit carriers work on custom/self-hosted Chat Completions routes,
    including remote endpoints.  Omission is ``auto`` and enables only a
    directly evidenced loopback llama.cpp contract.  ``none`` and ambiguous
    routes fail closed.  Require-side echo protocols are handled separately.
    """
    mode = configured_mode.strip().lower() if isinstance(configured_mode, str) else "auto"
    transport = str(api_mode or "").strip().lower()
    if transport not in {"", "chat_completions", "openai"}:
        return None
    if needs_reasoning_echo(provider, model, base_url):
        return None
    if mode == "none":
        return None
    if not _is_self_hosted_reasoning_replay_provider(provider):
        return None
    if mode in {"reasoning", "reasoning_content"}:
        return mode
    if mode != "auto" or not _is_loopback_reasoning_route(base_url):
        return None
    if str(detected_server_type or "").strip().lower() == "llamacpp":
        return "reasoning_content"
    return None


def stale_thinking_reaches_wire(
    api_mode: Any,
    provider: Any,
    model: Any,
    base_url: Any,
    *,
    reasoning_replay_field: Any = None,
) -> bool:
    """True when stale assistant reasoning text is actually replayed on the wire for the route.

    The single wire-truth predicate the compaction TRIGGER estimator and the tail-budget
    walks must share: if they disagree, a reasoning-heavy session can look over-threshold
    to preflight yet fully tail-protected to the walk — an infinite compaction loop.
    ``codex_responses`` never reads the text keys (continuity rides the encrypted sidecar).
    """
    if (api_mode or "") == "codex_responses":
        return False
    return (
        needs_reasoning_echo(provider, model, base_url)
        or reasoning_replay_field in {"reasoning", "reasoning_content"}
    )


def apply_reasoning_content_policy(
    source_msg: dict,
    api_msg: dict,
    needs_thinking_pad: bool,
    reasoning_replay_field: str | None = None,
) -> None:
    """Copy provider-facing reasoning fields onto an API replay message.

    ``needs_thinking_pad`` is the require-side flag (see
    ``needs_reasoning_echo`` / the agent's cached
    ``_needs_thinking_reasoning_pad``). ``reasoning_replay_field`` selects an
    opt-in soft replay carrier for endpoints that consume historical reasoning
    without requiring fabricated pads. Mutates ``api_msg`` in place.
    """
    if source_msg.get("role") != "assistant":
        return

    if reasoning_replay_field == "reasoning_content":
        normalized_reasoning = source_msg.get("reasoning")
        existing = source_msg.get("reasoning_content")
        replay = (
            existing
            if isinstance(existing, str) and existing.strip()
            else normalized_reasoning
            if isinstance(normalized_reasoning, str) and normalized_reasoning.strip()
            else None
        )
        api_msg.pop("reasoning", None)
        if replay is None:
            api_msg.pop("reasoning_content", None)
        else:
            api_msg["reasoning_content"] = replay
        return

    if reasoning_replay_field == "reasoning":
        normalized_reasoning = source_msg.get("reasoning")
        existing = source_msg.get("reasoning_content")
        replay = (
            normalized_reasoning
            if isinstance(normalized_reasoning, str) and normalized_reasoning.strip()
            else existing
            if isinstance(existing, str) and existing.strip()
            else None
        )
        api_msg.pop("reasoning_content", None)
        if replay is None:
            api_msg.pop("reasoning", None)
        else:
            api_msg["reasoning"] = replay
        return

    if not needs_thinking_pad:
        # Strict side: never carry the field — a reasoning primary pads history with " ",
        # then a fallback to Mistral/Cerebras/Groq replays the pad and 422s. Also drops a
        # non-string value (None after compaction): never pass null to the API.
        api_msg.pop("reasoning_content", None)
        return
    existing, reasoning = source_msg.get("reasoning_content"), source_msg.get("reasoning")
    # 1. Explicit reasoning_content already set. When the active provider enforces the thinking-mode
    #   echo-back (DeepSeek / Kimi / MiMo), preserve it verbatim — that includes their own space-placeholder
    #   written at creation time and any valid reasoning from the same provider. Sessions persisted BEFORE
    #   #17341 have empty-string placeholders pinned at creation time; DeepSeek V4 Pro rejects those with
    #   HTTP 400, so upgrade "" → " " on replay. When the active provider does NOT enforce echo-back, strip
    #   the field entirely. Strict OpenAI-compatible providers (Mistral, Cerebras, Groq, SambaNova, …)
    #   reject ANY reasoning_content key in input messages with HTTP 400/422 ("Extra inputs are not
    #   permitted"), even an empty string or a single-space pad. Stripping here covers the rebuild path;
    #   ``reapply_reasoning_echo`` covers the already-built api_messages path. Refs #45655.
    if isinstance(existing, str):
        # Explicit value: preserve verbatim, upgrading legacy "" to " " (DeepSeek V4 400s on "").
        api_msg["reasoning_content"] = existing or " "
    elif isinstance(reasoning, str) and reasoning and not source_msg.get("tool_calls"):
        # Healthy session: promote internal 'reasoning' → 'reasoning_content'.
        api_msg["reasoning_content"] = reasoning
    else:
        # tool_calls + 'reasoning' but no 'reasoning_content' means the reasoning came from
        # ANOTHER provider (DeepSeek's own build pins reasoning_content for tool-call turns):
        # pad without leaking foreign CoT. No reasoning at all: every assistant turn still needs
        # the field; " " (not "") because DeepSeek V4 rejects empty string.
        api_msg["reasoning_content"] = " "


def reapply_reasoning_echo(
    api_messages: list,
    needs_thinking_pad: bool,
    reasoning_replay_field: str | None = None,
    provider_boundary: bool = False,
) -> int:
    """Reconcile provider-facing reasoning fields for the active provider.

    ``api_messages`` is built once before the retry loop. Canonical source
    history is intentionally not imported here: a fallback may target a
    different provider/model, and forwarding another model's hidden trace would
    be a cross-provider privacy leak. Primary restoration rebuilds a fresh wire
    message list from canonical history on the next request instead. The
    ``provider_boundary`` means the already-built messages came from a different
    provider. In that case structured carriers and provider-specific hidden
    blocks are discarded before the destination policy is applied, so one
    provider's hidden trace cannot be forwarded to another opt-in replay route.
    The operation is idempotent and returns the number of assistant turns
    changed.
    """
    changed = 0
    for api_msg in api_messages:
        if not isinstance(api_msg, dict) or api_msg.get("role") != "assistant":
            continue
        before = dict(api_msg)
        if provider_boundary:
            api_msg.pop("reasoning_content", None)
            api_msg.pop("reasoning", None)
            api_msg.pop("reasoning_details", None)
            for field in REASONING_REPLAY_SIDECAR_FIELDS:
                api_msg.pop(field, None)
        source_msg = api_msg
        if provider_boundary and needs_thinking_pad:
            apply_reasoning_content_policy(
                source_msg, api_msg, needs_thinking_pad=True
            )
        elif reasoning_replay_field in {"reasoning", "reasoning_content"}:
            apply_reasoning_content_policy(
                source_msg,
                api_msg,
                needs_thinking_pad=False,
                reasoning_replay_field=reasoning_replay_field,
            )
        elif needs_thinking_pad:
            if not api_msg.get("reasoning_content"):
                apply_reasoning_content_policy(
                    api_msg, api_msg, needs_thinking_pad
                )
        else:
            api_msg.pop("reasoning_content", None)
            api_msg.pop("reasoning", None)
        if api_msg != before:
            changed += 1
    return changed


# Image / multimodal parts are deliberately NOT consolidated here: per-adapter handling is
# format-specific SYNTAX. The one shared image POLICY is ``_strip_images_from_messages``.
