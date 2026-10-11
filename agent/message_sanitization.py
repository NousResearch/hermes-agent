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
from typing import Any, Callable, Iterable, NamedTuple

from agent.agent_runtime_helpers_placeholders import _INTERRUPTED_PLACEHOLDER, hidden_interrupt_placeholder_row
from agent.message_metadata import DB_ROW_SNAPSHOT
from agent.vision_message_prep import _provider_model_key

logger = logging.getLogger(__name__)

# Lone surrogates are invalid UTF-8 and crash json.dumps in the OpenAI SDK; also used for
# CLI paste scrubbing.
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

# Keys handled explicitly by _sanitize_messages; every OTHER key is swept generically.
# The durable snapshot is an immutable compare-and-swap version, not message payload.
_MESSAGE_CORE_KEYS = frozenset({"content", "name", "tool_calls", "role", DB_ROW_SNAPSHOT})


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
    continuation. Mutates in place; True if a closing turn was appended.

    Only the placeholder closes silently: with no real text (or just the bare interrupt
    placeholder) the row is hidden from the user — ``api_content`` carries the LLM-visible
    text (substituted at API-build time by ``substitute_api_content``), ``content=""`` +
    ``display_kind="hidden"`` keep it out of rendered transcripts, matching the
    ``_INTERRUPTED_PLACEHOLDER`` shape in ``turn_api_call.py``. A caller-supplied banner
    (truncation notices, partial-delivery text) stays visible: it is the turn's only
    user-facing explanation."""
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "tool":
        return False
    text = final_response if isinstance(final_response, str) else ""
    from agent.message_metadata import append_message

    stripped = text.strip()
    if not stripped or stripped == _INTERRUPTED_PLACEHOLDER:
        append_message(messages, hidden_interrupt_placeholder_row())
    else:
        append_message(messages, {"role": "assistant", "content": stripped})
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
    "_SURROGATE_RE", "_escape_invalid_chars_in_json_strings", "_repair_tool_call_arguments",
    "_sanitize_messages_non_ascii", "_sanitize_messages_surrogates", "_sanitize_structure_non_ascii",
    "_sanitize_structure_surrogates", "_sanitize_surrogates", "_sanitize_tools_non_ascii",
    "_strip_images_from_messages", "_strip_non_ascii", "apply_reasoning_content_policy",
    "close_interrupted_tool_sequence", "coalesce_tool_call_id", "coerce_tool_name",
    "deterministic_call_id", "matches_reasoning_echo_family", "needs_reasoning_echo",
    "normalize_provider_tool_call_ids", "reapply_reasoning_echo", "reasoning_echo_family",
    "reasoning_replay_route", "record_reasoning_field_rejection", "rejected_reasoning_carriers",
    "route_reasoning_carriers",
    "sanitize_outbound_kwargs", "stale_thinking_reaches_wire", "strip_images_for_rejecting_model",
    "tool_call_id_variants", "tool_result_id_variants", "uniquify_tool_call_ids",
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


def uniquify_tool_call_ids(tool_calls: list, taken: Iterable[str] = ()) -> list:
    """Ensure every tool call in one assistant turn has an id no other call in the session has.

    Some providers reuse one id across a batch, and some name every call ``call_0`` turn after
    turn; the pre-API sanitizer then keeps only the first call/result pair per id, strict
    providers reject duplicates, and the desktop binds a tool card to the wrong call. ``taken``
    is the ids already in this session's history: they are never rewritten (prompt cache), so
    the incoming call is renamed instead. Collisions get a deterministic ``<id>_d<n>`` suffix
    (never uuid4 — cache-prefix stability). Mutates entries (SDK models / SimpleNamespace /
    dicts) in place. Blank ids are left for the deterministic fallback in
    ``build_assistant_message``.
    """
    seen: set = set(taken)
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
            "Model reused tool call id %s; renamed the duplicate to %s (tool=%s) to keep call/result "
            "pairing lossless.", cid, new_id, _fn_name,
        )
    return tool_calls


_PROVIDER_TOOL_ID_PREFIXES = ("chatcmpl-tool-",)

def normalize_provider_tool_call_ids(tool_calls: list) -> list:
    """Rewrite known provider ids when a parallel batch would be rejected on replay.

    The digest is deterministic so persisted messages and prompt-cache prefixes remain
    stable. Composite Responses ids retain their response-item half.
    """
    if len(tool_calls or []) < 2:
        return tool_calls
    # Gate on the effective id serialization and result pairing use (stripped, blank call_id
    # falls back to id), not on raw fields.
    if not all(coalesce_tool_call_id(tc).startswith(_PROVIDER_TOOL_ID_PREFIXES) for tc in tool_calls):
        return tool_calls
    logger.warning("Normalized provider-minted parallel tool-call ids for replay compatibility")
    for tc in tool_calls:
        # Rewrite each field's call half separately: ``id`` may carry the response-item
        # half while ``call_id`` is bare, and that half must survive.
        for key in ("id", "call_id"):
            value = _tc_field(tc, key)
            if not isinstance(value, str):
                continue
            primary, sep, item = value.strip().partition("|")
            primary = primary.strip()
            if not primary.startswith(_PROVIDER_TOOL_ID_PREFIXES):
                continue
            # surrogatepass: provider JSON can carry lone surrogates; strict utf-8 would raise,
            # and errors=replace would collapse distinct ids onto one digest.
            digest = hashlib.sha256(primary.encode("utf-8", "surrogatepass")).hexdigest()[:12]
            _set_provider_tool_id(tc, key, f"call_{digest}{sep}{item}")
    return tool_calls


def _set_provider_tool_id(tc: Any, key: str, value: str) -> None:
    # transports.types.ToolCall exposes call_id as a read-only view of provider_data;
    # write the backing value so id and call_id stay in agreement.
    if isinstance(getattr(type(tc), key, None), property) and isinstance(getattr(tc, "provider_data", None), dict):
        tc.provider_data[key] = value
    else:
        _tc_set(tc, key, value)


# -- reasoning replay policy: single owner; adapters keep only SYNTAX ----------------------
# Every model reasons and intends that reasoning to be replayed, so every chat-completions
# route gets its stored reasoning back BY DEFAULT, on the carrier(s) that route reads. Two
# tables own the decision:
#   * ``_REASONING_ECHO_RULES`` — the MUST-ECHO tier: a tool-call turn without a non-empty
#     ``reasoning_content`` is an HTTP 400 there, so a missing value is padded with " " (never
#     "": DeepSeek V4 rejects it, #17341). Kimi is host-driven on purpose (aggregators
#     re-exporting kimi do not need the pad).
#   * ``_REASONING_CARRIER_ROUTES`` — which carriers a route reads (first match wins). Strict
#     hosts come first: their schemas reject unknown message keys with 400/422 ("Extra inputs
#     are not permitted" / "property X is unsupported" / "no such field", #45655, #70233).
#     Every other route defaults to ``reasoning_content``, which is the de-facto standard.
# A route that still rejects a field is learned at runtime (``record_reasoning_field_rejection``)
# and the field is dropped for that (provider, host, model) only. Stored history keeps every
# carrier; only wire copies are shaped here.
_REASONING_ECHO_RULES: tuple = (
    # (family, exact providers (raw), exact providers (lowered), model substrings (lowered), hosts)
    ("kimi", frozenset({"kimi-coding", "kimi-coding-cn"}), frozenset(), (), ("api.kimi.com", "moonshot.ai", "moonshot.cn")),
    ("deepseek", frozenset(), frozenset({"deepseek"}), ("deepseek",), ("api.deepseek.com",)),
    ("mimo", frozenset(), frozenset({"xiaomi"}), ("mimo",), ("api.xiaomimimo.com", "xiaomimimo.com")),
    # Portal stealth model that 400s on a tool-call turn without reasoning_content (f86276d2ebc).
    ("missingno", frozenset(), frozenset(), ("stealth/missingno",), ()),
)
_REASONING_ECHO_RULE_BY_FAMILY = {rule[0]: rule for rule in _REASONING_ECHO_RULES}

RC, R, RD = "reasoning_content", "reasoning", "reasoning_details"
REASONING_CARRIERS = (RC, R, RD)
_ALL_CARRIERS = frozenset(REASONING_CARRIERS)
_DEFAULT_CARRIERS = frozenset({RC})
_LOCAL_CARRIERS = frozenset({RC, R})  # llama.cpp/SGLang read reasoning_content, vLLM/Ollama read reasoning

_REASONING_CARRIER_ROUTES: tuple = (
    # (carriers, providers (lowered), hosts) — evidence per row in the PR body / FINDINGS.
    (frozenset(), frozenset({"mistral", "groq", "cerebras", "sambanova"}),
     ("mistral.ai", "groq.com", "cerebras.ai", "sambanova.ai", "hunyuan.cloud.tencent.com")),
    # Fireworks' ChatMessage schema is additionalProperties:false with reasoning_content only.
    (frozenset({RC}), frozenset({"fireworks"}), ("fireworks.ai",)),
    # OpenRouter-format gateways read the unified array, the string and the alias.
    (_ALL_CARRIERS, frozenset({"openrouter", "kilocode", "ai-gateway", "nous"}),
     ("openrouter.ai", "kilo.ai", "ai-gateway.vercel.sh", "nousresearch.com")),
    # Vendors that document reasoning_details replay next to reasoning_content.
    (frozenset({RC, RD}), frozenset({"novita", "tencent-tokenhub", "minimax", "minimax-cn"}),
     ("novita.ai", "tokenhub.tencentmaas.com", "tokenhub-intl.tencentcloudmaas.com", "minimax.io", "minimaxi.com")),
    (_LOCAL_CARRIERS, frozenset({"ollama-cloud", "custom", "lmstudio", "upstage"}), ("ollama.com", "upstage.ai")),
)


class ReasoningReplayRoute(NamedTuple):
    """``pad``: must-echo tier (missing reasoning_content -> " "). ``carriers``: the reasoning
    keys an assistant wire copy carries; ``None`` outside chat_completions, where the native
    adapters own reasoning replay (only the must-echo pad is applied, details are untouched)."""

    pad: bool
    carriers: frozenset | None


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


def reasoning_echo_family(provider: Any, model: Any, base_url: Any) -> str | None:
    """``"kimi"`` / ``"deepseek"`` / ``"mimo"`` (first match in table order) when the endpoint
    enforces reasoning_content echo-back on tool-call turns, else ``None``."""
    families = (rule[0] for rule in _REASONING_ECHO_RULES)
    return next((f for f in families if matches_reasoning_echo_family(f, provider, model, base_url)), None)


def needs_reasoning_echo(provider: Any, model: Any, base_url: Any) -> bool:
    """True when the endpoint requires reasoning_content echo-back (the must-echo tier)."""
    return reasoning_echo_family(provider, model, base_url) is not None


def route_reasoning_carriers(provider: Any, base_url: Any) -> frozenset:
    """Reasoning carriers the chat-completions route reads (before any runtime rejection)."""
    from utils import base_url_host_matches

    provider_lower = (provider or "").strip().lower()
    carriers = next(
        (c for c, providers, hosts in _REASONING_CARRIER_ROUTES
         if provider_lower in providers or any(base_url_host_matches(base_url, h) for h in hosts)),
        None,
    )
    if carriers is None:
        from agent.model_metadata import is_local_endpoint

        carriers = _LOCAL_CARRIERS if base_url and is_local_endpoint(str(base_url)) else _DEFAULT_CARRIERS
    if RD not in carriers and carriers and _profile_declares_native_details(provider_lower):
        carriers = carriers | {RD}
    return carriers


def _profile_declares_native_details(provider_lower: str) -> bool:
    """A profile declaring ``native_reasoning_details_type`` consumes replayed details by contract."""
    if not provider_lower:
        return False
    from providers import get_provider_profile

    return bool(getattr(get_provider_profile(provider_lower), "native_reasoning_details_type", None))


def reasoning_replay_route(
    api_mode: Any, provider: Any, model: Any, base_url: Any, *, echo_opt_in: bool = False,
    rejected: Iterable[str] = (),
) -> ReasoningReplayRoute:
    """Replay decision for one route. Deterministic in its inputs, so the wire prefix is byte-stable
    across turns and changes only on a route switch or a recorded rejection (``rejected``)."""
    pad = bool(echo_opt_in) or needs_reasoning_echo(provider, model, base_url)
    if (api_mode or "chat_completions") != "chat_completions":
        return ReasoningReplayRoute(pad, None)
    carriers = route_reasoning_carriers(provider, base_url) - frozenset(rejected)
    # A strict route never gets a reasoning key, not even the must-echo pad.
    return ReasoningReplayRoute(pad and RC in carriers, carriers)


def stale_thinking_reaches_wire(api_mode: Any, provider: Any, model: Any, base_url: Any) -> bool:
    """True when stale assistant reasoning text is actually replayed on the wire for the route.

    The single wire-truth predicate the compaction TRIGGER estimator and the tail-budget
    walks must share: if they disagree, a reasoning-heavy session can look over-threshold
    to preflight yet fully tail-protected to the walk — an infinite compaction loop.
    ``codex_responses`` never reads the text keys (continuity rides the encrypted sidecar).
    A runtime rejection is not visible here; both sides then over-charge identically.
    """
    if (api_mode or "") == "anthropic_messages":
        from agent.anthropic_thinking_policy import native_anthropic_preserves_prior_thinking
        if native_anthropic_preserves_prior_thinking(base_url, model):
            return True
    if not api_mode:
        # No route facts (a bare compressor): only the must-echo tier is known to replay.
        return needs_reasoning_echo(provider, model, base_url)
    route = reasoning_replay_route(api_mode, provider, model, base_url)
    if route.carriers is None:
        return (api_mode or "") != "codex_responses" and route.pad
    return bool(route.carriers & {RC, R})


def native_anthropic_accounting_projection(messages: Any) -> tuple[Any, tuple[str, ...]]:
    """Return the native Anthropic wire shadow plus readable replay thinking out-of-band.

    Canonical history may retain storage-only reasoning alongside signed replay carriers.
    Native conversion prefers ordered anthropic_content_blocks over reasoning_details and
    never sends reasoning itself. The generic message estimator therefore receives only
    ordinary wire-shaped fields, while readable thinking is returned separately for the
    explicit Anthropic accounting seam. Opaque signature/data bytes are never priced.
    """
    if not isinstance(messages, list):
        return messages, ()

    from agent.anthropic_message_convert import assistant_replay_carrier

    projected = []
    replayed_thinking: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            projected.append(message)
            continue

        # Mirror _convert_assistant_message's actual inputs instead of starting
        # from canonical storage. Context selection is allowed to return canonical
        # rows, which can contain timestamp/finish_reason/api_content and other
        # local metadata that native Anthropic never sees.
        shadow = {"role": "assistant"}
        for key in ("content", "tool_calls", "reasoning_content", "cache_control"):
            if key in message:
                shadow[key] = message[key]
        # Canonical input (preflight, tail walk) still holds the api_content sidecar that
        # build_api_messages substitutes into content; post-build input already carries it in
        # content. Charge it either way: an ordered turn, or a context-selection clone the
        # converter reads raw, then overcounts, never undercounts.
        sidecar = message.get("api_content")
        if isinstance(sidecar, str) and sidecar:
            shadow["content"] = sidecar

        _, carrier = assistant_replay_carrier(message)
        # The converter ignores reasoning_content for an ordered turn and only injects it when the
        # details carrier holds no thinking, so it must not be charged in addition to the carrier.
        if carrier:
            shadow.pop("reasoning_content", None)

        replayed_thinking.extend(
            block["thinking"]
            for block in carrier
            if block.get("type") == "thinking" and isinstance(block.get("thinking"), str) and block["thinking"]
        )
        projected.append(shadow)
    return projected, tuple(replayed_thinking)


def _replayable_reasoning_text(msg: dict) -> str | None:
    """The turn's reasoning text: a non-blank ``reasoning_content`` wins over ``reasoning``."""
    for key in (RC, R):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _apply_legacy_must_echo_pad(source_msg: dict, api_msg: dict) -> None:
    """Native-adapter wires (``carriers=None``): the pre-replay contract, byte-for-byte. An explicit
    value is kept verbatim (legacy "" upgraded to " "); a tool-call turn holding only another
    provider's ``reasoning`` is padded instead of converted into a native thinking block."""
    existing, reasoning = source_msg.get(RC), source_msg.get(R)
    if isinstance(existing, str):
        api_msg[RC] = existing or " "
    elif isinstance(reasoning, str) and reasoning and not source_msg.get("tool_calls"):
        api_msg[RC] = reasoning
    else:
        api_msg[RC] = " "


def apply_reasoning_content_policy(
    source_msg: dict, api_msg: dict, needs_thinking_pad: bool, carriers: frozenset | None = None,
) -> None:
    """Shape an assistant replay message's reasoning keys for the active route (mutates ``api_msg``).

    ``needs_thinking_pad`` is the must-echo flag; ``carriers`` the keys the route reads
    (``reasoning_replay_route``). ``carriers=None`` is a non-chat-completions wire: only the
    must-echo ``reasoning_content`` rides there and ``reasoning_details`` is left to the adapter.
    Keys the route does not read are removed, so a strict host never sees one (#45655, #70233).
    """
    if source_msg.get("role") != "assistant":
        return
    text = _replayable_reasoning_text(source_msg)
    effective = carriers if carriers is not None else (frozenset({RC}) if needs_thinking_pad else frozenset())
    if carriers is None and needs_thinking_pad:
        _apply_legacy_must_echo_pad(source_msg, api_msg)
    elif RC in effective and (text is not None or needs_thinking_pad):
        # Must-echo tier: every assistant turn carries the field; " " (not "") when there is no
        # text, because DeepSeek V4 rejects empty string (#17341) and a legacy "" pad upgrades.
        api_msg[RC] = text if text is not None else " "
    else:
        # Also drops a non-string value (None after compaction) and a whitespace pad written
        # for a must-echo provider: never pass null, never replay a pad as reasoning.
        api_msg.pop(RC, None)
    if R in effective and text is not None:
        api_msg[R] = text
    else:
        api_msg.pop(R, None)
    if carriers is not None and RD not in carriers:
        # Private ``*.native_assistant`` records (Gemini/Copilot carriers) survive: the transport lifts
        # them onto the route that reads them and filters them from every other wire.
        private = [d for d in api_msg.pop(RD, None) or () if isinstance(d, dict)
                   and str(d.get("type") or "").endswith(".native_assistant")]
        if private:
            api_msg[RD] = private


def _reasoning_shape(msg: dict) -> tuple:
    return tuple(msg.get(key) for key in REASONING_CARRIERS)


def reapply_reasoning_echo(api_messages: list, needs_thinking_pad: bool, carriers: frozenset | None = None) -> int:
    """Reconcile already-built assistant turns with the ACTIVE route.

    ``api_messages`` is built once under the primary; a mid-conversation fallback or a recorded
    field rejection changes the route, so baked-in keys are reconciled both ways: TO a must-echo
    provider the pad is re-applied (else 400), TO a strict one the keys are stripped (else 422),
    and a carrier the new route reads is restored from the text that is still present.
    Idempotent. Returns the number of assistant turns changed.
    """
    changed = 0
    for api_msg in api_messages:
        if api_msg.get("role") != "assistant":
            continue
        before = _reasoning_shape(api_msg)
        if carriers is None and needs_thinking_pad and api_msg.get(RC):
            continue
        apply_reasoning_content_policy(api_msg, api_msg, needs_thinking_pad, carriers)
        changed += _reasoning_shape(api_msg) != before
    return changed


# Provider error bodies (lowercased) for "this request carries a key my schema does not know".
_UNKNOWN_FIELD_PHRASES = (
    "extra inputs are not permitted", "is unsupported", "no such field", "unknown field",
    "unrecognized request argument", "unrecognized field", "unknown parameter", "unexpected field",
    "additional propert", "unknown key",
    # Same class of wire rejection from an alias collision: a schema that treats `reasoning` as an alias of
    # `reasoning_content` rejects a message carrying both with "duplicate field `reasoning_content`" (serde).
    "duplicate field",
)
_NAMED_FIELD_RES = {
    RC: re.compile(r"(?<![\w])reasoning_content(?![\w])"),
    RD: re.compile(r"(?<![\w])reasoning_details(?![\w])"),
    # The bare word is also a top-level request param; only a message-path mention counts.
    R: re.compile(r"messages[^\n]{0,80}?(?<![\w])reasoning(?![\w])"),
}


def rejected_reasoning_fields(error_body: Any, sent_messages: Any) -> frozenset:
    """Reasoning keys to drop after an unknown-field rejection that names one this request carried.

    Strict schemas report one offending key per error, so the drop is widened to the keys the
    same schema will reject next: a rejected ``reasoning_details`` / ``reasoning`` takes the
    other non-standard carrier too (``reasoning_content`` still carries the text); a rejected
    ``reasoning_content`` (the most widely accepted alias) takes all three. One retry, not three.
    """
    body = str(error_body or "").lower()
    if not any(phrase in body for phrase in _UNKNOWN_FIELD_PHRASES):
        return frozenset()
    sent = {
        key for msg in (sent_messages if isinstance(sent_messages, list) else ())
        if isinstance(msg, dict) and msg.get("role") == "assistant" for key in REASONING_CARRIERS if key in msg
    }
    named = {key for key in sent if _NAMED_FIELD_RES[key].search(body)}
    if not named:
        return frozenset()
    return frozenset(sent & (_ALL_CARRIERS if RC in named else {R, RD}))


def reasoning_route_key(agent: Any) -> tuple[str, str, str]:
    """``(provider, base_url host, model)``: one gateway serves models on different upstream lanes,
    so a rejection learned for one model must not strip carriers from its siblings."""
    from utils import base_url_hostname

    return (
        (getattr(agent, "provider", "") or "").strip().lower(),
        base_url_hostname(getattr(agent, "base_url", "") or "") or "",
        (getattr(agent, "model", "") or "").strip(),
    )


_REJECTION_MODEL_CONFIG_KEY = "reasoning_rejected_carriers"


def rejected_reasoning_carriers(agent: Any) -> frozenset:
    """Reasoning keys the active (provider, host, model) rejected in this session.

    Persisted in the session's ``model_config`` so a resumed process (``--resume``, a gateway
    restart) never re-learns a rejection with another 400. Loaded once per ``session_id``.
    """
    routes = agent.__dict__.setdefault("_reasoning_rejecting_routes", {})
    session_id = getattr(agent, "session_id", None)
    if session_id and getattr(agent, "_reasoning_rejections_loaded_for", None) != session_id:
        agent._reasoning_rejections_loaded_for = session_id
        getter = getattr(getattr(agent, "_session_db", None), "get_session_model_config_value", None)
        stored = getter(session_id, _REJECTION_MODEL_CONFIG_KEY, []) if callable(getter) else []
        for entry in stored if isinstance(stored, list) else ():
            if isinstance(entry, list) and len(entry) == 4 and isinstance(entry[3], list):
                routes.setdefault(tuple(entry[:3]), set()).update(k for k in entry[3] if k in _ALL_CARRIERS)
    return frozenset(routes.get(reasoning_route_key(agent), ()))


def record_reasoning_field_rejection(agent: Any, error_body: Any, sent_messages: Any) -> frozenset:
    """Remember the reasoning keys this route rejected; returns the NEW ones (empty = no retry).

    The next ``build_api_request`` re-shapes ``api_messages`` without them for this
    (provider, host, model) for the rest of the session. A repeat rejection of an already
    recorded key returns empty, so recovery can never loop. History is never touched.
    """
    if getattr(agent, "api_mode", "chat_completions") != "chat_completions":
        return frozenset()
    fields = rejected_reasoning_fields(error_body, sent_messages)
    if needs_reasoning_echo(getattr(agent, "provider", ""), getattr(agent, "model", ""), getattr(agent, "base_url", "")):
        # Must-echo routes 400 on any tool turn WITHOUT reasoning_content; a body naming it is a
        # value complaint ("must not be empty"), never a schema that lacks the field.
        fields -= {RC}
    if not fields:
        return frozenset()
    new = fields - rejected_reasoning_carriers(agent)
    if not new:
        return frozenset()
    routes = agent._reasoning_rejecting_routes
    routes.setdefault(reasoning_route_key(agent), set()).update(new)
    patcher = getattr(getattr(agent, "_session_db", None), "patch_session_model_config", None)
    if callable(patcher) and getattr(agent, "session_id", None) and not getattr(agent, "_persist_disabled", False):
        patcher(agent.session_id, {_REJECTION_MODEL_CONFIG_KEY: [[*key, sorted(v)] for key, v in sorted(routes.items())]})
    return frozenset(new)


# Image / multimodal parts are deliberately NOT consolidated here: per-adapter handling is
# format-specific SYNTAX. The one shared image POLICY is ``_strip_images_from_messages``.
