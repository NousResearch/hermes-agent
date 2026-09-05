"""Codex Responses API adapter: stateless format conversion and normalization for the
OpenAI Responses API (OpenAI Codex, xAI, GitHub Models and other compatible endpoints)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, TypeGuard

from agent.message_sanitization import deterministic_call_id
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

logger = logging.getLogger(__name__)


def _classify_responses_issuer(
    *, is_xai_responses: bool = False, is_github_responses: bool = False, is_codex_backend: bool = False,
    base_url: Optional[str] = None,
) -> str:
    """Stable identifier for the endpoint that mints ``reasoning.encrypted_content``. Blobs are sealed to their
    issuer (HTTP 400 ``invalid_encrypted_content``), so stamping lets replay drop foreign blobs after a model switch."""
    for flag, kind in ((is_xai_responses, "xai_responses"), (is_github_responses, "github_responses"), (is_codex_backend, "codex_backend")):
        if flag:
            return kind
    return f"other:{base_url}" if base_url else "other"


# Per-process throttle for the cross-issuer skip warning.
_CROSS_ISSUER_WARN_EMITTED = False

# Codex/Harmony tool-call serialization leaked into assistant text (no structured function_call).
_TOOL_CALL_LEAK_PATTERN = re.compile(r"(?:^|[\s>|])to=functions\.[A-Za-z_][\w.]*", re.IGNORECASE)

# The Codex backend rejects literal Harmony wire tokens (``invalid_prompt: Request
# blocked.``). Fullwidth bars survive format-character stripping and stay legible.
_HARMONY_CONTROL_TOKEN_RE = re.compile(r"<\|(start|end|channel|message|constrain|return|call)\|>")
_FULLWIDTH_PIPE = "\uff5c"

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_IMAGE_PART_TYPES = {"image_url", "input_image"}
_OUTPUT_TEXT_TYPES = {"output_text", "text"}
_ASSISTANT_IMAGE_PLACEHOLDER = "[Assistant image omitted during replay]"
_INCOMPLETE_STATUSES = {"queued", "in_progress", "incomplete"}
_RESPONSE_MESSAGE_STATUSES = {"completed", "incomplete", "in_progress"}

# input[].id / function names longer than this are a non-retryable 400 ("string too
# long"). Codex message ids can run 400+ chars; Hermes ``msg_...`` ids stay under the cap.
_MAX_RESPONSES_ITEM_ID_LENGTH = 64
_VALID_RESPONSES_FN_NAME_RE = re.compile(r"[a-zA-Z0-9_-]{1,64}")

# Provider-executed built-in tools: declared by ``type`` alone, run server-side,
# reported via the ``*_call`` output items below; preflight passes them through.
_RESPONSES_BUILTIN_TOOL_TYPES = {
    "web_search", "web_search_preview", "file_search", "code_interpreter", "image_generation", "computer_use_preview",
    "local_shell",
}

# Server-side ``*_call`` output items. xAI leaves these ``in_progress`` even when the
# response is ``completed``, so they must NOT flip the incomplete verdict (else every
# server-search turn burns 3 fruitless continuation retries).
_SERVER_SIDE_TOOL_CALL_TYPES = {
    "web_search_call", "file_search_call", "code_interpreter_call",
    "image_generation_call", "computer_call", "local_shell_call", "mcp_call",
}


def _nonblank(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_str(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value)


def _str_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _lower_or_none(value: Any) -> Optional[str]:
    return value.strip().lower() if isinstance(value, str) else None


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a dict or an attribute-style (SDK/SimpleNamespace) object."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, default)


def _part_type(part: Dict[str, Any]) -> str:
    return str(part.get("type") or "").strip().lower()


def _text_type_for(role: str) -> str:
    return "output_text" if role == "assistant" else "input_text"


def _coerce_arguments(arguments: Any) -> str:
    """Normalize replayed tool-call arguments to a non-empty JSON string."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    elif not isinstance(arguments, str):
        arguments = str(arguments)
    return arguments.strip() or "{}"


def _neutralize_harmony_tokens(text: str) -> str:
    """Keep Harmony source readable without emitting reserved wire tokens."""
    if not text or "<" not in text or "|" not in text:
        return text
    if not any(unicodedata.category(char) == "Cf" for char in text):
        return _HARMONY_CONTROL_TOKEN_RE.sub(rf"<{_FULLWIDTH_PIPE}\1{_FULLWIDTH_PIPE}>", text)
    # The backend strips Unicode format controls (e.g. U+200B) before its reserved-token
    # check, so match on the visible text and rewrite the original spans.
    original_positions = [i for i, char in enumerate(text) if unicodedata.category(char) != "Cf"]
    visible_text = "".join(text[i] for i in original_positions)
    result, cursor = [], 0
    for match in _HARMONY_CONTROL_TOKEN_RE.finditer(visible_text):
        start, end = original_positions[match.start()], original_positions[match.end() - 1] + 1
        result += [text[cursor:start], f"<{_FULLWIDTH_PIPE}{match.group(1)}{_FULLWIDTH_PIPE}>"]
        cursor = end
    return "".join(result) + text[cursor:]


def _neutralize_harmony_structure(value: Any) -> Any:
    """Neutralize JSON-like values (tuples → lists). A reserved token in an object *key* is
    rejected, not rewritten — renaming could desync a tool schema from the executor contract."""
    if isinstance(value, str):
        return _neutralize_harmony_tokens(value)
    if isinstance(value, (list, tuple)):
        return [_neutralize_harmony_structure(item) for item in value]
    if isinstance(value, dict):
        if any(isinstance(key, str) and _neutralize_harmony_tokens(key) != key for key in value):
            raise ValueError(
                "Reserved Harmony tokens in a JSON object key cannot be "
                "neutralized without changing its contract."
            )
        return {key: _neutralize_harmony_structure(item) for key, item in value.items()}
    return value


# --- Multimodal content helpers ---------------------------------------------

def _iter_content_parts(content: list) -> Iterator[tuple[str, Any]]:
    """Yield ``("text", str)`` / ``("image", part)`` for recognized chat parts."""
    for part in content:
        if isinstance(part, str) and part:
            yield "text", part
        elif isinstance(part, dict):
            ptype = _part_type(part)
            if ptype in _TEXT_PART_TYPES and _nonempty_str(part.get("text")):
                yield "text", part["text"]
            elif ptype in _IMAGE_PART_TYPES:
                yield "image", part


def _input_image_part(part: Dict[str, Any], role: str = "user", *, keep_empty_url: bool) -> Optional[Dict[str, Any]]:
    """Responses image part from a chat/Responses image part (``image_url`` may be a str or
    ``{url, detail}``). Assistant → text placeholder (an assistant ``input_image`` 400s every
    replay); user → ``input_image``, None for an empty url unless ``keep_empty_url``."""
    if role == "assistant":
        return {"type": "output_text", "text": _ASSISTANT_IMAGE_PLACEHOLDER}
    url, detail = part.get("image_url"), part.get("detail")
    if isinstance(url, dict):
        url, detail = url.get("url"), url.get("detail", detail)
    if not _nonempty_str(url) and not keep_empty_url:
        return None
    image_part: Dict[str, Any] = {"type": "input_image", "image_url": str(url or "")}
    if _nonblank(detail):
        image_part["detail"] = detail.strip()
    return image_part


def _chat_content_to_responses_parts(content: Any, *, role: str = "user") -> List[Dict[str, Any]]:
    """Chat-style multimodal content → Responses API input parts ([] if not a list). Text is
    ``input_text`` (user) / ``output_text`` (assistant) — the API rejects the wrong type per role;
    ``input_image`` is only legal on user messages (see :func:`_input_image_part`)."""
    text_type = _text_type_for(role)
    converted: List[Dict[str, Any]] = []
    for kind, payload in _iter_content_parts(_as_list(content)):
        if kind == "text":
            converted.append({"type": text_type, "text": payload})
        elif (part := _input_image_part(payload, role, keep_empty_url=False)) is not None:
            converted.append(part)
    return converted


def _summarize_user_message_for_log(content: Any, *, sep: str = " ") -> str:
    """Flatten message content to plain text: text parts joined with ``sep`` (``" "`` for logs; ``"\\n"`` for memory
    providers feeding regexes), images → ``[N image(s)]`` marker, ``""`` for None, ``str(content)`` for other scalars."""
    if not isinstance(content, list):
        try:
            return _str_or_empty(content)
        except Exception:
            return ""
    parts = list(_iter_content_parts(content))
    text_bits = [payload for kind, payload in parts if kind == "text"]
    image_count = len(parts) - len(text_bits)
    note = f"[{image_count} image{'s' if image_count != 1 else ''}]" if image_count else ""
    return " ".join(bit for bit in (note, sep.join(text_bits).strip()) if bit)


# --- ID helpers ---------------------------------------------------------------

def _clamp_responses_call_id(call_id: str) -> str:
    """Keep ``call_id`` within the API's 64-char cap (the codex app-server namespaces MCP call ids past it). The
    surrogate is a pure function of the original so a ``function_call`` and its ``function_call_output`` agree."""
    if len(call_id) <= _MAX_RESPONSES_ITEM_ID_LENGTH:
        return call_id
    return f"call_{hashlib.sha256(call_id.encode('utf-8', errors='replace')).hexdigest()[:32]}"


def _sanitize_replayed_fn_name(name: str) -> str:
    """Coerce a *replayed* ``function_call.name`` to ``^[a-zA-Z0-9_-]{1,64}$`` (an invalid stored
    name 400s every later turn). Invalid runs collapse to ``_``; all-invalid → "fn". Apply ONLY to
    replayed items, never live tool definitions (schema names must match the dispatch registry)."""
    if not isinstance(name, str):
        return "fn"
    if _VALID_RESPONSES_FN_NAME_RE.fullmatch(name):
        return name
    coerced = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "_", name.strip())).strip("_")
    return coerced[:64] or "fn"


def _canonical_call_id_from_fc(response_item_id: Any) -> Optional[str]:
    """Map an ``fc_…`` item id to its canonical ``call_<suffix>``. Both sides of a replayed
    pair must derive the SAME call_id, or an oversized pair clamps to two surrogates."""
    if isinstance(response_item_id, str) and response_item_id.startswith("fc_") and len(response_item_id) > 3:
        return f"call_{response_item_id[3:]}"
    return None


def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
    """Split a stored tool id into (call_id, response_item_id)."""
    value = raw_id.strip() if isinstance(raw_id, str) else ""
    if "|" in value:
        call_id, response_item_id = value.split("|", 1)
        return call_id.strip() or None, response_item_id.strip() or None
    if not value:
        return None, None
    return (None, value) if value.startswith("fc_") else (value, None)


def _resolve_call_id(
    raw_call_id: Any, raw_item_id: Any, fn_name: str, arguments: Any, index: int, *, canonicalize_fc: bool,
) -> str:
    """Pick a non-blank call_id: explicit -> embedded in ``call|fc`` id -> (replay only)
    canonical ``call_<fc suffix>`` -> deterministic hash of name/arguments/index."""
    embedded_call_id, embedded_response_item_id = _split_responses_tool_id(raw_item_id)
    call_id = raw_call_id if _nonblank(raw_call_id) else embedded_call_id
    if not _nonblank(call_id) and canonicalize_fc:
        call_id = _canonical_call_id_from_fc(embedded_response_item_id)
    if not _nonblank(call_id):
        call_id = deterministic_call_id(fn_name, arguments, index)
    return call_id.strip()


def _derive_responses_function_call_id(call_id: str, response_item_id: Optional[str] = None) -> str:
    """Build a valid Responses `function_call.id` (must start with `fc_`)."""
    if isinstance(response_item_id, str) and response_item_id.strip().startswith("fc_"):
        return response_item_id.strip()
    source = (call_id or "").strip()
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
    for candidate in (source, sanitized):
        if candidate.startswith("fc_") or (candidate.startswith("call_") and len(candidate) > len("call_")):
            return candidate if candidate.startswith("fc_") else f"fc_{candidate[len('call_'):]}"
    if sanitized:
        return f"fc_{sanitized[:48]}"
    seed = source or str(response_item_id or "") or uuid.uuid4().hex
    return f"fc_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:24]}"


# --- Schema conversion --------------------------------------------------------

def _responses_tools(tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
    """Convert chat-completions tool schemas to Responses function-tool schemas."""
    fns = [item.get("function", {}) if isinstance(item, dict) else {} for item in tools or []]
    converted = [
        {
            "type": "function", "name": fn["name"], "description": fn.get("description", ""), "strict": False,
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        for fn in fns if _nonblank(fn.get("name"))
    ]
    return converted or None


# --- Message format conversion (chat history -> Responses input) --------------

def _normalize_responses_message_status(value: Any, *, default: str = "completed") -> str:
    """Normalize a replayed assistant message status, modulo case/hyphen spelling, so incomplete Codex
    continuation turns are not falsely marked completed."""
    status = value.strip().lower().replace("-", "_").replace(" ", "_") if isinstance(value, str) else None
    return status if status in _RESPONSE_MESSAGE_STATUSES else default


def _message_item(
    content: List[Dict[str, Any]], *, status: str, item_id: Optional[str] = None, phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Assistant ``message`` item; ``id``/``phase`` are added only when non-empty."""
    item: Dict[str, Any] = {"type": "message", "role": "assistant", "status": status, "content": content}
    item.update({k: v for k, v in (("id", item_id), ("phase", phase)) if v})
    return item


def _assistant_message_item(raw: Dict[str, Any], content: List[Dict[str, Any]], *, is_github_responses: bool) -> Dict[str, Any]:
    """Replayable assistant ``message`` item from a stored one. ``id`` is kept only when short enough and never for
    GitHub Copilot (ids bind to a backend connection; stale → 401); ``phase`` is preserved per OpenAI's cache guidance."""
    item_id, phase = raw.get("id"), raw.get("phase")
    keep_id = not is_github_responses and _nonblank(item_id) and len(item_id.strip()) <= _MAX_RESPONSES_ITEM_ID_LENGTH
    return _message_item(
        content, status=_normalize_responses_message_status(raw.get("status")),
        item_id=item_id.strip() if keep_id else None, phase=phase.strip() if _nonblank(phase) else None,
    )


def _replay_reasoning_items(
    msg: Dict[str, Any], *, seen_item_ids: set, current_issuer_kind: Optional[str], native_compaction_eligible: bool,
) -> List[Dict[str, Any]]:
    """Replay persisted encrypted reasoning/compaction items for one assistant turn. Skips duplicate
    ids, ``compaction`` checkpoints unless THIS request carries ``context_management`` (else a persisted
    checkpoint erases pre-checkpoint history on a model that cannot decrypt it), and items stamped by
    another issuer (HTTP 400); unstamped legacy items pass. ``id`` (store=False lookups 404) and
    ``_issuer_kind`` are stripped."""
    global _CROSS_ISSUER_WARN_EMITTED
    replayed: List[Dict[str, Any]] = []
    for ri in _as_list(msg.get("codex_reasoning_items")):
        if not (isinstance(ri, dict) and ri.get("encrypted_content")):
            continue
        item_id = ri.get("id")
        if (item_id and item_id in seen_item_ids) or (ri.get("type") == "compaction" and not native_compaction_eligible):
            continue
        item_issuer = ri.get("_issuer_kind")
        if current_issuer_kind is not None and item_issuer is not None and item_issuer != current_issuer_kind:
            if not _CROSS_ISSUER_WARN_EMITTED:
                logger.warning(
                    "Dropping reasoning item minted by %s while calling %s — encrypted_content is sealed to "
                    "its issuer. This happens when a session switches model providers mid-conversation.",
                    item_issuer, current_issuer_kind,
                )
                _CROSS_ISSUER_WARN_EMITTED = True
            continue
        replayed.append({k: v for k, v in ri.items() if k not in ("id", "_issuer_kind")})
        if item_id:
            seen_item_ids.add(item_id)
    return replayed


def _replay_message_items(msg: Dict[str, Any], *, is_github_responses: bool) -> List[Dict[str, Any]]:
    """Replay exact assistant message items (id/phase) for prefix-cache hits."""
    replayed: List[Dict[str, Any]] = []
    for raw_item in _as_list(msg.get("codex_message_items")):
        if not (isinstance(raw_item, dict) and raw_item.get("type") == "message" and raw_item.get("role") == "assistant"):
            continue
        content = [
            {"type": "output_text", "text": _str_or_empty(part.get("text", ""))}
            for part in _as_list(raw_item.get("content"))
            if isinstance(part, dict) and str(part.get("type") or "").strip() in _OUTPUT_TEXT_TYPES
        ]
        if content:
            replayed.append(_assistant_message_item(raw_item, content, is_github_responses=is_github_responses))
    return replayed


def _replay_tool_call_items(msg: Dict[str, Any], *, start_index: int) -> List[Dict[str, Any]]:
    """Convert an assistant message's ``tool_calls`` into ``function_call`` items."""
    replayed: List[Dict[str, Any]] = []
    for tc in _as_list(msg.get("tool_calls")):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        fn_name, arguments = fn.get("name"), fn.get("arguments", "{}")
        if not _nonblank(fn_name):
            continue
        index = start_index + len(replayed)
        call_id = _resolve_call_id(tc.get("call_id"), tc.get("id"), fn_name, str(arguments), index, canonicalize_fc=True)
        replayed.append({
            "type": "function_call", "call_id": _clamp_responses_call_id(call_id),
            "name": _sanitize_replayed_fn_name(fn_name), "arguments": _coerce_arguments(arguments),
        })
    return replayed


def _tool_output_items(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a tool-role message to ``[function_call_output]`` (``[]`` if unpairable)."""
    raw_tool_call_id = msg.get("tool_call_id")
    call_id, tool_response_item_id = _split_responses_tool_id(raw_tool_call_id)
    if not _nonblank(call_id):
        # Legacy fc_-only ids canonicalize to the same ``call_<suffix>`` the
        # assistant side synthesizes, so a >64-char pair clamps identically.
        call_id = _canonical_call_id_from_fc(tool_response_item_id)
        if call_id is None and _nonblank(raw_tool_call_id):
            call_id = raw_tool_call_id.strip()
    if not _nonblank(call_id):
        return []
    # ``output`` may be a string or an ``input_text``/``input_image`` array.
    tool_content = msg.get("content")
    is_parts = isinstance(tool_content, list)
    output_value: Any = (_chat_content_to_responses_parts(tool_content) or "") if is_parts else str(tool_content or "")
    return [{"type": "function_call_output", "call_id": _clamp_responses_call_id(call_id), "output": output_value}]


def _chat_messages_to_responses_input(
    messages: List[Dict[str, Any]], *, is_xai_responses: bool = False, is_github_responses: bool = False,
    replay_encrypted_reasoning: bool = True, current_issuer_kind: Optional[str] = None,
    native_compaction_eligible: bool = False,
) -> List[Dict[str, Any]]:
    """Convert internal chat-style messages to Responses input items.

    ``is_xai_responses``: signature compatibility only (xAI DOES replay encrypted reasoning).
    ``replay_encrypted_reasoning``: per-session kill switch, threaded False by
    ``AIAgent._disable_codex_reasoning_replay`` after an ``invalid_encrypted_content`` 400.
    ``is_github_responses``: drops ``id`` from replayed message items (Copilot 401s on stale ids).
    ``current_issuer_kind``: cross-issuer guard; foreign-stamped items drop, legacy items replay.
    ``native_compaction_eligible``: THIS request carries ``context_management``; gates both replaying ``compaction``
    checkpoints and ``prune_pre_checkpoint_items``. Checkpoints persist across model swaps / compression flips / resume,
    so without the gate one checkpoint would erase pre-checkpoint history on a model that cannot decrypt it (lossless:
    local history is never truncated).

    Earlier (PR #26644, May 2026) we believed xAI's OAuth/SuperGrok ``/v1/responses`` surface rejected
    replayed ``encrypted_content`` reasoning items minted by prior turns, and we stripped them. That
    decision was wrong — xAI explicitly relies on Hermes threading encrypted reasoning back across turns for
    cross-turn coherence (the whole point of their partnership integration). We now replay encrypted
    reasoning on every Responses transport (xAI, native Codex, custom relays) and let xAI tell us explicitly
    if a specific surface ever rejects a payload.
    The Copilot backend (api.githubcopilot.com/responses) binds these ids to a specific backend "connection"
    — credential-pool rotation, a gateway restart, or routine load-balancer churn between turns all
    invalidate it — and rejects a stale id with HTTP 401 "input item ID does not belong to this connection"
    even for short ids (see #32716). ``phase``/ ``status``/``content`` are still replayed; only ``id`` is
    unsafe to reuse across a Copilot connection.
    ``native_compaction_eligible`` mirrors, for THIS request, the decision made by
    ``native_compaction.native_compaction_context_management`` — it is True only when that gate returned a
    payload, i.e. when the request actually carries ``context_management``. It controls two things that must
    never outlive the gate: replaying ``type: "compaction"`` checkpoint items, and restructuring the wire
    around them (``prune_pre_checkpoint_items``). Checkpoints are persisted in the ``codex_reasoning_items``
    sidecar and survive a mid-session model swap, a ``compression.enabled: false`` flip, the rejection kill
    switch and a resumed session; without this flag a single captured checkpoint would keep deleting every
    pre-checkpoint item from every later request, on a model that cannot decrypt the blob (#85914). Default
    False = pre-feature wire, which is also correct for every caller that never sends ``context_management``
    (auxiliary/compression client, ad-hoc ``convert_messages``). Dropping the checkpoint costs nothing:
    Hermes' local history is never truncated by native compaction, so the full conversation is still on the
    wire.
    """
    items: List[Dict[str, Any]] = []
    # Parallel to ``items``: source chat message per item. Pruning reads a summary
    # carrier's provenance from the source; the converted item may be a lossy shape.
    # Pruning needs this to read a canonical summary carrier's up-to-date, provenance-tagged content
    # directly — the converted `item` can be a lossy shape (stale exact-replay, or a typed
    # `function_call_output` wrapper) that no longer carries it (#90976).
    item_sources: List[Optional[Dict[str, Any]]] = []
    seen_item_ids: set = set()
    def emit(new_items: List[Dict[str, Any]], msg: Dict[str, Any]) -> None:
        items.extend(new_items)
        item_sources.extend([msg] * len(new_items))
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            emit(_tool_output_items(msg), msg)
            continue
        if role not in {"user", "assistant"}:
            continue
        content = msg.get("content", "")
        content_parts = _chat_content_to_responses_parts(content, role=role)  # [] unless a list
        text_type = _text_type_for(role)
        content_text = (
            "".join(p["text"] for p in content_parts if p["type"] == text_type)
            if isinstance(content, list) else _str_or_empty(content)
        )
        if role == "user":
            emit([{"role": role, "content": content_parts or content_text}], msg)
            continue
        reasoning_items = [] if not replay_encrypted_reasoning else _replay_reasoning_items(
            msg, seen_item_ids=seen_item_ids, current_issuer_kind=current_issuer_kind,
            native_compaction_eligible=native_compaction_eligible,
        )
        emit(reasoning_items, msg)
        message_items = _replay_message_items(msg, is_github_responses=is_github_responses)
        emit(message_items, msg)
        if not message_items:
            # Every reasoning item needs a following item (else missing_following_item), hence the "" fallback.
            fallback = content_parts or (content_text if content_text.strip() else "" if reasoning_items else None)
            if fallback is not None:
                emit([{"role": "assistant", "content": fallback}], msg)
        emit(_replay_tool_call_items(msg, start_index=len(items)), msg)
    # The server renders nothing placed before a compaction item, so pre-checkpoint history is
    # dead weight and plaintext asks / merged summaries silently vanish. Keep the newest checkpoint
    # first, retain pre-checkpoint USER and SUMMARY messages within a token budget, leave the tail.
    # Native server-side compaction: when a replayed checkpoint is present, restructure the wire around it.
    # Gated on the CURRENT request's native eligibility, not merely on the presence of a checkpoint: a
    # persisted checkpoint outlives the gate, and pruning for a request that carries no
    # ``context_management`` deletes history the server never compacted. ``item_sources`` (parallel to
    # ``items``) carries the raw chat message each converted item came from. A canonical summary carrier's
    # content can be lost or gone stale by the time it becomes a Responses item — a merge-into-tail
    # tool-result carrier becomes a typed ``function_call_output`` (no ``content``/``role`` at all), and a
    # merge-into-tail assistant carrier can be shadowed by a stale exact ``codex_message_items`` replay from
    # before the merge rewrote its content. Pruning reads the source message's own up-to-date,
    # provenance-tagged content directly instead of trying to recover it from whatever shape the conversion
    # produced (#90976).
    if not native_compaction_eligible:
        return items
    from agent.native_compaction import prune_pre_checkpoint_items
    return prune_pre_checkpoint_items(items, item_sources=item_sources)


class ResponsesRouteFlags(NamedTuple):
    """Which special Responses-API route an agent is talking to. Single owner of the
    codex/xai/github predicates — every site must call :func:`classify_responses_route`.

    Every site that needs these flags (request kwargs build, preflight estimation, silent- reject hints)
    must call :func:`classify_responses_route` instead of re-implementing the string comparisons inline —
    inline copies drift (backend-identity class: #22548/#70893/#59561/#72468).
    """
    is_codex_backend: bool
    is_xai_responses: bool
    is_github_responses: bool


def classify_responses_route(agent: Any) -> ResponsesRouteFlags:
    """Classify the agent's Responses route from provider + base URL. Host checks are
    exact-host-or-subdomain, never substring (``evil.com/models.github.ai`` is not GitHub)."""
    from utils import base_url_hostname
    provider = getattr(agent, "provider", None)
    base_url = str(getattr(agent, "base_url", "") or "")
    hostname = str(getattr(agent, "_base_url_hostname", "") or "").lower() or base_url_hostname(base_url)
    lower = str(getattr(agent, "_base_url_lower", "") or base_url).lower()
    def _host_is(domain: str) -> bool:
        return hostname == domain or hostname.endswith("." + domain)
    return ResponsesRouteFlags(
        is_codex_backend=provider == "openai-codex" or (_host_is("chatgpt.com") and "/backend-api/codex" in lower),
        is_xai_responses=provider in {"xai", "xai-oauth"} or hostname == "api.x.ai",
        is_github_responses=_host_is("models.github.ai") or _host_is("githubcopilot.com"),
    )


def estimate_native_responses_preflight_tokens(
    agent: Any, messages: List[Dict[str, Any]], *, system_prompt: str = "", tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[int]:
    """Estimate tokens for the checkpoint-pruned Responses payload (the full transcript overstates a natively compacted
    session and fires local compression needlessly). None when native compaction is not proven eligible or conversion fails.

    Automatic preflight previously counted the full durable transcript. On a natively compacted Codex
    session that overstates the wire by several times and fires local compression against history the main
    request will never send (#96155).
    """
    if getattr(agent, "api_mode", None) != "codex_responses" or not isinstance(messages, list):
        return None
    route = classify_responses_route(agent)._asdict()
    from agent.native_compaction import native_compaction_context_management
    if not native_compaction_context_management(agent, **route):
        return None
    try:
        items = _chat_messages_to_responses_input(
            messages, is_xai_responses=route["is_xai_responses"], is_github_responses=route["is_github_responses"],
            replay_encrypted_reasoning=bool(getattr(agent, "_codex_reasoning_replay_enabled", True)),
            current_issuer_kind=_classify_responses_issuer(base_url=getattr(agent, "base_url", None), **route),
            native_compaction_eligible=True,
        )
    except Exception:
        logger.debug("native Responses preflight conversion failed; falling back to generic estimate", exc_info=True)
        return None
    from agent.model_metadata import estimate_request_tokens_rough
    return estimate_request_tokens_rough(items, system_prompt=system_prompt or "", tools=tools)


# --- Input preflight / validation --------------------------------------------

_PreflightCtx = NamedTuple("_PreflightCtx", [
    ("sanitize_text", Callable[[str], str]), ("sanitize_harmony_tokens", bool), ("is_github_responses", bool), ("seen_ids", set),
])


def _preflight_function_call(item: Dict[str, Any], idx: int, ctx: _PreflightCtx) -> Dict[str, Any]:
    call_id, name = item.get("call_id"), item.get("name")
    if not _nonblank(call_id):
        raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
    if not _nonblank(name):
        raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")
    return {
        "type": "function_call", "call_id": call_id.strip(), "name": _sanitize_replayed_fn_name(name),
        "arguments": ctx.sanitize_text(_coerce_arguments(item.get("arguments", "{}"))),
    }


def _preflight_function_call_output(item: Dict[str, Any], idx: int, ctx: _PreflightCtx) -> Dict[str, Any]:
    call_id = item.get("call_id")
    if not _nonblank(call_id):
        raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
    output = item.get("output", "")
    if isinstance(output, list):
        # Multimodal tool result: keep recognised input_text/input_image parts, drop the rest (4xx otherwise).
        cleaned: List[Dict[str, Any]] = []
        for part in output:
            ptype = part.get("type") if isinstance(part, dict) else None
            if ptype == "input_text" and _nonempty_str(part.get("text")):
                cleaned.append({"type": "input_text", "text": ctx.sanitize_text(part["text"])})
            elif ptype == "input_image" and _nonempty_str(part.get("image_url")):
                cleaned.append(_input_image_part(part, keep_empty_url=False))
        output_value: Any = cleaned or ""
    else:
        output_value = ctx.sanitize_text(_str_or_empty(output))
    return {"type": "function_call_output", "call_id": call_id.strip(), "output": output_value}


def _preflight_encrypted(item: Dict[str, Any], idx: int, ctx: _PreflightCtx) -> Optional[Dict[str, Any]]:
    """``reasoning`` / ``compaction`` items: opaque, issuer-sealed; forward only API-defined fields."""
    encrypted = item.get("encrypted_content")
    if not _nonempty_str(encrypted):
        return None
    if item["type"] == "compaction":
        return {"type": "compaction", "encrypted_content": encrypted}
    # ``id`` is used only for local dedup and NOT forwarded (store=False → server-side 404).
    item_id = item.get("id")
    if _nonempty_str(item_id):
        if item_id in ctx.seen_ids:
            return None
        ctx.seen_ids.add(item_id)
    summary = _as_list(item.get("summary"))
    return {
        "type": "reasoning", "encrypted_content": encrypted,
        "summary": _neutralize_harmony_structure(summary) if ctx.sanitize_harmony_tokens else summary,
    }


def _preflight_message(item: Dict[str, Any], idx: int, ctx: _PreflightCtx) -> Dict[str, Any]:
    if item.get("role") != "assistant":
        raise ValueError(f"Codex Responses input[{idx}] message items must have role='assistant'.")
    content = item.get("content")
    if not isinstance(content, list):
        raise ValueError(f"Codex Responses input[{idx}] message item must have content list.")
    normalized_content = []
    for part_idx, part in enumerate(content):
        if not isinstance(part, dict):
            raise ValueError(f"Codex Responses input[{idx}] message content[{part_idx}] must be an object.")
        part_type = part.get("type")
        if part_type not in _OUTPUT_TEXT_TYPES:
            raise ValueError(
                f"Codex Responses input[{idx}] message content[{part_idx}] has unsupported type {part_type!r}."
            )
        normalized_content.append({"type": "output_text", "text": ctx.sanitize_text(_str_or_empty(part.get("text", "")))})
    if not normalized_content:
        raise ValueError(f"Codex Responses input[{idx}] message item must contain at least one text part.")
    return _assistant_message_item(item, normalized_content, is_github_responses=ctx.is_github_responses)


def _preflight_role_message(item: Dict[str, Any], idx: int, ctx: _PreflightCtx) -> Dict[str, Any]:
    """Untyped ``user``/``assistant`` role message — the only legal shape besides typed items."""
    role = item.get("role")
    if role not in {"user", "assistant"}:
        raise ValueError(
            f"Codex Responses input[{idx}] has unsupported item shape (type={item.get('type')!r}, role={role!r})."
        )
    content = item.get("content", "")
    if not isinstance(content, list):
        return {"role": role, "content": ctx.sanitize_text(_str_or_empty(content))}
    # Parts are already Responses-shaped; validate and re-type text for the role.
    # Unlike history conversion, empty text / empty image urls are kept, not dropped.
    text_type = _text_type_for(role)
    validated: List[Dict[str, Any]] = []
    for part_idx, part in enumerate(content):
        if isinstance(part, str):
            if part:
                validated.append({"type": text_type, "text": ctx.sanitize_text(part)})
        elif not isinstance(part, dict):
            raise ValueError(f"Codex Responses input[{idx}].content[{part_idx}] must be an object or string.")
        elif (ptype := _part_type(part)) in _TEXT_PART_TYPES:
            text = part.get("text", "")
            text = text if isinstance(text, str) else str(text or "")
            validated.append({"type": text_type, "text": ctx.sanitize_text(text)})
        elif ptype in _IMAGE_PART_TYPES:
            validated.append(_input_image_part(part, role, keep_empty_url=True))
        else:
            raise ValueError(
                f"Codex Responses input[{idx}].content[{part_idx}] has unsupported type {part.get('type')!r}."
            )
    return {"role": role, "content": validated}


_PREFLIGHT_ITEM_HANDLERS: Dict[str, Callable[..., Optional[Dict[str, Any]]]] = {
    "function_call": _preflight_function_call, "function_call_output": _preflight_function_call_output,
    "reasoning": _preflight_encrypted, "compaction": _preflight_encrypted, "message": _preflight_message,
}


def _preflight_codex_input_items(
    raw_items: Any, *, is_github_responses: bool = False, sanitize_harmony_tokens: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError("Codex Responses input must be a list of input items.")
    sanitize_text = _neutralize_harmony_tokens if sanitize_harmony_tokens else (lambda text: text)
    ctx = _PreflightCtx(sanitize_text, sanitize_harmony_tokens, is_github_responses, set())
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"Codex Responses input[{idx}] must be an object.")
        item_type = item.get("type")
        handler = _PREFLIGHT_ITEM_HANDLERS.get(item_type) if isinstance(item_type, str) else None
        normalized_item = (handler or _preflight_role_message)(item, idx, ctx)
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized


def _preflight_tool(tool: Any, idx: int) -> Dict[str, Any]:
    if not isinstance(tool, dict):
        raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
    tool_type = tool.get("type")
    if tool_type in _RESPONSES_BUILTIN_TOOL_TYPES:  # provider-executed built-ins carry no name/parameters
        return dict(tool)
    if tool_type != "function":
        raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")
    name, parameters = tool.get("name"), tool.get("parameters")
    for ok, what in ((_nonblank(name), "a valid name"), (isinstance(parameters, dict), "valid parameters")):
        if not ok:
            raise ValueError(f"Codex Responses tools[{idx}] is missing {what}.")
    return {
        "type": "function", "name": name.strip(), "description": _str_or_empty(tool.get("description", "")),
        "strict": bool(tool.get("strict", False)), "parameters": parameters,
    }


# Optional scalar request fields, in wire order: (key, accept(value), coerce). Values
# failing ``accept`` are silently dropped.
_PREFLIGHT_OPTIONAL_FIELDS: tuple[tuple[str, Callable[[Any], bool], Optional[Callable[[Any], Any]]], ...] = (
    ("reasoning", lambda v: isinstance(v, dict), None),
    ("include", lambda v: isinstance(v, list), None),
    ("service_tier", _nonblank, str.strip),
    # Responses text controls (verbosity, structured-output format).
    ("text", lambda v: isinstance(v, dict) and bool(v), None),
    ("max_output_tokens", lambda v: isinstance(v, (int, float)) and v > 0, int),
    ("timeout", lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < float("inf"), float),
    ("temperature", lambda v: isinstance(v, (int, float)), float),
    # Cache routing/retention and tool-dispatch hints pass through as-is.
    *(
        (key, lambda v: v is not None, None)
        for key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key", "prompt_cache_retention")
    ),
    # Native compaction directive; eligibility is resolved in agent/native_compaction.py.
    ("context_management", lambda v: isinstance(v, list) and bool(v), None),
)

_PREFLIGHT_ALLOWED_KEYS = {
    "model", "instructions", "input", "tools", "store", "extra_headers", "extra_body",
    *(key for key, _, _ in _PREFLIGHT_OPTIONAL_FIELDS),
}


def _optional_dict(api_kwargs: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    value = api_kwargs.get(key)
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"Codex Responses request '{key}' must be an object.")
    return value


def _preflight_codex_api_kwargs(
    api_kwargs: Any, *, allow_stream: bool = False, is_github_responses: bool = False,
    sanitize_harmony_tokens: bool = False,
) -> Dict[str, Any]:
    if not isinstance(api_kwargs, dict):
        raise ValueError("Codex Responses request must be a dict.")
    if missing := sorted(key for key in ("model", "instructions", "input") if key not in api_kwargs):
        raise ValueError(f"Codex Responses request missing required field(s): {', '.join(missing)}.")
    model = api_kwargs.get("model")
    if not _nonblank(model):
        raise ValueError("Codex Responses request 'model' must be a non-empty string.")
    instructions = _str_or_empty(api_kwargs.get("instructions")).strip() or DEFAULT_AGENT_IDENTITY
    if sanitize_harmony_tokens:
        instructions = _neutralize_harmony_tokens(instructions)
    input_items = _preflight_codex_input_items(
        api_kwargs.get("input"), is_github_responses=is_github_responses, sanitize_harmony_tokens=sanitize_harmony_tokens,
    )
    normalized: Dict[str, Any] = {
        "model": model.strip(), "instructions": instructions, "input": input_items, "store": False,
    }
    tools = api_kwargs.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("Codex Responses request 'tools' must be a list when provided.")
        normalized_tools = [_preflight_tool(tool, idx) for idx, tool in enumerate(tools)]
        normalized["tools"] = _neutralize_harmony_structure(normalized_tools) if sanitize_harmony_tokens else normalized_tools
    if api_kwargs.get("store", False) is not False:
        raise ValueError("Codex Responses contract requires 'store' to be false.")
    for key, accept, coerce in _PREFLIGHT_OPTIONAL_FIELDS:
        value = api_kwargs.get(key)
        if accept(value):
            normalized[key] = coerce(value) if coerce else value
    extra_headers = _optional_dict(api_kwargs, "extra_headers") or {}
    if not all(_nonblank(key) for key in extra_headers):
        raise ValueError("Codex Responses request 'extra_headers' keys must be non-empty strings.")
    normalized_headers = {key.strip(): str(value) for key, value in extra_headers.items() if value is not None}
    if normalized_headers:
        normalized["extra_headers"] = normalized_headers
    # extra_body is verbatim: xAI carries ``prompt_cache_key`` as a body-level
    # field, and the SDK serializes extra_body without per-field checks.
    extra_body = _optional_dict(api_kwargs, "extra_body")
    if extra_body:
        normalized["extra_body"] = dict(extra_body)
    stream = api_kwargs.get("stream")
    if not allow_stream and "stream" in api_kwargs:
        raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")
    if allow_stream and stream is not None:
        if stream is not True:
            raise ValueError("Codex Responses 'stream' must be true when set.")
        normalized["stream"] = True
    # Defense-in-depth slash-enum strip for xAI (rejects ``Qwen/Qwen3.5`` enum values);
    # gated on the model name because native Codex accepts slashes.
    is_xai_model = str(api_kwargs.get("model") or "").lower().startswith(("grok-", "x-ai/grok-"))
    if is_xai_model and normalized.get("tools"):
        try:
            from tools.schema_sanitizer import strip_slash_enum
            normalized["tools"], _ = strip_slash_enum(normalized["tools"])
        except Exception:
            pass  # Best-effort — the caller-level sanitization should have handled it
    allowed_keys = _PREFLIGHT_ALLOWED_KEYS | ({"stream"} if allow_stream else set())
    if unexpected := sorted(key for key in api_kwargs if key not in allowed_keys):
        raise ValueError(f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}.")
    return normalized


# --- Response extraction helpers ----------------------------------------------

def _text_chunks(parts: Any, types: Optional[set] = None) -> List[str]:
    """Non-empty ``.text`` of each part (optionally filtered by ``.type``); [] if not a list."""
    selected = [part for part in _as_list(parts) if types is None or getattr(part, "type", None) in types]
    return [text for text in (getattr(part, "text", None) for part in selected) if _nonempty_str(text)]


def _extract_responses_message_text(item: Any) -> str:
    """Extract assistant text from a Responses message output item."""
    return "".join(_text_chunks(getattr(item, "content", None), _OUTPUT_TEXT_TYPES)).strip()


def _extract_responses_reasoning_text(item: Any) -> str:
    """Compact reasoning text from a Responses reasoning item (summary, else ``text``)."""
    chunks = _text_chunks(getattr(item, "summary", None))
    text = getattr(item, "text", None)
    return "\n".join(chunks).strip() if chunks else (text.strip() if isinstance(text, str) else "")


def _format_responses_error(error_obj: Any, response_status: str) -> str:
    """``"<code>: <message>"`` for a ``response.error`` payload (dict or object), else whichever
    is present, else ``str(error_obj)``, else a status-based default."""
    def field(name: str) -> str:
        value = _field(error_obj, name)
        return str(value).strip() if isinstance(value, str) or value else ""
    code_str, message_str = field("code"), field("message")
    if code_str and message_str:
        return f"{code_str}: {message_str}"
    return message_str or code_str or (str(error_obj) if error_obj else f"Responses API returned status '{response_status}'")


# --- Full response normalization ----------------------------------------------

def _response_tool_call(item: Any, item_type: str, index: int) -> SimpleNamespace:
    """Build a chat-style tool_call from a ``function_call``/``custom_tool_call`` item."""
    fn_name = getattr(item, "name", "") or ""
    arguments = getattr(item, "arguments" if item_type == "function_call" else "input", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    raw_item_id = getattr(item, "id", None)
    call_id = _resolve_call_id(getattr(item, "call_id", None), raw_item_id, fn_name, arguments, index, canonicalize_fc=False)
    fc_id = _derive_responses_function_call_id(call_id, raw_item_id if isinstance(raw_item_id, str) else None)
    return SimpleNamespace(
        id=call_id, call_id=call_id, response_item_id=fc_id, type="function",
        function=SimpleNamespace(name=fn_name, arguments=arguments),
    )


def _capture_encrypted_item(item: Any, item_type: str, issuer_kind: Optional[str]) -> Optional[Dict[str, Any]]:
    """``{type, encrypted_content[, _issuer_kind]}`` for replay, or None without a blob. Reasoning
    items also carry ``id`` + ``summary`` (required by the API on replay); transient ``rs_tmp_`` skip."""
    encrypted = getattr(item, "encrypted_content", None)
    if not _nonempty_str(encrypted):
        return None
    raw_item: Dict[str, Any] = {"type": item_type, "encrypted_content": encrypted}
    if issuer_kind:
        raw_item["_issuer_kind"] = issuer_kind
    if item_type != "reasoning":
        return raw_item
    item_id = getattr(item, "id", None)
    if isinstance(item_id, str) and item_id.startswith("rs_tmp_"):
        logger.debug("Skipping transient Codex reasoning item during normalization: %s", item_id)
        return None
    if _nonempty_str(item_id):
        raw_item["id"] = item_id
    summary = getattr(item, "summary", None)
    if isinstance(summary, list):
        texts = (getattr(part, "text", None) for part in summary)
        raw_item["summary"] = [{"type": "summary_text", "text": text} for text in texts if isinstance(text, str)]
    return raw_item


class _OutputScan:
    """Accumulated view of one Responses ``output`` list (phase 1 of normalization)."""

    def __init__(self, response_status: Optional[str]) -> None:
        self.content_parts, self.reasoning_parts, self.tool_calls = [], [], []
        self.reasoning_items_raw, self.message_items_raw = [], []
        self.has_incomplete_items = response_status in _INCOMPLETE_STATUSES
        self.saw_streaming_or_item_incomplete = response_status in {"queued", "in_progress"}
        self.saw_commentary_phase = self.saw_final_answer_phase = self.saw_reasoning_item = False

    def scan(self, output: List[Any], issuer_kind: Optional[str]) -> None:
        for item in output:
            item_type = getattr(item, "type", None)
            item_status = _lower_or_none(getattr(item, "status", None))
            if item_status in _INCOMPLETE_STATUSES and item_type not in _SERVER_SIDE_TOOL_CALL_TYPES:
                self.has_incomplete_items = True
                self.saw_streaming_or_item_incomplete = True
            if item_type == "message":
                self._message(item, item_status)
            elif item_type in {"reasoning", "compaction"}:
                if item_type == "reasoning":
                    self.saw_reasoning_item = True
                    reasoning_text = _extract_responses_reasoning_text(item)
                    if reasoning_text:
                        self.reasoning_parts.append(reasoning_text)
                # Compaction checkpoints ride the codex_reasoning_items sidecar (persistence,
                # replay, cross-issuer guard and kill switch for free).
                raw_item = _capture_encrypted_item(item, item_type, issuer_kind)
                if raw_item is not None:
                    self.reasoning_items_raw.append(raw_item)
                    if item_type == "compaction":
                        logger.info(
                            "Native Responses compaction item captured (%d chars encrypted).", len(raw_item["encrypted_content"]),
                        )
            elif item_type == "custom_tool_call" or (item_type == "function_call" and item_status not in _INCOMPLETE_STATUSES):
                self.tool_calls.append(_response_tool_call(item, item_type, len(self.tool_calls)))

    def _message(self, item: Any, item_status: Optional[str]) -> None:
        normalized_phase = _lower_or_none(getattr(item, "phase", None))
        is_commentary_phase = normalized_phase in {"commentary", "analysis"}
        self.saw_commentary_phase = self.saw_commentary_phase or is_commentary_phase
        self.saw_final_answer_phase = self.saw_final_answer_phase or normalized_phase in {"final_answer", "final"}
        message_text = _extract_responses_message_text(item)
        if not message_text:
            return
        # commentary/analysis text is mid-turn narration, never the final answer: route it
        # to the reasoning channel; the exact item is still preserved for replay/cache.
        (self.reasoning_parts if is_commentary_phase else self.content_parts).append(message_text)
        item_id = getattr(item, "id", None)
        self.message_items_raw.append(_message_item(
            [{"type": "output_text", "text": message_text}], status=_normalize_responses_message_status(item_status),
            item_id=item_id if isinstance(item_id, str) else None, phase=normalized_phase,
        ))


def _normalize_codex_response(response: Any, *, issuer_kind: Optional[str] = None) -> tuple[Any, str]:
    """Normalize a Responses API object to ``(assistant_message, finish_reason)``.
    ``issuer_kind`` is stamped onto captured reasoning items for cross-issuer replay drops."""
    response_status = _lower_or_none(getattr(response, "status", None))
    incomplete_reason = str(_field(getattr(response, "incomplete_details", None), "reason", "") or "").strip().lower()
    response_incomplete_content_filter = response_status == "incomplete" and incomplete_reason == "content_filter"
    output = getattr(response, "output", None)
    if not isinstance(output, list) or not output:
        # Codex can deliver the whole answer via stream events with an empty output.
        out_text = getattr(response, "output_text", None)
        out_text = out_text.strip() if isinstance(out_text, str) else ""
        if out_text:
            msg = "Codex response has empty output but output_text is present (%d chars); synthesizing output item."
            logger.debug(msg, len(out_text))
            content: List[Any] = [SimpleNamespace(type="output_text", text=out_text)]
        elif response_incomplete_content_filter:
            # Provider safety block, not a partial answer: finish content_filter, not incomplete.
            content = []
        else:
            raise RuntimeError("Responses API returned no output items")
        response.output = output = [
            SimpleNamespace(type="message", role="assistant", status="completed", content=content),
        ]
    if response_status in {"failed", "cancelled"}:
        raise RuntimeError(_format_responses_error(getattr(response, "error", None), response_status))
    scan = _OutputScan(response_status)
    scan.scan(output, issuer_kind)
    tool_calls, reasoning_parts = scan.tool_calls, scan.reasoning_parts
    final_text = "\n".join(scan.content_parts).strip()
    if not final_text and (scan.saw_final_answer_phase or not scan.saw_commentary_phase):
        out_text = getattr(response, "output_text", "")
        final_text = out_text.strip() if isinstance(out_text, str) else final_text
    # Tool-call leak recovery: gpt-5.x sometimes emits the intended ``function_call`` as plain Harmony text
    # (``to=functions.foo {json}``). Treat as incomplete so the continuation re-elicits a real call; clear the garbage.
    leaked_tool_call_text = bool(final_text and not tool_calls and _TOOL_CALL_LEAK_PATTERN.search(final_text))
    if leaked_tool_call_text:
        logger.warning(
            "Codex response contains leaked tool-call text in assistant content (no structured function_call "
            "items). Treating as incomplete so the continuation path can re-elicit a proper tool call. "
            "Leaked snippet: %r", final_text[:300],
        )
        final_text = ""
    # xAI grok-4.x sometimes puts the final answer inside the reasoning item after a ``<response>`` delimiter; without
    # salvage the reasoning-only rule marks the turn incomplete and every continuation is byte-identical. Promote the tail.
    if issuer_kind == "xai_responses" and not final_text and not tool_calls and reasoning_parts:
        joined_reasoning = "\n\n".join(reasoning_parts)
        marker = joined_reasoning.rfind("<response>")
        salvaged = joined_reasoning[marker + len("<response>"):].split("</response>", 1)[0].strip() if marker != -1 else ""
        if salvaged:
            logger.warning(
                "xAI response delivered its final answer inside the reasoning channel "
                "(<response> delimiter); promoting %d chars to assistant content.", len(salvaged),
            )
            final_text = salvaged
            reasoning_prefix = joined_reasoning[:marker].strip()
            reasoning_parts = [reasoning_prefix] if reasoning_prefix else []
    assistant_message = SimpleNamespace(
        content=final_text, tool_calls=tool_calls,
        reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
        reasoning_content=None, reasoning_details=None,
        codex_reasoning_items=scan.reasoning_items_raw or None, codex_message_items=scan.message_items_raw or None,
    )
    # Reasoning-only: for Codex/xAI/GitHub, status=completed means "still thinking" → incomplete so the continuation
    # retries. Other backends trust response.status — forcing incomplete there stalls for minutes on a final state.
    reasoning_only = (scan.reasoning_items_raw or reasoning_parts or scan.saw_reasoning_item) and not final_text
    trusted_final = (
        response_status == "completed" and issuer_kind not in ("codex_backend", "xai_responses", "github_responses")
    )
    if tool_calls:
        finish_reason = "tool_calls"
    elif response_incomplete_content_filter:
        finish_reason = "content_filter"
    elif (
        leaked_tool_call_text
        or scan.saw_streaming_or_item_incomplete
        or ((scan.has_incomplete_items or scan.saw_commentary_phase) and not scan.saw_final_answer_phase)
        or (reasoning_only and not trusted_final)
    ):
        finish_reason = "incomplete"
    else:
        finish_reason = "stop"
    return assistant_message, finish_reason
