from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

logger = logging.getLogger(__name__)


def _classify_responses_issuer(
    *,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
    is_codex_backend: bool = False,
    base_url: Optional[str] = None,
) -> str:
    """Stable identifier for the Responses endpoint that mints encrypted_content."""
    if is_xai_responses:
        return "xai_responses"
    if is_github_responses:
        return "github_responses"
    if is_codex_backend:
        return "codex_backend"
    if base_url:
        return f"other:{base_url}"
    return "other"


# Throttle the per-process cross-issuer skip warning so we don't flood logs
# when a long history contains many stale-issuer reasoning blocks.
_CROSS_ISSUER_WARN_EMITTED = False


# Matches Codex/Harmony tool-call serialization that occasionally leaks into
# assistant-message content when the model fails to emit a structured
# ``function_call`` item.  Accepts the common forms:
#
#   to=functions.exec_command
#   assistant to=functions.exec_command
#   <|channel|>commentary to=functions.exec_command
#
# ``to=functions.<name>`` is the stable marker — the optional ``assistant`` or
# Harmony channel prefix varies by degeneration mode.  Case-insensitive to
# cover lowercase/uppercase ``assistant`` variants.
_TOOL_CALL_LEAK_PATTERN = re.compile(
    r"(?:^|[\s>|])to=functions\.[A-Za-z_][\w.]*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Multimodal content helpers
# ---------------------------------------------------------------------------

def _chat_content_to_responses_parts(content: Any, *, role: str = "user") -> List[Dict[str, Any]]:
    """Convert chat-style multimodal content to Responses API input parts."""
    text_type = "output_text" if role == "assistant" else "input_text"
    if not isinstance(content, list):
        return []
    converted: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                converted.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip().lower()
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                converted.append({"type": text_type, "text": text})
            continue
        if ptype in {"image_url", "input_image"}:
            image_ref = part.get("image_url")
            detail = part.get("detail")
            if isinstance(image_ref, dict):
                url = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url = image_ref
            if not isinstance(url, str) or not url:
                continue
            image_part: Dict[str, Any] = {"type": "input_image", "image_url": url}
            if isinstance(detail, str) and detail.strip():
                image_part["detail"] = detail.strip()
            converted.append(image_part)
    return converted


def _summarize_user_message_for_log(content: Any, *, sep: str = " ") -> str:
    """Flatten message content to a plain-text summary."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_bits: List[str] = []
        image_count = 0
        for part in content:
            if isinstance(part, str):
                if part:
                    text_bits.append(part)
                continue
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "").strip().lower()
            if ptype in {"text", "input_text", "output_text"}:
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_bits.append(text)
            elif ptype in {"image_url", "input_image"}:
                image_count += 1
        summary = sep.join(text_bits).strip()
        if image_count:
            note = f"[{image_count} image{'s' if image_count != 1 else ''}]"
            summary = f"{note} {summary}" if summary else note
        return summary
    try:
        return str(content)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Generate a deterministic call_id from tool call content."""
    seed = f"{fn_name}:{arguments}:{index}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"call_{digest}"


def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
    """Split a stored tool id into (call_id, response_item_id)."""
    if not isinstance(raw_id, str):
        return None, None
    value = raw_id.strip()
    if not value:
        return None, None
    if "|" in value:
        call_id, response_item_id = value.split("|", 1)
        call_id = call_id.strip() or None
        response_item_id = response_item_id.strip() or None
        return call_id, response_item_id
    if value.startswith("fc_"):
        return None, value
    return value, None


def _derive_responses_function_call_id(
    call_id: str,
    response_item_id: Optional[str] = None,
) -> str:
    """Build a valid Responses `function_call.id` (must start with `fc_`)."""
    if isinstance(response_item_id, str):
        candidate = response_item_id.strip()
        if candidate.startswith("fc_"):
            return candidate

    source = (call_id or "").strip()
    if source.startswith("fc_"):
        return source
    if source.startswith("call_") and len(source) > len("call_"):
        return f"fc_{source[len('call_'):]}"

    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
    if sanitized.startswith("fc_"):
        return sanitized
    if sanitized.startswith("call_") and len(sanitized) > len("call_"):
        return f"fc_{sanitized[len('call_'):]}"
    if sanitized:
        return f"fc_{sanitized[:48]}"

    seed = source or str(response_item_id or "") or uuid.uuid4().hex
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
    return f"fc_{digest}"


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

def _responses_tools(tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
    """Convert chat-completions tool schemas to Responses function-tool schemas."""
    if not tools:
        return None

    converted: List[Dict[str, Any]] = []
    for item in tools:
        fn = item.get("function", {}) if isinstance(item, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "strict": False,
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted or None


# Provider-executed built-in tool *declaration* types accepted on the
# Responses ``tools`` array.  These are declared by ``type`` alone (no
# client-side name/parameters schema) and run server-side — the provider
# owns the implementation and reports progress via the matching ``*_call``
# output items.  Hermes injects xAI's native ``web_search`` for the xAI
# transport (see agent/transports/codex.py); the rest are listed so the
# preflight validator passes them through rather than rejecting them as
# \"unsupported type\".  Mirrors the ``*_call`` item-type set used in
# _normalize_codex_response.
_RESPONSES_BUILTIN_TOOL_TYPES = {
    "web_search",
    "web_search_preview",
    "file_search",
    "code_interpreter",
    "image_generation",
    "computer_use_preview",
    "local_shell",
}


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------

_RESPONSE_MESSAGE_STATUSES = {"completed", "incomplete", "in_progress"}

# The Responses API rejects input[].id longer than this with a non-retryable
# HTTP 400 (\"string too long\"). Codex-issued assistant message ids are
# server-assigned base64 blobs that can run 400+ chars, while Hermes-minted
# ids (msg_...) stay well under this cap and are worth keeping for
# prefix-cache hits. Drop only the oversized ones on replay.
_MAX_RESPONSES_ITEM_ID_LENGTH = 64


def _normalize_responses_message_status(value: Any, *, default: str = "completed") -> str:
    """Normalize a Responses assistant message status for replay."""
    if isinstance(value, str):
        status = value.strip().lower().replace("-", "_").replace(" ", "_")
        if status in _RESPONSE_MESSAGE_STATUSES:
            return status
    return default


def _chat_messages_to_responses_input(
    messages: List[Dict[str, Any]],
    *,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
    replay_encrypted_reasoning: bool = True,
    current_issuer_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert internal chat-style messages to Responses input items."""
    items: List[Dict[str, Any]] = []
    seen_item_ids: set = set()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue

        if role in {"user", "assistant"}:
            content = msg.get("content", "")
            if isinstance(content, list):
                content_parts = _chat_content_to_responses_parts(content, role=role)
                text_type = "output_text" if role == "assistant" else "input_text"
                content_text = "".join(
                    p.get("text", "") for p in content_parts if p.get("type") == text_type
                )
            else:
                content_parts = []
                content_text = str(content) if content is not None else ""

            if role == "assistant":
                codex_reasoning = (
                    msg.get("codex_reasoning_items")
                    if replay_encrypted_reasoning
                    else None
                )
                has_codex_reasoning = False
                if isinstance(codex_reasoning, list):
                    for ri in codex_reasoning:
                        if isinstance(ri, dict) and ri.get("encrypted_content"):
                            item_id = ri.get("id")
                            if item_id and item_id in seen_item_ids:
                                continue
                            item_issuer = ri.get("_issuer_kind")
                            if (
                                current_issuer_kind is not None
                                and item_issuer is not None
                                and item_issuer != current_issuer_kind
                            ):
                                global _CROSS_ISSUER_WARN_EMITTED
                                if not _CROSS_ISSUER_WARN_EMITTED:
                                    logger.warning(
                                        "Dropping reasoning item minted by %s while "
                                        "calling %s — encrypted_content is sealed to "
                                        "its issuer.",
                                        item_issuer, current_issuer_kind,
                                    )
                                    _CROSS_ISSUER_WARN_EMITTED = True
                                continue
                            replay_item = {
                                k: v for k, v in ri.items()
                                if k not in ("id", "_issuer_kind")
                            }
                            items.append(replay_item)
                            if item_id:
                                seen_item_ids.add(item_id)
                            has_codex_reasoning = True

                codex_message_items = msg.get("codex_message_items")
                replayed_message_items = 0
                if isinstance(codex_message_items, list):
                    for raw_item in codex_message_items:
                        if not isinstance(raw_item, dict):
                            continue
                        if raw_item.get("type") != "message" or raw_item.get("role") != "assistant":
                            continue
                        raw_content_parts = raw_item.get("content")
                        if not isinstance(raw_content_parts, list):
                            continue

                        normalized_content_parts = []
                        for part in raw_content_parts:
                            if not isinstance(part, dict):
                                continue
                            part_type = str(part.get("type") or "").strip()
                            if part_type not in {"output_text", "text"}:
                                continue
                            text = part.get("text", "")
                            if text is None:
                                text = ""
                            if not isinstance(text, str):
                                text = str(text)
                            normalized_content_parts.append({"type": "output_text", "text": text})

                        if not normalized_content_parts:
                            continue

                        replay_item = {
                            "type": "message",
                            "role": "assistant",
                            "status": _normalize_responses_message_status(raw_item.get("status")),
                            "content": normalized_content_parts,
                        }
                        item_id = raw_item.get("id")
                        if (
                            not is_github_responses
                            and isinstance(item_id, str)
                            and item_id.strip()
                        ):
                            stripped_id = item_id.strip()
                            if len(stripped_id) <= _MAX_RESPONSES_ITEM_ID_LENGTH:
                                replay_item["id"] = stripped_id
                        phase = raw_item.get("phase")
                        if isinstance(phase, str) and phase.strip():
                            replay_item["phase"] = phase.strip()
                        items.append(replay_item)
                        replayed_message_items += 1

                if replayed_message_items > 0:
                    pass
                elif content_parts:
                    items.append({"role": "assistant", "content": content_parts})
                elif content_text.strip():
                    items.append({"role": "assistant", "content": content_text})
                elif has_codex_reasoning:
                    items.append({"role": "assistant", "content": ""})

                # #69280: Omit tool_calls key entirely if the list is empty.
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        fn_name = fn.get("name")
                        if not isinstance(fn_name, str) or not fn_name.strip():
                            continue

                        embedded_call_id, embedded_response_item_id = _split_responses_tool_id(
                            tc.get("id")
                        )
                        call_id = tc.get("call_id")
                        if not isinstance(call_id, str) or not call_id.strip():
                            call_id = embedded_call_id
                        if not isinstance(call_id, str) or not call_id.strip():
                            if (
                                isinstance(embedded_response_item_id, str)
                                and embedded_response_item_id.startswith("fc_")
                            ):
                                call_id = f"call_{embedded_response_item_id[3:]}"
                        if not isinstance(call_id, str) or not call_id.strip():
                            call_id = _deterministic_call_id(fn_name, fn.get("arguments", ""))

                        items.append({
                            "type": "function_call",
                            "id": _derive_responses_function_call_id(
                                call_id,
                                response_item_id=embedded_response_item_id,
                            ),
                            "name": fn_name,
                            "arguments": fn.get("arguments", ""),
                        })
