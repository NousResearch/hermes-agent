"""Shared helpers for HTTP-style ingress sources.

This module now holds two narrow families of ingress logic:

1. Webhook-style envelope helpers that build/schedule ``MessageEvent`` objects.
2. Request-normalization helpers shared by HTTP API ingress surfaces.

The goal is deliberately modest: provide a proven internal seam after parsing and
before execution, without changing public handler behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - import guard mirrors api_server optional dependency handling
    web = None  # type: ignore[assignment]

from gateway.platforms.base import MessageEvent, MessageType, SessionSource

MAX_NORMALIZED_TEXT_LENGTH = 65_536
MAX_CONTENT_LIST_SIZE = 1_000

_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_FILE_PART_TYPES = frozenset({"file", "input_file"})


class IngressAdapter(Protocol):
    """Minimal adapter surface required by the shared ingress helpers."""

    _background_tasks: set[asyncio.Task]

    def build_source(
        self,
        *,
        chat_id: str,
        chat_name: str | None = None,
        chat_type: str = "dm",
        user_id: str | None = None,
        user_name: str | None = None,
        **kwargs: Any,
    ) -> SessionSource: ...

    async def handle_message(self, event: MessageEvent): ...


@dataclass(frozen=True)
class IngressEnvelope:
    """Normalized internal representation for HTTP-originated message ingress."""

    text: str
    message_id: str | None
    chat_id: str
    chat_name: str | None = None
    chat_type: str = "webhook"
    user_id: str | None = None
    user_name: str | None = None
    raw_payload: Any = None
    internal: bool = False


@dataclass(frozen=True)
class NormalizedSessionChatRequest:
    """Normalized request payload shared by session chat sync + streaming endpoints."""

    user_message: Any
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class NormalizedRunRequest:
    """Normalized /v1/runs request body before response-store expansion."""

    raw_input: Any
    user_message: str
    conversation_history: List[Dict[str, str]]
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass(frozen=True)
class NormalizedChatCompletionsRequest:
    """Normalized /v1/chat/completions request body."""

    user_message: Any
    history: List[Dict[str, Any]]
    conversation_messages: List[Dict[str, Any]]
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class NormalizedResponsesRequest:
    """Normalized /v1/responses request body before response-store chaining."""

    raw_input: Any
    input_messages: List[Dict[str, Any]]
    user_message: Any
    conversation_history: List[Dict[str, Any]]
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[str] = None
    store: bool = True
    history_from_input_messages: bool = False


def build_ingress_message_event(adapter: IngressAdapter, envelope: IngressEnvelope) -> MessageEvent:
    """Build a ``MessageEvent`` for an ingress envelope using an adapter's source factory."""

    source = adapter.build_source(
        chat_id=envelope.chat_id,
        chat_name=envelope.chat_name,
        chat_type=envelope.chat_type,
        user_id=envelope.user_id,
        user_name=envelope.user_name,
    )
    return MessageEvent(
        text=envelope.text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=envelope.raw_payload,
        message_id=envelope.message_id,
        internal=envelope.internal,
    )


def schedule_ingress_event(adapter: IngressAdapter, event: MessageEvent) -> asyncio.Task | None:
    """Schedule a background ingress event and register it with the adapter."""

    task = asyncio.create_task(adapter.handle_message(event))
    try:
        adapter._background_tasks.add(task)
    except TypeError:
        # Mirror the base adapter's test-friendly behavior: some tests stub
        # task creation with lightweight sentinels that aren't hashable.
        return None
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(adapter._background_tasks.discard)
    return task


def schedule_ingress_envelope(adapter: IngressAdapter, envelope: IngressEnvelope) -> asyncio.Task | None:
    """Build and schedule an ingress envelope in one step."""

    event = build_ingress_message_event(adapter, envelope)
    return schedule_ingress_event(adapter, event)


def _openai_error(
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


def _coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like API payload values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        if normalized in _FALSE_REQUEST_BOOL_STRINGS:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _normalize_chat_content(
    content: Any, *, _max_depth: int = 10, _depth: int = 0,
) -> str:
    """Normalize OpenAI chat message content into a plain text string."""
    if _depth > _max_depth:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content

    if isinstance(content, list):
        parts: List[str] = []
        items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
        for item in items:
            if isinstance(item, str):
                if item:
                    parts.append(item[:MAX_NORMALIZED_TEXT_LENGTH])
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", "")
                    if text:
                        try:
                            parts.append(str(text)[:MAX_NORMALIZED_TEXT_LENGTH])
                        except Exception:
                            pass
            elif isinstance(item, list):
                nested = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
                if nested:
                    parts.append(nested)
            if sum(len(p) for p in parts) >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        result = "\n".join(parts)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result

    try:
        result = str(content)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result
    except Exception:
        return ""


def _normalize_multimodal_content(content: Any) -> Any:
    """Validate and normalize multimodal content for API-style ingress."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content
    if not isinstance(content, list):
        return _normalize_chat_content(content)

    items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
    normalized_parts: List[Dict[str, Any]] = []

    for part in items:
        if isinstance(part, str):
            if part:
                trimmed = part[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
            continue

        if not isinstance(part, dict):
            continue

        raw_type = part.get("type")
        part_type = str(raw_type or "").strip().lower()

        if part_type in _TEXT_PART_TYPES:
            text = part.get("text")
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if text:
                trimmed = text[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
            continue

        if part_type in _IMAGE_PART_TYPES:
            detail = part.get("detail")
            image_ref = part.get("image_url")
            if isinstance(image_ref, dict):
                url_value = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url_value = image_ref
            if not isinstance(url_value, str) or not url_value.strip():
                raise ValueError("invalid_image_url:Image parts must include a non-empty image URL.")
            url_value = url_value.strip()
            lowered = url_value.lower()
            if lowered.startswith("data:"):
                if not lowered.startswith("data:image/") or "," not in url_value:
                    raise ValueError(
                        "unsupported_content_type:Only image data URLs are supported. "
                        "Non-image data payloads are not supported."
                    )
            elif not (lowered.startswith("http://") or lowered.startswith("https://")):
                raise ValueError(
                    "invalid_image_url:Image inputs must use http(s) URLs or data:image/... URLs."
                )
            image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
            if detail is not None:
                if not isinstance(detail, str) or not detail.strip():
                    raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
                image_part["image_url"]["detail"] = detail.strip()
            normalized_parts.append(image_part)
            continue

        if part_type in _FILE_PART_TYPES:
            raise ValueError(
                "unsupported_content_type:Inline image inputs are supported, "
                "but uploaded files and document inputs are not supported on this endpoint."
            )

        raise ValueError(
            f"unsupported_content_type:Unsupported content part type {raw_type!r}. "
            "Only text and image_url/input_image parts are supported."
        )

    if not normalized_parts:
        return ""
    if all(p.get("type") == "text" for p in normalized_parts):
        return "\n".join(p["text"] for p in normalized_parts if p.get("text"))
    return normalized_parts


def _content_has_visible_payload(content: Any) -> bool:
    """True when content has any text or image attachment."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in _TEXT_PART_TYPES and str(part.get("text") or "").strip():
                    return True
                if ptype in _IMAGE_PART_TYPES:
                    return True
    return False


def _multimodal_validation_error(exc: ValueError, *, param: str) -> Any:
    """Translate a multimodal validation ``ValueError`` into a 400 response."""
    raw = str(exc)
    code, _, message = raw.partition(":")
    if not message:
        code, message = "invalid_content_part", raw
    assert web is not None
    return web.json_response(
        _openai_error(message, code=code, param=param),
        status=400,
    )


def _session_chat_user_message(body: Dict[str, Any], *, param: str = "message") -> tuple[Any, Any]:
    """Parse and normalize session chat ``message`` / ``input`` like chat completions."""
    user_message = body.get("message") or body.get("input")
    if not _content_has_visible_payload(user_message):
        assert web is not None
        return None, web.json_response(
            _openai_error("Missing 'message' field", code="missing_message"),
            status=400,
        )
    try:
        return _normalize_multimodal_content(user_message), None
    except ValueError as exc:
        return None, _multimodal_validation_error(exc, param=param)


def _normalize_session_chat_request(
    body: Dict[str, Any], *, message_param: str = "message"
) -> tuple[Optional[NormalizedSessionChatRequest], Any]:
    """Normalize shared session-chat request fields used by sync + SSE handlers."""
    user_message, err = _session_chat_user_message(body, param=message_param)
    if err is not None:
        return None, err
    system_prompt = body.get("system_message") or body.get("instructions")
    if system_prompt is not None and not isinstance(system_prompt, str):
        assert web is not None
        return None, web.json_response(
            _openai_error("system_message must be a string", code="invalid_system_message"),
            status=400,
        )
    return NormalizedSessionChatRequest(
        user_message=user_message,
        system_prompt=system_prompt,
    ), None


def _normalize_run_request(
    body: Dict[str, Any],
) -> tuple[Optional[NormalizedRunRequest], Any]:
    """Normalize /v1/runs body fields prior to previous_response expansion."""
    raw_input = body.get("input")
    if not raw_input:
        assert web is not None
        return None, web.json_response(_openai_error("Missing 'input' field"), status=400)

    user_message = raw_input
    conversation_history: List[Dict[str, str]] = []
    if isinstance(raw_input, list):
        last_message = raw_input[-1] if raw_input else None
        if isinstance(last_message, dict):
            user_message = _normalize_chat_content(last_message.get("content", ""))
        else:
            user_message = _normalize_chat_content(last_message)
    elif not isinstance(raw_input, str):
        user_message = _normalize_chat_content(raw_input)

    if not user_message:
        assert web is not None
        return None, web.json_response(_openai_error("No user message found in input"), status=400)

    raw_history = body.get("conversation_history")
    if raw_history:
        if not isinstance(raw_history, list):
            assert web is not None
            return None, web.json_response(
                _openai_error("'conversation_history' must be an array of message objects"),
                status=400,
            )
        for i, entry in enumerate(raw_history):
            if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                assert web is not None
                return None, web.json_response(
                    _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                    status=400,
                )
            conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
    elif isinstance(raw_input, list) and len(raw_input) > 1:
        for msg in raw_input[:-1]:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                conversation_history.append(
                    {
                        "role": str(msg["role"]),
                        "content": _normalize_chat_content(msg["content"]),
                    }
                )

    return NormalizedRunRequest(
        raw_input=raw_input,
        user_message=str(user_message),
        conversation_history=conversation_history,
        instructions=body.get("instructions"),
        previous_response_id=body.get("previous_response_id"),
        session_id=body.get("session_id"),
    ), None


def _normalize_chat_completions_request(
    body: Dict[str, Any],
) -> tuple[Optional[NormalizedChatCompletionsRequest], Any]:
    """Normalize /v1/chat/completions request fields prior to session handling."""
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        assert web is not None
        return None, web.json_response(
            {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
            status=400,
        )

    system_prompt = None
    conversation_messages: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        raw_content = msg.get("content", "") if isinstance(msg, dict) else ""
        if role == "system":
            content = _normalize_chat_content(raw_content)
            if system_prompt is None:
                system_prompt = content
            else:
                system_prompt = system_prompt + "\n" + content
        elif role in {"user", "assistant"}:
            try:
                content = _normalize_multimodal_content(raw_content)
            except ValueError as exc:
                return None, _multimodal_validation_error(exc, param=f"messages[{idx}].content")
            conversation_messages.append({"role": role, "content": content})

    user_message: Any = ""
    history: List[Dict[str, Any]] = []
    if conversation_messages:
        user_message = conversation_messages[-1].get("content", "")
        history = conversation_messages[:-1]

    if not _content_has_visible_payload(user_message):
        assert web is not None
        return None, web.json_response(
            {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
            status=400,
        )

    return NormalizedChatCompletionsRequest(
        user_message=user_message,
        history=history,
        conversation_messages=conversation_messages,
        system_prompt=system_prompt,
    ), None


def _normalize_responses_request(
    body: Dict[str, Any],
) -> tuple[Optional[NormalizedResponsesRequest], Any]:
    """Normalize /v1/responses request fields before response-store chaining."""
    raw_input = body.get("input")
    if raw_input is None:
        assert web is not None
        return None, web.json_response(_openai_error("Missing 'input' field"), status=400)

    instructions = body.get("instructions")
    previous_response_id = body.get("previous_response_id")
    conversation = body.get("conversation")
    store = _coerce_request_bool(body.get("store"), default=True)

    if conversation and previous_response_id:
        assert web is not None
        return None, web.json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

    input_messages: List[Dict[str, Any]] = []
    if isinstance(raw_input, str):
        input_messages = [{"role": "user", "content": raw_input}]
    elif isinstance(raw_input, list):
        for idx, item in enumerate(raw_input):
            if isinstance(item, str):
                input_messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                try:
                    content = _normalize_multimodal_content(item.get("content", ""))
                except ValueError as exc:
                    return None, _multimodal_validation_error(exc, param=f"input[{idx}].content")
                input_messages.append({"role": role, "content": content})
    else:
        assert web is not None
        return None, web.json_response(_openai_error("'input' must be a string or array"), status=400)

    conversation_history: List[Dict[str, Any]] = []
    history_from_input_messages = False
    raw_history = body.get("conversation_history")
    if raw_history:
        if not isinstance(raw_history, list):
            assert web is not None
            return None, web.json_response(
                _openai_error("'conversation_history' must be an array of message objects"),
                status=400,
            )
        for i, entry in enumerate(raw_history):
            if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                assert web is not None
                return None, web.json_response(
                    _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                    status=400,
                )
            try:
                entry_content = _normalize_multimodal_content(entry["content"])
            except ValueError as exc:
                return None, _multimodal_validation_error(exc, param=f"conversation_history[{i}].content")
            conversation_history.append({"role": str(entry["role"]), "content": entry_content})
    else:
        conversation_history = list(input_messages[:-1])
        history_from_input_messages = True

    user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
    if not _content_has_visible_payload(user_message):
        assert web is not None
        return None, web.json_response(_openai_error("No user message found in input"), status=400)

    return NormalizedResponsesRequest(
        raw_input=raw_input,
        input_messages=input_messages,
        user_message=user_message,
        conversation_history=conversation_history,
        instructions=instructions,
        previous_response_id=previous_response_id,
        conversation=conversation,
        store=store,
        history_from_input_messages=history_from_input_messages,
    ), None

from typing import Any, Dict

def clean_log_value(value: Any, *, max_len: int = 200) -> str:
    """Sanitize request metadata before it reaches security logs."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:max_len]

def extract_request_audit_context(request: Any) -> Dict[str, str]:
    """Return non-secret source metadata for security/audit warnings."""
    peer_ip = ""
    try:
        peer = request.transport.get_extra_info("peername") if request.transport else None
        if isinstance(peer, (tuple, list)) and peer:
            peer_ip = str(peer[0])
    except Exception:
        peer_ip = ""

    return {
        "remote": clean_log_value(getattr(request, "remote", "") or peer_ip),
        "peer_ip": clean_log_value(peer_ip),
        "forwarded_for": clean_log_value(request.headers.get("X-Forwarded-For", "")),
        "real_ip": clean_log_value(request.headers.get("X-Real-IP", "")),
        "method": clean_log_value(request.method, max_len=16),
        "path": clean_log_value(getattr(request, "path_qs", ""), max_len=500),
        "user_agent": clean_log_value(request.headers.get("User-Agent", ""), max_len=300),
    }

def extract_request_audit_log_suffix(request: Any) -> str:
    ctx = extract_request_audit_context(request)
    fields = [f"{key}={value!r}" for key, value in ctx.items() if value]
    return " ".join(fields) if fields else "source='unknown'"
