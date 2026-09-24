"""Provider-encoded stream error decoding (split out of
:mod:`agent.chat_completion_helpers`): the SSE/JSON error-event parser, the
OpenAI-style ``error`` body normaliser, and :class:`ProviderStreamError` -- the
exception the retry loop reads to tell a provider-side error frame apart from a
transport drop.
"""

from __future__ import annotations

import contextlib
import json
import re
from types import SimpleNamespace
from typing import Any, Optional

from agent.error_classifier import (
    PROVIDER_STREAM_EMPTY_FRAME_ERROR_CODE, PROVIDER_STREAM_NON_JSON_ERROR_CODE)
from agent.message_sanitization import _sanitize_surrogates


_PROVIDER_STREAM_ERROR_FINISH_REASONS = {"error", "error_finish"}
_PROVIDER_STREAM_SSE_FIELDS = {"event", "data", "id", "retry"}
_PROVIDER_STREAM_ERROR_TEXT_LIMIT = 4096


class ProviderStreamError(Exception):
    """Provider encoded an API error as streaming content instead of an SDK error."""

    def __init__(self, *, status_code: Optional[int], body: dict, raw_text: str, headers: Any = None):
        self.status_code = status_code
        self.body = body
        self.raw_text = raw_text
        self.response = SimpleNamespace(headers=headers or {})
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        error_obj = self.body.get("error", {}) if isinstance(self.body, dict) else {}
        if not isinstance(error_obj, dict):
            error_obj = {}
        parts = ["Provider stream returned an error event"]
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        if error_obj.get("code"):
            parts.append(str(error_obj["code"]))
        text = " - ".join(parts)
        if error_obj.get("message"):
            text += f": {error_obj['message']}"
        return text


def _status_code_from_value(value: Any) -> Optional[int]:
    if isinstance(value, int) and 100 <= value < 600:
        return value
    if not isinstance(value, str):
        return None
    match = re.search(r"(?:HTTP_STATUS/)?\b([1-5]\d\d)\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _status_code_from_payload(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    candidates = [payload.get(k) for k in ("status_code", "status", "http_status")]
    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        candidates.extend(error_obj.get(k) for k in ("status_code", "status", "http_status", "code"))
    candidates.append(payload.get("code"))
    for candidate in candidates:
        status_code = _status_code_from_value(candidate)
        if status_code is not None:
            return status_code
    return None


def _json_object_from_text(text: str) -> Optional[dict]:
    stripped = (text or "").strip()
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        if stripped.startswith("{"):
            decoded = json.loads(stripped)
            return decoded if isinstance(decoded, dict) else None
    return None


def _parse_provider_sse_events(text: str) -> list[dict]:
    """Parse provider text that looks like Server-Sent Events."""
    events: list[dict] = []
    current = {"event": None, "data": [], "comments": [], "fields": {}}

    def _flush_current():
        nonlocal current
        if any(current.values()):
            status_candidates = list(current["comments"]) + [
                current["fields"][key]
                for key in ("status", "status_code", "http_status")
                if key in current["fields"]
            ]
            events.append({
                "event": current["event"],
                "data": "\n".join(current["data"]),
                "comments": list(current["comments"]),
                "fields": dict(current["fields"]),
                "status_code": next(
                    (s for s in map(_status_code_from_value, status_candidates) if s is not None), None),
            })
        current = {"event": None, "data": [], "comments": [], "fields": {}}

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip("\r")
        if line == "":
            _flush_current()
            continue
        if line.startswith(":"):
            current["comments"].append(line[1:].strip())
            continue

        field, sep, value = line.partition(":")
        if not sep:
            current["fields"][field.strip().lower()] = ""
            continue
        field = field.strip().lower()
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            current["event"] = value.strip()
        elif field == "data":
            current["data"].append(value)
        else:
            current["fields"][field] = value

    _flush_current()
    return events


def _provider_error_body(payload: dict, status_code: Optional[int]) -> dict:
    """Normalize common provider error payloads to OpenAI-style body.error."""
    if not isinstance(payload, dict):
        payload = {}
    elif isinstance(payload.get("error"), dict):
        return payload
    code = (payload.get("code") or payload.get("error_code") or payload.get("type")
            or (f"HTTP_{status_code}" if status_code else "provider_stream_error"))
    message = (payload.get("message") or payload.get("error_description") or payload.get("error")
               or "Provider stream returned an error event.")
    normalized_error = {"message": str(message)}
    if code:
        normalized_error["code"] = str(code)
    for key in ("request_id", "param", "type"):
        if payload.get(key):
            normalized_error[key] = payload[key]
    return {"error": normalized_error}


def _provider_stream_error_from_json_decode_error(error: json.JSONDecodeError, *,
    response: Any = None) -> ProviderStreamError:
    """Preserve plain-text SSE data rejected inside the OpenAI SDK: on a non-JSON
    ``event: error`` the SDK raises from ``sse.json()`` before yielding a chunk,
    but ``JSONDecodeError.doc`` still carries the provider's original message.

    An EMPTY ``doc`` is the other case: the frame carried no payload at all
    (``data:`` / ``event: ping`` / ``id:`` alone — legal SSE keepalives and no-ops),
    which the SDK's ``json.loads`` rejects the same way. A gateway that is degrading
    answers EVERY streaming request with such frames, so this is not the provider's
    malformed payload and must not be reported as one: it gets its own code and
    the stream helper recovers by retrying without streaming."""
    from agent.redact import redact_sensitive_text
    raw_text = str(getattr(error, "doc", "") or "").strip()
    headers = getattr(response, "headers", None) if response is not None else None
    if not raw_text:
        return ProviderStreamError(
            status_code=None,
            body=_provider_error_body(
                {"code": PROVIDER_STREAM_EMPTY_FRAME_ERROR_CODE,
                    "message": "Provider stream returned an empty SSE data frame (keepalive with no payload)."},
                None,
            ),
            raw_text="",
            headers=headers,
        )
    safe_text = redact_sensitive_text(_sanitize_surrogates(raw_text), force=True)
    safe_text = safe_text[:_PROVIDER_STREAM_ERROR_TEXT_LIMIT]
    return ProviderStreamError(
        status_code=None,
        body=_provider_error_body(
            {"code": PROVIDER_STREAM_NON_JSON_ERROR_CODE,
                "message": safe_text or "Provider stream returned non-JSON SSE data."},
            None,
        ),
        raw_text=safe_text,
        headers=headers,
    )


def _is_provider_stream_empty_frame_error(exc: BaseException) -> bool:
    """True for the translated contentless-SSE-frame error. Re-streaming cannot help
    (a degraded gateway answers every stream that way), so the caller must change channel."""
    body = getattr(exc, "body", None)
    error_obj = body.get("error") if isinstance(body, dict) else None
    return isinstance(error_obj, dict) and error_obj.get("code") == PROVIDER_STREAM_EMPTY_FRAME_ERROR_CODE


def _iter_provider_stream_chunks(stream, *, response: Any = None):
    """Yield SDK chunks while translating SDK-level SSE decode failures."""
    try:
        yield from stream
    except json.JSONDecodeError as error:
        stream_response = response() if callable(response) else response
        if stream_response is None:
            stream_response = getattr(stream, "response", None)
        raise _provider_stream_error_from_json_decode_error(error, response=stream_response) from error


def _payload_has_error_shape(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("error"), (dict, str)):
        return True
    return bool(payload.get("message")) and bool(
        payload.get("code") or payload.get("error_code") or _status_code_from_payload(payload) is not None)


def _provider_stream_text_may_be_sse(text: str) -> bool:
    """Return True while pending text still looks like an SSE control block."""
    stripped = (text or "").lstrip()
    if not stripped:
        return False

    lines = stripped.splitlines()
    trailing_newline = stripped.endswith(("\n", "\r"))
    saw_sse_field = False

    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r")
        if line == "":
            continue
        if line.startswith(":"):
            saw_sse_field = True
            continue

        field, sep, _value = line.partition(":")
        field_name = field.strip().lower()
        if sep and field_name in _PROVIDER_STREAM_SSE_FIELDS:
            saw_sse_field = True
            continue

        is_last_incomplete = index == len(lines) - 1 and not trailing_newline
        if is_last_incomplete and any(
            sse_field.startswith(field_name) for sse_field in _PROVIDER_STREAM_SSE_FIELDS):
            return True
        return False

    return saw_sse_field


def _provider_stream_error_from_text(text: str, finish_reason: Optional[str], *,
    response: Any = None) -> Optional[ProviderStreamError]:
    """Convert provider-streamed error text into an exception for retry logic."""
    if not text:
        return None

    if str(finish_reason or "").lower() not in _PROVIDER_STREAM_ERROR_FINISH_REASONS:
        return None

    headers = getattr(response, "headers", None) if response is not None else None

    def _error(payload: dict, status_code: Optional[int]) -> ProviderStreamError:
        return ProviderStreamError(status_code=status_code, body=_provider_error_body(payload, status_code),
            raw_text=text, headers=headers)

    for event in _parse_provider_sse_events(text):
        is_error_event = str(event.get("event") or "").strip().lower() == "error"
        payload = _json_object_from_text(event.get("data") or "") or {}
        status_code = event.get("status_code") or _status_code_from_payload(payload)
        # The finish_reason is an error here, so an error event always qualifies;
        # a non-error event needs an error-shaped payload or an HTTP error code.
        if (status_code is not None and status_code >= 400) or is_error_event or _payload_has_error_shape(payload):
            return _error(payload, status_code)

    payload = _json_object_from_text(text)
    if payload is not None:
        return _error(payload, _status_code_from_payload(payload))

    if text.strip():
        return _error({}, None)
    return None
