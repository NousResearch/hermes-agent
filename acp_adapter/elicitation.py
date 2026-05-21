"""ACP elicitation helpers.

This module intentionally keeps ACP elicitation-specific behavior isolated from
permission/authorization handling. Capability detection accepts generated SDK
models as well as dict-like fallback data because older SDKs or clients may
preserve unstable capability fields in metadata.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, cast

__all__ = [
    "OTHER_LABEL",
    "build_clarify_requested_schema",
    "create_form_elicitation",
    "extract_clarify_answer",
    "make_elicitation_clarify_callback",
    "supports_form_elicitation",
]

logger = logging.getLogger(__name__)

OTHER_LABEL = "Other (type your answer)"


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _model_dump(value: Any) -> dict[str, Any] | None:
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        return None
    try:
        data = dump(by_alias=True, exclude_none=True)
    except TypeError:
        data = dump()
    return data if isinstance(data, dict) else None


def _metadata_elicitation(client_capabilities: object) -> Any:
    for metadata_key in ("meta", "_meta", "field_meta"):
        meta = _get_attr_or_key(client_capabilities, metadata_key)
        elicitation = _get_attr_or_key(meta, "elicitation")
        if elicitation is not None:
            return elicitation
        elicitation = _get_attr_or_key(meta, "acp.elicitation")
        if elicitation is not None:
            return elicitation
    return None


def build_clarify_requested_schema(*, question: str, choices: list[str] | None) -> dict[str, Any]:
    """Build a simple ACP form schema for Hermes clarify prompts."""
    if choices:
        cleaned_choices = [str(choice).strip() for choice in choices if str(choice).strip()]
        enum_values = cleaned_choices[:4]
        if OTHER_LABEL not in enum_values:
            enum_values.append(OTHER_LABEL)
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "title": question,
                    "enum": enum_values,
                },
                "other_answer": {
                    "type": "string",
                    "title": "Other answer",
                    "description": "Fill this only if you selected Other.",
                },
            },
            "required": ["answer"],
        }

    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "title": question,
                "minLength": 1,
                "maxLength": 4000,
            }
        },
        "required": ["answer"],
    }


async def create_form_elicitation(
    conn: object,
    *,
    session_id: str,
    question: str,
    choices: list[str] | None,
) -> object:
    """Send an ACP form elicitation request and return the client response."""
    requested_schema = build_clarify_requested_schema(question=question, choices=choices)

    typed_helper = getattr(conn, "create_elicitation", None)
    if callable(typed_helper):
        result = typed_helper(
            session_id=session_id,
            mode="form",
            message=question,
            requested_schema=requested_schema,
        )
        return await cast(Awaitable[object], result)

    params = {
        "sessionId": session_id,
        "mode": "form",
        "message": question,
        "requestedSchema": requested_schema,
    }

    raw_conn = getattr(conn, "_conn", conn)
    send_request = getattr(raw_conn, "send_request", None)
    if not callable(send_request):
        raise RuntimeError("ACP connection does not support elicitation/create requests")
    result = send_request("elicitation/create", params)
    return await cast(Awaitable[object], result)


def extract_clarify_answer(response: object) -> str:
    """Convert an ACP elicitation response into Hermes' clarify string result."""
    if response is None:
        return "[clarify prompt could not be delivered]"

    action = _get_attr_or_key(response, "action")
    if action == "decline":
        return "[user declined the clarification]"
    if action == "cancel":
        return "[user cancelled the clarification]"
    if action != "accept":
        return "[clarify prompt could not be delivered]"

    content = _get_attr_or_key(response, "content")
    if content is None:
        return "[clarify prompt could not be delivered]"

    answer = _get_attr_or_key(content, "answer", "")
    other = _get_attr_or_key(content, "other_answer", "")
    if str(answer).strip() == OTHER_LABEL and str(other).strip():
        return str(other).strip()
    return str(answer or "").strip()


def _format_timeout_sentinel(timeout: float) -> str:
    seconds = max(1, int(timeout))
    if seconds < 60:
        return f"[user did not respond within {seconds}s]"
    minutes = max(1, int(seconds / 60))
    return f"[user did not respond within {minutes}m]"


def make_elicitation_clarify_callback(
    conn: object,
    loop: asyncio.AbstractEventLoop,
    session_id: str,
    *,
    timeout: float = 120.0,
) -> Callable[[str, list[str] | None], str]:
    """Create a blocking Hermes clarify callback backed by ACP elicitation."""

    def _callback(question: str, choices: list[str] | None = None) -> str:
        from agent.async_utils import safe_schedule_threadsafe

        future = safe_schedule_threadsafe(
            create_form_elicitation(
                conn,
                session_id=session_id,
                question=question,
                choices=choices,
            ),
            loop,
            logger=logger,
            log_message="ACP elicitation: failed to schedule on loop",
        )
        if future is None:
            return "[clarify prompt could not be delivered]"

        try:
            response = future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            logger.warning("ACP elicitation timed out")
            return _format_timeout_sentinel(timeout)
        except Exception as exc:
            future.cancel()
            logger.warning("ACP elicitation failed: %s", exc)
            return "[clarify prompt could not be delivered]"
        return extract_clarify_answer(response)

    return _callback


def supports_form_elicitation(client_capabilities: object) -> bool:
    """Return whether ACP client capabilities advertise form elicitation."""
    if client_capabilities is None:
        return False

    elicitation = _get_attr_or_key(client_capabilities, "elicitation")
    if elicitation is None:
        elicitation = _metadata_elicitation(client_capabilities)

    if elicitation is None:
        return False

    if isinstance(elicitation, dict):
        if elicitation == {}:
            return True
        return "form" in elicitation

    if getattr(elicitation, "form", None) is not None:
        return True

    data = _model_dump(elicitation)
    if data is not None:
        return data == {} or "form" in data

    return False
