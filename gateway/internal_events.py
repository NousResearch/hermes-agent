"""Typed host boundary for plugin-triggered gateway continuation events.

The public plugin API creates these values; gateway code carries them through
the normal adapter queue and agent turn path.  Keeping the marker typed and
closed prevents a caller from using the extension as a generic role override.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Optional


GATEWAY_SYSTEM_EVENT_KINDS: Final[frozenset[str]] = frozenset(
    {
        "external_tool_completed",
        "external_tool_error",
        "external_tool_pending_decision",
    }
)

_MAX_CONTENT_CHARS = 16_384
_MAX_SESSION_KEY_CHARS = 512
_MAX_IDENTIFIER_CHARS = 256
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:@/+~-]+$")
_MESSAGE_MARKER_VERSION = 1
_EXPECTED_ROUTE_FIELDS: Final[frozenset[str]] = frozenset(
    {"profile_name", "platform", "user_id", "chat_id", "topic_id"}
)


@dataclass(frozen=True)
class GatewayExpectedRoute:
    """Immutable identity tuple supplied by an authorized plugin caller."""

    profile_name: str
    platform: str
    user_id: str
    chat_id: str
    topic_id: str


@dataclass(frozen=True)
class DesktopExpectedDestination:
    profile_name: str
    session_id: str


@dataclass(frozen=True)
class GatewaySystemEvent:
    """Validated event admitted by the host-owned plugin API."""

    session_key: str
    expected_session_id: str
    event_id: str
    event_kind: str
    plugin_id: str
    receipt_id: str
    expected_route: Optional[GatewayExpectedRoute]
    eligibility_check: Callable[[], bool] = field(repr=False, compare=False)
    desktop_destination: Optional[DesktopExpectedDestination] = None
    completion_check: Optional[Callable[[], bool]] = field(default=None, repr=False, compare=False)


def _bounded_identifier(value: Any, *, name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > max_chars or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


def _snapshot_expected_route(value: Any) -> GatewayExpectedRoute:
    if not isinstance(value, Mapping):
        raise ValueError("expected_route must contain the complete route tuple")
    try:
        keys = set(value)
    except Exception as exc:
        raise ValueError("expected_route keys are invalid") from exc
    if keys != _EXPECTED_ROUTE_FIELDS:
        raise ValueError("expected_route must contain the complete route tuple")
    normalized: dict[str, str] = {}
    for name in _EXPECTED_ROUTE_FIELDS:
        try:
            raw = value[name]
        except Exception as exc:
            raise ValueError(f"expected_route.{name} is unavailable") from exc
        if not isinstance(raw, str):
            raise ValueError(f"expected_route.{name} must be a string")
        raw = raw.strip()
        if name != "topic_id" and not raw:
            raise ValueError(f"expected_route.{name} is empty")
        if len(raw) > _MAX_IDENTIFIER_CHARS:
            raise ValueError(f"expected_route.{name} exceeds the host limit")
        normalized[name] = raw
    return GatewayExpectedRoute(**normalized)


def create_gateway_system_event(
    *,
    content: Any,
    session_key: Any,
    expected_session_id: Any,
    event_id: Any,
    event_kind: Any,
    plugin_id: Any,
    expected_route: Any,
    eligibility_check: Any,
) -> tuple[str, GatewaySystemEvent]:
    """Validate an external request and return its content plus typed marker."""

    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if not content.strip() or len(content) > _MAX_CONTENT_CHARS:
        raise ValueError("content is empty or exceeds the host limit")

    session_key = _bounded_identifier(
        session_key, name="session_key", max_chars=_MAX_SESSION_KEY_CHARS
    )
    expected_session_id = _bounded_identifier(
        expected_session_id,
        name="expected_session_id",
        max_chars=_MAX_IDENTIFIER_CHARS,
    )
    event_id = _bounded_identifier(
        event_id, name="event_id", max_chars=_MAX_IDENTIFIER_CHARS
    )
    plugin_id = _bounded_identifier(
        plugin_id, name="plugin_id", max_chars=_MAX_IDENTIFIER_CHARS
    )
    if not isinstance(event_kind, str) or event_kind not in GATEWAY_SYSTEM_EVENT_KINDS:
        raise ValueError("event_kind is not supported")
    route = _snapshot_expected_route(expected_route)
    if not callable(eligibility_check):
        raise ValueError("eligibility_check must be callable")

    return content, GatewaySystemEvent(
        session_key=session_key,
        expected_session_id=expected_session_id,
        event_id=event_id,
        event_kind=event_kind,
        plugin_id=plugin_id,
        receipt_id=uuid.uuid4().hex,
        expected_route=route,
        eligibility_check=eligibility_check,
    )


def create_desktop_system_event(*, content: Any, destination: Any, event_id: Any,
                                event_kind: Any, plugin_id: Any, eligibility_check: Any, completion_check: Any = None):
    """Create the same closed developer event for an exact native physical session."""
    if not isinstance(destination, Mapping) or set(destination) != {"profile_name", "session_id"}:
        raise ValueError("desktop destination requires profile_name and session_id")
    profile = _bounded_identifier(destination["profile_name"], name="profile_name", max_chars=256)
    session = _bounded_identifier(destination["session_id"], name="session_id", max_chars=256)
    if not isinstance(content, str) or not content.strip() or len(content) > _MAX_CONTENT_CHARS:
        raise ValueError("content is invalid")
    if event_kind not in GATEWAY_SYSTEM_EVENT_KINDS or not callable(eligibility_check):
        raise ValueError("event authority is invalid")
    return content, GatewaySystemEvent(
        session_key=session, expected_session_id=session,
        event_id=_bounded_identifier(event_id, name="event_id", max_chars=256),
        event_kind=event_kind, plugin_id=_bounded_identifier(plugin_id, name="plugin_id", max_chars=256),
        receipt_id=uuid.uuid4().hex, expected_route=None, eligibility_check=eligibility_check,
        desktop_destination=DesktopExpectedDestination(profile, session), completion_check=completion_check)


def gateway_system_event_is_eligible(event: Any) -> bool:
    """Run the process-local admission predicate and fail closed.

    The callback is a deliberately synchronous trust boundary.  Awaitables are
    rejected so gateway admission never performs plugin I/O or leaves work
    scheduled on an unrelated event loop.
    """

    if not isinstance(event, GatewaySystemEvent):
        return False
    try:
        result = event.eligibility_check()
    except Exception:
        return False
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        return False
    return result is True


def gateway_system_event_message(
    event: GatewaySystemEvent,
    content: Any,
) -> dict[str, Any]:
    """Build the append-only developer-role message persisted for this event."""

    if not isinstance(event, GatewaySystemEvent):
        raise ValueError("gateway system event marker is invalid")
    if not isinstance(content, str) or not content or len(content) > _MAX_CONTENT_CHARS:
        raise ValueError("gateway system event content is invalid")
    return {
        "role": "developer",
        "content": content,
        "display_kind": "internal_notification",
        "display_metadata": {
            "schema_version": _MESSAGE_MARKER_VERSION,
            "event_kind": event.event_kind,
            "event_id": event.event_id,
            "plugin_id": event.plugin_id,
        },
    }


def new_gateway_event_receipt() -> concurrent.futures.Future:
    """Return the cross-thread Future exposed to a plugin caller."""

    return concurrent.futures.Future()


def gateway_event_result(
    status: str,
    *,
    event: Optional[GatewaySystemEvent] = None,
    receipt_id: Optional[str] = None,
) -> Mapping[str, Any]:
    """Create the bounded mapping returned by a terminal receipt."""

    result: dict[str, Any] = {"status": status}
    resolved_receipt_id = event.receipt_id if event is not None else receipt_id
    if resolved_receipt_id:
        result["receipt_id"] = str(resolved_receipt_id)[:_MAX_IDENTIFIER_CHARS]
    return result


def resolve_gateway_event_receipt(
    receipt: Any,
    status: str,
    *,
    event: Optional[GatewaySystemEvent] = None,
) -> bool:
    """Resolve a receipt exactly once; return whether this call won."""

    if not isinstance(receipt, concurrent.futures.Future) or receipt.done():
        return False
    try:
        receipt.set_result(gateway_event_result(status, event=event))
    except concurrent.futures.InvalidStateError:
        return False
    return True


def resolve_message_event_receipt(message_event: Any, status: str) -> bool:
    """Resolve the typed receipt carried by a ``MessageEvent`` if present."""

    event = getattr(message_event, "gateway_system_event", None)
    receipt = getattr(message_event, "gateway_event_receipt", None)
    if not isinstance(event, GatewaySystemEvent):
        return False
    return resolve_gateway_event_receipt(receipt, status, event=event)


def admit_message_event_receipt(message_event: Any) -> bool:
    """Atomically mark a typed receipt running at actual model admission.

    Queued receipts deliberately remain pending.  A caller cancellation that
    wins before this boundary prevents the event from executing.
    """

    event = getattr(message_event, "gateway_system_event", None)
    receipt = getattr(message_event, "gateway_event_receipt", None)
    if not isinstance(event, GatewaySystemEvent):
        return False
    if not isinstance(receipt, concurrent.futures.Future):
        return False
    if receipt.running():
        return True
    if receipt.done():
        return False
    try:
        return receipt.set_running_or_notify_cancel()
    except RuntimeError:
        return receipt.running()
