"""Verified gateway turn origin for plugin-owned tools.

This is process-local authority, not model context. An inbound projector may retain
only a bounded JSON claim across native admission; the original platform SDK object
is never stored in the admission or made available to a tool.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
import json

from hermes_state_runtime import RuntimeStoreError

TRUSTED_TURN_ORIGIN_API_VERSION = 1
MAX_ORIGIN_CLAIM_BYTES = 16 * 1024


@dataclass(frozen=True)
class TrustedTurnOrigin:
    """Identity of one admitted, human-authored gateway turn.

    ``claims`` contains only independently validated plugin projections. A
    missing claim is never permission to infer identity from the conversation.
    """

    kind: str
    platform: str
    profile: str | None
    workspace_id: str | None
    channel_id: str
    user_id: str | None
    message_ts: str | None
    thread_ts: str | None
    admission_id: str | None
    admitted: bool
    internal: bool
    claims: Mapping[str, Any]


_origin: ContextVar[TrustedTurnOrigin | None] = ContextVar("trusted_turn_origin", default=None)


def current_origin() -> TrustedTurnOrigin | None:
    """Return this execution's trusted origin, or ``None`` outside a human turn."""
    return _origin.get()


@contextmanager
def bind_turn_origin(origin: TrustedTurnOrigin | None):
    """Bind one execution, resetting even on cancellation or recursive turns."""
    token = _origin.set(origin)
    try:
        yield
    finally:
        _origin.reset(token)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _json_claim(value: Any) -> Any:
    """Detach caller-owned data and enforce a hard serialized bound."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeStoreError("invalid_params", field="origin_claim") from exc
    if len(encoded) > MAX_ORIGIN_CLAIM_BYTES:
        raise RuntimeStoreError("invalid_params", field="origin_claim_size")
    return json.loads(encoded)


def _projector(runner, source):
    """Resolve the owning profile's single projector for this platform."""
    from hermes_cli.plugins import get_plugin_manager
    platform = getattr(getattr(source, "platform", None), "value", None)
    if not isinstance(platform, str):
        return None
    with runner._profile_scope_for_source(source):
        return get_plugin_manager()._inbound_origin_projectors.get(platform)


def _binding(event):
    source = event.source
    return [source.platform.value, source.profile, source.scope_id, source.chat_id,
            source.user_id, source.thread_id, event.message_id]


def capture_origin_claim(runner, event):
    """Snapshot a plugin's narrow event projection before the first admission await."""
    if event.internal or event.source is None:
        return None
    projector = _projector(runner, event.source)
    if projector is None:
        return None
    plugin_id, capture, validate = projector
    try:
        claim = capture(event)
    except Exception as exc:
        raise RuntimeStoreError("permission_denied", field="origin_projector") from exc
    if claim is None:
        return None
    detached = _json_claim(claim)
    try:
        valid = isinstance(detached, dict) and validate(deepcopy(detached), event.source)
    except Exception as exc:
        raise RuntimeStoreError("permission_denied", field="origin_projector") from exc
    if not valid:
        raise RuntimeStoreError("invalid_params", field="origin_claim")
    return _json_claim({"version": TRUSTED_TURN_ORIGIN_API_VERSION,
                        "plugin": plugin_id, "platform": event.source.platform.value,
                        "binding": _binding(event), "claim": detached})


def restore_origin_claim(runner, event, projected):
    """Reject an invalid or unavailable projector instead of trusting a stored claim."""
    projected = _json_claim(projected)
    if not isinstance(projected, dict) or set(projected) != {"version", "plugin", "platform", "binding", "claim"}:
        raise RuntimeStoreError("invalid_params", field="origin_claim")
    if (type(projected["version"]) is not int
            or projected["version"] != TRUSTED_TURN_ORIGIN_API_VERSION
            or projected["platform"] != event.source.platform.value
            or projected["binding"] != _binding(event)
            or event.internal):
        raise RuntimeStoreError("invalid_params", field="origin_claim")
    projector = _projector(runner, event.source)
    if projector is None or projector[0] != projected["plugin"]:
        raise RuntimeStoreError("permission_denied", field="origin_projector")
    detached = _json_claim(projected["claim"])
    try:
        valid = isinstance(detached, dict) and projector[2](deepcopy(detached), event.source)
    except Exception as exc:
        raise RuntimeStoreError("permission_denied", field="origin_projector") from exc
    if not valid:
        raise RuntimeStoreError("permission_denied", field="origin_claim")
    event._trusted_origin_claim = {"plugin": projector[0], "claim": detached,
                                   "binding": _binding(event)}


def origin_for_event(runner, event, source):
    """Build an immutable origin only for the current admitted event."""
    if event is None or event.internal or event.source is None:
        return None
    from gateway.session_ingress import executing_admission
    if not (event._gateway_accepted is True or executing_admission.get()):
        return None
    # A channel pin may adjust the effective source, but it cannot change the
    # inbound principal, workspace, channel or thread.
    inbound = event.source
    if (inbound.platform != source.platform or inbound.profile != source.profile
            or inbound.chat_id != source.chat_id or inbound.user_id != source.user_id
            or inbound.scope_id != source.scope_id or inbound.thread_id != source.thread_id):
        return None
    stored = getattr(event, "_trusted_origin_claim", None)
    if stored is None and not getattr(event, "_native_origin_restored", False):
        projected = capture_origin_claim(runner, event)
        stored = None if projected is None else {"plugin": projected["plugin"],
                                                  "claim": projected["claim"],
                                                  "binding": projected["binding"]}
    if stored is not None and (not isinstance(stored, dict)
                               or stored.get("binding") != _binding(event)):
        return None
    claims = {} if stored is None else {stored["plugin"]: stored["claim"]}
    return TrustedTurnOrigin(
        kind="gateway", platform=inbound.platform.value, profile=inbound.profile,
        workspace_id=inbound.scope_id, channel_id=inbound.chat_id, user_id=inbound.user_id,
        message_ts=str(event.message_id) if event.message_id is not None else None,
        thread_ts=str(inbound.thread_id) if inbound.thread_id is not None else None,
        admission_id=getattr(event, "_native_admission_id", None),
        admitted=True, internal=False, claims=_freeze(claims),
    )
