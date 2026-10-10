"""Stable transcript ownership for accepted gateway input."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any

from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


_Namespace = tuple[str, str | None, str | None, str, str | None]


def _namespace(source: SessionSource) -> _Namespace:
    return (str(source.platform.value), source.profile, source.scope_id, source.chat_id, source.thread_id)


@dataclass(frozen=True)
class _InputOwner:
    namespace: _Namespace
    identifier: str
    pending_uid: str | None = None

    @property
    def owner(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([*self.namespace, self.identifier])))

    @classmethod
    def restore(cls, source: SessionSource, uid: str, state: Any) -> _InputOwner:
        if not isinstance(state, dict) or set(state) != {"namespace", "identifier", "pending_uid", "owner"}:
            raise ValueError("pending input owner requires its complete scope")
        namespace = _namespace(source)
        identifier = state["identifier"]
        if (state["namespace"] != list(namespace) or state["pending_uid"] != uid
                or not isinstance(identifier, str) or not identifier
                or identifier.startswith("pending:") and identifier != f"pending:{uid}"):
            raise ValueError("pending input owner does not match its scope")
        saved = cls(namespace, identifier, uid)
        if state["owner"] != saved.owner:
            raise ValueError("pending input owner receipt does not match its scope")
        return saved


def _event_owner(event: MessageEvent, source: SessionSource) -> _InputOwner:
    namespace = _namespace(source)
    uid = getattr(event, "_pending_snapshot_uid", None)
    saved = getattr(event, "_gateway_input_owner", None)
    if (isinstance(saved, _InputOwner) and saved.namespace == namespace
            and (saved.pending_uid is None or saved.pending_uid == uid)):
        return saved
    if not uid and not event.message_id:
        uid = uuid.uuid4().hex
        setattr(event, "_pending_snapshot_uid", uid)
    identifier = f"pending:{uid}" if uid else str(event.message_id)
    saved = _InputOwner(namespace, identifier, uid)
    setattr(event, "_gateway_input_owner", saved)
    return saved


def gateway_input_owner(event: MessageEvent | None, source: SessionSource) -> str:
    if event is not None:
        return _event_owner(event, source).owner
    return str(uuid.uuid4())


def capture_gateway_input_owner(event: MessageEvent) -> dict[str, Any]:
    saved = _event_owner(event, event.source)
    saved = replace(saved, pending_uid=getattr(event, "_pending_snapshot_uid"))
    setattr(event, "_gateway_input_owner", saved)
    return {"namespace": list(saved.namespace), "identifier": saved.identifier,
            "pending_uid": saved.pending_uid, "owner": saved.owner}


def restore_gateway_input_owner(event: MessageEvent, state: Any) -> None:
    saved = _InputOwner.restore(event.source, getattr(event, "_pending_snapshot_uid"), state)
    setattr(event, "_gateway_input_owner", saved)


def recorded_gateway_input_owner(source: SessionSource, uid: str, state: Any) -> str:
    return _InputOwner.restore(source, uid, state).owner
