"""Typed A2A ingress facts, not an operator approval or a tool permission.

Only trusted Python ingress code constructs these objects. Never derive them from
message text, source names, environment variables or model tool arguments.
"""
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import hashlib
from threading import RLock
from weakref import WeakValueDictionary


class A2AAuthMethod(str, Enum):
    PEER_BEARER = 'peer_bearer'
    SHARED_BEARER = 'shared_bearer'
    LOCAL_SOCKET = 'local_socket'


@dataclass(frozen=True)
class A2AAuthentication:
    peer: str
    method: A2AAuthMethod
    request_id: str = field(default_factory=lambda: uuid4().hex)


# Native admission serializes events before the turn runs. Keep only an opaque
# reference in that owner-only envelope, not a reconstructible authentication
# claim. The live ingress/event/turn owns the strong reference. Restart, expired
# requests and cross-process forwarding therefore cannot manufacture provenance.
_live: WeakValueDictionary[tuple, A2AAuthentication] = WeakValueDictionary()
_lock = RLock()


def _event_key(request_id, event):
    source = event.source
    return (request_id, source.platform.value, source.profile or '', source.chat_id,
            source.user_id, event.message_id, hashlib.sha256(event.text.encode()).digest())


def capture_auth_reference(event) -> str | None:
    auth = event.a2a_auth
    if not isinstance(auth, A2AAuthentication):
        return None
    if event.source.platform.value != 'a2a' or event.source.user_id != auth.peer:
        return None
    with _lock:
        _live[_event_key(auth.request_id, event)] = auth
    return auth.request_id


def restore_auth_reference(reference, event) -> A2AAuthentication | None:
    if not isinstance(reference, str):
        return None
    with _lock:
        return _live.get(_event_key(reference, event))
