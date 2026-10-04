"""Process-local broker for model-blind gateway secret input.

Adapters render a private input surface and resolve an opaque request id. The
submitted value stays on the waiting entry; it is never a MessageEvent, session
row, tool argument, or adapter result.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

SECURE_INPUT_TIMEOUT_SECONDS = 180.0
_MAX_VALUE_CHARS = 128


@dataclass
class SecureInputRequest:
    request_id: str
    session_key: str
    session_id: str
    run_generation: int
    task_id: str
    expected_user_id: str
    chat_id: str
    scope_id: str
    site: str
    profile_home: str
    expires_at: float
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _value: Optional[str] = field(default=None, repr=False)
    _claimed: bool = field(default=False, repr=False)
    _resolved: bool = field(default=False, repr=False)
    _cancelled: bool = field(default=False, repr=False)


_lock = threading.Lock()
_pending: dict[str, SecureInputRequest] = {}
# Run generations are monotonic and never reset for a routing session. A high-water fence is both
# race-free and durable: a stopped worker cannot resurrect its prompt after an arbitrary delay.
_closed_through: dict[str, int] = {}


def _expired(request: SecureInputRequest, now: Optional[float] = None) -> bool:
    return (time.monotonic() if now is None else now) >= request.expires_at


def _remove_locked(request: SecureInputRequest) -> None:
    if _pending.get(request.request_id) is request:
        _pending.pop(request.request_id, None)


def _cancel_locked(request: SecureInputRequest) -> None:
    _remove_locked(request)
    request._cancelled = True
    request._value = None
    request._event.set()


def _purge_expired_locked(now: Optional[float] = None) -> None:
    current = time.monotonic() if now is None else now
    for request in list(_pending.values()):
        if _expired(request, current):
            _cancel_locked(request)


def register(
    *,
    session_key: str,
    session_id: str,
    run_generation: int,
    task_id: str,
    expected_user_id: str,
    chat_id: str,
    scope_id: str,
    site: str,
    profile_home: str,
    timeout: Optional[float] = None,
) -> SecureInputRequest:
    """Register one bounded request bound to the exact gateway run and transport scope."""
    lifetime = SECURE_INPUT_TIMEOUT_SECONDS if timeout is None else max(0.0, float(timeout))
    request = SecureInputRequest(
        request_id="secure_" + secrets.token_urlsafe(18),
        session_key=str(session_key or ""),
        session_id=str(session_id or ""),
        run_generation=int(run_generation or 0),
        task_id=str(task_id or ""),
        expected_user_id=str(expected_user_id or ""),
        chat_id=str(chat_id or ""),
        scope_id=str(scope_id or ""),
        site=str(site or "")[:200],
        profile_home=str(profile_home or ""),
        expires_at=time.monotonic() + lifetime,
    )
    required = (
        request.session_key,
        request.session_id,
        request.run_generation,
        request.task_id,
        request.expected_user_id,
        request.chat_id,
        request.scope_id,
        request.profile_home,
    )
    if not all(required):
        raise ValueError(
            "secure input requires session, run, task, user, chat, transport scope, and profile"
        )
    with _lock:
        _purge_expired_locked()
        if request.run_generation <= _closed_through.get(request.session_key, 0):
            raise RuntimeError("secure input run is no longer active")
        _pending[request.request_id] = request
    return request


def claim(
    request_id: str,
    *,
    user_id: str,
    chat_id: str,
    scope_id: str,
) -> Optional[SecureInputRequest]:
    """Exclusively claim a live request for its original user and transport."""
    with _lock:
        _purge_expired_locked()
        request = _pending.get(str(request_id or ""))
        if request is None or request._claimed or request._resolved or request._cancelled:
            return None
        if request.expected_user_id != str(user_id or ""):
            return None
        if request.chat_id != str(chat_id or ""):
            return None
        if request.scope_id != str(scope_id or ""):
            return None
        request._claimed = True
        return request


def resolve(
    request_id: str,
    value: str,
    *,
    user_id: str,
    scope_id: str,
) -> Optional[SecureInputRequest]:
    """Resolve a claimed request once, retaining it until its owner atomically consumes it."""
    submitted = str(value or "").strip()
    if not submitted or len(submitted) > _MAX_VALUE_CHARS:
        return None
    with _lock:
        _purge_expired_locked()
        request = _pending.get(str(request_id or ""))
        if request is None or not request._claimed or request._resolved or request._cancelled:
            return None
        if request.expected_user_id != str(user_id or ""):
            return None
        if request.scope_id != str(scope_id or ""):
            return None
        request._value = submitted
        request._resolved = True
        request._event.set()
        return request


def cancel_claimed(
    request_id: str,
    *,
    user_id: str,
    scope_id: str,
) -> Optional[SecureInputRequest]:
    """Cancel only from the identity that exclusively claimed the private surface."""
    with _lock:
        _purge_expired_locked()
        request = _pending.get(str(request_id or ""))
        if request is None or not request._claimed or request._resolved or request._cancelled:
            return None
        if request.expected_user_id != str(user_id or ""):
            return None
        if request.scope_id != str(scope_id or ""):
            return None
        _cancel_locked(request)
        return request


def wait(request: SecureInputRequest) -> str:
    """Wait through the deadline and atomically consume a still-live resolved value."""
    request._event.wait(max(0.0, request.expires_at - time.monotonic()))
    with _lock:
        live = _pending.get(request.request_id) is request
        if not live or request._cancelled or _expired(request) or not request._resolved:
            if live:
                _cancel_locked(request)
            return ""
        _remove_locked(request)
        value = request._value or ""
        request._value = None
        return value


def cancel(request: SecureInputRequest) -> None:
    with _lock:
        if _pending.get(request.request_id) is request:
            _cancel_locked(request)
        else:
            request._value = None


def clear_run(session_key: str, run_generation: int) -> None:
    """Fence and cancel only the displaced turn, never a newer replacement sharing its routing key."""
    generation = int(run_generation)
    with _lock:
        _purge_expired_locked()
        _closed_through[session_key] = max(generation, _closed_through.get(session_key, 0))
        requests = [
            request
            for request in _pending.values()
            if request.session_key == session_key
            and request.run_generation == generation
        ]
        for request in requests:
            _cancel_locked(request)


def clear_session(session_key: str) -> None:
    """Cancel every request at an explicit conversation boundary."""
    with _lock:
        requests = [request for request in _pending.values() if request.session_key == session_key]
        for request in requests:
            _cancel_locked(request)


def is_pending(request: SecureInputRequest) -> bool:
    with _lock:
        _purge_expired_locked()
        return _pending.get(request.request_id) is request


def pending_count() -> int:
    with _lock:
        _purge_expired_locked()
        return len(_pending)
