"""Per-process unlock state for external password managers.

An unlock is a session token minted by the manager's CLI from the master
password (``op signin --raw`` / ``bw unlock --raw``). The token lives in
process memory only, keyed by profile, backend and owning session, and expires
after an idle TTL or an explicit lock. The master password is consumed by the CLI call and
dropped; nothing is written to disk or env.

The surface owns the prompt: ``set_unlock_prompt_callback`` is installed by
the CLI panel / TUI gateway bridge for the current thread, exactly like the
sudo-password callback. Headless contexts (cron, webhook, api_server,
single-query) install none and the vault stays locked — the same posture
approvals take where nobody can answer.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Dict, Optional

_IDLE_TTL_S = 30 * 60

_lock = threading.Lock()
_CacheKey = tuple[str, str, str, str]  # profile home, backend, owner namespace, session id
_sessions: Dict[_CacheKey, tuple[str, float]] = {}
_callback_tls = threading.local()

UnlockPrompt = Callable[[str, str], str]  # (backend_name, display_name) -> master password ("" = cancelled)
# (origin, site label) -> {"identifier": str, "password": str} or None when the user declines. The
# surface owns the masked fields; the tool stores the answer in the local vault and fills at once.
SaveLoginPrompt = Callable[[str, str], Optional[Dict[str, str]]]


def set_unlock_prompt_callback(cb: Optional[UnlockPrompt]) -> None:
    """Register the current surface's masked master-password prompt (per-thread slot)."""
    _callback_tls.prompt = cb


def get_unlock_prompt_callback() -> Optional[UnlockPrompt]:
    return getattr(_callback_tls, "prompt", None)


# (site, hint) -> the one-time code the user reads off their phone/email/app, "" when declined.
CodePrompt = Callable[[str, str], str]


def set_code_prompt_callback(cb: Optional[CodePrompt]) -> None:
    """Register the surface's "enter the code {site} sent you" prompt, per thread."""
    _callback_tls.code = cb


def get_code_prompt_callback() -> Optional[CodePrompt]:
    return getattr(_callback_tls, "code", None)


def set_save_login_prompt_callback(cb: Optional[SaveLoginPrompt]) -> None:
    """Register the surface's "save this login" prompt (identifier + masked password), per thread."""
    _callback_tls.save_login = cb


def get_save_login_prompt_callback() -> Optional[SaveLoginPrompt]:
    return getattr(_callback_tls, "save_login", None)


def _session_identity() -> Optional[tuple[str, str]]:
    from agent.delegation_context import is_delegated_child_context

    if is_delegated_child_context():
        # Delegates have no human-owned unlock surface or matching vault teardown hook. Do not
        # borrow the parent's token or retain a new interactive unlock after the child finishes.
        # Local vault access and separately configured service-account auth do not use this cache.
        return None
    identity = _current_session.get()
    if identity is not None:
        return identity
    # CLI turns and their copied tool-worker contexts bind this value. The public getter's
    # legacy os.environ fallback is NOT authority to borrow another turn's token.
    from tools.approval_context import _approval_session_key
    sid = _approval_session_key.get()
    return ("session", sid) if sid and sid != "default" else None


def _key(backend: str) -> Optional[_CacheKey]:
    from hermes_constants import get_hermes_home
    identity = _session_identity()
    return (str(get_hermes_home()), backend, *identity) if identity is not None else None


# Lock generation per key: ``lock()`` bumps it, and an unlock that started before the bump must
# not commit its token afterwards (a slow `bw unlock` child would otherwise silently undo an
# acknowledged Lock).
_generation: Dict[_CacheKey, int] = {}
# A queued worker must not recreate an ended owner, even if no unlock had started at teardown.
_released_sessions: set[tuple[str, str]] = set()
_current_session: ContextVar[Optional[tuple[str, str]]] = ContextVar("vault_current_session", default=None)


@dataclass(frozen=True)
class _UnlockAttempt:
    key: _CacheKey
    generation: int


def set_current_session_id(session_id: Optional[str]) -> None:
    """Bind the owning conversation; copied tool-worker contexts retain this identity."""
    _current_session.set(("session", session_id) if session_id and session_id != "default" else None)


@contextmanager
def settings_session_scope(owner_id: Optional[str]):
    """RPC adapter only: Settings has a server-derived owner, never a conversation's authority."""
    # An empty explicit identity suppresses a possibly inherited conversation/CLI context.
    token = _current_session.set(("settings", owner_id or ""))
    try:
        yield
    finally:
        _current_session.reset(token)


def _live(backend: str, *, touch: bool) -> Optional[str]:
    key = _key(backend)
    with _lock:
        if key is None or not key[3] or key[2:] in _released_sessions:
            return None
        entry = _sessions.get(key)
        if entry is None:
            return None
        token, last = entry
        if time.monotonic() - last > _IDLE_TTL_S:
            del _sessions[key]
            return None
        if touch:
            _sessions[key] = (token, time.monotonic())
        return token


def get_session_token(backend: str) -> Optional[str]:
    """Token for a real manager call; refreshes the idle timer."""
    return _live(backend, touch=True)


def begin_unlock(backend: str) -> _UnlockAttempt:
    """Bind an in-flight unlock to its exact owner and lock generation before spawning the CLI."""
    key = _key(backend)
    with _lock:
        if key is None or not key[3] or key[2:] in _released_sessions:
            raise RuntimeError("Unlock requires an active owning session; retry from the current surface")
        return _UnlockAttempt(key, _generation.setdefault(key, 0))


def store_session_token(backend: str, token: str, generation: Optional[_UnlockAttempt] = None) -> bool:
    """Commit an unlock. Returns False (and drops the token) when a Lock happened since ``begin_unlock``."""
    key = _key(backend)
    with _lock:
        if key is None or not key[3] or key[2:] in _released_sessions:
            return False
        if generation is not None and generation != _UnlockAttempt(key, _generation.get(key, 0)):
            return False
        _sessions[key] = (token, time.monotonic())
        return True


def lock(backend: Optional[str] = None) -> None:
    """Explicit Lock revokes ALL owners in the current profile, including pending unlocks."""
    from hermes_constants import get_hermes_home
    home = str(get_hermes_home())
    with _lock:
        # Bump the generation for every key the lock names (not only the ones holding a token):
        # an unlock that is still running for this backend must see the lock when it returns.
        keys = {k for k in list(_sessions) + list(_generation) if k[0] == home and (backend is None or k[1] == backend)}
        for key in keys:
            _forget(key)


def release_session(session_id: str, *, namespace: str = "session") -> None:
    """An owner ended: revoke its tokens and pending/queued unlocks, not its siblings'."""
    identity = (namespace, session_id)
    with _lock:
        _released_sessions.add(identity)
        for key in {k for k in list(_sessions) + list(_generation) if k[2:] == identity}:
            _forget(key)


def _forget(key: _CacheKey) -> None:
    _sessions.pop(key, None)
    _generation[key] = _generation.get(key, 0) + 1


def lock_all_profiles() -> None:
    """Process shutdown: drop every token."""
    with _lock:
        for key in set(_sessions) | set(_generation):
            _forget(key)


def is_unlocked(backend: str) -> bool:
    """Status probe: does NOT extend the idle TTL (only real manager calls do)."""
    return _live(backend, touch=False) is not None


def can_prompt_here() -> bool:
    """False in contexts where no human can answer (cron, webhook, api_server, -q)."""
    from tools.approval_context import (
        _is_cron_approval_context,
        _is_single_query_approval_context,
        _is_unattended_platform_approval_context,
    )
    if _is_cron_approval_context() or _is_unattended_platform_approval_context() or _is_single_query_approval_context():
        return False
    return get_unlock_prompt_callback() is not None
