"""Per-process unlock state for external password managers.

An unlock is a session token minted by the manager's CLI from the master
password (``op signin --raw`` / ``bw unlock --raw``). The token lives in
process memory only, keyed by backend, and expires after an idle TTL or an
explicit lock. The master password itself is consumed by the CLI call and
dropped; nothing is written to disk or env. KeePassXC mints no token, so for
that backend the stored session credential is the database key itself — same
in-process, profile-scoped lifetime, never on disk and never in the env.

The surface owns the prompt: ``set_unlock_prompt_callback`` is installed by
the CLI panel / TUI gateway bridge for the current thread, exactly like the
sudo-password callback. Headless contexts (cron, webhook, api_server,
single-query) install none and the vault stays locked — the same posture
approvals take where nobody can answer.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

_IDLE_TTL_S = 30 * 60

_lock = threading.Lock()
_sessions: Dict[tuple[str, str], tuple[str, float]] = {}   # (profile home, backend) → (token, last_used)
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


def _key(backend: str) -> tuple[str, str]:
    # Tokens are profile-scoped: a Desktop gateway hosts several profiles in one process and
    # profile B must never reuse (or lock) profile A's manager session.
    from hermes_constants import get_hermes_home
    return (str(get_hermes_home()), backend)


# Lock generation per key: ``lock()`` bumps it, and an unlock that started before the bump must
# not commit its token afterwards (a slow `bw unlock` child would otherwise silently undo an
# acknowledged Lock).
_generation: Dict[tuple[str, str], int] = {}
# Which gateway session performed the unlock; the token is released when THAT session ends,
# not when any sibling session in the profile is torn down.
_owner_session: Dict[tuple[str, str], Optional[str]] = {}
_current_session_tls = threading.local()


def set_current_session_id(session_id: Optional[str]) -> None:
    """Gateway surfaces bind the session running on this thread so an unlock records its owner."""
    _current_session_tls.sid = session_id


def _live(backend: str, *, touch: bool) -> Optional[str]:
    key = _key(backend)
    with _lock:
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


def begin_unlock(backend: str) -> int:
    """Snapshot the lock generation before spawning the manager CLI; pass it to ``store_session_token``."""
    with _lock:
        return _generation.get(_key(backend), 0)


def store_session_token(backend: str, token: str, generation: Optional[int] = None) -> bool:
    """Commit an unlock. Returns False (and drops the token) when a Lock happened since ``begin_unlock``."""
    key = _key(backend)
    with _lock:
        if generation is not None and generation != _generation.get(key, 0):
            return False
        _sessions[key] = (token, time.monotonic())
        _owner_session[key] = getattr(_current_session_tls, "sid", None)
        return True


def lock(backend: Optional[str] = None) -> None:
    """Forget the current profile's session for one backend (or all of them when None)."""
    home = _key("")[0]
    with _lock:
        # Bump the generation for every key the lock names (not only the ones holding a token):
        # an unlock that is still running for this backend must see the lock when it returns.
        keys = {k for k in list(_sessions) + list(_generation) if k[0] == home and (backend is None or k[1] == backend)}
        if backend is not None:
            keys.add((home, backend))
        for key in keys:
            _forget(key)


def release_session(session_id: str) -> None:
    """A gateway session ended: drop only the tokens that session unlocked."""
    with _lock:
        for key in [k for k, sid in _owner_session.items() if sid == session_id]:
            _forget(key)


def _forget(key: tuple[str, str]) -> None:
    _sessions.pop(key, None)
    _owner_session.pop(key, None)
    _generation[key] = _generation.get(key, 0) + 1


def lock_all_profiles() -> None:
    """Process shutdown: drop every token."""
    with _lock:
        for key in list(_sessions):
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
