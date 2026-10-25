"""Per-process unlock state for external password managers.

An unlock is a session token minted by the manager's CLI from the master
password (``op signin --raw`` / ``bw unlock --raw``). The token lives in
process memory only, keyed by (profile home, backend), and is dropped at an
idle TTL (by a timer, not at the next call) or an explicit lock. It is held
inside a closure, so no module attribute exposes it to a plain import or a
dump of module state, and it reaches only the manager CLI call that uses it
(as that child's ``BW_SESSION`` / ``OP_SESSION*``), never ``os.environ``.
That stops casual and accidental access, not code running in this process,
which can always read its memory: the boundary for third-party plugins is
``plugins.isolation: host``. The master password itself is consumed by the
CLI call and dropped; nothing is written to disk or env.

The surface owns the prompt: ``set_unlock_prompt_callback`` is installed by
the CLI panel / TUI gateway bridge for the current thread, exactly like the
sudo-password callback. Headless contexts (cron, webhook, api_server,
single-query) install none and the vault stays locked — the same posture
approvals take where nobody can answer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, NamedTuple, Optional

_IDLE_TTL_S = 30 * 60

_callback_tls = threading.local()

UnlockPrompt = Callable[[str, str], str]  # (backend_name, display_name) -> master password ("" = cancelled)
# (origin, site label) -> {"identifier": str, "password": str} or None when the user declines. The
# surface owns the masked fields; the tool stores the answer in the local vault and fills at once.
SaveLoginPrompt = Callable[[str, str], Optional[dict[str, str]]]


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


_current_session_tls = threading.local()


def set_current_session_id(session_id: Optional[str]) -> None:
    """Gateway surfaces bind the session running on this thread so an unlock records its owner."""
    _current_session_tls.sid = session_id


def get_current_session_id() -> Optional[str]:
    """The session bound on this thread (``tools.thread_context`` carries it to tool workers)."""
    return getattr(_current_session_tls, "sid", None)


_Key = tuple[str, str]  # (profile home, backend)


@dataclass(slots=True)
class _Token:
    value: str = field(repr=False)
    last_used: float
    owner: Optional[str]  # the gateway session that unlocked; released when THAT session ends


class _TokenOps(NamedTuple):
    live: Callable[[_Key, bool], Optional[str]]
    begin: Callable[[_Key], int]
    commit: Callable[[_Key, str, Optional[int], Optional[str]], bool]
    lock_profile: Callable[[str, Optional[str]], None]
    release: Callable[[str], None]
    lock_everything: Callable[[], None]


def _token_store() -> _TokenOps:
    """The token state, reachable only through the closures returned here.

    The lock generation per key is bumped by every lock, and an unlock that started before the bump
    must not commit its token afterwards (a slow `bw unlock` child would otherwise silently undo an
    acknowledged Lock). Sibling sessions in a profile never release each other's tokens.
    """
    guard = threading.Lock()
    tokens: dict[_Key, _Token] = {}
    generations: dict[_Key, int] = {}
    timer: Optional[threading.Timer] = None

    def forget(key: _Key) -> None:
        tokens.pop(key, None)
        generations[key] = generations.get(key, 0) + 1

    def arm(now: float) -> None:
        # One daemon timer for the earliest expiry; a touch only moves expiry later, so the timer may
        # fire early, find nothing expired and re-arm.
        nonlocal timer
        if timer is None and tokens:
            delay = min(t.last_used for t in tokens.values()) + _IDLE_TTL_S - now
            timer = threading.Timer(max(delay, 0.0) + 0.01, expire)
            timer.daemon = True
            timer.start()

    def settle() -> None:
        nonlocal timer
        if not tokens and timer is not None:
            timer.cancel()
            timer = None

    def expire() -> None:
        nonlocal timer
        with guard:
            if timer is threading.current_thread():  # not a newer timer armed while this one waited
                timer = None
            now = time.monotonic()
            for key in [k for k, t in tokens.items() if now - t.last_used > _IDLE_TTL_S]:
                del tokens[key]
            arm(now)

    def live(key: _Key, touch: bool) -> Optional[str]:
        with guard:
            token = tokens.get(key)
            if token is None:
                return None
            now = time.monotonic()
            if now - token.last_used > _IDLE_TTL_S:
                del tokens[key]
                settle()
                return None
            if touch:
                token.last_used = now
            return token.value

    def begin(key: _Key) -> int:
        with guard:
            return generations.get(key, 0)

    def commit(key: _Key, value: str, generation: Optional[int], owner: Optional[str]) -> bool:
        with guard:
            if generation is not None and generation != generations.get(key, 0):
                return False
            now = time.monotonic()
            tokens[key] = _Token(value, now, owner)
            arm(now)
            return True

    def lock_profile(home: str, backend: Optional[str]) -> None:
        with guard:
            # Bump the generation for every key the lock names (not only the ones holding a token):
            # an unlock that is still running for this backend must see the lock when it returns.
            keys = {k for k in [*tokens, *generations] if k[0] == home and (backend is None or k[1] == backend)}
            if backend is not None:
                keys.add((home, backend))
            for key in keys:
                forget(key)
            settle()

    def release(session_id: str) -> None:
        with guard:
            for key in [k for k, t in tokens.items() if t.owner == session_id]:
                forget(key)
            settle()

    def lock_everything() -> None:
        with guard:
            for key in list(tokens):
                forget(key)
            settle()

    return _TokenOps(live, begin, commit, lock_profile, release, lock_everything)


_tokens = _token_store()


def get_session_token(backend: str) -> Optional[str]:
    """Token for a real manager call; refreshes the idle timer."""
    return _tokens.live(_key(backend), True)


def begin_unlock(backend: str) -> int:
    """Snapshot the lock generation before spawning the manager CLI; pass it to ``store_session_token``."""
    return _tokens.begin(_key(backend))


def store_session_token(backend: str, token: str, generation: Optional[int] = None) -> bool:
    """Commit an unlock. Returns False (and drops the token) when a Lock happened since ``begin_unlock``."""
    return _tokens.commit(_key(backend), token, generation, get_current_session_id())


def lock(backend: Optional[str] = None) -> None:
    """Forget the current profile's session for one backend (or all of them when None)."""
    _tokens.lock_profile(_key("")[0], backend)


def release_session(session_id: str) -> None:
    """A gateway session ended: drop only the tokens that session unlocked."""
    _tokens.release(session_id)


def lock_all_profiles() -> None:
    """Process shutdown: drop every token."""
    _tokens.lock_everything()


def is_unlocked(backend: str) -> bool:
    """Status probe: does NOT extend the idle TTL (only real manager calls do)."""
    return _tokens.live(_key(backend), False) is not None


def can_prompt_here() -> bool:
    """False in contexts where no human can answer (cron, webhook, api_server, -q)."""
    from tools.approval_context import _no_user_can_answer
    if _no_user_can_answer():
        return False
    return get_unlock_prompt_callback() is not None
