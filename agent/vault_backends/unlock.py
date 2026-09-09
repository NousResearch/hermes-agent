"""Per-process unlock state for external password managers.

An unlock is a session token minted by the manager's CLI from the master
password (``op signin --raw`` / ``bw unlock --raw``). The token lives in
process memory only, keyed by backend, and expires after an idle TTL or an
explicit lock. The master password itself is consumed by the CLI call and
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
from typing import Callable, Dict, Optional

_IDLE_TTL_S = 30 * 60

_lock = threading.Lock()
_sessions: Dict[str, tuple[str, float]] = {}   # backend name → (token, last_used)
_callback_tls = threading.local()

UnlockPrompt = Callable[[str, str], str]  # (backend_name, display_name) -> master password ("" = cancelled)


def set_unlock_prompt_callback(cb: Optional[UnlockPrompt]) -> None:
    """Register the current surface's masked master-password prompt (per-thread slot)."""
    _callback_tls.prompt = cb


def get_unlock_prompt_callback() -> Optional[UnlockPrompt]:
    return getattr(_callback_tls, "prompt", None)


def get_session_token(backend: str) -> Optional[str]:
    with _lock:
        entry = _sessions.get(backend)
        if entry is None:
            return None
        token, last = entry
        if time.monotonic() - last > _IDLE_TTL_S:
            del _sessions[backend]
            return None
        _sessions[backend] = (token, time.monotonic())
        return token


def store_session_token(backend: str, token: str) -> None:
    with _lock:
        _sessions[backend] = (token, time.monotonic())


def lock(backend: Optional[str] = None) -> None:
    """Forget one backend's session (or every one when *backend* is None)."""
    with _lock:
        if backend is None:
            _sessions.clear()
        else:
            _sessions.pop(backend, None)


def is_unlocked(backend: str) -> bool:
    return get_session_token(backend) is not None


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
