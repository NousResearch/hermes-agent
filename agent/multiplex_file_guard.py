"""File tools must not cross served profile homes under multiplex (4d).

Every profile in a multiplexed gateway runs as the same OS uid, so the file
tools of one bot could read and write the other bot's ``config.yaml``,
``SOUL.md``, skills, memories, cron jobs and caches. ``agent.file_safety``
already blocks credentials and the session/memory stores under any home;
this adds the profile boundary itself: with multiplexing on, a path inside
another **served** profile's home is denied for ``read_file`` /
``search_files`` / ``write_file`` / ``patch``. The default profile's home is
the HERMES root minus its ``profiles/`` subtree.

``terminal`` is admin-only and remains outside this boundary, as documented.
Denials are written to the ``data_access_audit`` trail.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, Iterable, Optional, Tuple

_SERVED_HOMES: Tuple[str, ...] = ()

# One file-tool call can evaluate the same path several times (is_write_denied
# then get_write_denied_error; a search re-checks every hit), so identical
# denials within a short window are audited once.
_AUDIT_WINDOW_SECONDS = 5.0
_audit_lock = threading.Lock()
_recent_audits: Dict[Tuple[str, str], float] = {}


def audit_denial(action: str, target: str) -> None:
    """Append a ``data_access_audit`` line, de-duplicated over a short window."""
    now = time.monotonic()
    key = (action, target)
    with _audit_lock:
        last = _recent_audits.get(key)
        if last is not None and now - last < _AUDIT_WINDOW_SECONDS:
            return
        _recent_audits[key] = now
        if len(_recent_audits) > 512:
            for k, t in list(_recent_audits.items()):
                if now - t >= _AUDIT_WINDOW_SECONDS:
                    _recent_audits.pop(k, None)
    try:
        from agent.data_access_audit import record_access

        record_access(tool="file", action=action, target=target)
    except Exception:
        pass


def _real(path) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def set_served_profile_homes(homes: Iterable) -> None:
    """Record the multiplexer's served profile homes (gateway startup)."""
    global _SERVED_HOMES
    seen = []
    for home in homes:
        if not home:
            continue
        real = _real(home)
        if real not in seen:
            seen.append(real)
    _SERVED_HOMES = tuple(seen)


def served_profile_homes() -> Tuple[str, ...]:
    return _SERVED_HOMES


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def cross_profile_denial(path, *, verb: str = "Access") -> Optional[str]:
    """Return a denial when *path* lies in another served profile's home."""
    try:
        from agent.secret_scope import is_multiplex_active

        if not is_multiplex_active() or not _SERVED_HOMES:
            return None
        from hermes_constants import get_hermes_home

        target = _real(path)
        active = _real(get_hermes_home())
        for home in _SERVED_HOMES:
            if home == active or not _under(target, home):
                continue
            # A home that contains the profiles/ tree is the default (root)
            # profile: other profiles' homes live under it and are judged on
            # their own entries — including the active one.
            if _under(target, os.path.join(home, "profiles")):
                continue
            # The active profile lives inside this home (e.g. a named profile
            # nested under the root) — the subtree is the active profile's.
            if _under(target, active) and _under(active, home):
                continue
            audit_denial("cross_profile_denied", str(path))
            return (
                f"{verb} denied: '{path}' belongs to another profile served by "
                "this gateway. Each bot may only use files in its own profile."
            )
    except Exception:
        return None
    return None
