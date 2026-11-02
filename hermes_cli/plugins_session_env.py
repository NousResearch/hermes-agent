"""Per-session environment values that plugins give a session's tool subprocesses (``ctx.register_session_env``).

Tools run in the shared gateway, so a variable of the client that shows a session (a terminal pane's
id, say) is not in the gateway's environment. A plugin declares the names and a resolver; every child
the session context reaches (terminal foreground and background, ``execute_code``'s ``terminal()``)
calls the resolvers of the active profile with the child's ``HERMES_SESSION_ID`` when it is spawned.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

Resolver = Callable[[str], Optional[Mapping[str, Optional[str]]]]

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_LOCK = threading.Lock()
_REGISTRATIONS: dict[object, tuple[str, frozenset[str], Resolver]] = {}


def register_session_env(scope: str, names: Iterable[str], resolver: Resolver) -> Callable[[], None]:
    """Record ``resolver`` for ``names`` in profile ``scope``; return its release. ``HERMES_*`` stays core's."""
    names = [names] if isinstance(names, str) else list(names)
    if not names or any(not isinstance(n, str) or not _NAME_RE.fullmatch(n) or n.upper().startswith("HERMES_")
                        for n in names):
        raise ValueError("session env names must be valid environment variable names outside HERMES_*")
    if not callable(resolver):
        raise TypeError("session env resolver must be callable")
    token = object()
    with _LOCK:
        _REGISTRATIONS[token] = (scope, frozenset(names), resolver)

    def release() -> None:
        with _LOCK:
            _REGISTRATIONS.pop(token, None)
    return release


def session_env_names() -> frozenset[str]:
    """Every registered name, all profiles: none may be baked into a shared shell snapshot."""
    with _LOCK:
        return frozenset(n for _, names, _ in _REGISTRATIONS.values() for n in names)


def resolve_session_env(session_id: str) -> dict[str, Optional[str]]:
    """The active profile's values for ``session_id`` (``None``: unset the name). A resolver that raises,
    or returns something else than a map, contributes nothing; names it did not declare are ignored."""
    with _LOCK:
        registrations = tuple(_REGISTRATIONS.values())
    if not session_id or not registrations:
        return {}
    from hermes_constants import hermes_home_key
    scope = hermes_home_key()
    out: dict[str, Optional[str]] = {}
    for owner, names, resolver in registrations:
        if owner != scope:
            continue
        try:
            values = {k: v for k, v in (resolver(session_id) or {}).items()
                      if k in names and (v is None or isinstance(v, str))}
        except Exception:
            logger.warning("Session env resolver %r failed for %s", resolver, session_id, exc_info=True)
            continue
        out.update(values)
    return out
