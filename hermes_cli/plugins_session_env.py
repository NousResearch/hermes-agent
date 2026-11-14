"""Per-session environment values that plugins give a session's tool subprocesses (``ctx.register_session_env``).

Tools run in the shared gateway, so a variable of the client that shows a session (a terminal pane's
id, say) is not in the gateway's environment. A plugin declares the names and a resolver; every child
the session context reaches (terminal foreground and background, ``execute_code``'s ``terminal()``)
calls the resolvers of the active profile with the child's ``HERMES_SESSION_ID`` when it is spawned.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Callable, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

Resolver = Callable[[str], Optional[Mapping[str, Optional[str]]]]

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_LOCK = threading.Lock()
_REGISTRATIONS: dict[object, tuple[str, Optional[str], frozenset[str], Resolver]] = {}  # (scope, owner, names, resolver)
# Every name declared per profile scope, kept past its plugin's unload: a shell snapshot taken while a
# value was injected may hold it, and the name must stay excluded so no later command restores it.
# Another profile's shells never see these names (a multiplexed gateway serves several profiles).
_DECLARED: dict[str, set[str]] = {}


def _env_key(name: str) -> str:
    """The child-env key for ``name``: Windows' environment is case-insensitive and ``os.environ``
    holds it upper-cased, so ``pane`` must replace or unset an inherited ``PANE``, never sit beside it."""
    return name.upper() if os.name == "nt" else name


def register_session_env(scope: str, names: Iterable[str], resolver: Resolver, *,
                         owner: Optional[str] = None) -> Callable[[], None]:
    """Record ``resolver`` for ``names`` in profile ``scope``; return its release. ``HERMES_*`` stays core's.
    A name a live registration of another ``owner`` (plugin) in ``scope`` declares is refused (``ValueError``
    naming the holder): one plugin per name, never a silent last-wins between plugins. One owner's own
    overlapping registrations stack: the latest live one's value wins."""
    names = [names] if isinstance(names, str) else list(names)
    if not names or any(not isinstance(n, str) or not _NAME_RE.fullmatch(n) or n.upper().startswith("HERMES_")
                        for n in names):
        raise ValueError("session env names must be valid environment variable names outside HERMES_*")
    if not callable(resolver):
        raise TypeError("session env resolver must be callable")
    token = object()
    keys = frozenset(_env_key(n) for n in names)
    with _LOCK:
        clash = next(((sorted(keys & held)[0], holder) for other, holder, held, _ in _REGISTRATIONS.values()
                      if other == scope and holder != owner and keys & held), None)
        if clash is None:
            _REGISTRATIONS[token] = (scope, owner, keys, resolver)
            _DECLARED.setdefault(scope, set()).update(keys)
    if clash is not None:
        raise ValueError(f"session env {clash[0]!r} is already registered by {clash[1] or 'another plugin'!r}")

    def release() -> None:
        with _LOCK:
            _REGISTRATIONS.pop(token, None)
    return release


def session_env_names() -> frozenset[str]:
    """Every name declared in the active profile, unloaded plugins' included: none may be baked into (or
    restored from) a shared shell snapshot."""
    from hermes_constants import hermes_home_key
    scope = hermes_home_key()
    with _LOCK:
        return frozenset(_DECLARED.get(scope, ()))


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
    for owner, _, names, resolver in registrations:
        if owner != scope:
            continue
        try:
            values = {_env_key(k): v for k, v in (resolver(session_id) or {}).items()
                      if isinstance(k, str) and _env_key(k) in names and (v is None or isinstance(v, str))}
        except Exception:
            logger.warning("Session env resolver %r failed for %s", resolver, session_id, exc_info=True)
            continue
        out.update(values)
    return out
