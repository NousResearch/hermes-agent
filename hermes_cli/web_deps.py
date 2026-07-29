"""Shared late-binding dependency seam for extracted dashboard routers.

Why this exists
---------------
``hermes_cli/web_server.py`` owns all dashboard runtime state: the ephemeral
``_SESSION_TOKEN``, the ``DASHBOARD_HEALTH`` singleton, config helpers, and a
large set of private helper functions the route handlers call.  Extracted
``APIRouter`` modules under ``hermes_cli/web_routers/`` need those helpers, but

* importing ``web_server`` at module import time from a router module would be
  a circular import (``web_server`` imports the router modules to mount them),
  and
* re-homing the helpers/state here would break the many tests (and any third
  party code) that ``monkeypatch.setattr(web_server, "_helper", ...)``.

Design: **late binding, state stays in web_server.**  ``late(name)`` returns a
thin proxy that resolves ``hermes_cli.web_server.<name>`` *at call time*.  This
is cycle-safe (the import happens inside the call, long after both modules are
initialised) and keeps ``web_server``'s runtime behaviour byte-identical:
monkeypatching an attribute on ``web_server`` is still authoritative because
every call re-reads the attribute from the live module.
"""

from __future__ import annotations

import sys
from typing import Any


def _server():
    """Return the live ``hermes_cli.web_server`` module (imported on demand)."""
    mod = sys.modules.get("hermes_cli.web_server")
    if mod is None:  # pragma: no cover - routers are only mounted by web_server
        import hermes_cli.web_server as mod  # type: ignore[no-redef]
    return mod


def late(name: str):
    """Late-binding proxy for a callable defined on ``web_server``.

    The returned wrapper looks up ``web_server.<name>`` on every call, so
    async/sync nature, monkeypatched replacements, and module state are all
    resolved at call time — never frozen at import time.
    """

    def _proxy(*args: Any, **kwargs: Any):
        return getattr(_server(), name)(*args, **kwargs)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


def late_attr(name: str) -> Any:
    """Read ``web_server.<name>`` right now (for non-callable state reads)."""
    return getattr(_server(), name)


class LateState:
    """Live proxy for module-level state owned by ``web_server``.

    Extracted routers can't ``from web_server import _mcp_oauth_flows`` —
    that would freeze the object at import time (breaking tests that mutate
    or replace it on ``web_server``) and be a circular import besides.  Some
    of the state is also defined *after* the router's ``include_router``
    point in web_server's body, so even a late module-import wouldn't see it
    yet.  This proxy forwards every operation the extracted handlers actually
    perform — attribute access, item get/set/del, iteration, membership,
    ``len``/truthiness, ``with``-blocks (locks), and rich comparisons
    (numeric limits) — to ``web_server.<name>`` resolved at operation time,
    so mutating or monkeypatching the attribute on ``web_server`` stays
    authoritative.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def _target(self) -> Any:
        return getattr(_server(), object.__getattribute__(self, "_name"))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._target(), attr)

    def __getitem__(self, key: Any) -> Any:
        return self._target()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._target()[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._target()[key]

    def __contains__(self, item: Any) -> bool:
        return item in self._target()

    def __iter__(self):
        return iter(self._target())

    def __len__(self) -> int:
        return len(self._target())

    def __bool__(self) -> bool:
        return bool(self._target())

    def __enter__(self):
        return self._target().__enter__()

    def __exit__(self, *exc):
        return self._target().__exit__(*exc)

    def __eq__(self, other: Any) -> bool:
        return self._target() == other

    def __ne__(self, other: Any) -> bool:
        return self._target() != other

    def __lt__(self, other: Any) -> bool:
        return self._target() < other

    def __le__(self, other: Any) -> bool:
        return self._target() <= other

    def __gt__(self, other: Any) -> bool:
        return self._target() > other

    def __ge__(self, other: Any) -> bool:
        return self._target() >= other

    def __hash__(self) -> int:
        return hash(self._target())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LateState {object.__getattribute__(self, '_name')} -> {self._target()!r}>"


# --- Named accessors for the shared server state (call-time reads) ---------


def get_session_token() -> str:
    """Current dashboard session token (``web_server._SESSION_TOKEN``)."""
    return _server()._SESSION_TOKEN


def get_dashboard_health():
    """The ``DASHBOARD_HEALTH`` singleton owned by web_server."""
    return _server().DASHBOARD_HEALTH


def has_valid_session_token(request) -> bool:
    """Late-bound alias for ``web_server._has_valid_session_token``."""
    return _server()._has_valid_session_token(request)
