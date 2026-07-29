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
