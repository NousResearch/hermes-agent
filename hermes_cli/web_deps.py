"""Late-binding dependency seam for extracted dashboard routers.

``web_server`` owns the dashboard app and its process state; routers under
``web_routers/`` cannot import it at import time (web_server imports them to mount them —
a cycle) and must not copy its state (tests ``monkeypatch.setattr(web_server, ...)`` and
expect that to win). ``late(name)`` / ``LateState(name)`` resolve ``<module>.<name>`` *at
call time*; ``module`` defaults to ``web_server`` and may name a ``web_server_<concern>``
module whose helper tests monkeypatch there.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

WEB_SERVER = "hermes_cli.web_server"


def _server(module: str = WEB_SERVER):
    """Return the live ``module`` (imported on demand)."""
    mod = sys.modules.get(module)
    if mod is None:  # pragma: no cover - routers are only mounted by web_server
        mod = importlib.import_module(module)
    return mod


def late(name: str, module: str = WEB_SERVER):
    """Late-binding proxy for a callable defined on ``module``."""

    def _proxy(*args: Any, **kwargs: Any):
        return getattr(_server(module), name)(*args, **kwargs)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


class LateState:
    """Live proxy for module-level state owned by ``web_server``.

    Forwards attribute/item access, iteration, membership, len/truthiness, ``with`` (locks)
    and rich comparisons to ``web_server.<name>`` resolved at operation time — some state is
    defined *after* the router's ``include_router`` point, so a late import would miss it.
    """

    __slots__ = ("_name", "_module")

    def __init__(self, name: str, module: str = WEB_SERVER) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_module", module)

    def _target(self) -> Any:
        return getattr(_server(object.__getattribute__(self, "_module")),
                       object.__getattribute__(self, "_name"))

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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def get_dashboard_health():
    """The ``DASHBOARD_HEALTH`` singleton owned by web_server."""
    return _server().DASHBOARD_HEALTH

def get_session_token() -> str:
    """Current dashboard session token (``web_server._SESSION_TOKEN``)."""
    return _server()._SESSION_TOKEN

def has_valid_session_token(request) -> bool:
    """Late-bound alias for ``web_server._has_valid_session_token``."""
    return _server()._has_valid_session_token(request)

def late_attr(name: str) -> Any:
    """Read ``web_server.<name>`` right now (for non-callable state reads)."""
    return getattr(_server(), name)
# ---- END PLUGIN-COMPAT ----
