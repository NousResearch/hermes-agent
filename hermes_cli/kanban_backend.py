"""Pluggable kanban backend seam.

ONE resolution point for the kanban DB implementation. Call sites that need the
kanban backend resolve it lazily through :func:`get_kanban_db` instead of
importing :mod:`hermes_cli.kanban_db` (or its split modules) directly:

    from hermes_cli import kanban_backend
    kb = kanban_backend.get_kanban_db()
    kbc = kanban_backend.get_kanban_db_connect()

Active-backend selection (env-less, no new HERMES_* variables):

- ``plugins.kanban_backend`` in config.yaml — the registered plugin name
  (``PluginContext.register_kanban_backend``), matched case-insensitively.
- Unset: exactly today's behavior — the in-tree default modules
  (:mod:`hermes_cli.kanban_db`, :mod:`hermes_cli.kanban_db_connect`), resolved
  by importing them at call time (import-order safe: the default modules import
  each other at their tails, so binding them at seam import time would be a
  behavior change). Strictly additive: with no configured backend every call
  resolves to the same modules as before.

Fail-closed: a configured-but-unregistered backend name raises
:class:`KanbanBackendError` naming the misconfiguration — it never silently
falls back to the default (a half-loaded board would be worse).

Import-site conversion rules (documented contract, see PR):

- Inside functions: ``kb = kanban_backend.get_kanban_db()`` then use ``kb.X``.
- Module top-level ``from hermes_cli import kanban_db as kb``: binding at
  import time would freeze the default backend, so convert to a lazy
  ``get_kanban_db()`` call where the symbols are reachable as attributes.
- ``from kanban_db import X`` of STABLE symbols (constants like
  ``DEFAULT_BOARD``/exit codes, exception classes) is acceptable to keep —
  they are not overridden by alternate backends.
- Split-module helpers a backend must provide (``connect``, ``connect_closing``,
  ``init_db``, ``write_txn``, plus the same attribute surface it replaces) are
  resolved via :func:`get_kanban_db_connect` / :func:`get_kanban_db`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Config section/key read for the active backend: ``plugins.kanban_backend``.
KANBAN_BACKEND_CONFIG_KEY = ("plugins", "kanban_backend")

_DEFAULT_DB_MODULE = "hermes_cli.kanban_db"
_DEFAULT_CONNECT_MODULE = "hermes_cli.kanban_db_connect"

class KanbanBackendError(RuntimeError):
    """A configured kanban backend cannot be resolved (fail-closed, no fallback)."""


class _Registry:
    """Name -> backend-module registry.

    Lives in the seam (not :mod:`hermes_cli.plugins`) so the kanban stack never
    imports the plugin manager at module import time; ``register_kanban_backend``
    on the plugin context reaches in here, and this module only imports plugins
    lazily for the single exception class. Mirrors the shape of the
    ``agent.*_registry`` modules: plain process-global dict, mutex, and
    case-insensitive names (plugin names are user-facing).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._modules: dict[str, Any] = {}

    def register(self, name: str, module: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("kanban backend name must be a non-empty string")
        key = name.strip().lower()
        with self._lock:
            self._modules[key] = module

    def unregister(self, name: str) -> None:
        with self._lock:
            self._modules.pop(name.strip().lower(), None)

    def get(self, name: str) -> Any:
        with self._lock:
            return self._modules.get(name.strip().lower())

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._modules)


_registry = _Registry()


def register_backend(name: str, module: Any) -> None:
    """Register an alternate backend module under *name* (lowercased)."""
    _registry.register(name, module)


def unregister_backend(name: str) -> None:
    """Remove a backend registration (plugin unload / tests)."""
    _registry.unregister(name)


def registered_backends() -> list[str]:
    """Names of currently registered alternate backends (diagnostics/tests)."""
    return _registry.names()


def _configured_backend_name() -> Optional[str]:
    """Read ``plugins.kanban_backend`` from config.yaml; None when unset/blank."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        block = cfg.get("plugins") if isinstance(cfg, dict) else None
        raw = block.get("kanban_backend") if isinstance(block, dict) else None
    except Exception as exc:  # health: allow BLE001 -- broken config store must fall back to the default backend, exactly as an unset key would
        logger.debug("Could not read plugins.kanban_backend from config: %s",
                     exc, exc_info=True)
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if raw is not None:
        logger.warning("plugins.kanban_backend must be a string; ignoring %r", raw)
    return None


def _resolve_default(name: str):
    """Import and return a default backend module (import-order safe: at call time)."""
    import importlib
    return importlib.import_module(name)


def _raise_unconfigured(configured: str) -> None:
    raise KanbanBackendError(
        f"plugins.kanban_backend='{configured}' is configured but no kanban backend plugin has "
        f"registered that name (registered: {registered_backends() or ['none']}); install/enable "
        "the backend plugin or unset plugins.kanban_backend"
    )


def get_kanban_db() -> Any:
    """The active kanban DB module (plugin backend when configured, else the default)."""
    configured = _configured_backend_name()
    if configured is None:
        return _resolve_default(_DEFAULT_DB_MODULE)
    module = _registry.get(configured)
    if module is None:
        _raise_unconfigured(configured)
    return module


class _LazyBackendModule:
    """Module-like proxy resolving the active backend on EVERY attribute access.

    Lets converted call sites keep their historical ``kb.attr`` spelling at module
    top level without freezing the default backend at import time: ``import kanban.py``
    binds the proxy, each attribute read re-resolves. Attribute SET/DEL go to the
    proxy itself (monkeypatch support); GET prefers them, so a test patching
    ``kb.init_db`` is seen by every reader of the same proxy until restored.
    """

    def __init__(self, resolver: Callable[[], Any]) -> None:
        self._resolver = resolver

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolver(), name)

    def __repr__(self) -> str:
        try:
            return f"<lazy kanban backend -> {self._resolver()!r}>"
        except Exception as exc:  # health: allow BLE001 -- repr must never raise; name the misconfiguration instead
            return f"<lazy kanban backend: {exc}>"


def lazy_kanban_db() -> _LazyBackendModule:
    """A fresh module proxy over :func:`get_kanban_db` for module-level ``kb`` bindings."""
    return _LazyBackendModule(get_kanban_db)


def lazy_kanban_db_connect() -> _LazyBackendModule:
    """A fresh module proxy over :func:`get_kanban_db_connect` for module-level ``kbc`` bindings."""
    return _LazyBackendModule(get_kanban_db_connect)


def get_kanban_db_connect() -> Any:
    """The active kanban connection module (``connect`` / ``connect_closing`` / ``write_txn``)."""
    configured = _configured_backend_name()
    if configured is None:
        return _resolve_default(_DEFAULT_CONNECT_MODULE)
    module = _registry.get(configured)
    if module is None:
        _raise_unconfigured(configured)
    # A backend registers ONE module carrying the full surface; require the connection
    # entry points here so a misconfigured backend fails loudly at the seam itself.
    if not all(hasattr(module, attr) for attr in ("connect", "connect_closing", "write_txn")):
        raise KanbanBackendError(
            f"kanban backend '{configured}' does not expose connect()/connect_closing()/"
            "write_txn(); the plugin's backend module is misconfigured"
        )
    return module
