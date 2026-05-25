"""Runtime selector for Hermes kanban backends.

Importers continue using ``hermes_cli.kanban_db`` while this wrapper chooses
between the SQLite implementation (default) and the Postgres shim.

Selection order:
1. ``HERMES_KANBAN_BACKEND`` env var
2. ``kanban.backend`` in Hermes config
3. ``sqlite`` default
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

_BACKEND_ENV_VAR = "HERMES_KANBAN_BACKEND"
_DEFAULT_BACKEND = "sqlite"
_BACKEND_MODULES = {
    "sqlite": "hermes_cli.kanban_db_sqlite",
    "postgres": "hermes_cli.kanban_db_pg",
}
_BACKEND_FILES = {
    "sqlite": "kanban_db_sqlite.py",
    "postgres": "kanban_db_pg.py",
}

_PREVIOUS_EXPORTS = globals().get("_LOADED_BACKEND_EXPORTS", set())
for _name in list(_PREVIOUS_EXPORTS):
    globals().pop(_name, None)


def _normalize_backend_name(raw: Any) -> str:
    name = str(raw or "").strip().lower()
    if not name:
        return _DEFAULT_BACKEND
    if name not in _BACKEND_MODULES:
        valid = ", ".join(sorted(_BACKEND_MODULES))
        raise ValueError(
            f"unsupported kanban backend {raw!r}; expected one of: {valid}"
        )
    return name



def _config_backend_name() -> str | None:
    try:
        from hermes_cli.config import load_config
    except Exception:
        return None

    cfg = load_config() or {}
    section = cfg.get("kanban", {}) or {}
    if not isinstance(section, dict):
        return None
    backend = section.get("backend")
    if backend is None:
        return None
    return str(backend)



def _selected_backend_name() -> str:
    env_backend = os.environ.get(_BACKEND_ENV_VAR, "").strip()
    if env_backend:
        return _normalize_backend_name(env_backend)
    return _normalize_backend_name(_config_backend_name())


SELECTED_BACKEND = _selected_backend_name()
SELECTED_BACKEND_MODULE = _BACKEND_MODULES[SELECTED_BACKEND]
SELECTED_BACKEND_FILE = Path(__file__).with_name(_BACKEND_FILES[SELECTED_BACKEND])
_PRE_EXEC_KEYS = set(globals())
exec(
    compile(SELECTED_BACKEND_FILE.read_text(encoding="utf-8"), str(SELECTED_BACKEND_FILE), "exec"),
    globals(),
)
_CANONICAL_BACKEND_MODULE = importlib.import_module(SELECTED_BACKEND_MODULE)
for _name in dir(_CANONICAL_BACKEND_MODULE):
    _value = getattr(_CANONICAL_BACKEND_MODULE, _name)
    if isinstance(_value, type):
        globals()[_name] = _value
_LOADED_BACKEND_EXPORTS = {
    name
    for name in (set(globals()) - _PRE_EXEC_KEYS)
    if not (name.startswith("__") and name not in {"__all__", "__doc__"})
}



def get_selected_backend_name() -> str:
    return SELECTED_BACKEND



def get_selected_backend_module() -> str:
    return SELECTED_BACKEND_MODULE



def get_selected_backend_file() -> str:
    return str(SELECTED_BACKEND_FILE)


_PUBLIC_EXPORTS = {
    name
    for name in globals()
    if not name.startswith("_")
}
_PUBLIC_EXPORTS.update(
    {
        "get_selected_backend_name",
        "get_selected_backend_module",
        "get_selected_backend_file",
    }
)
__all__ = sorted(_PUBLIC_EXPORTS)
