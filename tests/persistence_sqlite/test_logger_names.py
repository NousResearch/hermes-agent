"""Every logger the SQLite backend writes to carries a name main's own modules log under for the same code, so the
per-component log views (``hermes logs --component``, gateway.log), which select records by logger name, show the
backend's records where they show main's."""

from __future__ import annotations

import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2] / "persistence" / "sqlite"

# The names main logs under in the modules this backend's code comes from.
_MAIN_LOGGER_NAMES = frozenset({
    "gateway.hosted_room_work_records",
    "gateway.hosted_rooms_legacy_import",
    "gateway.lifecycle_ledger",
    "gateway.session",
    "gateway.session_api",
    "hermes_cli.backup_restore",
    "hermes_cli.backup_sqlite",
    "hermes_cli.session_lost_and_found",
    "hermes_cli.timefmt",
    "hermes_state",
    "hermes_state_health",
    "hermes_state_holders",
    "hermes_state_lockowners",
    "hermes_state_reconcile",
    "hermes_state_registry",
    "tools.async_delegation",
})

_LEVELS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


def _backend_modules() -> list[Path]:
    return sorted(_BACKEND.rglob("*.py"))


def _logger_names(path: Path):
    """``(line, name)`` for every logger *path* obtains or logs through directly; a name that is not a string
    literal is given as its source text, the root logger as ``root``."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = ast.unparse(node.func)
        if func in ("logging.getLogger", "getLogger"):
            if not node.args:
                yield node.lineno, "root"
                continue
            arg = node.args[0]
            literal = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            yield node.lineno, arg.value if literal else ast.unparse(arg)
        elif isinstance(node.func, ast.Attribute) and func == f"logging.{node.func.attr}" and node.func.attr in _LEVELS:
            yield node.lineno, "root"


def test_every_backend_logger_carries_a_name_main_logs_under():
    found, strays = 0, []
    for path in _backend_modules():
        for line, name in _logger_names(path):
            found += 1
            if name not in _MAIN_LOGGER_NAMES:
                strays.append(f"{path.relative_to(_BACKEND.parents[1])}:{line}: {name}")
    assert found, "no logger found under persistence/sqlite"
    assert not strays, "backend loggers named outside main's names:\n" + "\n".join(strays)
