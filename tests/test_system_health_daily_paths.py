import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "system_health_daily.py"


def _load_module(monkeypatch, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    monkeypatch.setattr(sys, "argv", ["system_health_daily.py"])
    scripts_dir = REPO_ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    spec = importlib.util.spec_from_file_location("system_health_path_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_board_compat(monkeypatch, value):
    fake = types.ModuleType("_board_compat")
    setattr(fake, "build_board_db_map", lambda _slugs: {"ops": value})
    monkeypatch.setitem(sys.modules, "_board_compat", fake)


def _create_board(path: Path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE task_events (
                task_id TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            INSERT INTO tasks VALUES ('t_one', 'triage', 4102444800);
            INSERT INTO task_events VALUES ('t_one', 4102444800);
            """
        )


def test_check_kanban_accepts_string_path_configuration(monkeypatch, tmp_path):
    db_path = tmp_path / "board.db"
    _create_board(db_path)
    module = _load_module(monkeypatch, tmp_path / "home")
    _install_board_compat(monkeypatch, str(db_path))

    result = module.check_kanban()

    assert result is not None
    assert "1 triage task(s)" in result["body"]


def test_check_kanban_accepts_existing_path_object(monkeypatch, tmp_path):
    db_path = tmp_path / "board.db"
    _create_board(db_path)
    module = _load_module(monkeypatch, tmp_path / "home")
    _install_board_compat(monkeypatch, db_path)

    result = module.check_kanban()

    assert result is not None
    assert result["slug"] == "kanban-health"


def test_check_kanban_skips_nonexistent_string_path(monkeypatch, tmp_path):
    missing = tmp_path / "missing.db"
    module = _load_module(monkeypatch, tmp_path / "home")
    _install_board_compat(monkeypatch, str(missing))

    assert module.check_kanban() is None


@pytest.mark.parametrize("configured_path", [None, "", "   ", {"not": "a path"}])
def test_check_kanban_skips_missing_empty_or_malformed_paths(
    monkeypatch, tmp_path, configured_path
):
    module = _load_module(monkeypatch, tmp_path / "home")
    _install_board_compat(monkeypatch, configured_path)

    assert module.check_kanban() is None
