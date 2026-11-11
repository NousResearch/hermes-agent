"""Kanban path isolation regression for #153098."""

from pathlib import Path

import hermes_constants
from hermes_cli import kanban_db as kb


def test_named_board_under_nested_scratch_home_stays_in_scratch(tmp_path, monkeypatch):
    native_root = tmp_path / "user" / ".hermes"
    scratch = native_root / "profiles" / "worker" / "cache" / "scratch" / "run" / ".hermes"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(
        hermes_constants, "_get_platform_default_hermes_home", lambda: native_root
    )
    monkeypatch.setenv("HERMES_HOME", str(scratch))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)

    assert kb.kanban_home() == scratch
    assert kb.kanban_db_path(board="isolated") == (
        scratch / "kanban" / "boards" / "isolated" / "kanban.db"
    )
    assert not (native_root / "kanban.db").exists()
