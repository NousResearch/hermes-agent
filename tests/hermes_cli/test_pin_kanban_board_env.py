"""Tests for `_pin_kanban_board_env` helper invoked by `cmd_chat`.

Regression coverage for #20074: a chat session must export the active kanban
board into `HERMES_KANBAN_BOARD` at boot so subprocess shell-outs (e.g.
`hermes kanban …`) inherit the same board the in-process kanban tools resolve.
Without this, a concurrent `hermes kanban boards switch` from another session
can flip the global current-board file mid-turn and silently divert the
shell calls to a different DB.
"""
import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_kanban_board_env():
    """Snapshot `HERMES_KANBAN_BOARD` and restore it after the test.

    `_pin_kanban_board_env()` writes to ``os.environ`` directly, bypassing
    any ``monkeypatch.setenv`` tracking. Without this fixture the mutation
    leaks into subsequent tests and breaks anything that resolves a kanban
    path from the env (e.g. ``TestSharedBoardPaths`` in test_kanban_db.py).
    """
    prev = os.environ.get("HERMES_KANBAN_BOARD")
    os.environ.pop("HERMES_KANBAN_BOARD", None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HERMES_KANBAN_BOARD", None)
        else:
            os.environ["HERMES_KANBAN_BOARD"] = prev


def test_pin_writes_resolved_board_when_env_unset(monkeypatch):
    main_mod = importlib.import_module("hermes_cli.main")

    import hermes_cli.kanban_db as kdb
    monkeypatch.setattr(kdb, "get_current_board", lambda: "space")

    main_mod._pin_kanban_board_env()

    assert main_mod.os.environ.get("HERMES_KANBAN_BOARD") == "space"


def test_pin_does_not_overwrite_existing_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "preset")
    main_mod = importlib.import_module("hermes_cli.main")

    import hermes_cli.kanban_db as kdb

    def _explode():
        raise AssertionError("get_current_board must not be called when env is set")

    monkeypatch.setattr(kdb, "get_current_board", _explode)

    main_mod._pin_kanban_board_env()

    assert main_mod.os.environ.get("HERMES_KANBAN_BOARD") == "preset"


def test_pin_swallows_resolution_failures(monkeypatch):
    main_mod = importlib.import_module("hermes_cli.main")

    import hermes_cli.kanban_db as kdb

    def _boom():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(kdb, "get_current_board", _boom)

    main_mod._pin_kanban_board_env()

    assert "HERMES_KANBAN_BOARD" not in main_mod.os.environ


def test_pin_skips_when_allow_session_board_switch_true(monkeypatch):
    """When kanban.allow_session_board_switch is true, the env var is not pinned."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"allow_session_board_switch": True}},
    )
    main_mod = importlib.import_module("hermes_cli.main")

    import hermes_cli.kanban_db as kdb

    def _explode():
        raise AssertionError("get_current_board must not be called when config opt-out is set")

    monkeypatch.setattr(kdb, "get_current_board", _explode)

    main_mod._pin_kanban_board_env()

    assert "HERMES_KANBAN_BOARD" not in main_mod.os.environ


def test_pin_skips_when_allow_session_board_switch_true_even_with_env_set(monkeypatch):
    """Config opt-out takes priority — skips even when HERMES_KANBAN_BOARD is pre-set."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"allow_session_board_switch": True}},
    )
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "preset-by-dispatcher")
    main_mod = importlib.import_module("hermes_cli.main")

    import hermes_cli.kanban_db as kdb

    def _explode():
        raise AssertionError("get_current_board must not be called when config opt-out is set")

    monkeypatch.setattr(kdb, "get_current_board", _explode)

    main_mod._pin_kanban_board_env()
    # Should NOT have been overwritten — config opt-out means we leave it alone
    assert main_mod.os.environ.get("HERMES_KANBAN_BOARD") == "preset-by-dispatcher"


def test_pin_does_not_skip_when_allow_session_board_switch_false(monkeypatch):
    """When kanban.allow_session_board_switch is false or absent, the env var is pinned."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"allow_session_board_switch": False}},
    )
    main_mod = importlib.import_module("hermes_cli.main")

    import hermes_cli.kanban_db as kdb
    monkeypatch.setattr(kdb, "get_current_board", lambda: "space")

    main_mod._pin_kanban_board_env()

    assert main_mod.os.environ.get("HERMES_KANBAN_BOARD") == "space"


def test_set_current_board_updates_env(monkeypatch, tmp_path):
    """set_current_board() writes the file AND updates HERMES_KANBAN_BOARD."""
    import sys

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sys.modules.pop("hermes_cli.kanban_db", None)

    import hermes_cli.kanban_db as kdb

    slug = "test-board"
    old_env = os.environ.get("HERMES_KANBAN_BOARD")
    os.environ.pop("HERMES_KANBAN_BOARD", None)
    try:
        kdb.set_current_board(slug)
        assert os.environ.get("HERMES_KANBAN_BOARD") == slug
        assert (tmp_path / "kanban" / "current").read_text().strip() == slug
    finally:
        if old_env is not None:
            os.environ["HERMES_KANBAN_BOARD"] = old_env
        else:
            os.environ.pop("HERMES_KANBAN_BOARD", None)
