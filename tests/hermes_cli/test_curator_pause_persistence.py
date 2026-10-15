"""Tests for the curator pause flag's persistence contract (#139438).

``hermes curator pause`` used to print ``curator: paused`` and exit 0 even when the
state write failed (unwritable ``skills/`` dir), leaving the scheduler unpaused while
every reading surface (``hermes curator status``, the dashboard API) reports the flag
as the source of truth. ``save_state``/``set_paused`` now return whether the write
landed and the CLI exits non-zero on a lost pause.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)

    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    from hermes_cli import curator as cli_curator
    importlib.reload(cli_curator)

    yield {"curator": curator, "cli": cli_curator, "capsys": capsys}


def _break_state_write(curator_mod, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("simulated read-only filesystem")
    monkeypatch.setattr(curator_mod, "atomic_json_write", boom)


def test_set_paused_reports_failed_write(curator_env, monkeypatch):
    _break_state_write(curator_env["curator"], monkeypatch)
    assert curator_env["curator"].set_paused(True) is False
    assert curator_env["curator"].is_paused() is False


def test_set_paused_confirms_landed_write(curator_env):
    assert curator_env["curator"].set_paused(True) is True
    assert curator_env["curator"].is_paused() is True


def test_cli_pause_exits_nonzero_when_unpersisted(curator_env, monkeypatch):
    _break_state_write(curator_env["curator"], monkeypatch)
    rc = curator_env["cli"].cli_main(["pause"])
    captured = curator_env["capsys"].readouterr()
    assert rc == 1
    assert "NOT paused" in captured.err
    assert "curator: paused" not in captured.out
    assert curator_env["curator"].is_paused() is False


def test_cli_pause_succeeds_when_writable(curator_env):
    rc = curator_env["cli"].cli_main(["pause"])
    captured = curator_env["capsys"].readouterr()
    assert rc == 0
    assert "curator: paused" in captured.out
    assert curator_env["curator"].is_paused() is True


def test_cli_resume_exits_nonzero_when_unpersisted(curator_env, monkeypatch):
    curator_env["curator"].set_paused(True)
    _break_state_write(curator_env["curator"], monkeypatch)
    rc = curator_env["cli"].cli_main(["resume"])
    captured = curator_env["capsys"].readouterr()
    assert rc == 1
    assert "NOT resumed" in captured.err
    assert curator_env["curator"].is_paused() is True
