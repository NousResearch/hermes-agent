"""`hermes --tui --reasoning LEVEL` reaches the TUI backend (NousResearch/hermes-agent#107780).

``cmd_chat`` forwards the flag to ``_launch_tui``, which hands it to the TUI process as
``HERMES_TUI_REASONING`` (clearing any stale inherited value first) and, when combined with a resume,
says that the resumed session keeps its stored reasoning.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import main_tui_launch

LAUNCH_ENV = "HERMES_TUI_REASONING"


def _args(**overrides):
    base = {
        "cli": False,
        "continue_last": None,
        "in_dir": None,
        "model": None,
        "no_restore_cwd": False,
        "provider": None,
        "query": None,
        "resume": None,
        "safe_mode": False,
        "toolsets": None,
        "tui": True,
        "tui_dev": False,
        "worktree": False,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(mod, "_sync_bundled_skills_for_startup", lambda: False)
    monkeypatch.setattr(mod, "_pin_kanban_board_env", lambda: None)
    return mod


@pytest.fixture
def launched(main_mod, monkeypatch):
    """Capture the _launch_tui call instead of exec'ing Node."""
    captured = {}

    def fake_launch(resume_session_id=None, **kwargs):
        captured["resume"] = resume_session_id
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)
    return captured


@pytest.fixture
def child_env(main_mod, monkeypatch):
    """Run the real _launch_tui against a stubbed Node child; returns a callable yielding the child's env."""
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.delenv(LAUNCH_ENV, raising=False)
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv", lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")))
    captured = {}
    monkeypatch.setattr(main_mod.subprocess, "call", lambda argv, cwd=None, env=None: captured.update(env=env) or 1)

    def launch(*args, **kwargs):
        with pytest.raises(SystemExit):
            main_mod._launch_tui(*args, **kwargs)
        return captured["env"]

    return launch


def test_cmd_chat_forwards_explicit_reasoning_to_the_tui_launch(main_mod, launched):
    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(reasoning="medium"))
    assert launched["reasoning"] == "medium"


def test_cmd_chat_forwards_no_reasoning_when_the_flag_is_omitted(main_mod, launched):
    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args())
    assert launched["reasoning"] is None


def test_launch_exports_the_reasoning_value(child_env):
    assert child_env(reasoning=" medium ")[LAUNCH_ENV] == "medium"
    assert child_env(reasoning="none")[LAUNCH_ENV] == "none"


@pytest.mark.parametrize("reasoning", [None, "", "  "])
def test_launch_without_a_reasoning_flag_clears_an_inherited_value(child_env, monkeypatch, reasoning):
    monkeypatch.setenv(LAUNCH_ENV, "high")  # stale export from an outer shell or launcher
    assert LAUNCH_ENV not in child_env(reasoning=reasoning)


def test_resume_with_reasoning_prints_the_notice_and_keeps_the_launch_value(child_env, capsys):
    env = child_env("stored-session", reasoning="medium")
    err = capsys.readouterr().err
    assert "--reasoning applies to new sessions only" in err
    assert "keeps its stored reasoning" in err
    assert env["HERMES_TUI_RESUME"] == "stored-session"
    assert env[LAUNCH_ENV] == "medium"  # a later /new in this process is a new session


def test_no_notice_for_new_sessions_or_without_the_flag(child_env, capsys):
    child_env(reasoning="medium")
    child_env("stored-session")
    assert "--reasoning" not in capsys.readouterr().err


def test_cmd_chat_hands_a_resolved_resume_and_reasoning_to_the_launch(main_mod, launched, monkeypatch):
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda value: value)
    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(resume="stored-session", reasoning="low"))
    assert (launched["resume"], launched["reasoning"]) == ("stored-session", "low")
