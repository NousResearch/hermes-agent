"""Tests for `_print_curator_recent_run_notice`.

The notice prints the most recent curator run summary on `hermes update`,
exactly once per run. Show-once is enforced by stamping
`last_run_summary_shown_at` in curator state after printing.

Why this matters: the curator runs in the background (gateway tick + CLI
session start) so users normally never see the rename map. `hermes update`
is the high-attention surface where consolidations should land.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    from hermes_cli import main as hermes_main
    importlib.reload(hermes_main)

    yield {
        "curator": curator,
        "main": hermes_main,
        "capsys": capsys,
    }


def _set_state(curator_mod, **fields):
    state = curator_mod.load_state()
    state.update(fields)
    curator_mod.save_state(state)


def test_silent_when_no_curator_run_yet(curator_env):
    """First-run notice handles this case; recent-run notice stays silent."""
    curator_env["main"]._print_curator_recent_run_notice()
    out = curator_env["capsys"].readouterr().out
    assert "Skill curator — last run" not in out


def test_silent_when_summary_is_single_line(curator_env):
    """No archives = no rename map = nothing to surface. But still stamps shown."""
    now = datetime.now(timezone.utc).isoformat()
    _set_state(
        curator_env["curator"],
        last_run_at=now,
        last_run_summary="auto: no changes; llm: no change",
    )
    curator_env["main"]._print_curator_recent_run_notice()
    out = curator_env["capsys"].readouterr().out
    assert "Skill curator — last run" not in out
    # Should still mark shown so we don't reconsider on every update.
    state = curator_env["curator"].load_state()
    assert state["last_run_summary_shown_at"] == now


def test_prints_multiline_summary_with_rename_map(curator_env):
    """Multi-line summary (rename map appended) prints with timestamp + footer."""
    now = datetime.now(timezone.utc).isoformat()
    summary = (
        "auto: 1 marked stale; llm: consolidated 2 into 1\n"
        "archived 2 skill(s):\n"
        "  • pdf-extraction → document-tools\n"
        "  • docx-extraction → document-tools\n"
        "full report: hermes curator status"
    )
    _set_state(
        curator_env["curator"],
        last_run_at=now,
        last_run_summary=summary,
    )
    curator_env["main"]._print_curator_recent_run_notice()
    out = curator_env["capsys"].readouterr().out
    assert "Skill curator — last run" in out
    assert "pdf-extraction → document-tools" in out
    assert "docx-extraction → document-tools" in out
    assert "shows once per curator run" in out


def test_show_once_semantics(curator_env):
    """Calling twice prints once; second call is silent until a new run lands."""
    now = datetime.now(timezone.utc).isoformat()
    summary = (
        "auto: no changes; llm: consolidated 1 into 1\n"
        "archived 1 skill(s):\n"
        "  • old → new\n"
        "full report: hermes curator status"
    )
    _set_state(
        curator_env["curator"],
        last_run_at=now,
        last_run_summary=summary,
    )

    curator_env["main"]._print_curator_recent_run_notice()
    first = curator_env["capsys"].readouterr().out
    assert "old → new" in first

    curator_env["main"]._print_curator_recent_run_notice()
    second = curator_env["capsys"].readouterr().out
    assert second == "", "second call must be silent (already shown)"


def test_new_run_resets_show_once(curator_env):
    """A newer curator run with rename data prints again, even though one was already shown."""
    older = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    _set_state(
        curator_env["curator"],
        last_run_at=older,
        last_run_summary=(
            "auto: no changes; llm: consolidated 1 into 1\n"
            "archived 1 skill(s):\n"
            "  • thing-a → umbrella\n"
            "full report: hermes curator status"
        ),
    )
    curator_env["main"]._print_curator_recent_run_notice()
    curator_env["capsys"].readouterr()  # drain

    # New run lands.
    newer = datetime.now(timezone.utc).isoformat()
    _set_state(
        curator_env["curator"],
        last_run_at=newer,
        last_run_summary=(
            "auto: no changes; llm: consolidated 1 into 1\n"
            "archived 1 skill(s):\n"
            "  • thing-b → umbrella\n"
            "full report: hermes curator status"
        ),
    )
    curator_env["main"]._print_curator_recent_run_notice()
    out = curator_env["capsys"].readouterr().out
    assert "thing-b → umbrella" in out
    assert "thing-a" not in out  # only the newer run shows


def test_format_time_ago_buckets(curator_env):
    """Smoke test the time formatter — drives the `last run Xh ago` line."""
    fmt = curator_env["main"]._format_time_ago
    now = datetime.now(timezone.utc)
    assert fmt((now - timedelta(seconds=10)).isoformat()) == "just now"
    assert fmt((now - timedelta(minutes=5)).isoformat()) == "5m ago"
    assert fmt((now - timedelta(hours=3)).isoformat()) == "3h ago"
    assert fmt((now - timedelta(days=2)).isoformat()) == "2d ago"
    assert fmt("not-a-real-iso-string") == "recently"
