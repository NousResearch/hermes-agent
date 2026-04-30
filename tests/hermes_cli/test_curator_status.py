"""Tests for `hermes curator status` output.

Covers:
- y0shualee's "least recently active" semantic (view/patch/use all count as activity).
- The most-used / least-used rankings by activity_count so users can see which
  skills actually get exercised.
"""

from __future__ import annotations

import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_status_uses_last_activity_not_only_last_used(monkeypatch, capsys):
    import agent.curator as curator_state
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(curator_state, "load_state", lambda: {
        "paused": False,
        "last_run_at": None,
        "last_run_summary": "(none)",
        "run_count": 0,
    })
    monkeypatch.setattr(curator_state, "is_enabled", lambda: True)
    monkeypatch.setattr(curator_state, "get_interval_hours", lambda: 168)
    monkeypatch.setattr(curator_state, "get_stale_after_days", lambda: 30)
    monkeypatch.setattr(curator_state, "get_archive_after_days", lambda: 90)
    monkeypatch.setattr(skill_usage, "agent_created_report", lambda: [
        {
            "name": "recently-viewed",
            "state": "active",
            "pinned": False,
            "use_count": 0,
            "view_count": 3,
            "patch_count": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_used_at": None,
            "last_viewed_at": "2026-04-30T10:00:00+00:00",
            "last_patched_at": "2026-04-30T11:00:00+00:00",
            "last_activity_at": "2026-04-30T11:00:00+00:00",
            "activity_count": 4,
        }
    ])

    assert curator_cli._cmd_status(SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "least recently active" in out
    assert "activity=  4" in out
    assert "last_activity=never" not in out
    assert "last_used=never" not in out


@pytest.fixture
def curator_status_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with real agent-created skills on disk."""
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    from tools import skill_usage
    importlib.reload(skill_usage)
    from agent import curator
    importlib.reload(curator)
    from hermes_cli import curator as curator_cli
    importlib.reload(curator_cli)

    def _write_skill(name: str) -> None:
        d = skills / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\n"
            f"name: {name}\n"
            "description: test\n"
            "version: 1.0.0\n"
            "metadata:\n"
            "  hermes:\n"
            "    agent_created: true\n"
            "---\n"
            f"# {name}\n"
        )

    return {
        "home": home,
        "skills": skills,
        "make_skill": _write_skill,
        "skill_usage": skill_usage,
        "curator_cli": curator_cli,
    }


def _capture_status(curator_cli) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = curator_cli._cmd_status(Namespace())
    assert rc == 0
    return buf.getvalue()


def test_status_shows_most_and_least_used_sections(curator_status_env):
    env = curator_status_env
    env["make_skill"]("top-dog")
    env["make_skill"]("middling")
    env["make_skill"]("never-used")

    # Bump use_count differentially. All three counters (use/view/patch) feed
    # into activity_count, so bumping use alone is enough to make activity
    # diverge between skills.
    for _ in range(10):
        env["skill_usage"].bump_use("top-dog")
    for _ in range(2):
        env["skill_usage"].bump_use("middling")

    out = _capture_status(env["curator_cli"])

    # Both new sections present
    assert "most active (top 5):" in out
    assert "least active (top 5):" in out
    # y0shualee's section preserved
    assert "least recently active (top 5):" in out

    # most-active lists top-dog FIRST (highest activity_count)
    most_section = out.split("most active (top 5):")[1].split("\n\n")[0]
    top_line = most_section.strip().split("\n")[0]
    assert "top-dog" in top_line
    assert "activity= 10" in top_line

    # least-active lists never-used FIRST (activity=0)
    least_section = out.split("least active (top 5):")[1].split("\n\n")[0]
    bottom_line = least_section.strip().split("\n")[0]
    assert "never-used" in bottom_line
    assert "activity=  0" in bottom_line


def test_status_hides_most_active_when_all_zero(curator_status_env):
    """If no skills have any activity, skip the most-active block — it's noise.
    Least-active still shows so the user sees their catalog."""
    env = curator_status_env
    env["make_skill"]("a")
    env["make_skill"]("b")
    # No bumps.

    out = _capture_status(env["curator_cli"])

    # most-active section is hidden because the top is 0
    assert "most active (top 5):" not in out
    # least-active still renders — it's part of the catalog overview
    assert "least active (top 5):" in out


def test_status_no_skills_produces_clean_empty_output(curator_status_env):
    env = curator_status_env
    out = _capture_status(env["curator_cli"])
    assert "no agent-created skills" in out
    # None of the ranking sections render
    assert "most active" not in out
    assert "least active" not in out
