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
    monkeypatch.setattr(skill_usage, "curated_report", lambda: [
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
    # Mark all three as agent-created so they enter the curator's catalog.
    # Under the provenance-marker semantics, skills must be explicitly opted
    # into curator management (normally via the background-review fork when
    # it creates a skill through skill_manage).
    for n in ("top-dog", "middling", "never-used"):
        env["skill_usage"].mark_agent_created(n)

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
    # Mark both as agent-created so the catalog lists them. No bumps.
    env["skill_usage"].mark_agent_created("a")
    env["skill_usage"].mark_agent_created("b")

    out = _capture_status(env["curator_cli"])

    # most-active section is hidden because the top is 0
    assert "most active (top 5):" not in out
    # least-active still renders — it's part of the catalog overview
    assert "least active (top 5):" in out


def test_status_no_skills_produces_clean_empty_output(curator_status_env):
    env = curator_status_env
    out = _capture_status(env["curator_cli"])
    assert "no curator-managed skills" in out
    # None of the ranking sections render
    assert "most active" not in out
    assert "least active" not in out


def test_status_marks_missing_last_report_path(monkeypatch, capsys, tmp_path):
    import agent.curator as curator_state
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    missing_report = tmp_path / "stale-report"
    monkeypatch.setattr(curator_state, "load_state", lambda: {
        "paused": False,
        "last_run_at": None,
        "last_run_summary": "auto: no changes",
        "run_count": 1,
        "last_report_path": str(missing_report),
    })
    monkeypatch.setattr(curator_state, "is_enabled", lambda: True)
    monkeypatch.setattr(curator_state, "get_interval_hours", lambda: 168)
    monkeypatch.setattr(curator_state, "get_stale_after_days", lambda: 30)
    monkeypatch.setattr(curator_state, "get_archive_after_days", lambda: 90)
    monkeypatch.setattr(skill_usage, "curated_report", lambda: [])

    assert curator_cli._cmd_status(SimpleNamespace()) == 0

    out = capsys.readouterr().out
    assert f"last report:    {missing_report} (missing)" in out


# ---------------------------------------------------------------------------
# Unmanaged blind spot + adopt verb
# ---------------------------------------------------------------------------

def test_status_surfaces_unmanaged_skills(curator_status_env):
    """A skill with no provenance marker is invisible to every automatic
    transition, so status must SAY so rather than reporting only the managed
    count — otherwise a large library looks fully curated while much of it is
    untouchable."""
    env = curator_status_env
    env["make_skill"]("managed-one")
    env["make_skill"]("unmanaged-one")
    env["skill_usage"].mark_agent_created("managed-one")

    out = _capture_status(env["curator_cli"])

    assert "unmanaged (no provenance marker): 1 total" in out
    assert "curator adopt" in out


def test_status_reports_unmanaged_even_with_no_managed_skills(curator_status_env):
    """The early 'no curator-managed skills' return must not swallow the
    unmanaged summary — that combination is exactly the confusing case."""
    env = curator_status_env
    env["make_skill"]("unmanaged-one")

    out = _capture_status(env["curator_cli"])

    assert "no curator-managed skills" in out
    assert "unmanaged (no provenance marker): 1 total" in out


def test_status_omits_unmanaged_section_when_none(curator_status_env):
    env = curator_status_env
    env["make_skill"]("managed-one")
    env["skill_usage"].mark_agent_created("managed-one")

    out = _capture_status(env["curator_cli"])

    assert "unmanaged (no provenance marker)" not in out


def test_adopt_names_a_skill(curator_status_env):
    env = curator_status_env
    env["make_skill"]("legacy-one")
    cli = env["curator_cli"]

    rc = cli._cmd_adopt(SimpleNamespace(
        skill=["legacy-one"], all_unmanaged=False, dry_run=False, yes=False,
    ))

    assert rc == 0
    assert env["skill_usage"].get_record("legacy-one").get("created_by") == "agent"


def test_adopt_dry_run_writes_nothing(curator_status_env):
    env = curator_status_env
    env["make_skill"]("legacy-one")
    cli = env["curator_cli"]

    rc = cli._cmd_adopt(SimpleNamespace(
        skill=[], all_unmanaged=True, dry_run=True, yes=False,
    ))

    assert rc == 0
    assert env["skill_usage"].get_record("legacy-one").get("created_by") != "agent"


def test_adopt_all_unmanaged_requires_confirmation(curator_status_env, monkeypatch):
    """Bulk adoption makes skills archivable, so it must confirm by default."""
    env = curator_status_env
    env["make_skill"]("legacy-one")
    cli = env["curator_cli"]
    monkeypatch.setattr("builtins.input", lambda *_a: "n")

    rc = cli._cmd_adopt(SimpleNamespace(
        skill=[], all_unmanaged=True, dry_run=False, yes=False,
    ))

    assert rc == 1
    assert env["skill_usage"].get_record("legacy-one").get("created_by") != "agent"


def test_adopt_all_unmanaged_with_yes_skips_prompt(curator_status_env):
    env = curator_status_env
    env["make_skill"]("legacy-one")
    env["make_skill"]("legacy-two")
    cli = env["curator_cli"]

    rc = cli._cmd_adopt(SimpleNamespace(
        skill=[], all_unmanaged=True, dry_run=False, yes=True,
    ))

    assert rc == 0
    for n in ("legacy-one", "legacy-two"):
        assert env["skill_usage"].get_record(n).get("created_by") == "agent"


def test_adopt_rejects_names_combined_with_all_unmanaged(curator_status_env):
    env = curator_status_env
    env["make_skill"]("legacy-one")
    cli = env["curator_cli"]

    rc = cli._cmd_adopt(SimpleNamespace(
        skill=["legacy-one"], all_unmanaged=True, dry_run=False, yes=True,
    ))

    assert rc == 1


def test_adopt_with_no_target_is_an_error(curator_status_env):
    cli = curator_status_env["curator_cli"]

    rc = cli._cmd_adopt(SimpleNamespace(
        skill=[], all_unmanaged=False, dry_run=False, yes=False,
    ))

    assert rc == 1


def test_adopt_subcommand_is_registered():
    """The verb must be reachable through the real argparse tree, not just as a
    callable — a handler nobody can dispatch to is dead code."""
    import argparse

    import hermes_cli.curator as curator_cli

    parser = argparse.ArgumentParser()
    curator_cli.register_cli(parser)

    args = parser.parse_args(["adopt", "--all-unmanaged", "--dry-run"])
    assert args.func is curator_cli._cmd_adopt
    assert args.all_unmanaged is True
    assert args.dry_run is True
    assert args.skill == []

    named = parser.parse_args(["adopt", "alpha", "beta"])
    assert named.skill == ["alpha", "beta"]
    assert named.all_unmanaged is False


def test_list_unmanaged_subcommand_is_registered():
    import argparse

    import hermes_cli.curator as curator_cli

    parser = argparse.ArgumentParser()
    curator_cli.register_cli(parser)
    args = parser.parse_args(["list-unmanaged"])
    assert args.func is curator_cli._cmd_list_unmanaged


def test_list_unmanaged_itemizes_and_explains(curator_status_env):
    """`status` gives the count; this gives the names plus WHY each is
    unmanaged, so the user can decide what to adopt."""
    env = curator_status_env
    env["make_skill"]("legacy-one")
    env["make_skill"]("managed-one")
    env["skill_usage"].mark_agent_created("managed-one")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = env["curator_cli"]._cmd_list_unmanaged(Namespace())
    out = buf.getvalue()

    assert rc == 0
    assert "legacy-one" in out
    assert "managed-one" not in out
    assert "no marker" in out or "created_by:null" in out
    assert "curator adopt" in out


def test_list_unmanaged_reports_clean_state(curator_status_env):
    env = curator_status_env
    env["make_skill"]("managed-one")
    env["skill_usage"].mark_agent_created("managed-one")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = env["curator_cli"]._cmd_list_unmanaged(Namespace())

    assert rc == 0
    assert "no unmanaged skills" in buf.getvalue()

