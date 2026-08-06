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




# ---------------------------------------------------------------------------
# Unmanaged blind spot + adopt verb
# ---------------------------------------------------------------------------







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


