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


# ---------------------------------------------------------------------------
# Unmanaged blind spot + adopt verb
# ---------------------------------------------------------------------------


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


def test_pin_and_unpin_allow_bundled_when_prune_builtins_enabled(curator_status_env, monkeypatch, capsys):
    env = curator_status_env
    env["make_skill"]("bundled-one")
    (env["skills"] / ".bundled_manifest").write_text(
        "bundled-one:abc\n", encoding="utf-8",
    )
    monkeypatch.setattr(env["skill_usage"], "_prune_builtins_enabled", lambda: True)

    assert env["curator_cli"]._cmd_pin(Namespace(skill="bundled-one")) == 0
    assert env["skill_usage"].get_record("bundled-one")["pinned"] is True
    assert "pinned 'bundled-one'" in capsys.readouterr().out

    assert env["curator_cli"]._cmd_unpin(Namespace(skill="bundled-one")) == 0
    assert env["skill_usage"].get_record("bundled-one")["pinned"] is False
    assert "unpinned 'bundled-one'" in capsys.readouterr().out


def test_pin_still_refuses_hub_skill_even_when_prune_builtins_enabled(curator_status_env, monkeypatch, capsys):
    env = curator_status_env
    env["make_skill"]("hub-one")
    hub = env["skills"] / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text(
        '{"installed": {"hub-one": {}}}', encoding="utf-8",
    )
    monkeypatch.setattr(env["skill_usage"], "_prune_builtins_enabled", lambda: True)

    assert env["curator_cli"]._cmd_pin(Namespace(skill="hub-one")) == 1
    assert env["skill_usage"].load_usage() == {}
    out = capsys.readouterr().out
    assert "hub-one" in out
    assert "hub-installed skills are never curator-managed" in out


def test_pin_and_unpin_refuse_bundled_when_prune_builtins_disabled(curator_status_env, monkeypatch, capsys):
    env = curator_status_env
    env["make_skill"]("bundled-one")
    (env["skills"] / ".bundled_manifest").write_text(
        "bundled-one:abc\n", encoding="utf-8",
    )
    monkeypatch.setattr(env["skill_usage"], "_prune_builtins_enabled", lambda: False)

    assert env["curator_cli"]._cmd_pin(Namespace(skill="bundled-one")) == 1
    out = capsys.readouterr().out
    assert "cannot be pinned" in out
    assert "bundled built-ins require curator.prune_builtins=true" in out

    assert env["curator_cli"]._cmd_unpin(Namespace(skill="bundled-one")) == 1
    out = capsys.readouterr().out
    assert "cannot be unpinned" in out
    assert "bundled built-ins require curator.prune_builtins=true" in out
    assert env["skill_usage"].load_usage() == {}


@pytest.mark.parametrize("prune_builtins", [True, False])
@pytest.mark.parametrize("command", ["pin", "unpin"])
def test_hub_skill_is_refused_by_pin_and_unpin_either_way(curator_status_env, monkeypatch, capsys,
                                                          prune_builtins, command):
    env = curator_status_env
    env["make_skill"]("hub-one")
    hub = env["skills"] / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text(
        '{"installed": {"hub-one": {}}}', encoding="utf-8",
    )
    monkeypatch.setattr(env["skill_usage"], "_prune_builtins_enabled", lambda: prune_builtins)

    handler = env["curator_cli"]._cmd_pin if command == "pin" else env["curator_cli"]._cmd_unpin
    assert handler(Namespace(skill="hub-one")) == 1
    assert env["skill_usage"].load_usage() == {}
    out = capsys.readouterr().out
    assert "hub-installed skills are never curator-managed" in out


def test_pinning_a_bundled_skill_does_not_suggest_adopt(curator_status_env, monkeypatch, capsys):
    """`curator adopt` refuses bundled skills, so the unmanaged-skill hint must not point there."""
    env = curator_status_env
    env["make_skill"]("bundled-one")
    (env["skills"] / ".bundled_manifest").write_text(
        "bundled-one:abc\n", encoding="utf-8",
    )
    monkeypatch.setattr(env["skill_usage"], "_prune_builtins_enabled", lambda: True)

    assert env["curator_cli"]._cmd_pin(Namespace(skill="bundled-one")) == 0
    out = capsys.readouterr().out
    assert "will bypass auto-transitions" in out
    assert "adopt" not in out
