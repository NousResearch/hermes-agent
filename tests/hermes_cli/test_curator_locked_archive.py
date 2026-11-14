"""A user skill lock freezes a skill against EVERY write path, but the manual curator paths
(``hermes curator archive``, ``hermes curator prune``) only honoured ``pinned``: a locked skill
could still be relocated into ``.archive/`` by the agent through the terminal tool. The refusal
belongs at the mutation (``skill_usage.archive_skill``), which every caller shares — the CLI
commands, ``prune``, ``skill_manage``'s delete and the learning mutations.

The ``lock_env`` fixture mirrors ``tests/hermes_cli/test_curator_lock.py`` (a module-level import
of another test module's FIXTURE trips the repo health gate as F811), reduced to what these cases
need: an isolated profile home with reloaded modules and skill roots pinned to it.
"""

from __future__ import annotations

import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_skill(skills_dir: Path, name: str) -> Path:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\n# {name}\n", encoding="utf-8")
    return d


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    importlib.reload(hermes_constants)
    from tools import skill_usage
    importlib.reload(skill_usage)
    from agent import curator
    importlib.reload(curator)
    monkeypatch.setattr(curator, "_load_config", dict)
    monkeypatch.setattr(skill_usage, "_prune_builtins_enabled", lambda: False)
    from hermes_cli import curator as curator_cli
    importlib.reload(curator_cli)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills])
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills)
    return {"home": home, "skills": skills, "usage": skill_usage, "cli": curator_cli}


def _run(cli, *argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.cli_main(list(argv))
    return code, buf.getvalue()


def test_archive_skill_refuses_a_locked_skill(lock_env):
    usage = lock_env["usage"]
    skill_dir = _make_skill(lock_env["skills"], "coding-standards")
    usage.record_created("coding-standards", agent_created=True)
    assert usage.set_locked("coding-standards", True)

    ok, msg = usage.archive_skill("coding-standards")

    assert ok is False, msg
    assert "locked" in msg.lower()
    assert skill_dir.is_dir()
    assert not (lock_env["skills"] / ".archive" / "coding-standards").exists()


def test_archive_skill_still_archives_the_same_skill_once_unlocked(lock_env):
    usage = lock_env["usage"]
    _make_skill(lock_env["skills"], "coding-standards")
    usage.record_created("coding-standards", agent_created=True)
    assert usage.set_locked("coding-standards", True)
    assert usage.set_locked("coding-standards", False)

    ok, msg = usage.archive_skill("coding-standards")

    assert ok is True, msg
    assert (lock_env["skills"] / ".archive" / "coding-standards").is_dir()


def test_curator_archive_cli_refuses_a_locked_skill(lock_env):
    env = lock_env
    skill_dir = _make_skill(env["skills"], "coding-standards")
    env["usage"].record_created("coding-standards", agent_created=True)
    assert env["usage"].set_locked("coding-standards", True)

    code, out = _run(env["cli"], "archive", "coding-standards")

    assert code == 1, out
    assert "locked" in out.lower()
    assert "hermes curator unlock coding-standards" in out
    assert skill_dir.is_dir()


def test_prune_skips_locked_skills_and_still_archives_unlocked_ones(lock_env):
    env = lock_env
    usage = env["usage"]
    _make_skill(env["skills"], "frozen-helper")
    _make_skill(env["skills"], "idle-helper")
    old = "2020-01-01T00:00:00+00:00"
    for name in ("frozen-helper", "idle-helper"):
        usage.record_created(name, agent_created=True)
        data = usage.load_usage()
        # `last_activity_at` is DERIVED from the last use/view/patch stamp, so an idleness fixture
        # has to age the source field, not the derived one.
        data[name]["last_used_at"] = old
        assert usage.save_usage(data)
    assert usage.set_locked("frozen-helper", True)

    code, out = _run(env["cli"], "prune", "--days", "1", "--yes")

    assert code == 0, out
    assert "frozen-helper" in out
    assert "locked" in out.lower()
    assert (env["skills"] / "frozen-helper" / "SKILL.md").is_file()
    assert (env["skills"] / ".archive" / "idle-helper" / "SKILL.md").is_file()
    assert not (env["skills"] / ".archive" / "frozen-helper").exists()


def test_journey_delete_refuses_a_locked_skill(lock_env):
    """``hermes journey delete`` (the Desktop/TUI overlay) funnels through ``delete_node`` →
    ``archive_skill``, so it is the third route the lock now freezes. Nothing else pins it."""
    env = lock_env
    skill_dir = _make_skill(env["skills"], "coding-standards")
    env["usage"].record_created("coding-standards", agent_created=True)
    assert env["usage"].set_locked("coding-standards", True)

    from agent import learning_mutations

    result = learning_mutations.delete_node("coding-standards")

    assert result["ok"] is False, result
    assert "hermes curator unlock coding-standards" in result["message"]
    assert skill_dir.is_dir()
    assert not (env["skills"] / ".archive" / "coding-standards").exists()


def test_prune_reports_nothing_to_prune_when_every_candidate_is_locked(lock_env):
    env = lock_env
    usage = env["usage"]
    _make_skill(env["skills"], "frozen-helper")
    usage.record_created("frozen-helper", agent_created=True)
    data = usage.load_usage()
    data["frozen-helper"]["last_used_at"] = "2020-01-01T00:00:00+00:00"
    assert usage.save_usage(data)
    assert usage.set_locked("frozen-helper", True)

    code, out = _run(env["cli"], "prune", "--days", "1", "--yes")

    assert code == 0, out
    assert "nothing to prune" in out
    assert (env["skills"] / "frozen-helper" / "SKILL.md").is_file()
    sidecar = json.loads((env["skills"] / ".usage.json").read_text(encoding="utf-8"))
    assert sidecar["frozen-helper"]["locked"] is True
