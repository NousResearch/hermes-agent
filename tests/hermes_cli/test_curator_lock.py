"""`hermes curator lock|unlock` (#25083, #140332): a per-profile user declaration that the
autonomous passes may not change a skill's content. The flag lives beside ``pinned`` in the
active profile's ``skills/.usage.json``; ``curator status`` lists it."""

from __future__ import annotations

import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest


def _make_skill(skills_dir: Path, name: str) -> Path:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\n# {name}\n", encoding="utf-8")
    return d


@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with freshly reloaded modules (same shape as test_curator_pin_visibility)."""
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
    # `lock` resolves the name through skill_manage's finder; pin its search roots to this profile.
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills])
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills)
    return {"home": home, "skills": skills, "usage": skill_usage, "cli": curator_cli}


def _run(cli, *argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.cli_main(list(argv))
    return code, buf.getvalue()


def _sidecar(skills: Path) -> dict:
    path = skills / ".usage.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_lock_then_unlock_round_trip_on_a_hand_written_skill(lock_env):
    env = lock_env
    _make_skill(env["skills"], "coding-standards")

    code, out = _run(env["cli"], "lock", "coding-standards")
    assert code == 0, out
    assert "locked 'coding-standards'" in out
    assert "hermes curator unlock coding-standards" in out
    assert env["usage"].is_locked("coding-standards") is True
    rec = _sidecar(env["skills"])["coding-standards"]
    assert rec["locked"] is True
    assert rec["pinned"] is False  # the pin axis is untouched

    code, out = _run(env["cli"], "unlock", "coding-standards")
    assert code == 0, out
    assert "unlocked 'coding-standards'" in out
    assert env["usage"].is_locked("coding-standards") is False


def test_lock_refuses_a_name_that_does_not_resolve(lock_env):
    env = lock_env
    code, out = _run(env["cli"], "lock", "no-such-skill")
    assert code == 1
    assert "no skill named 'no-such-skill'" in out
    assert "no-such-skill" not in _sidecar(env["skills"])  # a typo never forges a record


def test_lock_resolves_a_categorized_path_and_records_the_directory_name(lock_env):
    env = lock_env
    _make_skill(env["skills"] / "ops", "deploy-runbook")
    code, out = _run(env["cli"], "lock", "ops/deploy-runbook")
    assert code == 0, out
    assert "locked 'deploy-runbook'" in out
    assert env["usage"].is_locked("deploy-runbook") is True


def test_status_lists_locked_skills(lock_env):
    env = lock_env
    _make_skill(env["skills"], "coding-standards")
    assert _run(env["cli"], "lock", "coding-standards")[0] == 0
    code, out = _run(env["cli"], "status")
    assert code == 0, out
    assert "locked (1): coding-standards" in out
