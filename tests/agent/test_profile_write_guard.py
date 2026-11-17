"""Opt-in per-profile self-modification guard (4e)."""
from __future__ import annotations

import json

import pytest

from agent import profile_write_guard as pwg
from agent.file_safety import get_write_denied_error, is_write_denied
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _home(tmp_path, name, guard_yaml):
    home = tmp_path / name
    ext = tmp_path / f"{name}-ext-skills"
    (ext / "kb-skill").mkdir(parents=True)
    (ext / "kb-skill" / "SKILL.md").write_text(
        "---\nname: kb-skill\ndescription: Use when x. y.\n---\nbody\n", encoding="utf-8"
    )
    (home / "skills" / "own-skill").mkdir(parents=True)
    (home / "skills" / "own-skill" / "SKILL.md").write_text(
        "---\nname: own-skill\ndescription: Use when x. y.\n---\nbody\n", encoding="utf-8"
    )
    (home / "config.yaml").write_text(
        f"{guard_yaml}skills:\n  external_dirs: [{ext}]\n", encoding="utf-8"
    )
    return home, ext


class _As:
    def __init__(self, home):
        self.home = home

    def __enter__(self):
        self.tok = set_hermes_home_override(str(self.home))

    def __exit__(self, *a):
        reset_hermes_home_override(self.tok)


@pytest.fixture()
def guarded(tmp_path, monkeypatch):
    home, ext = _home(
        tmp_path,
        "educator",
        "agent_write_guard:\n  enabled: true\n  extra_paths: [kb/locked, "
        + str(tmp_path / "abs-locked")
        + "]\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, ext, tmp_path


@pytest.mark.parametrize(
    "rel",
    ["config.yaml", "SOUL.md", "cron/jobs.json", "scripts/run.py", ".env", ".env.local", "kb/locked/a.md"],
)
def test_opted_in_profile_denies_protected_targets(guarded, rel):
    home, _, _ = guarded
    with _As(home):
        assert is_write_denied(str(home / rel))
        assert "agent_write_guard" in get_write_denied_error(str(home / rel))


def test_external_dirs_and_absolute_extra_paths_denied(guarded):
    home, ext, tmp = guarded
    with _As(home):
        assert is_write_denied(str(ext / "kb-skill" / "SKILL.md"))
        assert is_write_denied(str(tmp / "abs-locked" / "x.txt"))


def test_own_skills_and_workspace_allowed(guarded):
    home, _, tmp = guarded
    with _As(home):
        assert not is_write_denied(str(home / "skills" / "own-skill" / "SKILL.md"))
        assert not is_write_denied(str(home / "memories" / "MEMORY.md"))
        assert not is_write_denied(str(tmp / "workspace" / "draft.md"))


def test_bool_form_opts_in(tmp_path):
    home, _ = _home(tmp_path, "p", "agent_write_guard: true\n")
    with _As(home):
        assert is_write_denied(str(home / "SOUL.md"))


def test_not_opted_in_is_unchanged(tmp_path):
    home, ext = _home(tmp_path, "default", "")
    with _As(home):
        assert pwg.protected_roots() is None
        assert not is_write_denied(str(home / "SOUL.md"))
        assert not is_write_denied(str(ext / "kb-skill" / "SKILL.md"))


def test_guard_follows_owning_profile_not_process(tmp_path, monkeypatch):
    guarded_home, _ = _home(tmp_path, "educator", "agent_write_guard: true\n")
    plain_home, _ = _home(tmp_path, "default", "")
    monkeypatch.setenv("HERMES_HOME", str(guarded_home))
    with _As(plain_home):
        assert not is_write_denied(str(plain_home / "SOUL.md"))
    with _As(guarded_home):
        assert is_write_denied(str(guarded_home / "SOUL.md"))


def test_skill_manage_refuses_external_skill_patch(guarded):
    home, _, _ = guarded
    from tools.skill_manager_tool import skill_manage

    with _As(home):
        out = json.loads(
            skill_manage(
                operations=[{"name": "kb-skill", "action": "patch", "old_string": "body", "new_string": "b2"}]
            )
        )
        assert "agent_write_guard" in (out.get("error") or "")
        assert "error" not in json.loads(
            skill_manage(
                operations=[{"name": "own-skill", "action": "patch", "old_string": "body", "new_string": "b2"}]
            )
        )


def test_denial_audited(guarded):
    home, _, _ = guarded
    from agent import multiplex_file_guard as mfg

    mfg._recent_audits.clear()
    with _As(home):
        get_write_denied_error(str(home / "SOUL.md"))
        log = home / "audit" / "data-access.log"
        lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert any(l["action"] == "write_guard_denied" for l in lines)
