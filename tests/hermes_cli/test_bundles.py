"""Tests for hermes_cli/bundles.py — the `hermes bundles` CLI subcommand."""

import argparse
from pathlib import Path

import pytest

from hermes_cli.bundles import (
    bundles_command,
    register_cli,
)


@pytest.fixture
def bundles_env(tmp_path, monkeypatch):
    bundles_dir = tmp_path / "skill-bundles"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    # Reset module-level cache between tests.
    import agent.skill_bundles as mod
    mod._bundles_cache = {}
    mod._bundles_cache_mtime = None
    return bundles_dir, skills_dir


def _make_skill(skills_dir: Path, name: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _configure_protected_governance(home: Path) -> None:
    (home / "governance").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """\
skills:
  governance:
    registry_path: governance/skills-registry.yaml
    task_class: ardyn_engineering
    protected_task_classes:
      - ardyn_engineering
""",
        encoding="utf-8",
    )
    (home / "governance" / "skills-registry.yaml").write_text(
        """\
version: 1
skills:
  - name: SafeSkill
    classification: CURRENT
  - name: ToolTrust
    classification: COMPATIBILITY_ONLY
""",
        encoding="utf-8",
    )


def _parse(argv):
    parser = argparse.ArgumentParser()
    register_cli(parser)
    return parser.parse_args(argv)


class TestBundlesCli:



    def test_create_refuses_overwrite(self, bundles_env, capsys):
        bundles_command(_parse(["create", "dup", "--skill", "s1"]))
        capsys.readouterr()
        with pytest.raises(SystemExit) as ei:
            bundles_command(_parse(["create", "dup", "--skill", "s2"]))
        assert ei.value.code == 1
        out = capsys.readouterr().out
        assert "already exists" in out.lower() or "--force" in out.lower()


    def test_create_requires_skills(self, bundles_env, capsys, monkeypatch):
        # Simulate user pressing Ctrl-D immediately at the interactive prompt.
        monkeypatch.setattr("builtins.input", lambda *_a, **_kw: (_ for _ in ()).throw(EOFError()))
        with pytest.raises(SystemExit):
            bundles_command(_parse(["create", "empty"]))

    def test_list_hides_blocked_bundle_when_governance_protected(self, bundles_env, capsys, monkeypatch):
        bundles_dir, skills_dir = bundles_env
        home = bundles_dir.parent / "home"
        _configure_protected_governance(home)
        _make_skill(skills_dir, "SafeSkill")
        _make_skill(skills_dir, "ToolTrust")
        bundles_command(_parse(["create", "safe-pack", "--skill", "SafeSkill"]))
        capsys.readouterr()
        bundles_command(_parse(["create", "blocked-pack", "--skill", "ToolTrust"]))
        capsys.readouterr()

        monkeypatch.setenv("HERMES_HOME", str(home))
        bundles_command(_parse(["list"]))

        out = capsys.readouterr().out
        assert "/safe-pack" in out
        assert "/blocked-pack" not in out

    def test_list_hides_bundles_when_governance_config_is_malformed(self, bundles_env, capsys, monkeypatch):
        bundles_dir, skills_dir = bundles_env
        home = bundles_dir.parent / "home"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("skills:\n  governance: []\n", encoding="utf-8")
        _make_skill(skills_dir, "SafeSkill")
        bundles_command(_parse(["create", "safe-pack", "--skill", "SafeSkill"]))
        capsys.readouterr()

        monkeypatch.setenv("HERMES_HOME", str(home))
        bundles_command(_parse(["list"]))

        out = capsys.readouterr().out
        assert "/safe-pack" not in out
        assert "No bundles installed yet" in out

    def test_list_preserves_unprotected_behavior(self, bundles_env, capsys, monkeypatch):
        bundles_dir, skills_dir = bundles_env
        home = bundles_dir.parent / "home"
        _make_skill(skills_dir, "SafeSkill")
        bundles_command(_parse(["create", "safe-pack", "--skill", "SafeSkill"]))
        capsys.readouterr()

        monkeypatch.setenv("HERMES_HOME", str(home))
        bundles_command(_parse(["list"]))

        out = capsys.readouterr().out
        assert "/safe-pack" in out

