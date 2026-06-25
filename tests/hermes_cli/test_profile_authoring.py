"""Tests for profile distribution authoring helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.profile_authoring import (
    scaffold_distribution,
    slugify,
    validate_distribution,
)
from hermes_cli.profile_distribution import read_manifest


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


class TestScaffoldDistribution:
    def test_scaffold_creates_valid_distribution(self, profile_env):
        out = profile_env / "sql-reviewer"
        scaffold_distribution(
            out,
            name="SQL Reviewer",
            description="Reviews SQL migrations and writes rollback checklists.",
            author="Example Team",
        )

        manifest = read_manifest(out)
        assert manifest is not None
        assert manifest.name == "sql-reviewer"
        assert manifest.description == "Reviews SQL migrations and writes rollback checklists."
        assert (out / "SOUL.md").is_file()
        assert (out / "config.yaml").is_file()
        assert (out / "mcp.json").is_file()
        assert (out / ".env.EXAMPLE").is_file()
        assert (out / "templates" / "profile.params.yaml").is_file()
        assert validate_distribution(out).ok

    def test_scaffold_from_params_file(self, profile_env):
        params = profile_env / "params.yaml"
        params.write_text(
            "name: migration-guard\n"
            "display_name: Migration Guard\n"
            "description: Flags destructive database migrations.\n"
            "env_requires:\n"
            "  - name: DATABASE_URL\n"
            "    description: Optional database URL.\n"
            "    required: false\n",
            encoding="utf-8",
        )
        out = profile_env / "out"
        scaffold_distribution(out, params_file=params)

        manifest = read_manifest(out)
        assert manifest is not None
        assert manifest.name == "migration-guard"
        assert manifest.env_requires[0].name == "DATABASE_URL"
        assert "DATABASE_URL" in (out / ".env.EXAMPLE").read_text(encoding="utf-8")
        assert validate_distribution(out).ok

    def test_scaffold_refuses_existing_output_without_force(self, profile_env):
        out = profile_env / "existing"
        out.mkdir()
        with pytest.raises(FileExistsError):
            scaffold_distribution(out, name="x", description="desc")


class TestValidateDistribution:
    def test_validate_rejects_missing_manifest(self, tmp_path):
        result = validate_distribution(tmp_path)
        assert not result.ok
        assert any("missing required file: distribution.yaml" in err for err in result.errors)

    def test_validate_rejects_runtime_files(self, profile_env):
        out = profile_env / "profile"
        scaffold_distribution(out, name="runtime-test", description="Runtime test profile.")
        (out / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (out / "sessions").mkdir()
        (out / "sessions" / "session.json").write_text("{}\n", encoding="utf-8")

        result = validate_distribution(out)
        assert not result.ok
        assert any(".env" in err for err in result.errors)
        assert any("sessions" in err for err in result.errors)

    def test_validate_rejects_skill_without_frontmatter(self, profile_env):
        out = profile_env / "profile"
        scaffold_distribution(out, name="skill-test", description="Skill test profile.")
        skill = out / "skills" / "bad" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("# Missing frontmatter\n", encoding="utf-8")

        result = validate_distribution(out)
        assert not result.ok
        assert any("frontmatter" in err for err in result.errors)


class TestSlugify:
    def test_slugify_normalizes_names(self):
        assert slugify("SQL Migration Reviewer") == "sql-migration-reviewer"
        assert slugify("  A__B  ") == "a-b"

    def test_slugify_rejects_empty(self):
        with pytest.raises(ValueError):
            slugify("!!!")
