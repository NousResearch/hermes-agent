"""Regression for #128799: PM snapshots retain skills for their real consumers."""
from pathlib import Path

import pytest

from pm import workspace
from tools import skills_hub_official, skills_sync


def _source(tmp_path, include):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\n'
        f'[tool.setuptools.packages.find]\ninclude={include}\n', encoding="utf-8",
    )
    return source


def _skill(root, name="fixture"):
    directory = root / "testing" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f'---\nname: {name}\ndescription: Local fixture.\n---\n# Fixture\n', encoding="utf-8",
    )
    (directory / "references").mkdir()
    (directory / "references" / "guide.md").write_text("Local guide", encoding="utf-8")
    return directory


@pytest.mark.parametrize("include", ['["pm"]', '["*"]'])
@pytest.mark.parametrize("resource", ["skills", "optional-skills"])
def test_snapshot_preserves_skill_consumers(tmp_path, monkeypatch, include, resource):
    source = _source(tmp_path, include)
    skill = _skill(source / resource)
    (skill / ".env").write_text("FIXTURE_ONLY=1", encoding="utf-8")
    for name in (".git", "node_modules", "__pycache__"):
        (skill / name).mkdir()
        (skill / name / "ignored.txt").write_text("cache", encoding="utf-8")
    monkeypatch.delenv("HERMES_BUNDLED_SKILLS", raising=False)
    monkeypatch.delenv("HERMES_OPTIONAL_SKILLS", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    destination = tmp_path / "workspace"
    workspace._generate_pyproject([], destination, source=source)

    if resource == "optional-skills":
        monkeypatch.setattr(skills_hub_official, "__file__", str(source / "tools/skills_hub_official.py"))
        expected = skills_hub_official.OptionalSkillSource().list_local()
        assert [meta.name for meta in expected] == ["fixture"]
        monkeypatch.setattr(skills_hub_official, "__file__", str(destination / "tools/skills_hub_official.py"))
        catalog = skills_hub_official.OptionalSkillSource()
        assert catalog.list_local() == expected
        # Locally shipped skills must not require a remote tree/download to fetch.
        def no_remote(*args, **kwargs):
            pytest.fail("shipped skill incorrectly fell back to the network")
        monkeypatch.setattr(catalog, "_fetch_from_live_repo", no_remote)
        bundle = catalog.fetch("official/testing/fixture")
        assert bundle is not None
        assert bundle.files["SKILL.md"] == (skill / "SKILL.md").read_bytes()
        assert bundle.files[str(Path("references/guide.md"))] == b"Local guide"
        # Missing local skills retain the existing live-repo fallback.
        sentinel = object()
        monkeypatch.setattr(catalog, "_fetch_from_live_repo", lambda rel: sentinel)
        assert catalog.fetch("official/testing/not-shipped") is sentinel
        override = tmp_path / "override"
        _skill(override, "override")
        monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(override))
        assert [m.name for m in skills_hub_official.OptionalSkillSource().list_local()] == ["override"]
    else:
        monkeypatch.setattr(skills_sync, "__file__", str(destination / "tools/skills_sync.py"))
        result = skills_sync.sync_skills(quiet=True)
        assert result["copied"] == ["fixture"]
        installed = home / "skills/testing/fixture/SKILL.md"
        assert installed.read_bytes() == (skill / "SKILL.md").read_bytes()
        assert installed.with_name("references").joinpath("guide.md").read_text() == "Local guide"
        installed.write_text("User customization", encoding="utf-8")
        assert skills_sync.sync_skills(quiet=True)["copied"] == []
        assert installed.read_text() == "User customization"
        override = tmp_path / "override"
        _skill(override, "override")
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(override))
        assert skills_sync.sync_skills(quiet=True)["copied"] == ["override"]
        assert installed.read_text() == "User customization"

    copied = destination / skill.relative_to(source)
    for name in (".env", ".git", "node_modules", "__pycache__"):
        assert not (copied / name).exists()


@pytest.mark.parametrize("resource", ["skills", "optional-skills"])
@pytest.mark.parametrize("present", [False, True])
def test_skill_resource_absence_and_copy_failure(tmp_path, monkeypatch, resource, present):
    source = _source(tmp_path, '["pm"]')
    destination = tmp_path / "workspace"
    if present:
        _skill(source / resource)
        def unreadable(*args, **kwargs):
            raise PermissionError("skill resource unreadable")
        monkeypatch.setattr(workspace.shutil, "copytree", unreadable)
        with pytest.raises(PermissionError, match="skill resource unreadable"):
            workspace._generate_pyproject([], destination, source=source)
    else:
        workspace._generate_pyproject([], destination, source=source)
        assert not (destination / resource).exists()
