"""A catalog identifier must inspect the same official skill that it fetches."""

import pytest

from hermes_cli import skills_hub
from tools.skills_hub_official import OptionalSkillSource


@pytest.mark.parametrize("name", ["shared", "display-name"])
def test_official_catalog_selection_preserves_metadata_and_preview(tmp_path, monkeypatch, name):
    root = tmp_path / "optional-skills"
    for category in ("alpha", "beta"):
        directory = root / category / "shared"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {category} description\n---\n\n{category} body\n",
            encoding="utf-8",
        )
    source = OptionalSkillSource()
    source._optional_dir = root
    source._remote_dirs = {}
    monkeypatch.setattr(skills_hub, "_sources", lambda: [source])

    for entry in source.list_local():
        result = skills_hub.inspect_skill(entry.identifier)
        assert result is not None
        assert result["identifier"] == entry.identifier
        assert result["name"] == entry.name
        assert result["description"] == entry.description
        category = entry.identifier.split("/")[1]
        assert f"{category} body" in result["skill_md_preview"]


def test_remote_catalog_identifier_resolves_even_when_leaf_name_is_ambiguous(tmp_path):
    source = OptionalSkillSource()
    source._optional_dir = tmp_path / "optional-skills"
    source._remote_dirs = {"alpha/shared": True, "beta/shared": True, "gamma/unique": True}

    for relative in source._remote_dirs:
        meta = source.inspect(f"official/{relative}")
        assert meta is not None
        assert meta.identifier == f"official/{relative}"
        assert meta.path == f"optional-skills/{relative}"
    assert source.inspect("official/shared") is None
    assert source.inspect("official/unique").identifier == "official/gamma/unique"
