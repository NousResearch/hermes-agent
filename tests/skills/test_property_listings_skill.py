"""Property card recipes are optional, discoverable, and loadable on demand."""
import json
from pathlib import Path

from agent.prompt_builder import PLATFORM_HINTS
from tools.skills_hub_official import OptionalSkillSource


ROOT = Path(__file__).resolve().parents[2]


def test_property_recipe_is_not_paid_for_by_unrelated_desktop_sessions():
    hint = PLATFORM_HINTS["desktop"]
    assert "```listing" not in hint
    assert "MEDIA:" in hint and "::preview" in hint


def test_optional_catalog_fetch_preserves_a_usable_property_recipe(tmp_path, monkeypatch):
    source = OptionalSkillSource()
    source._optional_dir = ROOT / "optional-skills"
    matches = [m for m in source.list_local() if "property" in m.tags and "rental" in m.tags]
    assert matches, "Property tasks must be discoverable in the optional catalog"
    bundle = source.fetch(matches[0].identifier)
    assert bundle is not None
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    destination = tmp_path / "skills" / bundle.name
    destination.mkdir(parents=True)
    for name, data in bundle.files.items():
        (destination / name).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    loaded = json.loads(skill_view(bundle.name))
    assert loaded["success"], loaded
    content = loaded["content"]
    example = content.split("```listing\n", 1)[1].split("```", 1)[0]
    listing = json.loads(example)
    assert listing["address"] and listing["links"]
    assert {"price", "beds", "baths", "size", "note", "facts", "catches", "images"} <= listing.keys()
