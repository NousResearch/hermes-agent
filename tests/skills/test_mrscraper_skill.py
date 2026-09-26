"""The optional MrScraper skill can be discovered and loaded by Hermes."""

import json
from pathlib import Path

from tools.skills_hub_official import OptionalSkillSource
from tools.skills_tool import skill_view


REPO = Path(__file__).resolve().parents[2]


def test_optional_mrscraper_skill_discovery_and_load(tmp_path, monkeypatch):
    source = OptionalSkillSource()
    source._optional_dir = REPO / "optional-skills"
    source._remote_dirs = {}  # Exercise the shipped bundle without a live-repo lookup.

    matches = source.search("mrscraper")
    assert any(item.identifier == "official/research/mrscraper" for item in matches)

    bundle = source.fetch("official/research/mrscraper")
    assert bundle is not None
    assert bundle.name == "mrscraper"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    destination = tmp_path / "skills" / bundle.name
    destination.mkdir(parents=True)
    for name, data in bundle.files.items():
        (destination / name).write_bytes(data if isinstance(data, bytes) else data.encode())

    loaded = json.loads(skill_view(bundle.name))
    assert loaded["success"], loaded
    assert "MrScraper" in loaded["content"]
