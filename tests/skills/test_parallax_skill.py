"""Offline contracts for PARALLAX's optional bundle and runtime boundaries."""

import json
import socket
from pathlib import Path

import pytest


@pytest.fixture
def skill_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("Skill discovery and loading must work offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    return home


def stage_bundle(home):
    from tools.skills_hub_official import OptionalSkillSource

    source = OptionalSkillSource()
    matches = [entry for entry in source.list_local() if entry.name == "parallax"]
    assert len(matches) == 1
    bundle = source.fetch(matches[0].identifier)
    assert bundle is not None
    root = home / "skills" / "productivity" / bundle.name
    for relative, content in bundle.files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())
    return root, bundle


def test_optional_bundle_discovery_load_and_reference(skill_home):
    from tools.skills_tool import skill_view, skills_list

    before = json.loads(skills_list())
    assert before["success"]
    assert not any(item["name"] == "parallax" for item in before["skills"])
    root, bundle = stage_bundle(skill_home)
    listing = json.loads(skills_list(category="productivity"))
    assert listing["success"]
    entry = next(item for item in listing["skills"] if item["name"] == bundle.name)
    loaded = json.loads(skill_view(entry["name"]))
    assert loaded["success"]
    assert Path(loaded["skill_dir"]) == root
    assert loaded["description"] == entry["description"]
    assert loaded["content"] == (root / "SKILL.md").read_text(encoding="utf-8")
    references = loaded["linked_files"]["references"]
    assert references
    for relative in references:
        reference = json.loads(skill_view(entry["name"], file_path=relative))
        assert reference["success"]
        assert reference["content"] == (root / relative).read_text(encoding="utf-8")


def test_bundle_scans_safe_and_reference_reads_stay_contained(skill_home):
    from tools.skills_guard import scan_skill
    from tools.skills_tool import skill_view

    root, bundle = stage_bundle(skill_home)
    scan = scan_skill(root, source=bundle.source)
    assert scan.verdict == "safe", scan.summary
    secret = skill_home / "private.txt"
    secret.write_text("outside-skill-sentinel", encoding="utf-8")
    for relative in ("../../../private.txt", str(secret)):
        result = json.loads(skill_view(bundle.name, file_path=relative))
        assert not result["success"]
        assert "outside-skill-sentinel" not in json.dumps(result)
    # Rejection must not prevent subsequent legitimate reference access.
    reference = next(path for path in bundle.files if Path(path).parts[0] == "references")
    assert json.loads(skill_view(bundle.name, file_path=reference))["success"]
