import pytest

from agent.beta.specialists import SpecialistManifestError, SpecialistRegistry


def test_packaged_registry_contains_initial_specialists():
    registry = SpecialistRegistry.load()
    assert {specialist.id for specialist in registry.enabled()} == {
        "infra",
        "dba",
        "security",
        "monitoring",
        "dev",
        "devops",
        "memory",
        "qa-auditor",
    }
    assert registry.get("dba").memory_scope == "specialist:dba"


def test_invalid_manifest_has_clear_error(tmp_path):
    path = tmp_path / "specialists.yaml"
    path.write_text("specialists:\n  - id: Invalid ID\n", encoding="utf-8")
    with pytest.raises(SpecialistManifestError, match="invalid specialist at index 0"):
        SpecialistRegistry.load(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "specialists.yaml"
    manifest = """specialists:
  - &base
    id: dba
    name: DBA
    description: Database specialist
    capabilities: [database]
    keywords: [sql]
  - <<: *base
"""
    path.write_text(manifest, encoding="utf-8")
    with pytest.raises(SpecialistManifestError, match="duplicate specialist id: dba"):
        SpecialistRegistry.load(path)


def test_disabled_specialists_are_not_discoverable(tmp_path):
    path = tmp_path / "specialists.yaml"
    path.write_text(
        """specialists:
  - id: dba
    name: DBA
    description: Database specialist
    capabilities: [database]
    keywords: [sql]
    enabled: false
""",
        encoding="utf-8",
    )
    registry = SpecialistRegistry.load(path)
    assert registry.get("dba") is not None
    assert registry.enabled() == ()

