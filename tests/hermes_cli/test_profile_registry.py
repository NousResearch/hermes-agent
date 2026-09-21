"""P3.0 tests — authoritative profile registry contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from hermes_cli import profile_registry as pr


VALID_REGISTRY = {
    "schema_version": 1,
    "root": {"name": "KENSEI", "description": "orchestrator"},
    "profiles": [
        {
            "name": "octacon",
            "kind": "lead",
            "parent": "KENSEI",
            "lifecycle": "active",
            "domains": ["coding"],
            "gateway_unit": "hermes-gateway-octacon",
        },
        {
            "name": "octacon-backend",
            "kind": "worker",
            "parent": "octacon",
            "lifecycle": "active",
            "domains": ["backend"],
            "gateway_unit": None,
        },
    ],
}


def _write_registry(tmp_path: Path, doc) -> str:
    path = tmp_path / "profile-registry.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")
    return str(path)


def test_valid_registry_loads(tmp_path):
    path = _write_registry(tmp_path, VALID_REGISTRY)
    reg = pr.load_registry(path)
    assert reg["root"]["name"] == "KENSEI"
    names = [p["name"] for p in reg["profiles"]]
    assert names == sorted(names)  # deterministic ordering
    assert {"octacon", "octacon-backend"} <= set(names)


def test_duplicate_name_rejected(tmp_path):
    doc = VALID_REGISTRY.copy()
    doc["profiles"] = VALID_REGISTRY["profiles"] + [dict(VALID_REGISTRY["profiles"][1])]
    path = _write_registry(tmp_path, doc)
    with pytest.raises(pr.RegistryError, match="duplicate"):
        pr.load_registry(path)


@pytest.mark.parametrize("parent", ["ghost", "KENSEI-TYPO"])
def test_unknown_parent_rejected(tmp_path, parent):
    doc = VALID_REGISTRY.copy()
    doc["profiles"] = [
        dict(VALID_REGISTRY["profiles"][0], parent=parent),
        dict(VALID_REGISTRY["profiles"][1]),
    ]
    path = _write_registry(tmp_path, doc)
    with pytest.raises(pr.RegistryError, match="unknown parent"):
        pr.load_registry(path)


def test_self_parent_rejected(tmp_path):
    doc = VALID_REGISTRY.copy()
    doc["profiles"] = [dict(VALID_REGISTRY["profiles"][0], parent="octacon")]
    path = _write_registry(tmp_path, doc)
    with pytest.raises(pr.RegistryError, match="own parent"):
        pr.load_registry(path)


def test_multi_node_cycle_rejected(tmp_path):
    doc = VALID_REGISTRY.copy()
    a = {"name": "a", "kind": "worker", "parent": "b", "lifecycle": "active",
         "domains": [], "gateway_unit": None}
    b = {"name": "b", "kind": "worker", "parent": "a", "lifecycle": "active",
         "domains": [], "gateway_unit": None}
    doc["profiles"] = [a, b]
    path = _write_registry(tmp_path, doc)
    with pytest.raises(pr.RegistryError, match="cycle"):
        pr.load_registry(path)


@pytest.mark.parametrize("field,value", [
    ("kind", "manager"),
    ("kind", None),
    ("lifecycle", "zombie"),
])
def test_invalid_enum_rejected(tmp_path, field, value):
    doc = VALID_REGISTRY.copy()
    doc["profiles"] = [dict(VALID_REGISTRY["profiles"][0], **{field: value})]
    path = _write_registry(tmp_path, doc)
    with pytest.raises(pr.RegistryError):
        pr.load_registry(path)


def test_missing_required_field_rejected(tmp_path):
    doc = VALID_REGISTRY.copy()
    broken = dict(VALID_REGISTRY["profiles"][0])
    del broken["lifecycle"]
    doc["profiles"] = [broken]
    path = _write_registry(tmp_path, doc)
    with pytest.raises(pr.RegistryError, match="missing field"):
        pr.load_registry(path)


def test_deterministic_parse_and_render(tmp_path):
    path_a = _write_registry(tmp_path, VALID_REGISTRY)
    reg_a = pr.load_registry(path_a)
    md_a = pr.render_registry_markdown(reg_a)
    reg_b = pr.load_registry(path_a)
    md_b = pr.render_registry_markdown(reg_b)
    assert md_a == md_b
    # Row order matches sorted profile order byte-for-byte across runs.
    assert md_a.splitlines() == md_b.splitlines()


def test_render_markdown_parity(tmp_path):
    reg = pr.load_registry(_write_registry(tmp_path, VALID_REGISTRY))
    md = pr.render_registry_markdown(reg)
    for profile in reg["profiles"]:
        row = f"| {profile['name']} | {profile['kind']} | "
        assert row in md
        assert profile["lifecycle"] in md
        parent = profile["parent"] or reg["root"]["name"]
        assert parent in md


def test_filesystem_discrepancy_report_no_mutation(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "octacon").mkdir()
    (profiles_dir / "rogue-agent").mkdir()  # unregistered
    reg = pr.load_registry(_write_registry(tmp_path, VALID_REGISTRY))
    before = sorted(p.name for p in profiles_dir.iterdir())
    discrepancies = pr.compare_with_filesystem(reg, profiles_dir)
    kinds = {(d["kind"], d["profile"]) for d in discrepancies}
    assert ("unregistered_profile", "rogue-agent") in kinds
    after = sorted(p.name for p in profiles_dir.iterdir())
    assert before == after  # read-only: no directory created/deleted


def test_gateway_discrepancy_report_no_mutation(tmp_path):
    reg = pr.load_registry(_write_registry(tmp_path, VALID_REGISTRY))
    # Inventory is missing the declared octacon unit and has an unmapped one.
    discrepancies = pr.compare_with_gateway_inventory(
        reg, ["hermes-gateway", "hermes-gateway-ghost"]
    )
    kinds = {(d["kind"], d["profile"]) for d in discrepancies}
    assert ("declared_unit_not_in_inventory", "octacon") in kinds
    assert ("unregistered_gateway_unit", "") in kinds


def test_unregistered_profiles_not_silently_attached(tmp_path):
    """The renderer must never attach unknown profiles to the root."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "stray").mkdir()
    reg = pr.load_registry(_write_registry(tmp_path, VALID_REGISTRY))
    discrepancies = pr.compare_with_filesystem(reg, profiles_dir)
    stray = [d for d in discrepancies if d["profile"] == "stray"]
    assert stray and stray[0]["kind"] == "unregistered_profile"


def test_missing_registry_file(tmp_path):
    with pytest.raises(pr.RegistryError, match="not found"):
        pr.load_registry(tmp_path / "does-not-exist.yaml")


def test_repo_registry_itself_is_valid():
    """The committed governance/profile-registry.yaml must be valid."""
    repo_root = Path(__file__).resolve().parents[2]
    reg = pr.load_registry(repo_root / "governance" / "profile-registry.yaml")
    assert reg["profiles"]
    # No runtime configuration leaked into organisational metadata.
    forbidden = {"model", "provider", "toolsets", "skills", "memory", "runtime"}
    for profile in reg["profiles"]:
        assert forbidden & set(profile) == set()