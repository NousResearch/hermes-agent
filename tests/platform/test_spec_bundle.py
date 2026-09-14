"""Tenant bundle loading and cross-document validation."""

from __future__ import annotations

import shutil

import pytest

from nova.errors import SpecError
from nova.spec import load_bundle

from .conftest import EXAMPLE_BUNDLE


def _copy(tmp_path):
    dest = tmp_path / "bundle"
    shutil.copytree(EXAMPLE_BUNDLE, dest)
    return dest


def test_example_bundle_loads(bundle):
    assert bundle.tenant_id == "acme"
    assert [spec.id for spec in bundle.agents] == ["customer-support", "operations"]
    assert bundle.identity.product_name == "Acme Intelligence"


def test_instructions_are_resolved_from_the_bundle(bundle):
    assert "Acme Support Assistant" in bundle.agent("customer-support").instructions


def test_agent_lookup_reports_known_ids(bundle):
    with pytest.raises(SpecError, match="known agents"):
        bundle.agent("nope")


def test_enabled_agents_filters(tmp_path):
    root = _copy(tmp_path)
    path = root / "agents" / "operations.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("enabled: true", "enabled: false")
                    if "enabled:" in path.read_text(encoding="utf-8")
                    else path.read_text(encoding="utf-8") + "\nenabled: false\n", encoding="utf-8")
    loaded = load_bundle(root)
    assert [spec.id for spec in loaded.enabled_agents()] == ["customer-support"]


def test_missing_organization_is_reported(tmp_path):
    root = _copy(tmp_path)
    (root / "organization.yaml").unlink()
    with pytest.raises(SpecError, match="organization.yaml is required"):
        load_bundle(root)


def test_missing_agents_dir_is_reported(tmp_path):
    root = _copy(tmp_path)
    shutil.rmtree(root / "agents")
    with pytest.raises(SpecError, match="agents/ is required"):
        load_bundle(root)


def test_empty_agents_dir_is_reported(tmp_path):
    root = _copy(tmp_path)
    for path in (root / "agents").iterdir():
        path.unlink()
    with pytest.raises(SpecError, match="no .yaml files"):
        load_bundle(root)


def test_duplicate_agent_id_is_reported(tmp_path):
    root = _copy(tmp_path)
    source = (root / "agents" / "operations.yaml").read_text(encoding="utf-8")
    (root / "agents" / "copy.yaml").write_text(source, encoding="utf-8")
    with pytest.raises(SpecError, match="duplicate agent id"):
        load_bundle(root)


def test_delegation_to_unknown_agent_is_reported(tmp_path):
    root = _copy(tmp_path)
    path = root / "agents" / "customer-support.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("may_assign_to: [operations]", "may_assign_to: [ghost]"),
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="unknown agent"):
        load_bundle(root)


def test_branding_an_unknown_agent_is_reported(tmp_path):
    root = _copy(tmp_path)
    path = root / "identity.yaml"
    path.write_text(
        path.read_text(encoding="utf-8") + "  phantom:\n    display_name: Nobody\n", encoding="utf-8"
    )
    with pytest.raises(SpecError, match="unknown agent"):
        load_bundle(root)


def test_invalid_yaml_names_the_file(tmp_path):
    root = _copy(tmp_path)
    (root / "agents" / "customer-support.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    with pytest.raises(SpecError, match="not valid YAML"):
        load_bundle(root)


def test_identity_is_optional(tmp_path):
    root = _copy(tmp_path)
    (root / "identity.yaml").unlink()
    loaded = load_bundle(root)
    assert loaded.identity.product_name == "NOVA AI Workforce"


def test_bundle_digest_is_stable_and_meaningful(tmp_path):
    root = _copy(tmp_path)
    first = load_bundle(root).digest()
    assert first == load_bundle(root).digest()
    path = root / "identity.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("Acme Intelligence", "Other Name"), encoding="utf-8"
    )
    assert load_bundle(root).digest() != first


def test_missing_bundle_directory_is_reported(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        load_bundle(tmp_path / "nope")
