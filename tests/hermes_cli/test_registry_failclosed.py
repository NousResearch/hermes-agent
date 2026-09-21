"""C10 adversarial tests — registry fail-closed + dashboard parity."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from hermes_cli import profile_registry as pr


VALID = {
    "schema_version": 1,
    "root": {"name": "KENSEI", "description": "root"},
    "profiles": [
        {"name": "octacon", "kind": "lead", "parent": "KENSEI",
         "lifecycle": "active", "domains": [], "gateway_unit": "u-octacon"},
        {"name": "octacon-backend", "kind": "worker", "parent": "octacon",
         "lifecycle": "active", "domains": [], "gateway_unit": None},
    ],
}


class TestC10CoreNullParent:
    def test_null_parent_non_root_rejected(self, tmp_path):
        """RED: null-parent non-root profile must be rejected (one tree)."""
        doc = {
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "a", "kind": "worker", "parent": None,
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }
        p = tmp_path / "reg.yaml"
        p.write_text(yaml.safe_dump(doc))
        with pytest.raises(pr.RegistryError):
            pr.load_registry(p)


class TestC10DashboardFailClosed:
    @pytest.fixture
    def dash(self, tmp_path, monkeypatch):
        # R3-5: dashboard checkout supplied explicitly via EVIDENCE_SPINE_DASHBOARD.
        dash_root = os.environ.get("EVIDENCE_SPINE_DASHBOARD")
        if not dash_root or not Path(dash_root).is_dir():
            pytest.skip("EVIDENCE_SPINE_DASHBOARD not supplied (cross-repo test)")
        assert isinstance(dash_root, str)
        home = tmp_path / "hermes-home"
        (home / "profiles").mkdir(parents=True)
        (home / "governance").mkdir()
        import sys
        sys.path.insert(0, dash_root)
        from backend import profile_docs
        monkeypatch.setattr(profile_docs, "HERMES_HOME", home)
        monkeypatch.setattr(profile_docs, "PROFILES_DIR", home / "profiles")
        monkeypatch.setattr(profile_docs, "GATEWAY_PROFILES", ["octacon"])
        yield home, profile_docs
        sys.path.remove(dash_root)

    def _write_registry(self, home, doc):
        (home / "governance" / "profile-registry.yaml").write_text(
            yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")

    def test_invalid_enum_registry_fails_closed(self, dash):
        home, profile_docs = dash
        (home / "profiles" / "octacon").mkdir()
        doc = {
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "octacon", "kind": "emperor", "parent": "KENSEI",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }
        self._write_registry(home, doc)
        hierarchy = profile_docs.profile_hierarchy()
        node = next(n for n in hierarchy["nodes"] if n["name"] == "octacon")
        # Fail closed: not silently rendered with the invalid kind
        assert node.get("registry_state") != "registered"
        assert node.get("registry_state") in ("unregistered", "invalid_registry")

    def test_unknown_parent_registry_fails_closed(self, dash):
        home, profile_docs = dash
        (home / "profiles" / "octacon").mkdir()
        doc = {
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "octacon", "kind": "lead", "parent": "ghost-lead",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }
        self._write_registry(home, doc)
        hierarchy = profile_docs.profile_hierarchy()
        node = next(n for n in hierarchy["nodes"] if n["name"] == "octacon")
        assert node.get("registry_state") != "registered"

    def test_cycle_registry_fails_closed(self, dash):
        home, profile_docs = dash
        doc = {
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "a", "kind": "worker", "parent": "b",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
                {"name": "b", "kind": "worker", "parent": "a",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }
        self._write_registry(home, doc)
        hierarchy = profile_docs.profile_hierarchy()
        # No malformed graph silently formed: nodes are surfaced unregistered
        by_name = {n["name"]: n for n in hierarchy["nodes"]}
        for name in ("a", "b"):
            assert by_name.get(name, {}).get("registry_state") != "registered"

    def test_valid_registry_still_consumed(self, dash):
        home, profile_docs = dash
        (home / "profiles" / "octacon").mkdir()
        self._write_registry(home, VALID)
        hierarchy = profile_docs.profile_hierarchy()
        node = next(n for n in hierarchy["nodes"] if n["name"] == "octacon")
        assert node["registry_state"] == "registered"
        assert node["type"] == "lead"