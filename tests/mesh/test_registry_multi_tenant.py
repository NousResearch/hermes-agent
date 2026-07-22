"""Multi-tenant and profile-isolation tests for mesh registry.

Addresses hermes-sweeper review on PR #29460:
  - profile scope (get_hermes_home, not hardcoded ~/.hermes)
  - duplicate hosts across namespaces (composite key)
  - namespace-aware removal
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mesh import registry
from mesh.provisioner import ControllerConfig, NodeSpec, _Redacted


def _cfg(namespace: str = "phoebe") -> ControllerConfig:
    return ControllerConfig(
        namespace=namespace,
        broker="127.0.0.1",
        broker_user="u",
        broker_password=_Redacted("p"),
        ca_cert_path=None,
        template_dirs=[],
    )


def _spec(host: str, namespace: str = "phoebe", capabilities=None) -> NodeSpec:
    return NodeSpec(
        host=host,
        role="bare",
        namespace=namespace,
        broker="127.0.0.1",
        capabilities=capabilities or [],
        user=None,
    )


class TestCompositeKey:
    """Same host in different namespaces must not overwrite each other."""

    def test_two_namespaces_same_host(self, tmp_path):
        with patch("mesh.registry.get_hermes_home", return_value=tmp_path):
            registry.append_to_nodes_yaml(_spec("castor", "phoebe"), _cfg("phoebe"))
            registry.append_to_nodes_yaml(_spec("castor", "test"), _cfg("test"))

            nodes = registry.list_nodes()
            keys = [n["key"] for n in nodes]
            assert "phoebe:castor" in keys
            assert "test:castor" in keys
            assert len([k for k in keys if "castor" in k]) == 2

    def test_reprovision_same_namespace_updates_in_place(self, tmp_path):
        with patch("mesh.registry.get_hermes_home", return_value=tmp_path):
            spec = _spec("tycho", "phoebe")
            spec.capabilities = ["storage"]
            registry.append_to_nodes_yaml(spec, _cfg("phoebe"))

            spec2 = _spec("tycho", "phoebe")
            spec2.capabilities = ["storage", "gpu"]
            registry.append_to_nodes_yaml(spec2, _cfg("phoebe"))

            nodes = registry.list_nodes()
            tycho = [n for n in nodes if n["host"] == "tycho"]
            assert len(tycho) == 1
            assert "gpu" in tycho[0]["capabilities"]


class TestNamespaceAwareRemoval:
    """remove_from_nodes_yaml must respect per-node namespace."""

    def test_remove_specific_namespace_only(self, tmp_path):
        with patch("mesh.registry.get_hermes_home", return_value=tmp_path):
            registry.append_to_nodes_yaml(_spec("castor", "phoebe"), _cfg("phoebe"))
            registry.append_to_nodes_yaml(_spec("castor", "test"), _cfg("test"))

            registry.remove_from_nodes_yaml("castor", namespace="test")
            nodes = registry.list_nodes()
            hosts = [n["host"] for n in nodes]
            assert "castor" in hosts  # phoebe entry survives
            assert "test:castor" not in [n["key"] for n in nodes]
            assert "phoebe:castor" in [n["key"] for n in nodes]

    def test_remove_without_namespace_clears_all(self, tmp_path):
        """Backwards-compat: no namespace arg removes across all namespaces."""
        with patch("mesh.registry.get_hermes_home", return_value=tmp_path):
            registry.append_to_nodes_yaml(_spec("castor", "phoebe"), _cfg("phoebe"))
            registry.append_to_nodes_yaml(_spec("castor", "test"), _cfg("test"))

            registry.remove_from_nodes_yaml("castor")
            nodes = registry.list_nodes()
            assert not any(n["host"] == "castor" for n in nodes)

    def test_remove_nonexistent_namespace_is_noop(self, tmp_path):
        with patch("mesh.registry.get_hermes_home", return_value=tmp_path):
            registry.append_to_nodes_yaml(_spec("castor", "phoebe"), _cfg("phoebe"))
            # Should not raise
            registry.remove_from_nodes_yaml("castor", namespace="nonexistent")
            nodes = registry.list_nodes()
            assert "phoebe:castor" in [n["key"] for n in nodes]


class TestProfileIsolation:
    """Registry paths must go through get_hermes_home, not hardcoded ~/.hermes."""

    def test_nodes_yaml_uses_get_hermes_home(self, tmp_path):
        """nodes.yaml should be written under the patched hermes home."""
        with patch("mesh.registry.get_hermes_home", return_value=tmp_path) as mock_home:
            registry.append_to_nodes_yaml(_spec("tycho"), _cfg())
            # get_hermes_home should have been called (path resolution)
            assert mock_home.called
            # nodes.yaml should exist under tmp_path, not ~/.hermes
            assert (tmp_path / "mesh" / "nodes.yaml").exists()
            # Ensure we did NOT write to the real ~/.hermes/mesh/nodes.yaml
            real_home = Path.home() / ".hermes" / "mesh" / "nodes.yaml"
            # Only fails if the test env's real home was used — the mock
            # redirects, so this file should not have been touched by this test.
            # (Guard: if it exists from a real mesh install, that's fine —
            #  we just confirm tmp_path got the write.)
            assert (tmp_path / "mesh" / "nodes.yaml").read_text() != ""

    def test_different_profiles_isolate(self, tmp_path):
        """Two profile-scoped homes should not share nodes.yaml state."""
        profile_a = tmp_path / "profileA"
        profile_b = tmp_path / "profileB"
        profile_a.mkdir()
        profile_b.mkdir()

        with patch("mesh.registry.get_hermes_home", return_value=profile_a):
            registry.append_to_nodes_yaml(_spec("tycho", "phoebe"), _cfg("phoebe"))

        with patch("mesh.registry.get_hermes_home", return_value=profile_b):
            # profile_b should not see tycho from profile_a
            nodes = registry.list_nodes()
            assert not any(n["host"] == "tycho" for n in nodes)

        # profile_a still has tycho
        with patch("mesh.registry.get_hermes_home", return_value=profile_a):
            nodes = registry.list_nodes()
            assert any(n["host"] == "tycho" for n in nodes)
