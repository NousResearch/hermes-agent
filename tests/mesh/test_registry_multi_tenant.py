"""Basic multi-tenant and profile tests for mesh registry (addresses sweeper review)."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Adjust path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mesh import registry
from mesh.provisioner import ControllerConfig


def test_composite_key_prevents_overwrite(tmp_path, monkeypatch):
    """Same host in different namespaces should not overwrite."""
    # Patch get_hermes_home
    with patch("mesh.registry.get_hermes_home", return_value=tmp_path):
        cfg1 = ControllerConfig(namespace="phoebe", broker="127.0.0.1", broker_user="u", broker_password="p", ca_cert_path=None)
        cfg2 = ControllerConfig(namespace="test", broker="127.0.0.1", broker_user="u", broker_password="p", ca_cert_path=None)

        spec1 = MagicMock(host="castor", role="bare", capabilities=[], namespace="phoebe", user=None)
        spec2 = MagicMock(host="castor", role="bare", capabilities=[], namespace="test", user=None)

        registry.append_to_nodes_yaml(spec1, cfg1)
        registry.append_to_nodes_yaml(spec2, cfg2)

        nodes = registry.list_nodes()
        keys = [n["key"] for n in nodes]
        assert "phoebe:castor" in keys
        assert "test:castor" in keys
        assert len([k for k in keys if "castor" in k]) == 2


def test_remove_uses_namespace(monkeypatch, tmp_path):
    """Remove should respect per-node namespace for registry and topics."""
    with patch("mesh.registry.get_hermes_home", return_value=tmp_path):
        # Simplified: just check that remove_from_nodes_yaml accepts namespace
        registry.append_to_nodes_yaml(
            MagicMock(host="node1", role="bare", capabilities=[], namespace="phoebe", user=None),
            ControllerConfig(namespace="phoebe", broker="127.0.0.1", broker_user="u", broker_password="p", ca_cert_path=None)
        )
        registry.remove_from_nodes_yaml("node1", namespace="phoebe")
        nodes = registry.list_nodes()
        assert not any(n.get("host") == "node1" for n in nodes)
