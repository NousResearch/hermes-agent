"""WhatsApp bridge should prefer Hermes-managed Node over system PATH (#49242).

On Windows a broken/elevated/incompatible system Node on PATH could stop the
bridge from starting even though Hermes installed its own Node under
$HERMES_HOME/node. _resolve_node() prefers the managed binary, then HERMES_NODE,
then PATH.
"""

import os
from unittest.mock import patch

from gateway.platforms import whatsapp


def test_prefers_hermes_env_override(tmp_path, monkeypatch):
    node = tmp_path / "custom-node"
    node.write_text("#!/bin/sh\n")
    node.chmod(0o755)
    monkeypatch.setenv("HERMES_NODE", str(node))
    assert whatsapp._resolve_node() == str(node)


def test_prefers_managed_windows_node(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_NODE", raising=False)
    managed = tmp_path / "node" / "node.exe"
    managed.parent.mkdir(parents=True)
    managed.write_text("")
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        assert whatsapp._resolve_node() == str(managed)


def test_prefers_managed_posix_node(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_NODE", raising=False)
    managed = tmp_path / "node" / "bin" / "node"
    managed.parent.mkdir(parents=True)
    managed.write_text("")
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        assert whatsapp._resolve_node() == str(managed)


def test_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_NODE", raising=False)
    # Managed home with no node installed → fall back to PATH lookup.
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("gateway.platforms.whatsapp.shutil.which", return_value="/usr/bin/node") as which:
        assert whatsapp._resolve_node() == "/usr/bin/node"
        which.assert_called_with("node")
