"""Native MCP installs retain the reviewed server's write-approval policy."""

import pytest
import hermes_yaml as yaml


def _manifest(tmp_path, trust="untrusted"):
    path = tmp_path / "manifest.yaml"
    data = {
        "manifest_version": 1,
        "name": "demo",
        "description": "OAuth MCP with writes",
        "transport": {"type": "http", "url": "https://example.com/mcp"},
        "auth": {"type": "oauth"},
    }
    if trust is not None:
        data["trust"] = trust
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_cli_and_connection_card_preserve_oauth_write_approval(tmp_path, monkeypatch):
    from hermes_cli import mcp_catalog as mc
    from hermes_cli.config import load_config

    entry = mc._parse_manifest(_manifest(tmp_path))
    monkeypatch.setattr(mc, "_probe_tools", lambda name: None)
    mc.install_entry(entry)
    server = load_config()["mcp_servers"]["demo"]
    assert server["auth"] == "oauth"
    assert server["trust"] == "untrusted"
    assert mc.card_install_config(entry)["trust"] == "untrusted"


@pytest.mark.parametrize("trust", ["trusted", "", False, [], {}])
def test_invalid_declared_trust_cannot_silently_widen_access(tmp_path, trust):
    from hermes_cli import mcp_catalog as mc

    with pytest.raises(mc.CatalogError, match="trust"):
        mc._parse_manifest(_manifest(tmp_path, trust))


def test_existing_catalog_entries_keep_their_default_policy(tmp_path):
    from hermes_cli import mcp_catalog as mc

    entry = mc._parse_manifest(_manifest(tmp_path, None))
    assert "trust" not in mc._build_server_config(entry, None)


def test_native_oauth_connection_overrides_same_name_portable_server(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from hermes_cli import mcp_catalog as mc, plugins
    from hermes_cli.agent_plugins import load_agent_plugin
    from tools.mcp_tool_config import _portable_mcp_servers

    root = tmp_path / "plugin"
    root.mkdir()
    (root / "plugin.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        "name": "demo", "version": "1.0.0",
    }), encoding="utf-8")
    (root / "mcp.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
        "mcpServers": {"demo": {"type": "streamable-http", "url": "https://example.com/mcp"}},
    }), encoding="utf-8")
    package = load_agent_plugin(root, tmp_path / "plugin-data")
    assert not package.diagnostics
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: SimpleNamespace(
        get_portable_mcp_servers=lambda: package.mcp_servers))
    native = mc._build_server_config(mc._parse_manifest(_manifest(tmp_path)), None)
    servers = {"demo": native.copy()}
    _portable_mcp_servers(servers)
    assert servers == {"demo": native}
    assert servers["demo"]["auth"] == "oauth"
    assert servers["demo"]["trust"] == "untrusted"
