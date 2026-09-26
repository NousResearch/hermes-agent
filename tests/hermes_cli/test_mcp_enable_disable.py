"""`hermes mcp enable/disable <name>` — toggle a server without editing YAML.

Disable writes `enabled: false` to the server's config entry; enable removes the
key (the runtime treats absence as enabled). Unknown names print the not-found
hint and leave config untouched.
"""

import types

import pytest
import yaml


@pytest.fixture()
def mcp_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "mcp_servers": {"remote": {"url": "https://mcp.example.com/v1"}}}))
    return home


def _config(home):
    return yaml.safe_load((home / "config.yaml").read_text())


def test_disable_then_enable_roundtrip(mcp_home, capsys):
    from hermes_cli import mcp_config as mod

    mod.cmd_mcp_disable(types.SimpleNamespace(name="remote"))
    assert _config(mcp_home)["mcp_servers"]["remote"]["enabled"] is False

    mod.cmd_mcp_enable(types.SimpleNamespace(name="remote"))
    # Enable removes the key entirely — absence means enabled to the runtime.
    assert "enabled" not in _config(mcp_home)["mcp_servers"]["remote"]
    out = capsys.readouterr().out
    assert "Disabled 'remote'" in out and "Enabled 'remote'" in out


def test_unknown_server_leaves_config_untouched(mcp_home, capsys):
    from hermes_cli import mcp_config as mod

    before = _config(mcp_home)
    mod.cmd_mcp_disable(types.SimpleNamespace(name="nope"))
    assert _config(mcp_home) == before
    assert "not found" in capsys.readouterr().out
