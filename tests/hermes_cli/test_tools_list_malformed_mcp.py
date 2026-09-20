"""Read-only tools listing must survive legacy non-mapping MCP entries."""
from argparse import Namespace

import pytest
import yaml


@pytest.mark.parametrize("malformed", [["web", "terminal"], "https://example.invalid/mcp", 42, None, False])
def test_tools_list_skips_malformed_servers(tmp_path, monkeypatch, capsys, malformed):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.tools_config_mcp import tools_disable_enable_command

    config = {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "platform_toolsets": {"cli": []},
        "mcp_servers": {
            "legacy": malformed,
            "healthy": {"command": "unused", "tools": {"include": ["read_item"]}},
            "blocked": {"command": "unused", "tools": {"include": []}},
            "filtered": {"command": "unused", "tools": {"exclude": ["delete_item"]}},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    before = path.read_bytes()
    try:
        tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
    except AttributeError as exc:
        pytest.fail(f"Malformed server aborted the public tools-list path: {exc}")
    output = capsys.readouterr().out
    assert "legacy" in output and "skipped" in output.lower()
    assert "healthy  [include only: read_item]" in output
    assert "blocked  [include only: (none)]" in output
    assert "filtered  [excluded:" in output and "delete_item" in output
    assert path.read_bytes() == before


def test_tools_list_with_no_servers(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.tools_config_mcp import tools_disable_enable_command

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "platform_toolsets": {"cli": []}, "mcp_servers": {},
    }), encoding="utf-8")
    tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
    output = capsys.readouterr().out
    assert "Built-in toolsets (cli):" in output
    assert "MCP servers:" not in output
