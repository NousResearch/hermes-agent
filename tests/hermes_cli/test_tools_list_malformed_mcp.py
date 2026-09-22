"""Read-only tools listing must survive legacy non-mapping MCP entries."""
from argparse import Namespace

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def isolated_tools_list(tmp_path):
    def run(tools):
        # Set every home before importing product code; never inherit credentials.
        env = {key: value for key, value in os.environ.items()
               if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"}}
        for key in ("HOME", "USERPROFILE", "HERMES_HOME", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
            directory = tmp_path / key.lower()
            directory.mkdir(exist_ok=True)
            env[key] = str(directory)
        env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", '''
import json
import os
from argparse import Namespace
from pathlib import Path
import sys
import yaml
from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.tools_config_mcp import tools_disable_enable_command

path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
path.write_text(yaml.safe_dump({
    "_config_version": DEFAULT_CONFIG["_config_version"],
    "platform_toolsets": {"cli": []},
    "mcp_servers": {
        "subject": {"command": "unused", "tools": json.loads(sys.argv[1])},
        "later": {"command": "unused", "tools": {"include": ["read_item"]}},
    },
}, sort_keys=False), encoding="utf-8")
before = path.read_bytes()
try:
    tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
finally:
    assert path.read_bytes() == before, "tools list changed config bytes"
tools = json.loads(sys.argv[1])
if isinstance(tools, dict):
    from tools.mcp_tool_registration import _make_tool_filter
    predicate = _make_tool_filter("subject", {"tools": tools})
    names = ["read_item", "delete_item", "42", "other"]
    print("ACCEPTED=" + json.dumps([name for name in names if predicate(name)]))
''', json.dumps(tools)],
            cwd=Path(__file__).resolve().parents[2], env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "later  [include only: read_item]" in result.stdout
        return result.stdout
    return run


@pytest.mark.parametrize("tools", ["yes", ["read_item"], 42, True, None, False, [], ""])
def test_tools_list_survives_non_mapping_tools(isolated_tools_list, tools):
    output = isolated_tools_list(tools)
    assert "subject" in output
    if tools:
        assert f"subject.tools: expected a mapping, got {type(tools).__name__}" in output
        assert "all tools enabled" not in output


@pytest.mark.parametrize("tools, summary, accepted", [
    ({"include": "read_*", "exclude": ["read_item"]}, "[include only: read_*]", ["read_item"]),
    ({"exclude": "delete_*"}, "[excluded: delete_*]", ["read_item", "42", "other"]),
    ({"include": [42, "read_item", 42], "exclude": ["read_item"]},
     "[include only: 42, read_item]", ["read_item", "42"]),
    ({"exclude": [42, "delete_item", 42]}, "[excluded: 42, delete_item]", ["read_item", "other"]),
    ({"include": [], "exclude": ["delete_item"]}, "[include only: (none)]", []),
    ({"include": "", "exclude": ["delete_item"]}, "[include only: (none)]", []),
    ({"include": 42, "exclude": "delete_item"}, "[excluded: delete_item]", ["read_item", "42", "other"]),
    ({"exclude": 42}, "all tools enabled", ["read_item", "delete_item", "42", "other"]),
    ({"include": False, "exclude": True}, "all tools enabled", ["read_item", "delete_item", "42", "other"]),
    ({"include": {"read_item": True}, "exclude": {"delete_item": True}},
     "all tools enabled", ["read_item", "delete_item", "42", "other"]),
    ({"exclude": ""}, "[excluded: (none)]", ["read_item", "delete_item", "42", "other"]),
])
def test_tools_list_filter_summary_matches_runtime(isolated_tools_list, tools, summary, accepted):
    output = isolated_tools_list(tools)
    assert f"subject  {summary}" in output
    assert "ACCEPTED=" + json.dumps(accepted) in output


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
