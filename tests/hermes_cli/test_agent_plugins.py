"""Agent Plugins v1 portable package validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.agent_plugins import (
    MCP_SCHEMA_V1,
    PLUGIN_SCHEMA_V1,
    AgentPluginError,
    load_agent_plugin,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(**overrides: object) -> dict:
    value = {"$schema": PLUGIN_SCHEMA_V1, "name": "portable.test"}
    value.update(overrides)
    return value


def _write_skill(root: Path, directory: str = "summarize", **fields: object) -> Path:
    skill_dir = root / "skills" / directory
    skill_dir.mkdir(parents=True)
    metadata = {"name": directory, "description": "Summarizes reports."}
    metadata.update(fields)
    import yaml

    (skill_dir / "SKILL.md").write_text(
        f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\nInstructions.\n",
        encoding="utf-8",
    )
    return skill_dir


def test_loads_manifest_skill_and_stdio_server(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    _write_json(root / "plugin.json", _manifest(version="1.2.3"))
    skill_dir = _write_skill(root)
    _write_json(
        root / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "worker": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["${PLUGIN_ROOT}/server.py", "${UNKNOWN}"],
                    "env": {"CACHE": "${PLUGIN_DATA}/cache"},
                }
            },
        },
    )

    package = load_agent_plugin(root, tmp_path / "data")

    assert package.name == "portable.test"
    assert package.version == "1.2.3"
    assert package.skills[0].root == skill_dir.resolve()
    server = package.mcp_servers["worker"]
    assert server["command"] == "python"
    assert server["args"] == [str(root.resolve() / "server.py"), "${UNKNOWN}"]
    assert server["cwd"] == str(root.resolve())
    assert server["env"]["PLUGIN_ROOT"] == str(root.resolve())
    assert server["env"]["PLUGIN_DATA"] == str((tmp_path / "data").resolve())
    assert server["env"]["CACHE"] == str((tmp_path / "data").resolve() / "cache")
    assert (tmp_path / "data").is_dir()


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"name": "valid-name"},
        {"$schema": "https://example.test/schema.json", "name": "valid-name"},
        {"$schema": PLUGIN_SCHEMA_V1},
        {"$schema": PLUGIN_SCHEMA_V1, "name": "Bad_Name"},
        {"$schema": PLUGIN_SCHEMA_V1, "name": "a--b"},
        {"$schema": PLUGIN_SCHEMA_V1, "name": "a", "keywords": [1]},
        {"$schema": PLUGIN_SCHEMA_V1, "name": "a", "author": {"handle": "x"}},
    ],
)
def test_rejects_invalid_manifests(tmp_path: Path, manifest: object) -> None:
    _write_json(tmp_path / "plugin.json", manifest)
    with pytest.raises(AgentPluginError):
        load_agent_plugin(tmp_path, tmp_path / "data")


def test_unknown_fields_and_non_object_extensions_are_nonfatal(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "plugin.json",
        _manifest(unknown=True, extensions="ignored"),
    )
    package = load_agent_plugin(tmp_path, tmp_path / "data")
    assert len(package.diagnostics) == 2
    assert all(d.fatal is False for d in package.diagnostics)


def test_invalid_skill_does_not_hide_valid_sibling(tmp_path: Path) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_skill(tmp_path, "valid-skill", license="MIT", compatibility="Linux")
    _write_skill(tmp_path, "wrong-dir", name="different")

    package = load_agent_plugin(tmp_path, tmp_path / "data")

    assert [skill.name for skill in package.skills] == ["valid-skill"]
    assert any(d.scope == "skill:wrong-dir" for d in package.diagnostics)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("license", ["MIT"]),
        ("compatibility", 1),
        ("metadata", []),
        ("allowed-tools", ["terminal"]),
    ],
)
def test_rejects_invalid_optional_skill_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_skill(tmp_path, "bad-skill", **{field: value})
    package = load_agent_plugin(tmp_path, tmp_path / "data")
    assert package.skills == ()


def test_symlink_escape_is_isolated_to_component(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    _write_json(root / "plugin.json", _manifest())
    _write_skill(root)
    outside = tmp_path / "outside.json"
    _write_json(outside, {"$schema": MCP_SCHEMA_V1, "mcpServers": {}})
    (root / "mcp.json").symlink_to(outside)

    package = load_agent_plugin(root, tmp_path / "data")

    assert len(package.skills) == 1
    assert package.mcp_servers == {}
    assert any(d.scope == "mcp" for d in package.diagnostics)


def test_stdio_command_and_data_cwd_containment(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    _write_json(root / "plugin.json", _manifest())
    (root / "bin").mkdir()
    (root / "bin" / "server").write_text("", encoding="utf-8")
    _write_json(
        root / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "valid": {
                    "type": "stdio",
                    "command": "./bin/server",
                    "cwd": "${PLUGIN_DATA}/state",
                },
                "escape": {
                    "type": "stdio",
                    "command": "./../outside",
                },
            },
        },
    )
    package = load_agent_plugin(root, tmp_path / "data")
    assert set(package.mcp_servers) == {"valid"}
    assert package.mcp_servers["valid"]["cwd"] == str(
        (tmp_path / "data" / "state").resolve()
    )


def test_invalid_entries_and_unsupported_remote_preserve_valid_stdio(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_json(
        tmp_path / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "valid": {"type": "stdio", "command": "python"},
                "invalid": {"type": "stdio", "command": "python", "extra": True},
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.test/mcp",
                },
            },
        },
    )
    package = load_agent_plugin(tmp_path, tmp_path / "data")
    assert set(package.mcp_servers) == {"valid"}
    assert {d.scope for d in package.diagnostics} >= {
        "mcp:invalid",
        "mcp:remote",
    }


def test_invalid_mcp_top_level_preserves_skills(tmp_path: Path) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_skill(tmp_path)
    _write_json(
        tmp_path / "mcp.json",
        {"$schema": "https://example.test/mcp.json", "mcpServers": {}},
    )
    package = load_agent_plugin(tmp_path, tmp_path / "data")
    assert len(package.skills) == 1
    assert package.mcp_servers == {}
