"""Agent Plugins v1 portable package validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.agent_plugins import (
    MCP_SCHEMA_V1,
    PLUGIN_SCHEMA_V1,
    AgentPluginError,
    has_enabled_agent_plugin_mcp,
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
        ("compatibility", ""),
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
                "opaque-command": {
                    "type": "stdio",
                    "command": "./${PLUGIN_ROOT}",
                },
                "escape": {
                    "type": "stdio",
                    "command": "./../outside",
                },
                "mixed-root": {
                    "type": "stdio",
                    "command": "python",
                    "cwd": "./${PLUGIN_DATA}/state",
                },
            },
        },
    )
    package = load_agent_plugin(root, tmp_path / "data")
    assert set(package.mcp_servers) == {"opaque-command", "valid"}
    assert package.mcp_servers["valid"]["cwd"] == str(
        (tmp_path / "data" / "state").resolve()
    )
    assert (tmp_path / "data" / "state").is_dir()
    assert "agent_plugin" not in package.mcp_servers["valid"]
    assert package.mcp_servers["opaque-command"]["command"] == str(
        (root / "${PLUGIN_ROOT}").resolve()
    )


def test_stdio_cwd_directory_failure_isolated_to_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_json(
        tmp_path / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "broken": {
                    "type": "stdio",
                    "command": "python",
                    "cwd": "${PLUGIN_DATA}/broken",
                },
                "valid": {
                    "type": "stdio",
                    "command": "python",
                    "cwd": "${PLUGIN_DATA}/valid",
                },
            },
        },
    )
    data_root = (tmp_path / "data").resolve()
    original_mkdir = Path.mkdir

    def fail_broken(path: Path, *args: object, **kwargs: object) -> None:
        if path == data_root / "broken":
            raise PermissionError("broken cwd")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_broken)

    package = load_agent_plugin(tmp_path, data_root)

    assert set(package.mcp_servers) == {"valid"}
    assert (data_root / "valid").is_dir()
    assert any(d.scope == "mcp:broken" for d in package.diagnostics)


def test_malformed_skill_yaml_is_skipped(tmp_path: Path) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    skill = tmp_path / "skills" / "broken"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: broken\ndescription: [unterminated\n---\nBody.\n",
        encoding="utf-8",
    )

    package = load_agent_plugin(tmp_path, tmp_path / "data")

    assert package.skills == ()
    assert any(d.scope == "skill:broken" for d in package.diagnostics)


def test_data_directory_failure_preserves_valid_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_skill(tmp_path)
    _write_json(
        tmp_path / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {"worker": {"type": "stdio", "command": "python"}},
        },
    )
    data_root = tmp_path / "data"
    original_mkdir = Path.mkdir

    def fail_data_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if path == data_root:
            raise PermissionError("read-only profile")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_data_mkdir)

    package = load_agent_plugin(tmp_path, data_root)

    assert len(package.skills) == 1
    assert package.mcp_servers == {}
    assert any(d.scope == "mcp:worker" for d in package.diagnostics)


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
                "multi-token": {"type": "stdio", "command": "python -m worker"},
                "reserved-root": {
                    "type": "stdio",
                    "command": "python",
                    "env": {"PLUGIN_ROOT": "override"},
                },
                "reserved-data": {
                    "type": "stdio",
                    "command": "python",
                    "env": {"PLUGIN_DATA": "override"},
                },
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.test/mcp",
                    "headers": {"X-Tenant": "a", "x-tenant": "b"},
                },
            },
        },
    )
    package = load_agent_plugin(tmp_path, tmp_path / "data")
    assert set(package.mcp_servers) == {"valid"}
    assert {d.scope for d in package.diagnostics} >= {
        "mcp:invalid",
        "mcp:multi-token",
        "mcp:reserved-root",
        "mcp:reserved-data",
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


def test_enabled_portable_mcp_probe_does_not_load_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    plugin = home / "plugins" / "portable"
    plugin.mkdir(parents=True)
    _write_json(plugin / "plugin.json", _manifest())
    _write_json(
        plugin / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {"worker": {"type": "stdio", "command": "python"}},
        },
    )
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))

    assert has_enabled_agent_plugin_mcp(
        {"plugins": {"enabled": ["portable.test"]}}
    )
    assert not has_enabled_agent_plugin_mcp(
        {
            "plugins": {
                "enabled": ["portable.test"],
                "disabled": ["portable.test"],
            }
        }
    )


def test_portable_mcp_probe_counts_streamable_http_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    plugin = home / "plugins" / "portable"
    plugin.mkdir(parents=True)
    _write_json(plugin / "plugin.json", _manifest())
    _write_json(
        plugin / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.test/mcp",
                }
            },
        },
    )
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))

    assert has_enabled_agent_plugin_mcp(
        {"plugins": {"enabled": ["portable.test"]}}
    )


def test_portable_mcp_probe_ignores_unsupported_only_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    plugin = home / "plugins" / "portable"
    plugin.mkdir(parents=True)
    _write_json(plugin / "plugin.json", _manifest())
    _write_json(
        plugin / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "remote": {
                    "type": "sse",
                    "url": "https://example.test/sse",
                }
            },
        },
    )
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))

    assert not has_enabled_agent_plugin_mcp(
        {"plugins": {"enabled": ["portable.test"]}}
    )


def test_streamable_http_translates_to_native_remote_config(tmp_path: Path) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_json(
        tmp_path / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "deploy": {
                    "type": "streamable-http",
                    "url": "https://deploy.example.test/mcp",
                    "headers": {"X-Tenant": "public-tenant"},
                },
                "local": {
                    "type": "streamable-http",
                    "url": "http://127.0.0.1:8000/mcp",
                },
                "bare": {
                    "type": "streamable-http",
                    "url": "https://bare.example.test/mcp",
                },
            },
        },
    )

    package = load_agent_plugin(tmp_path, tmp_path / "data")

    assert set(package.mcp_servers) == {"deploy", "local", "bare"}
    deploy = package.mcp_servers["deploy"]
    assert deploy == {
        "url": "https://deploy.example.test/mcp",
        "headers": {"X-Tenant": "public-tenant"},
        "strict_redirect_headers": True,
    }
    assert "command" not in deploy
    assert package.mcp_servers["bare"] == {
        "url": "https://bare.example.test/mcp",
        "strict_redirect_headers": True,
    }


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://example.test/mcp",
        "https://user:pass@example.test/mcp",
        "https://example.test/mcp#fragment",
        "http://example.test/mcp",  # non-loopback plain HTTP
        "http://10.0.0.5/mcp",
        "https:///mcp",  # empty host
    ],
)
def test_streamable_http_rejects_invalid_urls(tmp_path: Path, url: str) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_json(
        tmp_path / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "remote": {"type": "streamable-http", "url": url},
                "valid": {"type": "stdio", "command": "python"},
            },
        },
    )

    package = load_agent_plugin(tmp_path, tmp_path / "data")

    assert set(package.mcp_servers) == {"valid"}
    assert any(d.scope == "mcp:remote" for d in package.diagnostics)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/mcp",
        "http://127.0.0.1/mcp",
        "http://[::1]:9000/mcp",
        "https://remote.example.test/mcp",
    ],
)
def test_streamable_http_accepts_loopback_http_and_https(
    tmp_path: Path, url: str
) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_json(
        tmp_path / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {"remote": {"type": "streamable-http", "url": url}},
        },
    )

    package = load_agent_plugin(tmp_path, tmp_path / "data")

    assert set(package.mcp_servers) == {"remote"}
    assert package.mcp_servers["remote"]["url"] == url


def test_sse_remains_unsupported(tmp_path: Path) -> None:
    _write_json(tmp_path / "plugin.json", _manifest())
    _write_json(
        tmp_path / "mcp.json",
        {
            "$schema": MCP_SCHEMA_V1,
            "mcpServers": {
                "legacy": {"type": "sse", "url": "https://legacy.example.test/sse"}
            },
        },
    )

    package = load_agent_plugin(tmp_path, tmp_path / "data")

    assert package.mcp_servers == {}
    assert any(
        d.scope == "mcp:legacy" and "not supported" in d.message
        for d in package.diagnostics
    )


def test_portable_mcp_probe_honors_native_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    plugin = home / "plugins" / "portable"
    plugin.mkdir(parents=True)
    _write_json(plugin / "plugin.json", _manifest())
    (plugin / "plugin.yaml").write_text("name: native\n", encoding="utf-8")
    _write_json(
        plugin / "mcp.json",
        {"$schema": MCP_SCHEMA_V1, "mcpServers": {}},
    )
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))

    assert not has_enabled_agent_plugin_mcp(
        {"plugins": {"enabled": ["portable.test"]}}
    )
