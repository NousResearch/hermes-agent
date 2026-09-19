"""Codex conversion contracts and the real profile-scoped discovery/connect path."""

from __future__ import annotations

import copy
import fnmatch
import json
from pathlib import Path

import pytest
import yaml

from hermes_cli.mcp_discovery_codex import normalize_codex_mcp


def test_stdio_conversion_preserves_settings_without_resolving_env(monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "must-not-be-copied")
    source = {
        "command": "node", "args": ["server.js", ""], "cwd": "/workspace",
        "enabled": False, "required": False, "startup_timeout_sec": 12.5, "tool_timeout_sec": 90,
        "env": {"LOG_LEVEL": "debug", "OVERRIDE": "explicit"},
        "env_vars": ["TEST_MCP_TOKEN", "OVERRIDE", {"name": "LOCAL_VALUE", "source": "local"}],
    }
    before = copy.deepcopy(source)
    result = normalize_codex_mcp(source)
    assert result == {
        "command": "node", "args": ["server.js", ""], "cwd": "/workspace", "enabled": False,
        "connect_timeout": 12.5, "timeout": 90,
        "env": {"LOG_LEVEL": "debug", "OVERRIDE": "explicit", "TEST_MCP_TOKEN": "${TEST_MCP_TOKEN}",
                "LOCAL_VALUE": "${LOCAL_VALUE}"},
    }
    result["args"].append("changed")
    result["env"]["LOG_LEVEL"] = "changed"
    assert source == before


def test_http_conversion_preserves_headers_and_auth_references(monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "must-not-be-copied")
    source = {
        "url": "https://example.test/mcp", "http_headers": {"X-Region": "eu"},
        "env_http_headers": {"X-Api-Key": "TEST_MCP_KEY"}, "bearer_token_env_var": "TEST_MCP_TOKEN",
        "auth": "oauth",
    }
    assert normalize_codex_mcp(source) == {
        "url": source["url"], "auth": "oauth",
        "headers": {"X-Region": "eu", "X-Api-Key": "${TEST_MCP_KEY}", "Authorization": "Bearer ${TEST_MCP_TOKEN}"},
    }


@pytest.mark.parametrize("policy", [
    {}, {"enabled_tools": []}, {"disabled_tools": []}, {"enabled_tools": ["read", "write"]},
    {"disabled_tools": ["write"]}, {"enabled_tools": ["read", "write"], "disabled_tools": ["write"]},
    {"enabled_tools": ["read"], "disabled_tools": ["read"]},
    {"enabled_tools": ["read*", "q?", "x[ab]"]}, {"disabled_tools": ["read*", "q?", "x[ab]"]},
])
def test_tool_filter_semantics_match_codex(policy):
    translated = normalize_codex_mcp({"command": "node", **policy}).get("tools", {})
    for name in ("read", "write", "other", "read*", "read_all", "q?", "qa", "x[ab]", "xa"):
        expected = ("enabled_tools" not in policy or name in policy["enabled_tools"]) and name not in policy.get("disabled_tools", [])
        include, exclude = translated.get("include"), translated.get("exclude", [])
        actual = any(fnmatch.fnmatchcase(name, pattern) for pattern in include) if include else not any(
            fnmatch.fnmatchcase(name, pattern) for pattern in exclude
        )
        assert actual == expected, (policy, name, translated)


@pytest.mark.parametrize("config", [
    {}, {"command": "node", "url": "https://example.test"}, {"command": 1}, {"command": " "},
    {"command": "node", "args": "server.js"}, {"command": "node", "args": [1]},
    {"command": "node", "env": ["KEY=value"]}, {"command": "node", "env": {"KEY": 1}},
    {"command": "node", "env_vars": "TOKEN"}, {"command": "node", "env_vars": ["${TOKEN}"]},
    {"command": "node", "env_vars": [{"name": "TOKEN", "source": "remote"}]},
    {"command": "node", "env_vars": [{"name": "TOKEN", "unsupported": True}]},
    {"command": "node", "env_vars": [{}]}, {"command": "node", "cwd": 3},
    {"command": "node", "enabled": "false"}, {"command": "node", "required": "false"},
    {"command": "node", "required": True}, {"command": "node", "startup_timeout_sec": True},
    {"command": "node", "startup_timeout_sec": 0}, {"command": "node", "tool_timeout_sec": -1},
    {"command": "node", "tool_timeout_sec": 10**1000},
    {"command": "node", "tool_timeout_sec": float("nan")}, {"command": "node", "tool_timeout_sec": float("inf")},
    {"command": "node", "enabled_tools": "read"}, {"command": "node", "disabled_tools": [3]},
    {"command": "node", "default_tools_approval_mode": "prompt"},
    {"command": "node", "tools": {"read": {"approval_mode": "prompt"}}},
    {"command": "node", "experimental_environment": "remote"},
    {"url": "https://example.test", "auth": "chatgpt"},
    {"url": "https://example.test", "http_headers_helper": ["do-not-execute"]},
    {"url": "https://example.test", "oauth": {"client_id": "custom-client"}},
    {"url": "https://example.test", "http_headers": {"X-Bad\nHeader": "value"}},
    {"url": "https://example.test", "http_headers": {"X-Test": "value\r\ninjected"}},
    {"url": "https://example.test", "env_http_headers": {"X-Test": "bad-variable"}},
    {"url": "https://example.test", "bearer_token_env_var": "${TOKEN}"},
    {"url": "https://example.test", "http_headers": {"X-Test": "one", "x-test": "two"}},
    {"url": "https://example.test", "http_headers": {"X-Test": "one"}, "env_http_headers": {"x-test": "TWO"}},
    {"url": "https://example.test", "http_headers": {"authorization": "Bearer inline-value"}, "bearer_token_env_var": "TOKEN"},
])
def test_incompatible_settings_are_rejected_without_echoing_values(config):
    with pytest.raises(ValueError) as error:
        normalize_codex_mcp(config)
    assert "inline-value" not in str(error.value)
    assert "do-not-execute" not in str(error.value)


@pytest.fixture
def codex_discovery(tmp_path, monkeypatch):
    # Real server/config/profile imports; only environmental I/O is isolated.
    import psutil
    root = tmp_path / ".hermes"
    target = root / "profiles" / "target" / "config.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("target_sibling: true\n", encoding="utf-8")
    (root / "config.yaml").write_text("default_sibling: true\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(psutil, "net_connections", lambda kind="inet": [])
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('[mcp_servers.shared]\ncommand = "node"\nargs = ["server.js"]\nenabled = false\n', encoding="utf-8")
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    try:
        yield config, target, client
    finally:
        client.close()


@pytest.mark.parametrize("outcome", ["success", "changed", "deleted", "collision", "credentials"])
def test_integration_codex_connect_revalidates_and_only_writes_target(codex_discovery, outcome):
    config, target, client = codex_discovery
    original = config.read_bytes()
    found = client.post("/api/mcp/discovery?profile=target")
    assert found.status_code == 200
    candidate = next(c for c in found.json()["candidates"] if c["name"] == "shared")
    assert candidate["source"].startswith("Codex (")
    assert candidate["connectable"] is True
    if outcome == "changed":
        # Even a newly explicit default (which conversion drops) changes the raw fingerprint.
        config.write_bytes(original + b"required = false\n")
    elif outcome == "deleted":
        config.unlink()
    elif outcome == "collision":
        target.write_text("mcp_servers:\n  shared:\n    command: other\n", encoding="utf-8")
    elif outcome == "credentials":
        config.write_bytes(original + b'env = { API_TOKEN = "inline-test-value" }\n')
    before = target.read_bytes()
    response = client.post("/api/mcp/discovery/connect?profile=target", json={"candidate_id": candidate["id"]})
    if outcome == "success":
        assert response.status_code == 200
        assert yaml.safe_load(target.read_text())["mcp_servers"]["shared"] == {
            "command": "node", "args": ["server.js"], "enabled": True,
        }
        assert yaml.safe_load(target.read_text())["target_sibling"] is True
        assert config.read_bytes() == original
    else:
        assert response.status_code == 409
        assert target.read_bytes() == before
    assert b"inline-test-value" not in response.content
    assert (target.parents[2] / "config.yaml").read_text() == "default_sibling: true\n"


def test_integration_codex_home_override_and_redaction(codex_discovery, tmp_path, monkeypatch):
    _, _, client = codex_discovery
    custom = tmp_path / "custom codex"
    custom.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(custom))
    (custom / "config.toml").write_text('''
[mcp_servers.refs]
url = "https://example.test/mcp"
bearer_token_env_var = "TEST_TOKEN"
env_http_headers = { "X-Api-Key" = "TEST_KEY" }
[mcp_servers.secret]
url = "https://example.test/mcp"
http_headers = { Cookie = "opaque-inline-cookie" }
[mcp_servers.unsupported]
command = "node"
http_headers_helper = ["never-execute-this"]
''', encoding="utf-8")
    response = client.post("/api/mcp/discovery?profile=target")
    assert response.status_code == 200
    by_name = {c["name"]: c for c in response.json()["candidates"]}
    assert "shared" not in by_name
    assert by_name["refs"]["connectable"] is True
    assert not by_name["secret"]["connectable"]
    assert not by_name["unsupported"]["connectable"]
    assert "opaque-inline-cookie" not in json.dumps(response.json())
    connected = client.post("/api/mcp/discovery/connect?profile=target", json={"candidate_id": by_name["refs"]["id"]})
    assert connected.status_code == 200
    saved = yaml.safe_load((tmp_path / ".hermes/profiles/target/config.yaml").read_text())["mcp_servers"]["refs"]
    assert saved["headers"] == {"Authorization": "Bearer ${TEST_TOKEN}", "X-Api-Key": "${TEST_KEY}"}


@pytest.mark.parametrize("invalid", ['[mcp_servers\nsecret-value = "bad"', 'mcp_servers = []'])
def test_integration_malformed_toml_does_not_hide_other_clients(codex_discovery, monkeypatch, invalid):
    config, _, client = codex_discovery
    config.write_text(invalid, encoding="utf-8")
    (Path.home() / ".claude.json").write_text(json.dumps({"mcpServers": {"healthy": {"command": "node"}}}), encoding="utf-8")
    response = client.post("/api/mcp/discovery?profile=target")
    assert response.status_code == 200
    assert any(c["name"] == "healthy" for c in response.json()["candidates"])
    assert response.json()["warnings"]
    assert "secret-value" not in response.text


@pytest.mark.parametrize("case", ["valid", "malformed", "oversized", "directory", "missing"])
def test_toml_reader_uses_existing_bounded_file_safety(tmp_path, case):
    from hermes_cli.mcp_discovery import _CONFIG_MAX_BYTES, _read_mapping

    path = tmp_path / "config.toml"
    if case == "valid":
        path.write_text('[mcp_servers.local]\ncommand = "node"\n', encoding="utf-8")
    elif case == "malformed":
        path.write_text('[mcp_servers\nsecret-value = "invalid"', encoding="utf-8")
    elif case == "oversized":
        path.write_bytes(b"#" * (_CONFIG_MAX_BYTES + 1))
    elif case == "directory":
        path.mkdir()
    warnings = []
    result = _read_mapping(path, yaml_file=False, toml_file=True, warnings=warnings)
    if case == "valid":
        assert result == {"mcp_servers": {"local": {"command": "node"}}}
        assert not warnings
    else:
        assert result is None
        assert bool(warnings) == (case != "missing")
    assert "secret-value" not in str(warnings)


def test_codex_config_path_honors_home_override(tmp_path, monkeypatch):
    from hermes_cli.mcp_discovery import _client_config_paths

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert dict(_client_config_paths())["Codex"] == tmp_path / ".codex" / "config.toml"
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom codex"))
    assert dict(_client_config_paths())["Codex"] == tmp_path / "custom codex" / "config.toml"
