"""
Tests for hermes_cli.mcp_config — ``hermes mcp`` subcommands.

These tests mock the MCP server connection layer so they run without
any actual MCP servers or API keys.
"""

import argparse
import json
import os
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect all config I/O to a temp directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home", lambda: tmp_path
    )
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(
        "hermes_cli.config.get_config_path", lambda: config_path
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_env_path", lambda: env_path
    )
    return tmp_path


def _make_args(**kwargs):
    """Build a minimal argparse.Namespace."""
    defaults = {
        "name": "test-server",
        "url": None,
        "mcp_command": None,
        "args": None,
        "auth": None,
        "preset": None,
        "env": None,
        "mcp_action": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _seed_config(tmp_path: Path, mcp_servers: dict):
    """Write a config.yaml with the given mcp_servers."""
    import yaml

    config = {"mcp_servers": mcp_servers, "_config_version": 9}
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


class FakeTool:
    """Mimics an MCP tool object returned by the SDK."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_list
# ---------------------------------------------------------------------------

class TestMcpList:
    def test_list_empty_config(self, tmp_path, capsys):
        from hermes_cli.mcp_config import cmd_mcp_list

        cmd_mcp_list()
        out = capsys.readouterr().out
        assert "No MCP servers configured" in out

    def test_list_with_servers(self, tmp_path, capsys):
        _seed_config(tmp_path, {
            "ink": {
                "url": "https://mcp.ml.ink/mcp",
                "enabled": True,
                "tools": {"include": ["create_service", "get_service"]},
            },
            "github": {
                "command": "npx",
                "args": ["@mcp/github"],
                "enabled": False,
            },
        })
        from hermes_cli.mcp_config import cmd_mcp_list

        cmd_mcp_list()
        out = capsys.readouterr().out
        assert "ink" in out
        assert "github" in out
        assert "2 selected" in out  # ink has 2 in include
        assert "disabled" in out  # github is disabled

    def test_list_enabled_default_true(self, tmp_path, capsys):
        """Server without explicit enabled key defaults to enabled."""
        _seed_config(tmp_path, {
            "myserver": {"url": "https://example.com/mcp"},
        })
        from hermes_cli.mcp_config import cmd_mcp_list

        cmd_mcp_list()
        out = capsys.readouterr().out
        assert "myserver" in out
        assert "enabled" in out


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_remove
# ---------------------------------------------------------------------------

class TestMcpRemove:
    def test_remove_existing_server(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "myserver": {"url": "https://example.com/mcp"},
        })
        monkeypatch.setattr("builtins.input", lambda _: "y")
        from hermes_cli.mcp_config import cmd_mcp_remove

        cmd_mcp_remove(_make_args(name="myserver"))

        out = capsys.readouterr().out
        assert "Removed" in out

        # Verify config updated
        from hermes_cli.config import load_config

        config = load_config()
        assert "myserver" not in config.get("mcp_servers", {})

    def test_remove_nonexistent(self, tmp_path, capsys):
        _seed_config(tmp_path, {})
        from hermes_cli.mcp_config import cmd_mcp_remove

        cmd_mcp_remove(_make_args(name="ghost"))
        out = capsys.readouterr().out
        assert "not found" in out

    def test_remove_cleans_oauth_tokens(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "oauth-srv": {"url": "https://example.com/mcp", "auth": "oauth"},
        })
        monkeypatch.setattr("builtins.input", lambda _: "y")
        # Also patch get_hermes_home in the mcp_config module namespace
        monkeypatch.setattr(
            "hermes_cli.mcp_config.get_hermes_home", lambda: tmp_path
        )

        # Create a fake token file
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir()
        token_file = token_dir / "oauth-srv.json"
        token_file.write_text("{}")

        from hermes_cli.mcp_config import cmd_mcp_remove

        cmd_mcp_remove(_make_args(name="oauth-srv"))
        assert not token_file.exists()


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_add
# ---------------------------------------------------------------------------

class TestMcpAdd:
    def test_add_no_transport(self, capsys):
        """Must specify --url or --command."""
        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(name="bad"))
        out = capsys.readouterr().out
        assert "Must specify" in out

    def test_add_http_server_all_tools(self, tmp_path, capsys, monkeypatch):
        """Add an HTTP server, accept all tools."""
        fake_tools = [
            FakeTool("create_service", "Deploy from repo"),
            FakeTool("list_services", "List all services"),
        ]

        def mock_probe(name, config, **kw):
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        # No auth, accept all tools
        inputs = iter(["n", ""])  # no auth needed, enable all
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(name="ink", url="https://mcp.ml.ink/mcp"))
        out = capsys.readouterr().out
        assert "Saved" in out
        assert "2/2 tools" in out

        # Verify config written
        from hermes_cli.config import load_config

        config = load_config()
        assert "ink" in config.get("mcp_servers", {})
        assert config["mcp_servers"]["ink"]["url"] == "https://mcp.ml.ink/mcp"

    def test_add_stdio_server(self, tmp_path, capsys, monkeypatch):
        """Add a stdio server."""
        fake_tools = [FakeTool("search", "Search repos")]

        def mock_probe(name, config, **kw):
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        inputs = iter([""])  # accept all tools
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(
            name="github",
            mcp_command="npx",
            args=["@mcp/github"],
        ))
        out = capsys.readouterr().out
        assert "Saved" in out

        from hermes_cli.config import load_config

        config = load_config()
        srv = config["mcp_servers"]["github"]
        assert srv["command"] == "npx"
        assert srv["args"] == ["@mcp/github"]

    def test_add_connection_failure_save_disabled(
        self, tmp_path, capsys, monkeypatch
    ):
        """Failed connection → option to save as disabled."""

        def mock_probe_fail(name, config, **kw):
            raise ConnectionError("Connection refused")

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe_fail
        )
        inputs = iter(["n", "y"])  # no auth, yes save disabled
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(name="broken", url="https://bad.host/mcp"))
        out = capsys.readouterr().out
        assert "disabled" in out

        from hermes_cli.config import load_config

        config = load_config()
        assert config["mcp_servers"]["broken"]["enabled"] is False

    def test_add_stdio_server_with_env(self, tmp_path, capsys, monkeypatch):
        """Stdio servers can persist explicit environment variables."""
        fake_tools = [FakeTool("search", "Search repos")]

        def mock_probe(name, config, **kw):
            assert config["env"] == {
                "MY_API_KEY": "secret123",
                "DEBUG": "true",
            }
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(
            name="github",
            mcp_command="npx",
            args=["@mcp/github"],
            env=["MY_API_KEY=secret123", "DEBUG=true"],
        ))
        out = capsys.readouterr().out
        assert "Saved" in out

        from hermes_cli.config import load_config

        config = load_config()
        srv = config["mcp_servers"]["github"]
        assert srv["env"] == {
            "MY_API_KEY": "secret123",
            "DEBUG": "true",
        }

    def test_add_stdio_server_rejects_invalid_env_name(self, capsys):
        """Invalid environment variable names are rejected up front."""
        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(
            name="github",
            mcp_command="npx",
            args=["@mcp/github"],
            env=["BAD-NAME=value"],
        ))
        out = capsys.readouterr().out
        assert "Invalid --env variable name" in out

    def test_add_http_server_rejects_env_flag(self, capsys):
        """The --env flag is only valid for stdio transports."""
        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(
            name="ink",
            url="https://mcp.ml.ink/mcp",
            env=["DEBUG=true"],
        ))
        out = capsys.readouterr().out
        assert "only supported for stdio MCP servers" in out

    def test_add_preset_fills_transport(self, tmp_path, capsys, monkeypatch):
        """A preset fills in command/args when no explicit transport given."""
        monkeypatch.setattr(
            "hermes_cli.mcp_config._MCP_PRESETS",
            {"testmcp": {"command": "npx", "args": ["-y", "test-mcp-server"], "display_name": "Test MCP"}},
        )
        fake_tools = [FakeTool("do_thing", "Does a thing")]

        def mock_probe(name, config, **kw):
            assert name == "myserver"
            assert config["command"] == "npx"
            assert config["args"] == ["-y", "test-mcp-server"]
            assert "env" not in config
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        from hermes_cli.mcp_config import cmd_mcp_add
        from hermes_cli.config import read_raw_config

        cmd_mcp_add(_make_args(name="myserver", preset="testmcp"))
        out = capsys.readouterr().out
        assert "Saved" in out

        config = read_raw_config()
        srv = config["mcp_servers"]["myserver"]
        assert srv["command"] == "npx"
        assert srv["args"] == ["-y", "test-mcp-server"]
        assert "env" not in srv

    def test_preset_does_not_override_explicit_command(self, tmp_path, capsys, monkeypatch):
        """Explicit transports win over presets."""
        monkeypatch.setattr(
            "hermes_cli.mcp_config._MCP_PRESETS",
            {"testmcp": {"command": "npx", "args": ["-y", "test-mcp-server"], "display_name": "Test MCP"}},
        )
        fake_tools = [FakeTool("search", "Search repos")]

        def mock_probe(name, config, **kw):
            assert config["command"] == "uvx"
            assert config["args"] == ["custom-server"]
            assert "env" not in config
            return [(t.name, t.description) for t in fake_tools]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        from hermes_cli.mcp_config import cmd_mcp_add
        from hermes_cli.config import read_raw_config

        cmd_mcp_add(_make_args(
            name="custom",
            preset="testmcp",
            mcp_command="uvx",
            args=["custom-server"],
        ))
        out = capsys.readouterr().out
        assert "Saved" in out

        config = read_raw_config()
        srv = config["mcp_servers"]["custom"]
        assert srv["command"] == "uvx"
        assert srv["args"] == ["custom-server"]
        assert "env" not in srv

    def test_unknown_preset_rejected(self, capsys):
        """An unknown preset name is rejected with a clear error."""
        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(name="foo", preset="nonexistent"))
        out = capsys.readouterr().out
        assert "Unknown MCP preset" in out


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_test
# ---------------------------------------------------------------------------

class TestMcpTest:
    def test_test_not_found(self, tmp_path, capsys):
        _seed_config(tmp_path, {})
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(_make_args(name="ghost"))
        out = capsys.readouterr().out
        assert "not found" in out

    def test_test_success(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "ink": {"url": "https://mcp.ml.ink/mcp"},
        })

        def mock_probe(name, config, **kw):
            return [("create_service", "Deploy"), ("list_services", "List all")]

        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server", mock_probe
        )
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(_make_args(name="ink"))
        out = capsys.readouterr().out
        assert "Connected" in out
        assert "Tools discovered: 2" in out


# ---------------------------------------------------------------------------
# Tests: env var interpolation
# ---------------------------------------------------------------------------

class TestEnvVarInterpolation:
    def test_interpolate_simple(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        from tools.mcp_tool import _interpolate_env_vars

        result = _interpolate_env_vars("Bearer ${MY_KEY}")
        assert result == "Bearer secret123"

    def test_interpolate_missing_var(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        from tools.mcp_tool import _interpolate_env_vars

        result = _interpolate_env_vars("Bearer ${MISSING_VAR}")
        assert result == "Bearer ${MISSING_VAR}"

    def test_interpolate_nested_dict(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "abc")
        from tools.mcp_tool import _interpolate_env_vars

        result = _interpolate_env_vars({
            "url": "https://example.com",
            "headers": {"Authorization": "Bearer ${API_KEY}"},
        })
        assert result["headers"]["Authorization"] == "Bearer abc"
        assert result["url"] == "https://example.com"

    def test_interpolate_list(self, monkeypatch):
        monkeypatch.setenv("ARG1", "hello")
        from tools.mcp_tool import _interpolate_env_vars

        result = _interpolate_env_vars(["${ARG1}", "static"])
        assert result == ["hello", "static"]

    def test_interpolate_non_string(self):
        from tools.mcp_tool import _interpolate_env_vars

        assert _interpolate_env_vars(42) == 42
        assert _interpolate_env_vars(True) is True
        assert _interpolate_env_vars(None) is None


# ---------------------------------------------------------------------------
# Tests: config helpers
# ---------------------------------------------------------------------------

class TestConfigHelpers:
    def test_save_and_load_mcp_server(self, tmp_path):
        from hermes_cli.mcp_config import _save_mcp_server, _get_mcp_servers

        _save_mcp_server("mysvr", {"url": "https://example.com/mcp"})
        servers = _get_mcp_servers()
        assert "mysvr" in servers
        assert servers["mysvr"]["url"] == "https://example.com/mcp"

    def test_remove_mcp_server(self, tmp_path):
        from hermes_cli.mcp_config import (
            _save_mcp_server,
            _remove_mcp_server,
            _get_mcp_servers,
        )

        _save_mcp_server("s1", {"command": "test"})
        _save_mcp_server("s2", {"command": "test2"})
        result = _remove_mcp_server("s1")
        assert result is True
        assert "s1" not in _get_mcp_servers()
        assert "s2" in _get_mcp_servers()

    def test_remove_nonexistent(self, tmp_path):
        from hermes_cli.mcp_config import _remove_mcp_server

        assert _remove_mcp_server("ghost") is False

    def test_env_key_for_server(self):
        from hermes_cli.mcp_config import _env_key_for_server

        assert _env_key_for_server("ink") == "MCP_INK_API_KEY"
        assert _env_key_for_server("my-server") == "MCP_MY_SERVER_API_KEY"


# ---------------------------------------------------------------------------
# Tests: dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_no_action_shows_list(self, tmp_path, capsys):
        from hermes_cli.mcp_config import mcp_command

        _seed_config(tmp_path, {})
        mcp_command(_make_args(mcp_action=None))
        out = capsys.readouterr().out
        assert "Commands:" in out or "No MCP servers" in out


# ---------------------------------------------------------------------------
# Tests: Task 7 consolidation — cmd_mcp_remove evicts manager cache,
# cmd_mcp_login forces re-auth
# ---------------------------------------------------------------------------


class TestMcpRemoveEvictsManager:
    def test_remove_evicts_in_memory_provider(self, tmp_path, capsys, monkeypatch):
        """After cmd_mcp_remove, the MCPOAuthManager no longer caches the provider."""
        _seed_config(tmp_path, {
            "oauth-srv": {"url": "https://example.com/mcp", "auth": "oauth"},
        })
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "hermes_cli.mcp_config.get_hermes_home", lambda: tmp_path
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
        reset_manager_for_tests()

        mgr = get_manager()
        mgr.get_or_build_provider(
            "oauth-srv", "https://example.com/mcp", None,
        )
        assert "oauth-srv" in mgr._entries

        from hermes_cli.mcp_config import cmd_mcp_remove
        cmd_mcp_remove(_make_args(name="oauth-srv"))

        assert "oauth-srv" not in mgr._entries


class TestMcpLogin:
    def test_login_rejects_unknown_server(self, tmp_path, capsys):
        _seed_config(tmp_path, {})
        from hermes_cli.mcp_config import cmd_mcp_login
        cmd_mcp_login(_make_args(name="ghost"))
        out = capsys.readouterr().out
        assert "not found" in out

    def test_login_rejects_non_oauth_server(self, tmp_path, capsys):
        _seed_config(tmp_path, {
            "srv": {"url": "https://example.com/mcp", "auth": "header"},
        })
        from hermes_cli.mcp_config import cmd_mcp_login
        cmd_mcp_login(_make_args(name="srv"))
        out = capsys.readouterr().out
        assert "not configured for OAuth" in out

    def test_login_rejects_stdio_server(self, tmp_path, capsys):
        _seed_config(tmp_path, {
            "srv": {"command": "npx", "args": ["some-server"]},
        })
        from hermes_cli.mcp_config import cmd_mcp_login
        cmd_mcp_login(_make_args(name="srv"))
        out = capsys.readouterr().out
        assert "no URL" in out or "not an OAuth" in out

