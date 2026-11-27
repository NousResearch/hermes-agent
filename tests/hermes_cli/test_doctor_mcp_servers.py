"""Tests for the ``MCP Servers`` preflight section of ``hermes doctor``."""

import contextlib
import io
import os
import sys
from pathlib import Path

import pytest
import yaml

import hermes_cli.doctor_mcp as doctor


def _run(monkeypatch, servers, *, resolve=None, safe_env=None):
    """Run the preflight against config and return its report."""
    monkeypatch.setattr(doctor, "load_config", lambda: {"mcp_servers": servers})
    if resolve is not None:
        monkeypatch.setattr(doctor, "_resolve_stdio_command", resolve)
    if safe_env is not None:
        monkeypatch.setattr(doctor, "_build_safe_env", safe_env)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        finding = doctor._check_mcp_servers(False)
    return buf.getvalue(), finding.manual_issues


def _executable(tmp_path, name="srv"):
    binary = tmp_path / name
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


class TestNoServers:
    def test_reports_when_nothing_configured(self, monkeypatch):
        out, issues = _run(monkeypatch, {})
        assert "No MCP servers configured" in out
        assert issues == []


class TestTransportValidation:
    @pytest.mark.parametrize("servers", [[], ["not", "a", "mapping"]])
    def test_mcp_servers_section_must_be_mapping(self, monkeypatch, servers):
        out, issues = _run(monkeypatch, servers)
        assert "mcp_servers: malformed configuration" in out
        assert "No MCP servers configured" not in out
        assert any("mapping of server names" in i for i in issues)

    def test_missing_transport_fails(self, monkeypatch):
        out, issues = _run(monkeypatch, {"broken": {"timeout": 30}})
        assert "no transport configured" in out
        assert any("broken" in i for i in issues)

    def test_non_mapping_entry_fails(self, monkeypatch):
        out, issues = _run(monkeypatch, {"weird": ["not", "a", "mapping"]})
        assert "malformed entry" in out
        assert any("weird" in i for i in issues)
        assert "No MCP servers configured" not in out

    def test_both_url_and_command_warns_and_prefers_http(self, monkeypatch):
        out, _ = _run(
            monkeypatch,
            {"mixed": {"url": "https://example.com/mcp", "command": "npx"}},
        )
        assert "both 'url' and 'command' set" in out
        # HTTP wins, so the entry is validated as an http server.
        assert "(http)" in out


class TestHttpServers:
    def test_valid_url_passes(self, monkeypatch):
        out, issues = _run(
            monkeypatch, {"remote": {"url": "https://mcp.example.com/mcp"}}
        )
        assert "MCP server 'remote' (http)" in out
        assert issues == []

    def test_invalid_url_fails(self, monkeypatch):
        out, issues = _run(monkeypatch, {"remote": {"url": "example.com/mcp"}})
        assert "invalid url" in out
        assert any("remote" in i for i in issues)

    def test_non_mapping_headers_fails_without_success(self, monkeypatch):
        out, issues = _run(
            monkeypatch,
            {"remote": {"url": "https://mcp.example.com", "headers": "Bearer x"}},
        )
        assert "'headers' is not a mapping" in out
        assert "url and headers look valid" not in out
        assert any("remote.headers" in i for i in issues)

    def test_non_string_header_value_fails(self, monkeypatch):
        out, issues = _run(
            monkeypatch,
            {"remote": {"url": "https://mcp.example.com", "headers": {"X-Retry": 3}}},
        )
        assert "invalid header name/value" in out
        assert "url and headers look valid" not in out
        assert any("remote.headers" in i for i in issues)


class TestStdioServers:
    def test_resolvable_command_passes(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path, "my-mcp")

        out, issues = _run(
            monkeypatch,
            {"local": {"command": "my-mcp", "args": ["--stdio"]}},
            resolve=lambda cmd, env: (str(binary), env),
        )
        assert "MCP server 'local' (stdio)" in out
        assert issues == []

    def test_unresolved_command_fails_with_path_hint(self, monkeypatch):
        # _resolve_stdio_command echoes the bare name back when nothing matches.
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "definitely-not-installed"}},
            resolve=lambda cmd, env: (cmd, env),
        )
        assert "not found on PATH" in out
        assert any("local.command" in i for i in issues)

    def test_resolved_but_missing_file_fails(self, monkeypatch, tmp_path):
        ghost = tmp_path / "gone" / "server"
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "server"}},
            resolve=lambda cmd, env: (str(ghost), env),
        )
        assert "command not found" in out
        assert any("local.command" in i for i in issues)

    def test_non_string_command_fails(self, monkeypatch):
        out, issues = _run(monkeypatch, {"local": {"command": None}})
        assert "must be a non-empty string" in out
        assert any("local" in i for i in issues)

    def test_non_list_args_fails_without_success(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path)
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "args": "--stdio"}},
            resolve=lambda cmd, env: (str(binary), env),
        )
        assert "invalid 'args'" in out
        assert "MCP server 'local' (stdio)" not in out
        assert any("local.args" in i for i in issues)

    def test_non_string_arg_fails(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path)
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "args": ["--port", 3000]}},
            resolve=lambda cmd, env: (str(binary), env),
        )
        assert "invalid 'args'" in out
        assert "MCP server 'local' (stdio)" not in out
        assert any("local.args" in i for i in issues)

    def test_empty_declared_env_warns(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path)
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "env": {"API_KEY": ""}}},
            resolve=lambda cmd, env: (str(binary), env),
            safe_env=lambda user_env: {"API_KEY": ""},
        )
        assert "empty env value(s)" in out
        assert "API_KEY" in out
        assert "✓ MCP server 'local'" not in out
        assert any("API_KEY" in i for i in issues)

    def test_unresolved_env_reference_fails(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path)
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "env": {"API_KEY": "${MISSING_KEY}"}}},
            resolve=lambda cmd, env: (str(binary), env),
            safe_env=lambda user_env: dict(user_env),
        )
        assert "unresolved env reference(s)" in out
        assert "API_KEY" in out
        assert "✓ MCP server 'local'" not in out
        assert any("API_KEY" in i for i in issues)

    @pytest.mark.parametrize("env", ["API_KEY=value", ["API_KEY=value"]])
    def test_non_mapping_env_fails(self, monkeypatch, tmp_path, env):
        binary = _executable(tmp_path)
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "env": env}},
            resolve=lambda cmd, child_env: (str(binary), child_env),
        )
        assert "'env' is not a mapping" in out
        assert "MCP server 'local' (stdio)" not in out
        assert any("local.env" in i for i in issues)

    def test_non_string_env_value_fails(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path)
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "env": {"API_KEY": None}}},
            resolve=lambda cmd, env: (str(binary), env),
        )
        assert "invalid env name/value" in out
        assert "MCP server 'local' (stdio)" not in out
        assert any("local.env" in i for i in issues)

    def test_safe_env_failure_is_not_replaced_with_parent_env(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path)

        def _boom(_user_env):
            raise ValueError("invalid environment")

        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "env": {"API_KEY": "set"}}},
            resolve=lambda cmd, env: (str(binary), env),
            safe_env=_boom,
        )
        assert "could not build child environment" in out
        assert "MCP server 'local' (stdio)" not in out
        assert any("local.env" in i for i in issues)

    def test_populated_declared_env_passes(self, monkeypatch, tmp_path):
        binary = _executable(tmp_path)
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv", "env": {"API_KEY": "set"}}},
            resolve=lambda cmd, env: (str(binary), env),
            safe_env=lambda user_env: {"API_KEY": "set"},
        )
        assert "MCP server 'local' (stdio)" in out
        assert issues == []

    def test_directory_command_fails(self, monkeypatch, tmp_path):
        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv"}},
            resolve=lambda cmd, env: (str(tmp_path), env),
        )
        assert "command not found" in out
        assert "MCP server 'local' (stdio)" not in out
        assert any("existing executable file" in i for i in issues)

    @pytest.mark.linux_only
    def test_non_executable_command_fails(self, monkeypatch, tmp_path):
        binary = tmp_path / "srv"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o644)

        out, issues = _run(
            monkeypatch,
            {"local": {"command": "srv"}},
            resolve=lambda cmd, env: (str(binary), env),
        )
        assert "command is not executable" in out
        assert "MCP server 'local' (stdio)" not in out
        assert any("executable" in i for i in issues)


class TestResilience:
    def test_config_read_failure_warns_but_does_not_raise(self, monkeypatch):
        def _boom():
            raise RuntimeError("config exploded")

        monkeypatch.setattr(doctor, "load_config", _boom)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            finding = doctor._check_mcp_servers(False)
        assert "Could not read mcp_servers" in buf.getvalue()
        assert finding.manual_issues

    @pytest.mark.parametrize("should_fix", [False, True])
    def test_makes_no_subprocess_or_socket_calls(self, monkeypatch, tmp_path, should_fix):
        """The preflight is static: it must not launch or dial anything."""
        import socket
        import subprocess

        from hermes_cli.env_loader import load_hermes_dotenv

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("DOCTOR_TEST_TOKEN", "")  # Restore the environment after dotenv loads it.
        servers = {
            "local": {"command": Path(sys.executable).name,
                      "env": {"PATH": str(Path(sys.executable).parent),
                              "API_KEY": "${DOCTOR_TEST_TOKEN}"}},
            "remote": {"url": "https://mcp.example.com/mcp",
                       "headers": {"Authorization": "Bearer ${DOCTOR_TEST_TOKEN}"}},
        }
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")
        (tmp_path / ".env").write_text("DOCTOR_TEST_TOKEN=preflight-private-canary\n", encoding="utf-8")
        load_hermes_dotenv(hermes_home=tmp_path, project_env=tmp_path / ".env")
        before = (tmp_path / "config.yaml").read_bytes()
        calls = []

        def _fail(*a, **k):
            calls.append(True)
            raise AssertionError("preflight must not spawn subprocesses")

        monkeypatch.setattr(subprocess, "run", _fail)
        monkeypatch.setattr(subprocess, "Popen", _fail)
        monkeypatch.setattr(socket, "create_connection", _fail)
        monkeypatch.setattr(socket.socket, "connect", _fail)
        monkeypatch.setattr(os, "system", _fail)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            finding = doctor._check_mcp_servers(should_fix)
        out = buf.getvalue()
        assert "(stdio)" in out and "(http)" in out
        assert finding.manual_issues == []
        assert calls == []  # Includes attempts swallowed by best-effort runtime helpers.
        assert "preflight-private-canary" not in out
        assert (tmp_path / "config.yaml").read_bytes() == before

    @pytest.mark.parametrize("enabled", [False, "false", "off"])
    def test_disabled_servers_do_not_need_launchable_config(self, tmp_path, monkeypatch, capsys, enabled):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {
            "disabled": {"enabled": enabled, "command": "not-installed", "env": {"TOKEN": "${UNSET_TOKEN}"}},
            "working": {"command": sys.executable},
        }}), encoding="utf-8")

        finding = doctor._check_mcp_servers(False)

        out = capsys.readouterr().out
        assert "disabled; launch checks skipped" in out
        assert "MCP server 'working' (stdio)" in out
        assert finding.manual_issues == []

    @pytest.mark.parametrize("entry", [
        {"command": "${DOCTOR_PRIVATE_VALUE}"},
        {"url": "ftp://user:${DOCTOR_PRIVATE_VALUE}@example.com/mcp"},
        {"url": "https://example.com/mcp", "headers": {"Authorization": "${DOCTOR_MISSING_VALUE}"}},
    ])
    def test_registered_check_reports_bad_config_without_values(self, tmp_path, monkeypatch, capsys, entry):
        from argparse import Namespace
        from hermes_cli import doctor as command

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("DOCTOR_PRIVATE_VALUE", "preflight-private-canary")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {
            "broken": entry, "working": {"command": sys.executable},
        }}), encoding="utf-8")
        checks = [(title, check) for title, check in command.DOCTOR_CHECKS if title == "MCP Servers"]
        assert checks, "the static check must be registered in hermes doctor"
        monkeypatch.setattr(command, "DOCTOR_CHECKS", checks)

        command.run_doctor(Namespace(fix=False, live=False))

        out = capsys.readouterr().out
        assert "mcp_servers.broken" in out  # Included in doctor's actionable summary.
        assert "MCP server 'working' (stdio)" in out
        assert "preflight-private-canary" not in out
