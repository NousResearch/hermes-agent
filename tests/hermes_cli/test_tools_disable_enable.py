"""Tests for hermes tools disable/enable/list command (backend)."""
from argparse import Namespace
from unittest.mock import patch

from hermes_cli.tools_config import build_tools_diagnostics, tools_disable_enable_command


# ── Built-in toolset disable ────────────────────────────────────────────────


class TestToolsDisableBuiltin:

    def test_disable_removes_toolset_from_platform(self):
        config = {"platform_toolsets": {"cli": ["web", "memory", "terminal"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="disable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]

    def test_disable_multiple_toolsets(self):
        config = {"platform_toolsets": {"cli": ["web", "memory", "terminal"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="disable", names=["web", "memory"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" not in saved["platform_toolsets"]["cli"]
        assert "terminal" in saved["platform_toolsets"]["cli"]

    def test_disable_already_absent_is_idempotent(self):
        config = {"platform_toolsets": {"cli": ["memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="disable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]


# ── Built-in toolset enable ─────────────────────────────────────────────────


class TestToolsEnableBuiltin:

    def test_enable_adds_toolset_to_platform(self):
        config = {"platform_toolsets": {"cli": ["memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="enable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert "web" in saved["platform_toolsets"]["cli"]

    def test_enable_already_present_is_idempotent(self):
        config = {"platform_toolsets": {"cli": ["web"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(tools_action="enable", names=["web"], platform="cli"))
        saved = mock_save.call_args[0][0]
        assert saved["platform_toolsets"]["cli"].count("web") == 1


# ── MCP tool disable ────────────────────────────────────────────────────────


class TestToolsDisableMcp:

    def test_disable_adds_to_exclude_list(self):
        config = {"mcp_servers": {"github": {"command": "npx"}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["github:create_issue"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "create_issue" in saved["mcp_servers"]["github"]["tools"]["exclude"]

    def test_disable_already_excluded_is_idempotent(self):
        config = {"mcp_servers": {"github": {"tools": {"exclude": ["create_issue"]}}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["github:create_issue"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert saved["mcp_servers"]["github"]["tools"]["exclude"].count("create_issue") == 1

    def test_disable_unknown_server_prints_error(self, capsys):
        config = {"mcp_servers": {}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config"):
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["unknown:tool"], platform="cli")
            )
        out = capsys.readouterr().out
        assert "MCP server 'unknown' not found in config" in out


# ── MCP tool enable ──────────────────────────────────────────────────────────


class TestToolsEnableMcp:

    def test_enable_removes_from_exclude_list(self):
        config = {"mcp_servers": {"github": {"tools": {"exclude": ["create_issue", "delete_branch"]}}}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="enable", names=["github:create_issue"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "create_issue" not in saved["mcp_servers"]["github"]["tools"]["exclude"]
        assert "delete_branch" in saved["mcp_servers"]["github"]["tools"]["exclude"]


# ── Mixed targets ────────────────────────────────────────────────────────────


class TestToolsMixedTargets:

    def test_disable_builtin_and_mcp_together(self):
        config = {
            "platform_toolsets": {"cli": ["web", "memory"]},
            "mcp_servers": {"github": {"command": "npx"}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(
                tools_action="disable",
                names=["web", "github:create_issue"],
                platform="cli",
            ))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "create_issue" in saved["mcp_servers"]["github"]["tools"]["exclude"]

    def test_builtin_toggle_does_not_persist_implicit_mcp_defaults(self):
        config = {
            "platform_toolsets": {"cli": ["web", "memory"]},
            "mcp_servers": {"exa": {"url": "https://mcp.exa.ai/mcp"}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(Namespace(
                tools_action="disable",
                names=["web"],
                platform="cli",
            ))
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]
        assert "exa" not in saved["platform_toolsets"]["cli"]


# ── List output ──────────────────────────────────────────────────────────────


class TestToolsList:

    def test_list_shows_enabled_toolsets(self, capsys):
        config = {"platform_toolsets": {"cli": ["web", "memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config):
            tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
        out = capsys.readouterr().out
        assert "web" in out
        assert "memory" in out

    def test_list_shows_mcp_excluded_tools(self, capsys):
        config = {
            "mcp_servers": {"github": {"tools": {"exclude": ["create_issue"]}}},
        }
        with patch("hermes_cli.tools_config.load_config", return_value=config):
            tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
        out = capsys.readouterr().out
        assert "github" in out
        assert "create_issue" in out


# ── Diagnostics ──────────────────────────────────────────────────────────────


class TestToolsDiagnose:

    def test_build_diagnostics_reports_visible_and_disabled_tools(self):
        config = {
            "platform_toolsets": {"cli": ["file"]},
            "tools": {"tool_search": {"enabled": "off"}},
        }

        diag = build_tools_diagnostics(config, "cli")

        assert diag["platform"] == "cli"
        assert "file" in diag["enabled_toolsets"]
        assert "read_file" in diag["tools_visible"]
        assert {
            "tool": "terminal",
            "toolset": "terminal",
            "reason": "toolset disabled",
        } in diag["filtered"]

    def test_build_diagnostics_reports_external_memory_and_context_tools(self):
        config = {
            "memory": {"provider": "stub-memory"},
            "context": {"engine": "stub-context"},
            "platform_toolsets": {
                "cli": ["file", "memory", "context_engine"]
            },
            "tools": {"tool_search": {"enabled": "off"}},
        }
        memory_schema = {
            "name": "stub_memory_recall",
            "description": "Recall",
            "parameters": {"type": "object", "properties": {}},
        }
        context_schema = {
            "name": "stub_context_expand",
            "description": "Expand",
            "parameters": {"type": "object", "properties": {}},
        }

        with patch(
            "hermes_cli.tools_config._diagnostic_memory_provider",
            return_value=("stub-memory", [memory_schema], None),
        ), patch(
            "hermes_cli.tools_config._diagnostic_context_engine",
            return_value=("stub-context", [context_schema], None),
        ):
            diag = build_tools_diagnostics(config, "cli")

        assert "stub_memory_recall" in diag["tools_visible"]
        assert "stub_context_expand" in diag["tools_visible"]
        assert diag["provider_tools"]["memory"]["injected"] == 1
        assert diag["provider_tools"]["context_engine"]["injected"] == 1

    def test_build_diagnostics_reports_external_family_toolset_gates(self):
        config = {
            "memory": {"provider": "stub-memory"},
            "context": {"engine": "stub-context"},
            "platform_toolsets": {"cli": ["file"]},
            "agent": {"disabled_toolsets": ["memory", "context_engine"]},
            "tools": {"tool_search": {"enabled": "off"}},
        }
        memory_schema = {"name": "gated_memory", "parameters": {}}
        context_schema = {"name": "gated_context", "parameters": {}}

        with patch(
            "hermes_cli.tools_config._diagnostic_memory_provider",
            return_value=("stub-memory", [memory_schema], None),
        ), patch(
            "hermes_cli.tools_config._diagnostic_context_engine",
            return_value=("stub-context", [context_schema], None),
        ):
            diag = build_tools_diagnostics(config, "cli")

        assert "gated_memory" not in diag["tools_visible"]
        assert "gated_context" not in diag["tools_visible"]
        assert diag["provider_tools"]["memory"]["skipped_reason"] == "toolset disabled"
        assert diag["provider_tools"]["context_engine"]["skipped_reason"] == "toolset disabled"

    def test_build_diagnostics_reports_activated_tool_search(self):
        from tools.registry import registry

        tool_name = "diagnose_deferred_tool"
        toolset = "mcp-diagnose-test"
        registry.register(
            name=tool_name,
            handler=lambda args, **kwargs: "{}",
            schema={
                "name": tool_name,
                "description": "Deferred diagnostic tool",
                "parameters": {"type": "object", "properties": {}},
            },
            toolset=toolset,
        )
        try:
            config = {
                "platform_toolsets": {"cli": [toolset]},
                "tools": {"tool_search": {"enabled": "on"}},
            }
            with patch(
                "hermes_cli.tools_config._get_platform_tools",
                return_value={toolset},
            ):
                diag = build_tools_diagnostics(config, "cli")
        finally:
            registry.deregister(tool_name)

        assert diag["tool_search"]["activated"] is True
        assert tool_name not in diag["tools_visible"]
        assert {"tool_search", "tool_describe", "tool_call"}.issubset(
            diag["tools_visible"]
        )
        assert {
            "tool": tool_name,
            "toolset": toolset,
            "reason": "deferred by tool search",
        } in diag["filtered"]

    def test_build_diagnostics_does_not_publish_process_global_resolution(
        self, monkeypatch
    ):
        import model_tools

        monkeypatch.setattr(model_tools, "_last_resolved_tool_names", ["runtime_tool"])
        config = {
            "platform_toolsets": {"cli": ["file"]},
            "tools": {"tool_search": {"enabled": "off"}},
        }

        build_tools_diagnostics(config, "cli")

        assert model_tools._last_resolved_tool_names == ["runtime_tool"]


# ── Validation ───────────────────────────────────────────────────────────────


class TestToolsValidation:

    def test_unknown_platform_prints_error(self, capsys):
        config = {}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config"):
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["web"], platform="invalid_platform")
            )
        out = capsys.readouterr().out
        assert "Unknown platform 'invalid_platform'" in out

    def test_unknown_toolset_prints_error(self, capsys):
        config = {"platform_toolsets": {"cli": ["web"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config"):
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["nonexistent_toolset"], platform="cli")
            )
        out = capsys.readouterr().out
        assert "Unknown toolset 'nonexistent_toolset'" in out

    def test_unknown_toolset_does_not_corrupt_config(self):
        config = {"platform_toolsets": {"cli": ["web", "memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["nonexistent_toolset"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "web" in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]

    def test_mixed_valid_and_invalid_applies_valid_only(self):
        config = {"platform_toolsets": {"cli": ["web", "memory"]}}
        with patch("hermes_cli.tools_config.load_config", return_value=config), \
             patch("hermes_cli.tools_config.save_config") as mock_save:
            tools_disable_enable_command(
                Namespace(tools_action="disable", names=["web", "bad_toolset"], platform="cli")
            )
        saved = mock_save.call_args[0][0]
        assert "web" not in saved["platform_toolsets"]["cli"]
        assert "memory" in saved["platform_toolsets"]["cli"]
