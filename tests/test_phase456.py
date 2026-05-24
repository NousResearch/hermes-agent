"""Tests for Phase 4-6 modules."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Phase 4: Config Unification Tests ───────────────────────────────

class TestConfigValidator:
    """Test config validation system."""

    def test_valid_terminal_config(self) -> None:
        from hermes_cli.config_validator import validate_config_section
        valid, errors = validate_config_section("terminal", {
            "env_type": "local",
            "timeout": 300,
        })
        assert valid
        assert errors == []

    def test_invalid_terminal_config(self) -> None:
        from hermes_cli.config_validator import validate_config_section
        valid, errors = validate_config_section("terminal", {
            "timeout": "not_a_number",
        })
        assert not valid
        assert any("timeout" in e["field"] for e in errors)

    def test_apply_defaults(self) -> None:
        from hermes_cli.config_validator import apply_config_defaults
        result = apply_config_defaults({})
        assert result["terminal"]["env_type"] == "local"
        assert result["terminal"]["timeout"] == 300
        assert result["logging"]["audit_enabled"] is True

    def test_validate_full_config(self) -> None:
        from hermes_cli.config_validator import validate_full_config
        valid, errors = validate_full_config({
            "terminal": {"env_type": "local"},
            "logging": {"audit_enabled": True},
        })
        assert valid

    def test_get_validation_report_pass(self) -> None:
        from hermes_cli.config_validator import get_validation_report
        report = get_validation_report({"terminal": {"env_type": "local"}})
        assert "passed" in report.lower()

    def test_get_validation_report_fail(self) -> None:
        from hermes_cli.config_validator import get_validation_report
        report = get_validation_report({"terminal": {"timeout": "invalid"}})
        assert "failed" in report.lower()


# ── Phase 5: Plugin SDK Tests ──────────────────────────────────────

class TestPluginSDK:
    """Test plugin SDK."""

    def test_plugin_context_creation(self) -> None:
        from hermes_cli.plugin_sdk import PluginContext, PluginInfo
        ctx = PluginContext(plugin_info=PluginInfo(name="test", version="1.0.0"))
        assert ctx.plugin_info.name == "test"

    def test_plugin_context_register_tool(self) -> None:
        from hermes_cli.plugin_sdk import PluginContext, PluginInfo
        registered = []

        def mock_register(**kwargs):
            registered.append(kwargs)

        ctx = PluginContext(
            plugin_info=PluginInfo(name="test"),
            _register_tool_fn=mock_register,
        )
        ctx.register_tool("my_tool", "Description", {"name": "my_tool"}, lambda: None)
        assert len(registered) == 1
        assert registered[0]["name"] == "my_tool"

    def test_plugin_context_register_hook(self) -> None:
        from hermes_cli.plugin_sdk import PluginContext, PluginInfo
        hooks = []

        def mock_register(name, handler):
            hooks.append((name, handler))

        ctx = PluginContext(
            plugin_info=PluginInfo(name="test"),
            _register_hook_fn=mock_register,
        )
        ctx.register_hook("post_tool_call", lambda: None)
        assert len(hooks) == 1
        assert hooks[0][0] == "post_tool_call"

    def test_plugin_context_logging(self) -> None:
        from hermes_cli.plugin_sdk import PluginContext, PluginInfo
        ctx = PluginContext(plugin_info=PluginInfo(name="my-plugin"))
        # Just verify it doesn't crash — logging setup varies by test env
        ctx.log_info("test message")
        ctx.log_warning("warning message")
        ctx.log_error("error message")

    def test_sdk_list_plugins(self) -> None:
        from hermes_cli.plugin_sdk import PluginInfo, get_sdk
        sdk = get_sdk()
        sdk.create_context(PluginInfo(name="plugin-a"))
        sdk.create_context(PluginInfo(name="plugin-b"))
        plugins = sdk.list_plugins()
        assert "plugin-a" in plugins
        assert "plugin-b" in plugins

    def test_get_plugin_home(self, monkeypatch, tmp_path: Path) -> None:
        from hermes_cli.plugin_sdk import PluginContext, PluginInfo
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        ctx = PluginContext(plugin_info=PluginInfo(name="test-plugin"))
        home = ctx.get_plugin_home()
        assert "test-plugin" in home


# ── Phase 6: Swarm Tests ───────────────────────────────────────────

class TestSwarm:
    """Test multi-agent swarm coordination."""

    def test_create_operation(self) -> None:
        from agent.swarm import SwarmCoordinator
        coord = SwarmCoordinator()
        op = coord.create_operation(
            "test-op",
            [
                {"name": "task-1", "description": "First task", "type": "search"},
                {"name": "task-2", "description": "Second task", "type": "write"},
            ],
        )
        assert len(op.tasks) == 2
        assert op.name == "test-op"

    def test_operation_progress(self) -> None:
        from agent.swarm import SwarmTask, SwarmOperation, TaskStatus
        op = SwarmOperation()
        op.tasks = [
            SwarmTask(status=TaskStatus.COMPLETED),
            SwarmTask(status=TaskStatus.RUNNING),
            SwarmTask(status=TaskStatus.PENDING),
        ]
        assert op.progress == pytest.approx(33.33, abs=1)

    def test_get_status_report(self) -> None:
        from agent.swarm import SwarmCoordinator
        coord = SwarmCoordinator()
        op = coord.create_operation("report-test", [
            {"name": "t1", "description": "test", "type": "search"},
        ])
        report = coord.get_status_report(op.id)
        assert report["operation"] == "report-test"
        assert report["total"] == 1
        assert "tasks" in report

    def test_register_handler(self) -> None:
        from agent.swarm import SwarmCoordinator
        coord = SwarmCoordinator()
        coord.register_handler("search", lambda: None)
        assert "search" in coord._handlers

    def test_get_swarm_coordinator_singleton(self) -> None:
        from agent.swarm import get_swarm_coordinator
        c1 = get_swarm_coordinator()
        c2 = get_swarm_coordinator()
        assert c1 is c2


# ── Phase 6: Self-Healing Tests ────────────────────────────────────

class TestSelfHealing:
    """Test self-healing mechanism."""

    def test_detect_rate_limit(self) -> None:
        from agent.self_healing import SelfHealingEngine, FailureType
        engine = SelfHealingEngine()
        error = Exception("Rate limit exceeded: 429")
        assert engine.detect_failure(error) == FailureType.API_RATE_LIMIT

    def test_detect_context_overflow(self) -> None:
        from agent.self_healing import SelfHealingEngine, FailureType
        engine = SelfHealingEngine()
        error = Exception("context length exceeded")
        assert engine.detect_failure(error) == FailureType.CONTEXT_OVERFLOW

    def test_detect_network_timeout(self) -> None:
        from agent.self_healing import SelfHealingEngine, FailureType
        engine = SelfHealingEngine()
        error = Exception("Connection timeout")
        assert engine.detect_failure(error) == FailureType.NETWORK_TIMEOUT

    def test_detect_auth_expired(self) -> None:
        from agent.self_healing import SelfHealingEngine, FailureType
        engine = SelfHealingEngine()
        error = Exception("Unauthorized: 401")
        assert engine.detect_failure(error) == FailureType.AUTH_EXPIRED

    def test_detect_unknown(self) -> None:
        from agent.self_healing import SelfHealingEngine
        engine = SelfHealingEngine()
        error = Exception("Some unknown error")
        assert engine.detect_failure(error) is None

    def test_recover_rate_limit(self) -> None:
        from agent.self_healing import SelfHealingEngine, FailureType
        engine = SelfHealingEngine(enabled=False)  # skip actual sleep
        action = engine.recover(FailureType.API_RATE_LIMIT)
        # With disabled, returns immediately
        assert action.success is False

    def test_health_report(self) -> None:
        from agent.self_healing import SelfHealingEngine
        engine = SelfHealingEngine()
        report = engine.get_health_report()
        assert "is_healthy" in report
        assert "recovery_count" in report
        assert "recent_recoveries" in report

    def test_get_self_healing_singleton(self) -> None:
        from agent.self_healing import get_self_healing_engine
        e1 = get_self_healing_engine()
        e2 = get_self_healing_engine()
        assert e1 is e2
