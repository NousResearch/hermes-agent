"""Tests for oneshot MCP tool discovery ordering (issue #38448)."""

from unittest.mock import MagicMock

import hermes_cli.tools_config as _tools_config_mod
import run_agent as _run_agent_mod
import tools.mcp_tool as _mcp_mod


def _cfg_dict(mcp_servers=None):
    """Return a minimal config dict the way oneshot._run_agent expects."""
    cfg = {
        "model": {"default": "test-model", "provider": "test-provider"},
        "platforms": {"cli": {"toolsets": ["core", "web"]}},
    }
    if mcp_servers is not None:
        cfg["mcp_servers"] = mcp_servers
    return cfg


def _stub_oneshot_dependencies(monkeypatch, cfg):
    """Patch dependencies that are unrelated to MCP discovery ordering."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {"provider": "test", "api_key": "***", "base_url": "http://x"},
    )
    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: MagicMock(),
    )
    monkeypatch.setattr("hermes_cli.oneshot.get_fallback_chain", lambda cfg: None)
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda model, current_provider: None,
    )

    mock_agent = MagicMock()
    mock_agent.chat.return_value = "ok"
    monkeypatch.setattr(_run_agent_mod, "AIAgent", lambda **kw: mock_agent)


class TestOneshotMcpDiscovery:
    def test_discover_mcp_tools_called_before_get_platform_tools(self, monkeypatch):
        """MCP tools must be registered before the oneshot toolset snapshot."""
        call_order = []
        _stub_oneshot_dependencies(
            monkeypatch,
            _cfg_dict(mcp_servers={"test-server": {"enabled": True}}),
        )

        def _fake_discover():
            call_order.append("discover_mcp_tools")
            return ["mcp_test_tool"]

        def _fake_get_platform_tools(cfg, platform):
            call_order.append("get_platform_tools")
            return {"core", "web"}

        monkeypatch.setattr(_mcp_mod, "discover_mcp_tools", _fake_discover)
        monkeypatch.setattr(_tools_config_mod, "_get_platform_tools", _fake_get_platform_tools)

        import hermes_cli.oneshot as oneshot_mod

        assert oneshot_mod._run_agent("test prompt") == "ok"
        assert call_order == ["discover_mcp_tools", "get_platform_tools"]

    def test_no_mcp_config_still_works(self, monkeypatch):
        """An absent mcp_servers section should remain a safe no-op."""
        _stub_oneshot_dependencies(monkeypatch, _cfg_dict())
        discover = MagicMock(return_value=[])
        monkeypatch.setattr(_mcp_mod, "discover_mcp_tools", discover)
        monkeypatch.setattr(_tools_config_mod, "_get_platform_tools", lambda cfg, platform: {"core"})

        import hermes_cli.oneshot as oneshot_mod

        assert oneshot_mod._run_agent("test prompt") == "ok"
        discover.assert_called_once_with()

    def test_mcp_discovery_error_does_not_abort_oneshot(self, monkeypatch):
        """A failing MCP server should not prevent the oneshot agent from running."""
        _stub_oneshot_dependencies(
            monkeypatch,
            _cfg_dict(mcp_servers={"broken-server": {"enabled": True}}),
        )
        monkeypatch.setattr(
            _mcp_mod,
            "discover_mcp_tools",
            MagicMock(side_effect=RuntimeError("server unavailable")),
        )
        monkeypatch.setattr(_tools_config_mod, "_get_platform_tools", lambda cfg, platform: {"core"})

        import hermes_cli.oneshot as oneshot_mod

        assert oneshot_mod._run_agent("test prompt") == "ok"
