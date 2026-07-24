"""ACP tool policy: compatibility vs profile-owned capability.

Covers the opt-in ``acp.tool_policy`` config that lets ACP hosts either keep
the coding-focused ``hermes-acp`` toolset or resolve the selected profile's
local CLI tool configuration through the canonical platform resolver.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from acp_adapter.session import (
    SessionManager,
    resolve_acp_enabled_toolsets,
    resolve_acp_tool_policy,
)
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# pure policy resolution
# ---------------------------------------------------------------------------


class TestResolveAcpToolPolicy:
    def test_absent_defaults_to_compat(self):
        assert resolve_acp_tool_policy({}) == "hermes-acp"
        assert resolve_acp_tool_policy(None) == "hermes-acp"

    def test_invalid_falls_back_to_compat(self):
        assert resolve_acp_tool_policy({"acp": {"tool_policy": "full"}}) == "hermes-acp"
        assert resolve_acp_tool_policy({"acp": {"tool_policy": 12}}) == "hermes-acp"
        assert resolve_acp_tool_policy({"acp": "profile"}) == "hermes-acp"

    def test_compat_aliases(self):
        for value in ("hermes-acp", "compat", "default", "", "  COMPAT  "):
            assert resolve_acp_tool_policy({"acp": {"tool_policy": value}}) == "hermes-acp"

    def test_profile_mode(self):
        assert resolve_acp_tool_policy({"acp": {"tool_policy": "profile"}}) == "profile"
        assert resolve_acp_tool_policy({"acp": {"tool_policy": " PROFILE "}}) == "profile"


class TestResolveAcpEnabledToolsets:
    def test_compat_mode_uses_hermes_acp_and_profile_mcp(self):
        config = {
            "acp": {"tool_policy": "hermes-acp"},
            "mcp_servers": {
                "olympus": {"command": "python", "enabled": True},
                "disabled": {"command": "python", "enabled": False},
            },
        }
        assert resolve_acp_enabled_toolsets(config) == [
            "hermes-acp",
            "mcp-olympus",
        ]

    def test_profile_mode_uses_cli_platform_policy(self, monkeypatch):
        config = {
            "acp": {"tool_policy": "profile"},
            "platform_toolsets": {
                "cli": [
                    "terminal",
                    "file",
                    "memory",
                    "skills",
                    "cronjob",
                    "kanban",
                    "delegation",
                ],
            },
            "mcp_servers": {},
            "agent": {},
        }

        # Avoid environment-gated auto-enables polluting the assertion surface.
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        monkeypatch.setattr(
            "hermes_cli.tools_config._xai_credentials_present",
            lambda: False,
        )

        enabled = resolve_acp_enabled_toolsets(config)
        assert "hermes-acp" not in enabled
        for required in (
            "terminal",
            "file",
            "memory",
            "skills",
            "cronjob",
            "kanban",
            "delegation",
        ):
            assert required in enabled

    def test_profile_mode_honours_disabled_and_default_off_toolsets(self, monkeypatch):
        config = {
            "acp": {"tool_policy": "profile"},
            "platform_toolsets": {
                # Explicit list without homeassistant/spotify — those stay off.
                "cli": ["terminal", "file", "memory"],
            },
            "agent": {
                "disabled_toolsets": ["memory"],
            },
            "mcp_servers": {},
        }
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        monkeypatch.setattr(
            "hermes_cli.tools_config._xai_credentials_present",
            lambda: False,
        )

        enabled = set(resolve_acp_enabled_toolsets(config))
        assert "terminal" in enabled
        assert "file" in enabled
        assert "memory" not in enabled
        assert "homeassistant" not in enabled
        assert "spotify" not in enabled

    def test_profile_mode_prefers_explicit_acp_platform_list(self, monkeypatch):
        config = {
            "acp": {"tool_policy": "profile"},
            "platform_toolsets": {
                "cli": ["terminal", "file", "cronjob", "kanban"],
                "acp": ["terminal", "memory"],
            },
            "mcp_servers": {},
            "agent": {},
        }
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        monkeypatch.setattr(
            "hermes_cli.tools_config._xai_credentials_present",
            lambda: False,
        )

        enabled = set(resolve_acp_enabled_toolsets(config))
        assert "terminal" in enabled
        assert "memory" in enabled
        assert "cronjob" not in enabled
        assert "kanban" not in enabled


# ---------------------------------------------------------------------------
# session construction / restore
# ---------------------------------------------------------------------------


def _fake_runtime_provider(requested=None, **kwargs):
    return {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.example/v1",
        "api_key": "***",
        "command": None,
        "args": [],
    }


def _patch_agent_stack(monkeypatch, config, captured):
    def fake_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model=kwargs.get("model"),
            enabled_toolsets=kwargs.get("enabled_toolsets"),
            session_cwd=None,
            _print_fn=None,
        )

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_runtime_provider,
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *a, **k: None)
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)


class TestSessionToolPolicy:
    def test_default_compat_session_keeps_hermes_acp(self, tmp_path, monkeypatch):
        captured = {}
        config = {
            "model": {"provider": "openrouter", "default": "test-model"},
            "mcp_servers": {},
        }
        _patch_agent_stack(monkeypatch, config, captured)
        db = SessionDB(tmp_path / "state.db")
        SessionManager(db=db).create_session(cwd="/work")
        assert captured["enabled_toolsets"] == ["hermes-acp"]

    def test_profile_mode_session_exposes_configured_cli_toolsets(
        self, tmp_path, monkeypatch
    ):
        captured = {}
        config = {
            "model": {"provider": "openrouter", "default": "test-model"},
            "acp": {"tool_policy": "profile"},
            "platform_toolsets": {
                "cli": ["terminal", "skills", "memory", "cronjob", "delegation"],
            },
            "mcp_servers": {},
            "agent": {},
        }
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        monkeypatch.setattr(
            "hermes_cli.tools_config._xai_credentials_present",
            lambda: False,
        )
        _patch_agent_stack(monkeypatch, config, captured)
        db = SessionDB(tmp_path / "state.db")
        SessionManager(db=db).create_session(cwd="/work")
        enabled = set(captured["enabled_toolsets"])
        assert "hermes-acp" not in enabled
        assert {"terminal", "skills", "memory", "cronjob", "delegation"} <= enabled

    def test_restored_session_uses_same_profile_policy(self, tmp_path, monkeypatch):
        """Create under profile policy, drop memory, restore — policy must re-apply."""
        captured_create = {}
        captured_restore = {}
        config = {
            "model": {"provider": "openrouter", "default": "test-model"},
            "acp": {"tool_policy": "profile"},
            "platform_toolsets": {
                "cli": ["terminal", "file", "cronjob"],
            },
            "mcp_servers": {},
            "agent": {},
        }
        monkeypatch.delenv("HASS_TOKEN", raising=False)
        monkeypatch.setattr(
            "hermes_cli.tools_config._xai_credentials_present",
            lambda: False,
        )

        def fake_agent(**kwargs):
            # First construction is create; after memory drop, restore rebuilds.
            if "enabled_toolsets" in kwargs:
                if not captured_create:
                    captured_create.update(kwargs)
                else:
                    captured_restore.update(kwargs)
            return SimpleNamespace(
                model=kwargs.get("model"),
                enabled_toolsets=kwargs.get("enabled_toolsets"),
                session_cwd=None,
                _print_fn=None,
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            _fake_runtime_provider,
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *a, **k: None)
        monkeypatch.setattr("run_agent.AIAgent", fake_agent)

        db = SessionDB(tmp_path / "state.db")
        manager = SessionManager(db=db)
        state = manager.create_session(cwd="/work")
        sid = state.session_id

        with manager._lock:
            del manager._sessions[sid]

        restored = manager.get_session(sid)
        assert restored is not None
        assert set(captured_create["enabled_toolsets"]) == set(
            captured_restore["enabled_toolsets"]
        )
        assert "cronjob" in captured_restore["enabled_toolsets"]
        assert "hermes-acp" not in captured_restore["enabled_toolsets"]

    def test_restore_still_rejects_non_acp_source(self, tmp_path, monkeypatch):
        config = {
            "model": {"provider": "openrouter", "default": "test-model"},
            "acp": {"tool_policy": "profile"},
            "platform_toolsets": {"cli": ["terminal"]},
            "mcp_servers": {},
            "agent": {},
        }
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            _fake_runtime_provider,
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *a, **k: None)
        monkeypatch.setattr(
            "run_agent.AIAgent",
            lambda **kwargs: SimpleNamespace(
                model="m",
                enabled_toolsets=kwargs.get("enabled_toolsets"),
                session_cwd=None,
                _print_fn=None,
            ),
        )

        db = SessionDB(tmp_path / "state.db")
        # Seed a non-ACP session row directly.
        db.create_session(session_id="desktop-sess-1", source="cli")
        manager = SessionManager(db=db)
        assert manager.get_session("desktop-sess-1") is None


# ---------------------------------------------------------------------------
# host MCP expansion blocked in profile mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_mode_ignores_host_mcp_servers(tmp_path, monkeypatch):
    pytest.importorskip("acp.schema")
    from acp.schema import McpServerStdio
    from acp_adapter.server import HermesACPAgent

    config = {
        "model": {"provider": "openrouter", "default": "test-model"},
        "acp": {"tool_policy": "profile"},
        "platform_toolsets": {"cli": ["terminal", "memory"]},
        "mcp_servers": {},
        "agent": {},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)

    agent = HermesACPAgent()
    agent.session_manager = SessionManager(
        agent_factory=lambda: SimpleNamespace(
            enabled_toolsets=["terminal", "memory"],
            disabled_toolsets=None,
            tools=[],
            valid_tool_names=set(),
            session_id="s1",
            model="m",
        ),
        db=SessionDB(tmp_path / "state.db"),
    )
    state = agent.session_manager.create_session(cwd="/work")
    original = list(state.agent.enabled_toolsets)

    register_calls = []

    def fake_register(config_map):
        register_calls.append(config_map)

    monkeypatch.setattr(
        "tools.mcp_tool.register_mcp_servers",
        fake_register,
    )

    server = McpServerStdio(name="host-extra", command="echo", args=[], env=[])
    await agent._register_session_mcp_servers(state, [server])
    assert register_calls == []
    assert state.agent.enabled_toolsets == original
    assert "mcp-host-extra" not in state.agent.enabled_toolsets
    assert "host-extra" not in state.agent.enabled_toolsets
