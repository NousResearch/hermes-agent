from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin


class _StubGateway(GatewaySlashCommandsMixin):
    def __init__(self, *, user_id: str = "admin"):
        self.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    extra={
                        "group_allow_admin_from": ["admin"],
                        "group_user_allowed_commands": ["ping", "status"],
                    },
                )
            }
        )
        self.adapters = {Platform.TELEGRAM: object()}
        self._failed_platforms = {}
        self._running_agents = {}
        self._startup_time = time.time() - 65
        self.source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id=user_id,
        )

    def _running_agent_count(self) -> int:
        return len(self._running_agents)

    def _session_key_for_source(self, source) -> str:
        return "session-key"

    def _queue_depth(self, session_key, adapter=None) -> int:
        return 0

    def event(self):
        return SimpleNamespace(source=self.source)


def test_ping_command_is_fast_and_model_free():
    gateway = _StubGateway()

    out = asyncio.run(gateway._handle_ping_command(gateway.event()))

    assert out.startswith("pong ✅")
    assert "Gateway: online" in out
    assert "Platform: telegram" in out
    assert "Active agents: 0" in out


def test_health_command_denies_non_admin_when_gating_enabled():
    gateway = _StubGateway(user_id="not-admin")

    out = asyncio.run(gateway._handle_health_command(gateway.event()))

    assert "admin-only" in out
    assert "Model route" not in out


def test_health_command_renders_redacted_summary_for_admin(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.run._hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"provider": "test-provider", "default": "test-model"}},
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg=None: "test-model")
    gateway = _StubGateway(user_id="admin")

    out = asyncio.run(gateway._handle_health_command(gateway.event()))

    assert "**Hermes health**" in out
    assert "Gateway: ✅ online" in out
    assert "Platforms: telegram" in out
    assert "Model route: test-provider / test-model" in out
    assert "Debug report: /debug (admin)" in out


def test_diagnostics_commands_registered_for_gateway():
    from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, resolve_command

    ping = resolve_command("ping")
    health = resolve_command("health")

    assert ping is not None and ping.gateway_only is True
    assert health is not None and health.gateway_only is True
    assert "ping" in ACTIVE_SESSION_BYPASS_COMMANDS
    assert "health" in ACTIVE_SESSION_BYPASS_COMMANDS
