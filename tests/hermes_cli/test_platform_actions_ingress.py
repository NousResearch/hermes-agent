"""Platform actions use the exact live ingress adapter across a routed profile turn."""

import asyncio
from unittest.mock import AsyncMock, patch

import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.ingress_context import bind_ingress_adapter, current_ingress_adapter
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session_identity import resolve_identity
from hermes_cli.platform_actions import PlatformActions
from hermes_constants import get_hermes_home


class _Telegram(BasePlatformAdapter):
    pass


_Telegram.__abstractmethods__ = frozenset()


def _adapter(runner):
    adapter = _Telegram.__new__(_Telegram)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_messages, adapter._active_sessions = {}, {}
    adapter._set_reaction = AsyncMock(return_value=True)
    adapter._running = True
    return adapter


def test_routed_plugin_action_uses_exact_ingress_and_fails_closed_outside_turn(
    tmp_path, monkeypatch
):
    from gateway.run import GatewayRunner

    home_a = tmp_path / ".hermes"
    home_b = home_a / "profiles" / "cfo"
    home_b.mkdir(parents=True)
    (home_a / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {}}), encoding="utf-8"
    )
    (home_b / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {}}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-be-used-as-fallback")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.profile_routes = parse_profile_routes([
        {
            "name": "cfo-topic",
            "platform": "telegram",
            "chat_id": "3232",
            "profile": "cfo",
        },
    ])
    runner.config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})
    }
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    ingress = _adapter(runner)
    runner.adapters = {Platform.TELEGRAM: ingress}
    runner._profile_adapters = {"cfo": {}}
    executor = runner._get_executor()
    source = ingress.build_source(chat_id="3232", chat_type="dm", user_id="u")

    with (
        patch("gateway.run._gateway_runner_ref", lambda: runner),
        patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", home_a), ("cfo", home_b)],
        ),
        patch(
            "hermes_cli.profiles.get_profile_dir",
            side_effect=lambda n: home_a if n == "default" else home_b,
        ),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
        patch(
            "hermes_cli.plugin_capabilities.plugin_capability_granted",
            return_value=True,
        ),
    ):
        identity = resolve_identity(source, runner=runner)
        assert identity.runtime_home == home_b and identity.adapter() is ingress
        assert get_hermes_home() == home_a

        async def run_plugin(*_args, **_kwargs):
            assert get_hermes_home() == home_b
            worker_home, worker_adapter = await runner._run_in_executor_with_context(
                lambda: (get_hermes_home(), current_ingress_adapter("telegram"))
            )
            assert worker_home == home_b and worker_adapter is ingress
            actions = PlatformActions("profile-plugin")
            exact = await actions.add_reaction("telegram", "3232", "9", "👍")
            wrong_platform = await actions.add_reaction("discord", "3232", "9", "👍")
            return exact, wrong_platform

        runner._run_agent_inner = run_plugin
        result, wrong_platform = asyncio.run(
            runner._run_agent("run", "", [], source, "session", session_key="key")
        )
        assert get_hermes_home() == home_a
        outside = asyncio.run(
            PlatformActions("profile-plugin").add_reaction(
                "telegram", "3232", "10", "👍"
            )
        )

    assert result == {"ok": True, "action": "add_reaction"}
    assert wrong_platform["error"] == "adapter_not_registered"
    ingress._set_reaction.assert_awaited_once_with("3232", "9", "👍")
    assert outside["error"] == "adapter_not_registered"
    executor.shutdown(wait=True)


def test_ingress_binding_does_not_bypass_plugin_capability(tmp_path):
    source = type("Source", (), {})()
    source.platform = Platform.TELEGRAM
    adapter = type(
        "Adapter", (), {"platform": Platform.TELEGRAM, "is_connected": True}
    )()
    adapter._set_reaction = AsyncMock(return_value=True)
    source._transport_adapter_ref = lambda: adapter

    with (
        bind_ingress_adapter(source),
        patch(
            "hermes_cli.plugin_capabilities.plugin_capability_granted",
            return_value=False,
        ),
    ):
        result = asyncio.run(
            PlatformActions("profile-plugin").add_reaction("telegram", "1", "2", "👍")
        )

    assert result["error"] == "capability_not_granted"
    adapter._set_reaction.assert_not_awaited()
