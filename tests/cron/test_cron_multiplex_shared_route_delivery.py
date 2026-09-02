"""Regression tests for #101113 — a credentialless satellite profile under
``gateway.profile_routes`` delivers cron output through the PRIMARY adapter
for exactly the targets the primary routes to it, and fails closed otherwise.

The multiplex ticker hands such a profile a ``SharedRouteAdapters`` view over
the primary adapter map; ``_deliver_result`` resolves a transport from it per
target using the same ``ProfileRoute.matches`` predicate as inbound routing.
"""
import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import yaml

from cron.scheduler import (
    SharedRouteAdapters,
    _deliver_result,
    _primary_profile_routes_for_current_home,
)
from gateway.config import Platform, PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

PRIMARY_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "fit", "platform": "discord", "chat_id": "1543065293755256852", "profile": "fitness"},
            {"name": "off", "platform": "discord", "chat_id": "999", "profile": "fitness", "enabled": False},
            {"name": "other", "platform": "discord", "chat_id": "777", "profile": "other"},
        ],
    }
}


def _job(chat_id: str) -> dict:
    return {"id": "a7ae1520356c", "name": "brief", "deliver": f"discord:{chat_id}"}


def _run(job, adapters):
    """Drive ``_deliver_result`` with a live loop and a real DeliveryRouter."""
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    standalone = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "DISCORD_BOT_TOKEN is not set"}

    config = MagicMock()
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, "hello", adapters=adapters, loop=loop)
    return error, standalone


def _primary_adapter():
    adapter = MagicMock()
    adapter.sent = []

    async def send(chat_id, content, metadata=None):
        adapter.sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    adapter.send = send
    return adapter


def test_satellite_routes_exact_target_through_primary_adapter(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(fitness_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
        # exact enabled route → primary adapter sends, no standalone attempt
        error, standalone = _run(_job("1543065293755256852"), shared)
        assert error is None, error
        assert primary.sent == ["1543065293755256852"]
        assert standalone == []

        # unmatched chat, disabled route, route for another profile → the
        # primary bot is NEVER used; delivery stays on the satellite's own
        # (credentialless) standalone path and reports its failure.
        for chat in ("424242", "999", "777"):
            primary.sent.clear()
            error, standalone = _run(_job(chat), shared)
            assert error is not None and "DISCORD_BOT_TOKEN" in error
            assert primary.sent == []
            assert standalone == [chat]
    finally:
        reset_hermes_home_override(token)


def test_shared_view_is_falsy_without_routes_or_primary_adapters():
    assert not SharedRouteAdapters({}, [])
    assert SharedRouteAdapters({Platform.DISCORD: object()}, []).get(Platform.DISCORD) is None
