"""Live-lane text send uses the already-authorized transport (#115656).

``_resolve_target_transport`` authorizes a credentialless satellite's exact
profile_routes target through the PRIMARY adapter (the SharedRouteAdapters
grant, #101113), building a transport with the satellite's own
``platforms.<p>`` block force-enabled. ``_live_send_text`` then rebuilt a
``DeliveryRouter`` over the plain ``target_adapters`` dict and re-resolved:
the satellite's real (disabled) block vetoed the grant, the second
resolution yielded None, and delivery fell through to the credentialless
standalone lane and failed.

The live send must go through the already-authorized ``t.transport`` — no
second resolution. These tests pin it with a REAL DeliveryRouter and a
satellite config whose own ``platforms.discord`` block is disabled: the only
way the send succeeds is via the authorized transport.
"""
import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import yaml

from cron.scheduler import _deliver_result
from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home
from gateway.config import Platform, PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

PRIMARY_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "fit", "platform": "discord", "chat_id": "1543065293755256852", "profile": "fitness"},
        ],
    }
}

CHAT_ID = "1543065293755256852"


def _job(chat_id: str) -> dict:
    return {"id": "a7ae1520356c", "name": "brief", "deliver": f"discord:{chat_id}"}


def _run(job, adapters):
    """Drive ``_deliver_result`` with a live loop and a real DeliveryRouter.

    The satellite's own ``platforms.discord`` block is DISABLED (a connector
    the credentialless satellite never runs) — the shape under which a second
    router-side resolution yields None.
    """
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
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=False)}
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


def _shared_view(tmp_path, monkeypatch, primary):
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    token = set_hermes_home_override(str(fitness_home))
    try:
        yield SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
    finally:
        reset_hermes_home_override(token)


def test_satellite_disabled_block_sends_through_authorized_transport(tmp_path, monkeypatch):
    """Exact route + disabled satellite block: live lane delivers, no standalone."""
    primary = _primary_adapter()
    for shared in _shared_view(tmp_path, monkeypatch, primary):
        error, standalone = _run(_job(CHAT_ID), shared)
    assert error is None, error
    assert primary.sent == [CHAT_ID]
    assert standalone == []


def test_satellite_disabled_block_still_fails_closed_off_route(tmp_path, monkeypatch):
    """Unmatched chat: the primary bot is NEVER used (#101113 fail-closed).

    With the satellite's own block disabled, an off-route chat fails closed
    at transport resolution (never reaching any sender) — the pin is that the
    primary adapter stays silent and the run reports an error.
    """
    primary = _primary_adapter()
    for shared in _shared_view(tmp_path, monkeypatch, primary):
        error, standalone = _run(_job("424242"), shared)
    assert error is not None
    assert primary.sent == []
    assert standalone == []
