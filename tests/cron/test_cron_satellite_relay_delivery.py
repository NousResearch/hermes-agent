"""A served profile with no bot of its own delivers cron output through the process's Relay.

On a Team Gateway deployment (``GATEWAY_RELAY_URL`` stamped) the connector owns every chat
platform, so no profile runs a native bot and every non-default profile is a credentialless
satellite. The multiplex ticker and the restart-safe drain hand such a profile a
``SharedRouteAdapters`` view of the primary's adapters. That view lent only route-matched NATIVE
bots, and the Relay is registered under ``Platform.RELAY``, so a satellite's ``deliver:
telegram:<chat>`` always failed with "relay-fronted and has no live gateway transport" while the
same job in the default profile delivered.

Real ``load_gateway_config`` under the satellite's home (its own config.yaml decides Relay
enablement), real ``_deliver_result``; only the Relay transport is a fake.
"""
import asyncio
from concurrent.futures import Future
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from cron.scheduler import _deliver_result
from cron.scheduler_preflight import shared_route_adapters_for_current_home
from gateway.config import Platform
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

CHAT = "5550001111"


class _Relay:
    """The process Relay: fronts Telegram, records each send with its logical platform."""

    def __init__(self):
        self.sent = []

    def fronts_platform(self, platform):
        return platform == Platform.TELEGRAM

    async def send_for_platform(self, platform, chat_id, content, metadata=None):
        self.sent.append((platform, chat_id))
        return {"success": True, "message_id": "m1"}


class _NativeTelegram:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(chat_id)
        return {"success": True, "message_id": "n1"}


@pytest.fixture
def satellite_home(tmp_path, monkeypatch):
    """A multiplexed root plus a ``fitness`` profile, on a Team Gateway deployment."""
    root = tmp_path / "root"
    home = root / "profiles" / "fitness"
    home.mkdir(parents=True)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://connector.example/relay")
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "telegram")
    for var in ("TELEGRAM_BOT_TOKEN", "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS"):
        monkeypatch.delenv(var, raising=False)

    def write(root_cfg, profile_cfg=None):
        (root / "config.yaml").write_text(yaml.safe_dump(root_cfg), encoding="utf-8")
        if profile_cfg is not None:
            (home / "config.yaml").write_text(yaml.safe_dump(profile_cfg), encoding="utf-8")
        return home

    return write


def _deliver_as(home, primary_adapters, deliver=f"telegram:{CHAT}"):
    """Build cron's view for *home* the way the ticker and drain do, then deliver through it."""
    job = {"id": "a7ae1520356c", "name": "brief", "deliver": deliver}
    standalone = []

    async def _standalone(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "STANDALONE"}

    def run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    class _Loop:
        def is_running(self):
            return True

    token = set_hermes_home_override(str(home))
    try:
        shared = shared_route_adapters_for_current_home(primary_adapters)
        with patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("gateway.outbound.send._send_to_platform", _standalone), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=run_coro):
            error = _deliver_result(job, "hello", adapters=shared, loop=_Loop())
    finally:
        reset_hermes_home_override(token)
    return shared, error, standalone


def test_satellite_delivers_through_the_process_relay(satellite_home):
    """The reported case: no ``profile_routes`` at all (Team Gateway routes profiles at the connector)."""
    home = satellite_home({"gateway": {"multiplex_profiles": True}})
    relay = _Relay()

    shared, error, standalone = _deliver_as(home, {Platform.RELAY: relay})

    assert shared, "the drain only swaps in the shared view when it is truthy"
    assert error is None, error
    assert relay.sent == [(Platform.TELEGRAM, CHAT)]
    assert standalone == []


def test_queued_satellite_delivery_drains_through_the_process_relay(satellite_home, monkeypatch):
    """The restart-safe worker queues a satellite's result; the primary gateway's drain must send it
    through the Relay (it used to keep the satellite's empty map, since the view was falsy)."""
    import threading
    from types import SimpleNamespace

    from cron import delivery_queue
    from gateway import run as gateway_run

    home = satellite_home({"gateway": {"multiplex_profiles": True}})
    (home / "cron").mkdir()
    relay = _Relay()

    token = set_hermes_home_override(str(home))
    try:
        delivery_queue.enqueue("exec-1", {"id": "job1", "name": "brief", "deliver": f"telegram:{CHAT}"}, "hello")
    finally:
        reset_hermes_home_override(token)

    loop = asyncio.new_event_loop()
    running = threading.Event()
    loop.call_soon(running.set)  # set only once run_forever dispatches: the drain needs is_running()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True), _profile_adapters={"fitness": {}})
    monkeypatch.setattr(gateway_run, "_handoff_watch_scopes", lambda _r: [("fitness", home)])
    try:
        assert running.wait(timeout=10), "event loop never started"
        with patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            gateway_run._drain_restart_safe_cron_deliveries({Platform.RELAY: relay}, loop, runner)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        assert not thread.is_alive(), "event loop thread did not stop"
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

    token = set_hermes_home_override(str(home))
    try:
        row = delivery_queue.get_status("exec-1")
    finally:
        reset_hermes_home_override(token)
    assert row["status"] == "delivered", row
    assert relay.sent == [(Platform.TELEGRAM, CHAT)]


def test_satellite_that_opted_out_of_the_relay_stays_out(satellite_home):
    """The profile's OWN ``platforms.relay.enabled: false`` keeps its output off the Relay."""
    home = satellite_home({"gateway": {"multiplex_profiles": True}},
                          {"platforms": {"relay": {"enabled": False}}})
    relay = _Relay()

    _shared, error, _standalone = _deliver_as(home, {Platform.RELAY: relay})

    assert error is not None
    assert relay.sent == []


def test_routed_native_bot_still_wins_over_the_relay(satellite_home):
    """An exact route to the primary's native bot keeps that bot (native-wins rule)."""
    home = satellite_home({"gateway": {"multiplex_profiles": True, "profile_routes": [
        {"name": "fit", "platform": "telegram", "chat_id": CHAT, "profile": "fitness"}]}})
    relay, native = _Relay(), _NativeTelegram()

    _shared, error, _standalone = _deliver_as(home, {Platform.RELAY: relay, Platform.TELEGRAM: native})

    assert error is None, error
    assert native.sent == [CHAT]
    assert relay.sent == []


def test_platform_the_relay_does_not_front_still_fails_closed(satellite_home):
    """Borrowing is per platform: a Discord target with no routed bot is not sent over the Relay."""
    home = satellite_home({"gateway": {"multiplex_profiles": True}})
    relay = _Relay()

    _shared, error, _standalone = _deliver_as(home, {Platform.RELAY: relay}, deliver="discord:123")

    assert error is not None
    assert relay.sent == []
