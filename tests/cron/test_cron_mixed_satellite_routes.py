"""Mixed satellites: a profile with SOME own adapters still reaches the host for routed platforms.

A multiplexed satellite may run its own bot on one platform (e.g. Telegram) and have no
credential for another (e.g. Slack) that the host gateway routes to it via ``profile_routes``.
Before, the ticker and the restart-safe drain consulted host routes only when the satellite had
NO own adapters, so the routed Slack target saw the satellite's own map and failed with
"platform 'slack' not configured/enabled".

Per platform:
* a platform the satellite holds its own adapter for is ALWAYS served by that adapter (never the
  host bot), even for a chat_id the host also routes to it;
* a platform it lacks reaches the host adapter only through an exact enabled host route;
* unrouted targets fail closed.
"""

import asyncio
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cron import scheduler as cron_scheduler
from cron import scheduler_preflight as sched_preflight
from cron.scheduler_delivery import _resolve_target_transport
from gateway import run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.profile_routing import parse_profile_routes

ROUTED = "C0BTH5ZQH61"
OWN_TG_CHAT = "-1001234"
ROUTES = parse_profile_routes([
    {"name": "slack-sat", "platform": "slack", "chat_id": ROUTED, "profile": "sat"},
    # A host telegram route for the same satellite must NOT hijack its own telegram bot.
    {"name": "tg-sat", "platform": "telegram", "chat_id": OWN_TG_CHAT, "profile": "sat"},
])


def _config(platforms):
    config = MagicMock()
    config.platforms = platforms
    config.get_home_channel = lambda p: None
    return config


def satellite_delivery_adapters(*args, **kwargs):
    # Imported lazily so the ticker/drain behaviour tests below collect (and fail on behaviour,
    # not on ImportError) against code that predates the helper.
    return sched_preflight.satellite_delivery_adapters(*args, **kwargs)


def _resolve(adapters, platform, chat_id, platforms=None):
    if platforms is None:
        platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    return _resolve_target_transport(
        {"id": "j"}, platform, platform.value, {"platform": platform.value, "chat_id": chat_id},
        adapters, _config(platforms))


class TestHelper:
    def test_routed_platform_missing_from_own_map_gets_host_adapter(self):
        host_slack, own_tg = object(), object()
        adapters = satellite_delivery_adapters({Platform.TELEGRAM: own_tg},
                                               {Platform.SLACK: host_slack}, routes=ROUTES)
        resolved, err = _resolve(adapters, Platform.SLACK, ROUTED)
        assert err is None, err
        assert resolved[2] is host_slack

    def test_unrouted_chat_on_missing_platform_fails_closed(self):
        adapters = satellite_delivery_adapters({Platform.TELEGRAM: object()},
                                               {Platform.SLACK: object()}, routes=ROUTES)
        resolved, err = _resolve(adapters, Platform.SLACK, "C0UNROUTED")
        assert resolved is None and "slack" in err

    def test_own_platform_always_uses_own_adapter(self):
        host_tg, own_tg = object(), object()
        adapters = satellite_delivery_adapters(
            {Platform.TELEGRAM: own_tg}, {Platform.TELEGRAM: host_tg, Platform.SLACK: object()},
            routes=ROUTES)
        for chat in (OWN_TG_CHAT, "-100999"):  # host-routed or not: own bot either way
            resolved, err = _resolve(adapters, Platform.TELEGRAM, chat)
            assert err is None, err
            assert resolved[2] is own_tg

    def test_own_platform_disabled_never_falls_to_host(self):
        host_tg = object()
        adapters = satellite_delivery_adapters(
            {Platform.TELEGRAM: object()}, {Platform.TELEGRAM: host_tg}, routes=ROUTES)
        resolved, err = _resolve(adapters, Platform.TELEGRAM, OWN_TG_CHAT,
                                 platforms={Platform.TELEGRAM: PlatformConfig(enabled=False)})
        assert resolved is None and "telegram" in err

    def test_no_routes_returns_own_map_unchanged(self):
        own = {Platform.TELEGRAM: object()}
        assert satellite_delivery_adapters(own, {Platform.SLACK: object()}, routes=[]) is own
        assert satellite_delivery_adapters(own, {}, routes=ROUTES) is own


def _primary_adapter(sent):
    class Adapter:
        async def send(self, chat_id, content, metadata=None):
            sent.append(chat_id)
            return {"success": True, "message_id": "m1"}
    return Adapter()


def _drive_deliver(job, adapters, platforms):
    from cron.scheduler import _deliver_result
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    standalone = []

    async def _standalone(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "SLACK_BOT_TOKEN is not set"}

    with patch("gateway.config.load_gateway_config", return_value=_config(platforms)), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _standalone), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, "hello", adapters=adapters, loop=loop)
    return error, standalone


def test_end_to_end_mixed_satellite_slack_delivery_uses_host_adapter():
    host_sent, own_sent = [], []
    adapters = satellite_delivery_adapters(
        {Platform.TELEGRAM: _primary_adapter(own_sent)},
        {Platform.SLACK: _primary_adapter(host_sent)}, routes=ROUTES)
    platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True),
                 Platform.SLACK: PlatformConfig(enabled=False)}
    error, standalone = _drive_deliver(
        {"id": "08b335c6f58a", "name": "probe", "deliver": f"slack:{ROUTED}"}, adapters, platforms)
    assert error is None, error
    assert host_sent == [ROUTED] and own_sent == [] and standalone == []


def _multiplex_tick_adapters(tmp_path, *, own, primary, routes):
    from cron.scheduler_provider import InProcessCronScheduler
    p_default, p_sat = tmp_path / "default", tmp_path / "sat"
    for d in (p_default, p_sat):
        (d / "cron").mkdir(parents=True)
    captured = []
    stop = threading.Event()
    with patch("cron.scheduler.tick", side_effect=lambda *a, **k: captured.append(k.get("adapters")) or 0), \
         patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None), \
         patch.object(sched_preflight, "_primary_profile_routes_for_current_home", lambda: routes):
        t = threading.Thread(target=InProcessCronScheduler().start, args=(stop,), kwargs={
            "interval": 0, "profile_homes": [("default", p_default), ("sat", p_sat)],
            "adapters": primary, "profile_adapters": {"sat": own}, "default_profile": "default",
        }, daemon=True)
        t.start()
        deadline = time.monotonic() + 10
        while len(captured) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        stop.set()
        t.join(timeout=5)
    assert len(captured) >= 2
    return captured[1]


def test_multiplex_ticker_mixed_satellite(tmp_path):
    host_slack, host_tg, own_tg = object(), object(), object()
    sat = _multiplex_tick_adapters(
        tmp_path, own={Platform.TELEGRAM: own_tg},
        primary={Platform.SLACK: host_slack, Platform.TELEGRAM: host_tg}, routes=ROUTES)
    resolved, err = _resolve(sat, Platform.SLACK, ROUTED)
    assert err is None and resolved[2] is host_slack
    resolved, err = _resolve(sat, Platform.SLACK, "C0UNROUTED")
    assert resolved is None
    resolved, err = _resolve(sat, Platform.TELEGRAM, OWN_TG_CHAT)
    assert err is None and resolved[2] is own_tg


def test_restart_safe_drain_mixed_satellite(tmp_path, monkeypatch):
    host_slack, host_tg, own_tg = object(), object(), object()
    primary = {Platform.SLACK: host_slack, Platform.TELEGRAM: host_tg}
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True),
                             _profile_adapters={"sat": {Platform.TELEGRAM: own_tg}})
    monkeypatch.setattr(gateway_run, "_handoff_watch_scopes", lambda _r: [("sat", tmp_path)])

    @contextmanager
    def fake_scope(_home):
        yield

    drained = []
    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(sched_preflight, "_primary_profile_routes_for_current_home", lambda: ROUTES)
    monkeypatch.setattr(cron_scheduler, "drain_delivery_queue", lambda a, _l: drained.append(a))
    gateway_run._drain_restart_safe_cron_deliveries(primary, object(), runner)

    assert len(drained) == 1
    sat = drained[0]
    resolved, err = _resolve(sat, Platform.SLACK, ROUTED)
    assert err is None and resolved[2] is host_slack
    resolved, err = _resolve(sat, Platform.SLACK, "C0UNROUTED")
    assert resolved is None
    resolved, err = _resolve(sat, Platform.TELEGRAM, OWN_TG_CHAT)
    assert err is None and resolved[2] is own_tg
