"""A satellite profile that connected ONLY a local transport is still credentialless.

Any local session of a served profile — the desktop app, a ``hermes`` run, a kanban worker —
lazily registers a ``LocalSessionAdapter`` under ``Platform.LOCAL`` in that profile's adapter map
(``gateway.session_local_recovery.local_adapter_map``). The multiplex cron ticker read that map as
"this profile owns its own transport", returned it as-is, and every delivery to a target its
``profile_routes`` map failed closed with ``platform 'telegram' not configured/enabled`` for as
long as any local session of that profile existed.

Regression: LOCAL (and API_SERVER/WEBHOOK) must not count as a messaging credential.
"""
import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml

from cron.scheduler import _deliver_result
from cron.scheduler_preflight import (
    SharedRouteAdapters, _primary_profile_routes_for_current_home, delivering_adapters,
)
from cron.scheduler_provider import _tick_adapters_for_profile
from gateway.config import Platform
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

CHAT = "111000111"
THREAD = "42"

PRIMARY_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "trading-topic", "platform": "telegram", "chat_id": CHAT,
             "thread_id": THREAD, "profile": "trading", "enabled": True},
        ],
    }
}


def _satellite_home(tmp_path, monkeypatch):
    root = tmp_path / "root"
    trading_home = root / "profiles" / "trading"
    trading_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    return trading_home


def _primary_telegram():
    adapter = MagicMock()
    adapter.sent = []

    async def send(chat_id, content, metadata=None):
        adapter.sent.append((chat_id, (metadata or {}).get("thread_id")))
        return {"success": True, "message_id": "m1"}

    adapter.send = send
    return adapter


def _job():
    return {"id": "9a1c0ff1ee77", "name": "brief", "deliver": f"telegram:{CHAT}:{THREAD}"}


def _deliver(job, adapters):
    """Drive ``_deliver_result`` with a live loop and a profile config with no telegram block."""
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    standalone = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "TELEGRAM_BOT_TOKEN is not set"}

    config = MagicMock()
    config.platforms = {}
    config.get_home_channel = lambda p: None
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, "hello", adapters=adapters, loop=loop)
    return error, standalone


def test_local_and_listener_transports_are_not_delivering():
    local = object()
    assert delivering_adapters({Platform.LOCAL: local}) == {}
    assert delivering_adapters({Platform.LOCAL: local, Platform.API_SERVER: object()}) == {}
    assert delivering_adapters(None) == {}
    telegram = object()
    assert delivering_adapters({Platform.LOCAL: local, Platform.TELEGRAM: telegram}) == {
        Platform.TELEGRAM: telegram}


def test_local_only_satellite_rides_the_routed_primary_adapter(tmp_path, monkeypatch):
    trading_home = _satellite_home(tmp_path, monkeypatch)
    primary = _primary_telegram()
    local = object()
    token = set_hermes_home_override(str(trading_home))
    try:
        view = _tick_adapters_for_profile(
            "trading", default_profile="default",
            adapters={Platform.TELEGRAM: primary},
            profile_adapters={"trading": {Platform.LOCAL: local}},
        )
        assert isinstance(view, SharedRouteAdapters), view
        # the exact routed target resolves to the primary bot …
        assert view.get(Platform.TELEGRAM, {"chat_id": CHAT, "thread_id": THREAD}) is primary
        # … and nothing else does (fail closed, never the wrong bot)
        assert view.get(Platform.TELEGRAM, {"chat_id": "999"}) is None
        assert view.get(Platform.TELEGRAM) is None

        # Before the fix the ticker returned the LOCAL-only map, which authorized no target:
        # this is the exact production failure (`platform 'telegram' not configured/enabled`,
        # nothing sent, nothing even attempted standalone).
        error, standalone = _deliver(_job(), {Platform.LOCAL: local})
        assert error is not None and "not configured/enabled" in error
        assert primary.sent == []

        # With the composed view the job delivers through the primary adapter.
        error, standalone = _deliver(_job(), view)
        assert error is None, error
        assert primary.sent == [(CHAT, THREAD)]
        assert standalone == []
    finally:
        reset_hermes_home_override(token)


def test_profile_with_its_own_messaging_adapter_never_borrows(tmp_path, monkeypatch):
    trading_home = _satellite_home(tmp_path, monkeypatch)
    own = {Platform.TELEGRAM: object()}
    token = set_hermes_home_override(str(trading_home))
    try:
        returned = _tick_adapters_for_profile(
            "trading", default_profile="default",
            adapters={Platform.TELEGRAM: _primary_telegram()},
            profile_adapters={"trading": dict(own)},
        )
        assert returned == own and not isinstance(returned, SharedRouteAdapters)
    finally:
        reset_hermes_home_override(token)


def test_default_profile_and_empty_maps_keep_their_old_behaviour(tmp_path, monkeypatch):
    trading_home = _satellite_home(tmp_path, monkeypatch)
    primary = {Platform.TELEGRAM: _primary_telegram()}
    token = set_hermes_home_override(str(trading_home))
    try:
        assert _tick_adapters_for_profile(
            None, default_profile="default", adapters=primary, profile_adapters={}) is primary
        assert _tick_adapters_for_profile(
            "default", default_profile="default", adapters=primary, profile_adapters={}) is primary
        # empty satellite map → shared-route grant, as before
        view = _tick_adapters_for_profile(
            "trading", default_profile="default", adapters=primary, profile_adapters={"trading": {}})
        assert isinstance(view, SharedRouteAdapters)
        # no primary adapters at all → whatever the satellite has, unchanged
        local = {Platform.LOCAL: object()}
        assert _tick_adapters_for_profile(
            "trading", default_profile="default", adapters=None,
            profile_adapters={"trading": local}) == local
    finally:
        reset_hermes_home_override(token)


def test_routes_helper_still_targets_the_trading_route_only(tmp_path, monkeypatch):
    trading_home = _satellite_home(tmp_path, monkeypatch)
    token = set_hermes_home_override(str(trading_home))
    try:
        routes = _primary_profile_routes_for_current_home()
        assert [(r.platform, r.chat_id, r.thread_id) for r in routes] == [("telegram", CHAT, THREAD)]
    finally:
        reset_hermes_home_override(token)
