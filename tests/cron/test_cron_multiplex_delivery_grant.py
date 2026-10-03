"""Regression tests for #128411 — an OUTBOUND-ONLY ``gateway.delivery_grants`` entry lets a
credentialless satellite profile's cron output (and, the reported case, its
``--failure-deliver`` alerts) reach a channel on ANOTHER profile's bot, without opening any
inbound door.

The two invariants pinned here are the whole security claim:

1. the grant is honored for exactly the target it names, and every other target on the same bot
   still fails closed to the satellite's own credentialless path; and
2. the grant is outbound-only — it creates no ``ProfileRoute``, so nothing about INBOUND routing
   changes, and ``outbound_only: false`` is refused outright at parse time.

Preflight is exercised for the reported symptom: a granted target must stop being blocked with
``delivery platform discord has no gateway credentials configured`` (which is what silently
killed the alerts), while an ungranted target on the same platform must still block.
"""
import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml

from cron.scheduler import _deliver_result
from cron.scheduler_preflight import (
    SharedRouteAdapters,
    _preflight_check_delivery,
    _primary_delivery_grants_for_current_home,
    _primary_profile_routes_for_current_home,
)
from gateway.config import Platform, PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

PRIMARY_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "delivery_grants": [
            {
                "name": "ops-pages",
                "bot_platform": "discord",
                "to_profiles": ["fitness"],
                "targets": ["discord:1543065293755256852"],
                "outbound_only": True,
            },
        ],
    },
}

GRANTED_CHAT = "1543065293755256852"


def _satellite_home(root):
    home = root / "profiles" / "fitness"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _primary_adapter():
    adapter = MagicMock()
    adapter.sent = []

    async def send(chat_id, content, metadata=None):
        adapter.sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    adapter.send = send
    return adapter


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


def _unconnected_gateway_config():
    config = MagicMock()
    config.get_connected_platforms.return_value = set()
    config.platforms = {}
    config.get_home_channel = lambda p: None
    return config


def test_grant_carries_one_exact_target_and_never_becomes_inbound(tmp_path, monkeypatch):
    root = tmp_path / "root"
    satellite_home = _satellite_home(root)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(satellite_home))
    try:
        view = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home())
        # The grantor named a target, so the view is usable at all — this is the whole reported
        # failure: with no routes and no grants the view was falsy and delivery never happened.
        assert view, "a grant must make the satellite view resolvable"

        error, standalone = _run(
            {"id": "j1", "name": "brief", "deliver": f"discord:{GRANTED_CHAT}"}, view)
        assert error is None, error
        assert primary.sent == [GRANTED_CHAT]
        assert standalone == []

        # Every other target on the SAME bot still fails closed to the satellite's own
        # credentialless path. A grant is not a channel prefix and not a bot-wide permit.
        for chat in ("424242", "1" + GRANTED_CHAT, GRANTED_CHAT[:-1]):
            primary.sent.clear()
            error, standalone = _run({"id": "j2", "deliver": f"discord:{chat}"}, view)
            assert error is not None and "DISCORD_BOT_TOKEN" in error, (chat, error)
            assert primary.sent == [], chat
            assert standalone == [chat]

        # A thread under the granted channel is a DIFFERENT target; the grant does not widen.
        primary.sent.clear()
        error, standalone = _run(
            {"id": "j3", "deliver": f"discord:{GRANTED_CHAT}:4242"}, view)
        assert error is not None and "DISCORD_BOT_TOKEN" in error, error
        assert primary.sent == []

        # A target-less lookup is never authorized, and another platform is never in scope.
        assert view.get(Platform.DISCORD) is None
        assert view.get(Platform.TELEGRAM, {"chat_id": GRANTED_CHAT}) is None
    finally:
        reset_hermes_home_override(token)


def test_grant_is_outbound_only(tmp_path, monkeypatch):
    """The grant must not be readable as an inbound capability: it creates no route, and an entry
    that opts OUT of ``outbound_only`` is dropped rather than honored."""
    from gateway.delivery_grants import parse_delivery_grants
    from gateway.profile_routing import parse_profile_routes

    root = tmp_path / "root"
    satellite_home = _satellite_home(root)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(satellite_home))
    try:
        # Inbound routing for the satellite is unchanged: the grant names a profile, not a route.
        assert parse_profile_routes(PRIMARY_YAML["gateway"].get("profile_routes") or []) == []

        grants = _primary_delivery_grants_for_current_home()
        assert [g.name for g in grants] == ["ops-pages"]

        # ``outbound_only: false`` is a refused config, not a wider one — nothing is parsed from it.
        assert parse_delivery_grants([{
            "bot_platform": "discord", "to_profiles": ["fitness"],
            "targets": ["discord:1"], "outbound_only": False,
        }]) == []
        # So is a grant with no platform, no grantee, or no target: each would otherwise be a
        # silent open invitation.
        for raw in (
            {"to_profiles": ["fitness"], "targets": ["discord:1"]},
            {"bot_platform": "discord", "targets": ["discord:1"]},
            {"bot_platform": "discord", "to_profiles": ["fitness"]},
        ):
            assert parse_delivery_grants([raw]) == [], raw

        # And preflight: the reported alert stop silently dying, plus the target it must NOT wave
        # through. The satellite reads discord as unconnected — it holds no credential itself.
        config = _unconnected_gateway_config()
        job = {"id": "j", "deliver": "local", "failure_deliver": f"discord:{GRANTED_CHAT}"}
        with patch("gateway.config.load_gateway_config", return_value=config):
            assert _preflight_check_delivery(job) is None, _preflight_check_delivery(job)
            ungranted = {"id": "j", "deliver": "local", "failure_deliver": "discord:424242"}
            blocked = _preflight_check_delivery(ungranted)
            assert blocked and "no gateway credentials" in blocked, blocked
    finally:
        reset_hermes_home_override(token)


def test_profile_with_a_bot_of_its_own_gets_no_primary_rescue(tmp_path, monkeypatch):
    """A profile connected on ANY platform is its own credential boundary and never borrows the
    primary's bot (``authz_mixin._is_shared_bot_satellite``), so the borrowed view preflight's
    rescue stands in for is never even built for it: ``tick_adapters_for`` returns the profile's
    own map, and a granted or routed target on another platform finds no adapter and is dropped.

    Preflight has to say so. Clearing it — as it did — is how a ``--failure-deliver`` alert stops
    firing without a word, which is worse than either failing or refusing.
    """
    root = tmp_path / "root"
    satellite_home = _satellite_home(root)
    config_yaml = {
        "gateway": {
            "multiplex_profiles": True,
            # Same grant as above, plus an inbound route to the same profile: BOTH escape hatches
            # are consulted for a target, and neither is reachable without the borrowed view.
            "profile_routes": [{
                "name": "ops-pages", "platform": "discord",
                "chat_id": GRANTED_CHAT, "profile": "fitness",
            }],
            "delivery_grants": PRIMARY_YAML["gateway"]["delivery_grants"],
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    token = set_hermes_home_override(str(satellite_home))
    try:
        config = MagicMock()
        # The satellite brought its own bot — its own platform reads connected, discord does not.
        config.get_connected_platforms.return_value = {Platform.TELEGRAM}
        config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
        config.get_home_channel = lambda p: None

        with patch("gateway.config.load_gateway_config", return_value=config):
            # Routed AND granted target: the grant is target-exact here, the route names this
            # chat, and both would have rescued before.
            for deliver in (f"discord:{GRANTED_CHAT}", "discord:424242"):
                reason = _preflight_check_delivery({"id": "j", "deliver": deliver})
                assert reason and "discord" in reason, (deliver, reason)
    finally:
        reset_hermes_home_override(token)
