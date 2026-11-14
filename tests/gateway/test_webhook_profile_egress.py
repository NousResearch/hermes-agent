"""Webhook egress under multiplex stays bound to the routed profile (#65939, #84266).

A ``/p/<profile>/`` route's reply, deliver_only message, home-channel fallback and
``github_comment`` credential all belong to THAT profile; a default-bound route never
borrows a secondary's adapter. Real ``GatewayAuthorizationMixin`` resolver, real
``load_gateway_config`` against a temp HERMES_HOME — no patched predicates.
"""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter


class _Runner(GatewayAuthorizationMixin):
    def __init__(self, adapters, profile_adapters, config=None):
        self.adapters = adapters
        self._profile_adapters = profile_adapters
        self._primary_profile_name = "default"
        self.config = config or GatewayConfig()


def _target():
    t = MagicMock()
    t.send = AsyncMock(return_value=SendResult(success=True))
    return t


def _webhook(runner) -> WebhookAdapter:
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": {}}))
    adapter.gateway_runner = runner
    return adapter


@pytest.fixture
def profile_homes(tmp_path, monkeypatch):
    """Default home with a DEFAULT-HOME Slack home channel; ``profiles/sec`` with SEC-HOME + its own GH_TOKEN."""
    home = tmp_path / ".hermes"
    sec = home / "profiles" / "sec"
    sec.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: home / "profiles")
    (sec / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\nplatforms:\n  slack:\n    enabled: true\n"
        "    home_channel:\n      platform: slack\n      chat_id: SEC-HOME\n")
    (sec / ".env").write_text("GH_TOKEN=sec-token\n")
    monkeypatch.setenv("GH_TOKEN", "default-token")  # multiplex: os.environ == the default profile
    default_cfg = GatewayConfig()
    default_cfg.platforms[Platform.SLACK] = PlatformConfig(
        enabled=True, home_channel=HomeChannel(platform=Platform.SLACK, chat_id="DEFAULT-HOME", name="Home"))
    return default_cfg


@pytest.mark.asyncio
async def test_routed_profile_delivers_via_its_own_adapter_and_home_channel(profile_homes):
    default, secondary = _target(), _target()
    adapter = _webhook(_Runner({Platform.SLACK: default}, {"sec": {Platform.SLACK: secondary}}, profile_homes))

    result = await adapter._deliver_cross_platform(
        "slack", "hi", {"deliver": "slack", "deliver_extra": {}, "profile": "sec"})

    assert result.success
    default.send.assert_not_awaited()
    assert secondary.send.await_args.args[0] == "SEC-HOME"


@pytest.mark.asyncio
async def test_delivery_fails_closed_instead_of_crossing_profiles(profile_homes):
    """Neither direction may borrow: a secondary route without the platform must not use the default
    bot, and a default-bound route must not use a platform parked only on a secondary."""
    default, secondary = _target(), _target()

    sec_without_slack = _webhook(_Runner({Platform.SLACK: default}, {"sec": {}}, profile_homes))
    res = await sec_without_slack._deliver_cross_platform(
        "slack", "hi", {"deliver": "slack", "deliver_extra": {"chat_id": "C1"}, "profile": "sec"})
    assert not res.success and "not connected" in res.error
    default.send.assert_not_awaited()

    default_without_slack = _webhook(_Runner({}, {"sec": {Platform.SLACK: secondary}}, profile_homes))
    res = await default_without_slack._deliver_cross_platform(
        "slack", "hi", {"deliver": "slack", "deliver_extra": {"chat_id": "C1"}, "profile": None})
    assert not res.success and "not connected" in res.error
    secondary.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_github_comment_authenticates_with_routed_profile_token(profile_homes, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["GH_TOKEN"] = kw["env"].get("GH_TOKEN") if kw.get("env") is not None else os.environ.get("GH_TOKEN")
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr("gateway.platforms.webhook.subprocess.run", fake_run)
    adapter = _webhook(_Runner({}, {"sec": {}}, profile_homes))

    res = await adapter._deliver_github_comment(
        "body", {"deliver": "github_comment", "profile": "sec", "deliver_extra": {"repo": "o/r", "pr_number": "7"}})

    assert res.success
    assert seen["GH_TOKEN"] == "sec-token"


def test_api_server_profile_callback_resolves_routed_profile_adapter_fail_closed():
    default, secondary = object(), object()
    api = APIServerAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    request = MagicMock()
    request.app = {}

    api.gateway_runner = _Runner({Platform("google_chat"): default}, {"sec": {Platform("google_chat"): secondary}})
    token = _api_request_profile.set("sec")
    try:
        assert api._get_platform_callback_adapter(request, "google_chat") is secondary
        api.gateway_runner = _Runner({Platform("google_chat"): default}, {"sec": {}})
        assert api._get_platform_callback_adapter(request, "google_chat") is None
    finally:
        _api_request_profile.reset(token)
    # No prefix: the primary map, unchanged.
    assert api._get_platform_callback_adapter(request, "google_chat") is default


@pytest.mark.asyncio
async def test_retained_destination_keeps_profile_and_freezes_its_home(profile_homes):
    from gateway.platforms.webhook_delivery import snapshot_destination

    default, secondary = _target(), _target()
    adapter = _webhook(_Runner({Platform.SLACK: default}, {"sec": {Platform.SLACK: secondary}}, profile_homes))
    delivery = snapshot_destination(adapter, {"deliver": "slack", "deliver_extra": {}, "profile": "sec"})
    assert delivery == {"deliver": "slack", "deliver_extra": {"chat_id": "SEC-HOME"}, "profile": "sec"}
    adapter._delivery_info["retained"] = delivery
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "profiles" / "sec" / "config.yaml").write_text("platforms: {}\n")
    result = await adapter.send("retained", "answer")
    assert result.success
    assert secondary.send.await_args.args[0] == "SEC-HOME"
    default.send.assert_not_awaited()
    adapter.gateway_runner._profile_adapters["sec"].clear()
    assert not (await adapter.send("retained", "answer")).success
    default.send.assert_not_awaited()


def test_destination_missing_profile_home_does_not_borrow_default(profile_homes):
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import RuntimeStoreError
    from gateway.platforms.webhook_delivery import snapshot_destination, validate_destination

    (get_hermes_home() / "profiles" / "sec" / "config.yaml").write_text("platforms: {}\n")
    adapter = _webhook(_Runner({}, {}, profile_homes))
    with pytest.raises(RuntimeStoreError):
        snapshot_destination(adapter, {"deliver": "slack", "deliver_extra": {}, "profile": "sec"})
    for profile in ([], {}, 1):
        with pytest.raises(RuntimeStoreError):
            validate_destination({"deliver": "log", "deliver_extra": {}, "profile": profile})


@pytest.mark.asyncio
@pytest.mark.parametrize("own", [{}, {Platform.LOCAL: object()}], ids=["no-adapters", "local-only"])
async def test_shared_bot_satellite_webhook_egress_is_route_scoped(profile_homes, tmp_path, own):
    """``sec`` owns no bot; one route maps slack:ROUTED to it. Its ``/p/sec/`` webhook rides the
    primary bot only to a target a route maps to ``sec`` (the cron rule, #147602): the routed chat
    succeeds, an unrouted chat and the home-channel fallback fail closed with no outbound send."""
    from unittest.mock import patch

    from gateway.profile_routing import parse_profile_routes

    profile_homes.multiplex_profiles = True
    profile_homes.profile_routes = parse_profile_routes(
        [{"name": "sec-room", "platform": "slack", "profile": "sec", "chat_id": "ROUTED"}])
    primary = _target()
    webhook = _webhook(_Runner({Platform.SLACK: primary}, {"sec": dict(own)}, profile_homes))

    async def deliver(extra):
        return await webhook._deliver_cross_platform(
            "slack", "hi", {"deliver": "slack", "deliver_extra": extra, "profile": "sec"})

    with patch("gateway.run._multiplex_profile_homes", return_value=[("sec", tmp_path / "sec")]):
        assert (await deliver({"chat_id": "ROUTED"})).success
        assert primary.send.await_args.args[0] == "ROUTED"
        primary.send.reset_mock()
        for extra in ({"chat_id": "UNROUTED"}, {}):  # {} → SEC-HOME home channel, also unrouted
            res = await deliver(extra)
            assert not res.success and "not routed" in res.error
        primary.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_bot_satellite_cannot_escape_routed_chat_via_thread_id(profile_homes, tmp_path):
    """Discord resolves ``thread_id`` as a global channel id, so a routed chat_id paired with a
    payload-rendered foreign ``thread_id`` must not ride the shared bot; a thread the route pins does.
    Telegram topic ids are chat-scoped, so a routed chat's topic still delivers (#147602)."""
    from unittest.mock import patch

    from gateway.profile_routing import parse_profile_routes

    profile_homes.multiplex_profiles = True
    profile_homes.profile_routes = parse_profile_routes([
        {"name": "d", "platform": "discord", "profile": "sec", "chat_id": "111"},
        {"name": "dt", "platform": "discord", "profile": "sec", "chat_id": "222", "thread_id": "333"},
        {"name": "t", "platform": "telegram", "profile": "sec", "chat_id": "-100"},
    ])
    discord, telegram = _target(), _target()
    webhook = _webhook(_Runner({Platform.DISCORD: discord, Platform.TELEGRAM: telegram}, {"sec": {}}, profile_homes))

    async def deliver(platform, extra):
        return await webhook._deliver_cross_platform(
            platform, "hi", {"deliver": platform, "deliver_extra": extra, "profile": "sec"})

    with patch("gateway.run._multiplex_profile_homes", return_value=[("sec", tmp_path / "sec")]):
        for extra in ({"chat_id": "111", "thread_id": "999"}, {"chat_id": "111", "message_thread_id": "999"}):
            res = await deliver("discord", extra)
            assert not res.success and "not routed" in res.error
        discord.send.assert_not_awaited()
        assert (await deliver("discord", {"chat_id": "222", "thread_id": "333"})).success
        assert discord.send.await_args.kwargs["metadata"] == {"thread_id": "333"}
        assert (await deliver("telegram", {"chat_id": "-100", "message_thread_id": "42"})).success
