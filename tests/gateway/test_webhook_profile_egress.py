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
