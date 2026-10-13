"""``channel_overrides.<id>.service_tier`` scopes a speed tier to one channel.
Precedence: session /fast > channel override > profile ``agent.service_tier``."""

import logging

import pytest

import gateway.run as gateway_run
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.gateway.test_fast_command import _make_runner

GENERAL = "100000000000000001"
OTHER = "100000000000000002"
_ASTRA_ON_CODEX = {"model": "gpt-6-astra", "provider": "openai-codex",
                   "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "***"}


def _source(chat_id, **kw):
    return SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type="group", user_id="u1", **kw)


def _runner(monkeypatch, overrides, profile_tier=None):
    runner = _make_runner()
    runner.config = GatewayConfig(platforms={
        Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides=overrides)})
    agent_cfg = {"service_tier": profile_tier} if profile_tier else {}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: {"agent": agent_cfg})
    return runner


def test_config_round_trip_keeps_service_tier_and_drops_unset():
    cfg = PlatformConfig.from_dict({"channel_overrides": {
        GENERAL: {"model": "gpt-6-astra", "provider": "openai-codex", "service_tier": "ultrafast"},
        OTHER: {"model": "gpt-6.1-sol", "provider": "openai-codex"}}})
    assert cfg.channel_overrides[GENERAL].service_tier == "ultrafast"
    assert cfg.channel_overrides[OTHER].service_tier is None
    dumped = cfg.to_dict()["channel_overrides"]
    assert dumped[GENERAL]["service_tier"] == "ultrafast"
    assert "service_tier" not in dumped[OTHER]


def test_channel_tier_applies_only_to_its_channel_and_threads(monkeypatch):
    runner = _runner(monkeypatch, {
        GENERAL: ChannelOverride(model="gpt-6-astra", provider="openai-codex", service_tier="ultrafast"),
        OTHER: ChannelOverride(model="gpt-6.1-sol", provider="openai-codex")})
    assert runner._resolve_session_service_tier(source=_source(GENERAL)) == "ultrafast"
    # Discord threads inherit the parent channel's override, as model overrides do.
    assert runner._resolve_session_service_tier(source=_source("777", thread_id="777", parent_chat_id=GENERAL)) == "ultrafast"
    assert runner._resolve_session_service_tier(source=_source(OTHER)) is None
    assert runner._resolve_session_service_tier(source=_source("123")) is None
    # No source (CLI-style callers): profile tier only.
    assert runner._resolve_session_service_tier(session_key="unrelated") is None


def test_session_fast_beats_channel_and_channel_beats_profile(monkeypatch):
    runner = _runner(monkeypatch, {
        GENERAL: ChannelOverride(service_tier="ultrafast"), OTHER: ChannelOverride(service_tier="normal")},
        profile_tier="fast")
    general = _source(GENERAL)
    assert runner._resolve_session_service_tier(source=_source(OTHER)) is None  # explicit normal pins Standard
    assert runner._resolve_session_service_tier(source=_source("123")) == "priority"  # profile tier
    key = runner._session_key_for_source(general)
    runner._set_session_service_tier_override(key, None)  # /fast normal in #general
    assert runner._resolve_session_service_tier(source=general) is None
    runner._set_session_service_tier_override(key, None, clear=True)
    assert runner._resolve_session_service_tier(source=general) == "ultrafast"


def test_unknown_channel_tier_warns_and_uses_profile(monkeypatch, caplog):
    runner = _runner(monkeypatch, {GENERAL: ChannelOverride(service_tier="warp")}, profile_tier="fast")
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        assert runner._resolve_session_service_tier(source=_source(GENERAL)) == "priority"
    assert "Unknown channel_overrides service_tier 'warp'" in caplog.text


def test_channel_ultrafast_reaches_the_turn_route_on_astra_codex(monkeypatch):
    runner = _runner(monkeypatch, {GENERAL: ChannelOverride(model="gpt-6-astra", provider="openai-codex",
                                                            service_tier="ultrafast")})
    runner._service_tier = runner._resolve_session_service_tier(source=_source(GENERAL))
    route = runner._resolve_turn_agent_config("hi", "gpt-6-astra", dict(_ASTRA_ON_CODEX, args=[]))
    assert route["request_overrides"] == {"service_tier": "ultrafast"}


@pytest.mark.asyncio
async def test_fast_status_in_the_channel_reports_the_channel_tier(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, {GENERAL: ChannelOverride(service_tier="ultrafast")})
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-6-astra")
    general = _source(GENERAL)
    runner._session_model_overrides[runner._session_key_for_source(general)] = dict(_ASTRA_ON_CODEX)
    monkeypatch.setattr(runner, "_try_send_choice_picker", _no_picker)
    await runner._handle_fast_command(MessageEvent(text="/fast status", source=general, message_id="m1"))
    assert runner._service_tier == "ultrafast"


async def _no_picker(*args, **kwargs):
    return False
