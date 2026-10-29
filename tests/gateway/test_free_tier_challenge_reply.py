"""What a messaging user is told when the free tier's browser check blocks a turn.

The trial runs every surface once the Desktop claimed it, but only the Desktop can clear the account
service's browser challenge: the gateway never waits on one and opens nothing. Under ``provider: auto``
the resolver swallows the pending check into a keyless stand-in, and the agent built on it fails as
"no provider configured". A turn that ends there must say where the check is finished; a turn whose
channel names its own working provider must run on it, untouched by the check.
"""
from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.hermes_cli.anon_portal import claimed_trial, install_portal, off_local_desktop


@pytest.fixture
def challenged_gateway(monkeypatch, tmp_path):
    from gateway import status
    from hermes_cli import anon_challenge

    nas = install_portal(monkeypatch, tmp_path)
    claimed_trial(monkeypatch)
    off_local_desktop(monkeypatch)  # a bot's turn, not the Desktop's
    monkeypatch.setattr(status, "owns_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr(anon_challenge, "CHALLENGE_WAIT_SECONDS", 0.05)
    nas.challenge_required = True  # the Desktop's clearance has lapsed: the next exchange is challenged
    nas.challenge_never_clears = True
    return nas


class _ChannelAgent:
    """Stands in for AIAgent on a route that has a key: records what it was built with."""
    built: dict = {}

    def __init__(self, **kwargs):
        type(self).built = kwargs
        self.model, self.provider = kwargs["model"], kwargs.get("provider")
        self.session_id = kwargs.get("session_id")
        self.tools = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200_000)
        self.session_prompt_tokens = self.session_completion_tokens = 0

    def run_conversation(self, _message, **_kwargs):
        return {"final_response": "answered by the channel's provider", "messages": []}


def _runner(channel_overrides=None):
    """A GatewayRunner double whose model/runtime resolution is the production one."""
    from gateway.run_turn import GatewayTurnMixin

    runner = MagicMock()
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        enabled=True, channel_overrides=channel_overrides or {})})
    runner._provider_routing = {}
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._session_db = runner._prefill_messages = None
    runner._pending_model_notes = runner._pending_skills_reload_notes = {}
    runner.session_store._entries = {}
    runner._pre_agent_fallback_notice = None
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._agent_config_signature.return_value = ("sig",)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = None
    runner._consume_pending_native_image_paths.return_value = []
    runner._consume_pending_turn_sidecar_notes.return_value = []
    runner._delivery_adapter_for.return_value = None
    for lane in ("_is_telegram_topic_lane", "_is_discord_auto_thread_lane", "_is_relay_discord_channel_lane"):
        getattr(runner, lane).return_value = False
    runner._resolve_session_key_or_none.return_value = "bot-session"
    runner._peek_session_state.return_value = None
    runner._sessions_map.return_value = {}
    runner._resolve_session_agent_runtime = types.MethodType(GatewayTurnMixin._resolve_session_agent_runtime, runner)
    runner._resolve_turn_agent_config.side_effect = lambda _msg, model, rt: {
        "model": model, "runtime": {k: v for k, v in rt.items() if k != "request_overrides"}}
    return runner


def _telegram_turn(runner, agent_class):
    from gateway.run_turn_runner import TurnRunner

    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1"),
        message="hi", history=[], session_id="sid", session_key="bot-session", user_config={},
        AIAgent=agent_class, resolve_display_setting=lambda *_a: False, _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False))
    return TurnRunner(runner, ctx).run_sync()


def test_a_messaging_turn_blocked_by_the_browser_check_says_to_open_the_desktop_app(challenged_gateway):
    from agent.i18n import t
    from run_agent import AIAgent

    result = _telegram_turn(_runner(), AIAgent)

    assert result["final_response"] == t("gateway.errors.free_tier_check_pending")
    assert (result["failure_reason"], result["failure_retryable"]) == ("auth", True)
    assert "/api/anonymous/challenge/status" not in [p for _, p in challenged_gateway.calls]


def test_a_channel_with_its_own_working_provider_answers_despite_the_pending_check(challenged_gateway):
    """The global route's pending check belongs to the trial; a channel override that names a working
    custom provider never uses the trial, so its turn runs on that provider."""
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(
        "custom_providers:\n- name: team-llm\n  base_url: http://127.0.0.1:9/v1\n  api_key: sk-team-llm-key\n")
    runner = _runner({"chat-1": ChannelOverride(provider="team-llm", model="team-model")})

    result = _telegram_turn(runner, _ChannelAgent)

    assert result["final_response"] == "answered by the channel's provider"
    assert not result.get("failed")
    assert (_ChannelAgent.built["base_url"], _ChannelAgent.built["api_key"]) == (
        "http://127.0.0.1:9/v1", "sk-team-llm-key")
