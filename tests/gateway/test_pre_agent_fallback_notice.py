"""A fallback resolved during gateway credential resolution (before any AIAgent exists) must carry a
user-visible notice through the agent's one-shot fallback-notice mechanism (#74349).

Drives the production entry point — ``GatewayTurnMixin._resolve_session_agent_runtime`` bound to the
runner, then ``TurnRunner.run_sync`` — so the pop in run_turn.py and the attach in run_turn_runner.py
are both pinned (a helper-only test stays green with either removed)."""
import types
import gateway.run  # Import bootstrap before the per-test home I/O guard is active.
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gateway.run_turn import GatewayTurnMixin
from gateway.session import Platform, SessionSource
from gateway.turn_context import TurnContext
from hermes_cli.auth import AuthError


class _RecordingAgent:
    built_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).built_kwargs = kwargs
        self.model = kwargs["model"]
        self.session_id = kwargs.get("session_id")
        self.tools = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200_000)
        self.session_prompt_tokens = self.session_completion_tokens = 0

    def run_conversation(self, _message, **_kwargs):
        return {"final_response": "ok", "messages": []}


def _runner_with_real_runtime_resolution():
    runner = MagicMock()
    runner.config = SimpleNamespace(streaming=None)
    runner._provider_routing = {}
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._session_db = runner._prefill_messages = None
    runner._pending_model_notes = runner._pending_skills_reload_notes = {}
    runner.session_store._entries = {}
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._agent_config_signature.return_value = ("sig",)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = None
    runner._consume_pending_native_image_paths.return_value = []
    runner._consume_pending_turn_sidecar_notes.return_value = []
    for lane in ("_is_telegram_topic_lane", "_is_discord_auto_thread_lane", "_is_relay_discord_channel_lane"):
        getattr(runner, lane).return_value = False
    # Production resolution: no /model override, no channel override.
    runner._resolve_session_key_or_none.return_value = "test-session-key"
    runner._peek_session_state.return_value = None
    runner._sessions_map.return_value = {}
    runner._resolve_session_agent_runtime = types.MethodType(GatewayTurnMixin._resolve_session_agent_runtime, runner)
    # Pass the resolved runtime straight through so the kwargs handed to AIAgent are the resolved ones.
    # (Production pops ``request_overrides`` out of the runtime into the route; mirror that so the
    # resolved kwargs never carry it twice.)
    runner._resolve_turn_agent_config.side_effect = lambda _msg, model, rt: {
        "model": model, "runtime": {k: v for k, v in rt.items() if k != "request_overrides"}}
    return runner


def test_credential_resolution_fallback_reaches_agent_notice_not_agent_kwargs():
    from gateway.run_turn_runner import TurnRunner

    fb = {"provider": "anthropic", "model": "claude-sonnet-5", "api_key": "k", "base_url": "u"}
    runner = _runner_with_real_runtime_resolution()
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="hi", history=[], session_id="sid", session_key="test-session-key", user_config={},
        AIAgent=_RecordingAgent, resolve_display_setting=lambda *_a: False, _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )
    def primary_auth_fails(**kw):
        if kw.get("requested") is None:  # the primary, resolved from config.yaml
            raise AuthError("expired")
        return dict(fb)  # the fallback entry, walked by resolve_runtime_with_fallback

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=primary_auth_fails), \
         patch("hermes_cli.runtime_provider._get_model_config",
               return_value={"provider": "openai-codex", "default": "gpt-5.6-sol"}), \
         patch("gateway.run._load_gateway_config",
               return_value={"fallback_providers": [{"provider": "anthropic", "model": "claude-sonnet-5"}]}), \
         patch("gateway.run._resolve_gateway_model", return_value="gpt-5.6-sol"), \
         patch("gateway.run._get_channel_override", return_value=None), \
         patch("gateway.run._current_max_iterations", return_value=100):
        result = TurnRunner(runner, ctx).run_sync()

    assert result["final_response"] == "ok"
    agent = ctx.agent_holder[0]
    notice = agent._pending_fallback_notice
    assert "openai-codex/gpt-5.6-sol" in notice and "anthropic/claude-sonnet-5" in notice
    assert "_fallback_notice" not in _RecordingAgent.built_kwargs
    # Consumed by the turn: a later resolution without fallback must not re-attach a stale notice.
    assert runner._pre_agent_fallback_notice is None


def test_model_override_fast_path_clears_stale_notice():
    """The /model-override fast path returns before the pop; a notice stashed by an earlier resolution
    (hygiene, inbound, another session) must not survive to attach to this session's turn."""
    runner = _runner_with_real_runtime_resolution()
    runner._pre_agent_fallback_notice = "⚠️ Provider fallback: stale"
    override = {"model": "claude-sonnet-5", "provider": "anthropic", "api_key": "k", "base_url": "u"}
    runner._peek_session_state.return_value = SimpleNamespace(conversation=SimpleNamespace(model_override=override))
    with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.6-sol"), \
         patch("gateway.run._credential_pool_for_provider", return_value=None):
        model, runtime = runner._resolve_session_agent_runtime(session_key="test-session-key")
    assert (model, runtime["provider"]) == ("claude-sonnet-5", "anthropic")
    assert runner._pre_agent_fallback_notice is None


def test_gateway_resolution_time_policy_survives_turn_route_to_constructor():
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from gateway.run_turn_runner import TurnRunner

    runner = _runner_with_real_runtime_resolution()
    runner._service_tier = "priority"
    runner._resolve_turn_agent_config = types.MethodType(GatewayTurnPrepareMixin._resolve_turn_agent_config, runner)
    ctx = TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="hi", history=[], session_id="sid", session_key="test-session-key", user_config={},
        AIAgent=_RecordingAgent, resolve_display_setting=lambda *_a: False, _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )

    def resolve(**kwargs):
        if kwargs.get("requested") is None:
            raise AuthError("expired")
        return {"provider": "custom", "api_key": "k", "base_url": "https://fallback.example/v1",
                "request_overrides": {"speed": "fast", "extra_body": {"service_tier": "priority", "marker": 1}}}

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
         patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "openai", "default": "primary"}), \
         patch("gateway.run._load_gateway_config", return_value={"fallback_providers": [
             {"provider": "custom", "model": "fallback", "service_tier_override": "normal"}]}), \
         patch("gateway.run._resolve_gateway_model", return_value="primary"), \
         patch("gateway.run._get_channel_override", return_value=None), \
         patch("gateway.run._current_max_iterations", return_value=100):
        assert TurnRunner(runner, ctx).run_sync()["final_response"] == "ok"
    assert _RecordingAgent.built_kwargs["fallback_service_tier_override"] == "normal"
    from tests.agent.fallback_policy_wire import first_wire_from_runtime
    wire, snapshot = first_wire_from_runtime(_RecordingAgent.built_kwargs)
    assert snapshot["fallback_service_tier_override"] == "normal"
    assert "service_tier" not in wire and "speed" not in wire
    assert "service_tier" not in wire["extra_body"] and "speed" not in wire["extra_body"]
    assert wire["extra_body"]["marker"] == "keep"


def test_gateway_cache_signature_changes_with_selected_fallback_tier():
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin

    runner = GatewayTurnPrepareMixin()
    setattr(runner, "_service_tier", None)
    base = {"provider": "deepseek", "args": [], "capabilities": {}}
    normal = GatewayTurnPrepareMixin._resolve_turn_agent_config(
        runner, "hi", "fallback", {**base, "fallback_service_tier_override": "normal"},
    )
    inherited = GatewayTurnPrepareMixin._resolve_turn_agent_config(
        runner, "hi", "fallback", {**base, "fallback_service_tier_override": None},
    )
    normal_sig = GatewayAgentCacheMixin._agent_config_signature(
        "fallback", normal["runtime"], ["file"], "",
    )
    inherited_sig = GatewayAgentCacheMixin._agent_config_signature(
        "fallback", inherited["runtime"], ["file"], "",
    )

    assert normal["runtime"]["fallback_service_tier_override"] == "normal"
    assert normal_sig != inherited_sig


def test_frozen_gateway_runtime_carries_selected_fallback_policy_to_first_request(monkeypatch):
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from tests.agent.fallback_policy_wire import first_wire_from_runtime

    def resolve(*, requested, **kw):
        if requested == "openai":
            raise AuthError("no primary key")
        return {"provider": "deepseek", "api_key": "test-key", "base_url": "https://api.deepseek.com/v1", "api_mode": "chat_completions"}

    frozen = {"model": {"default": "primary", "provider": "openai"},
              "fallback_providers": [{"provider": "deepseek", "model": "fallback", "service_tier_override": "normal"}]}
    policy = SimpleNamespace(effective=lambda _authority: (frozen, "primary"),
                             credential_ref=None, request_json="{}")
    runner = SimpleNamespace(session_authority=SimpleNamespace())
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    model, runtime = _resolve_policy_agent_runtime(runner, policy)
    assert model == "fallback"
    assert runtime["fallback_service_tier_override"] == "normal"
    wire, snapshot = first_wire_from_runtime({**runtime, "model": model})
    assert snapshot["fallback_service_tier_override"] == "normal"
    assert "service_tier" not in wire and "speed" not in wire
