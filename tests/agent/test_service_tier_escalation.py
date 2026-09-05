"""Per-turn OpenRouter service-tier escalation on streaming TTFT."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _effective_request_overrides
from agent.service_tier_escalation import (
    ServiceTierEscalationState,
    TtftObservation,
    accept_logical_request,
    apply_escalation_to_overrides,
    begin_escalation_turn,
    begin_logical_request,
    begin_request_ttft,
    bind_service_tier_escalation,
    end_request_ttft,
    finish_request_ttft,
    note_non_observation,
    rebase_escalation_runtime,
    reset_escalation_for_model_switch,
)
from hermes_constants import ServiceTierEscalationConfig


def _enabled_cfg(threshold=8.0, consecutive=1):
    return {
        "enabled": True,
        "ttft_threshold_seconds": threshold,
        "consecutive_slow_requests": consecutive,
    }


def _agent(**kwargs):
    agent = SimpleNamespace(
        service_tier=kwargs.get("service_tier", "flex"),
        request_overrides=dict(kwargs.get("request_overrides", {})),
        model=kwargs.get("model", "google/gemini-flash"),
        provider=kwargs.get("provider", "openrouter"),
        base_url=kwargs.get("base_url", "https://openrouter.ai/api/v1"),
        _service_tier_session_pinned=kwargs.get("pinned", False),
        platform=kwargs.get("platform", "cli"),
        _persist_disabled=kwargs.get("persist_disabled", False),
        _delegate_depth=kwargs.get("delegate_depth", 0),
        is_subagent=kwargs.get("is_subagent", False),
        _interrupt_requested=False,
        _block_service_tier_escalation=kwargs.get("block_escalation", False),
    )
    bind_service_tier_escalation(agent, kwargs.get("escalation", _enabled_cfg()))
    return agent


def _slow_obs():
    obs = TtftObservation(clock=lambda: 0.0)
    obs.t_send = 0.0
    obs.t_first = 20.0
    obs.open_count = 1
    return obs


class TestServiceTierEscalationStateMachine:
    def test_ladder_flex_default_priority_and_cap(self):
        state = ServiceTierEscalationState(
            ServiceTierEscalationConfig(enabled=True, ttft_threshold_seconds=8.0, consecutive_slow_requests=1),
            base_tier="flex",
        )
        state.observe_ttft(9.0, model="m")
        assert state.effective_tier is None
        state.observe_ttft(9.0, model="m")
        assert state.effective_tier == "priority"
        state.observe_ttft(9.0, model="m")
        assert state.effective_tier == "priority"

    def test_fast_response_resets_streak_keeps_tier(self):
        state = ServiceTierEscalationState(
            ServiceTierEscalationConfig(enabled=True, ttft_threshold_seconds=8.0, consecutive_slow_requests=2),
            base_tier="flex",
        )
        state.observe_ttft(9.0, model="m")
        assert state.streak == 1
        assert state.effective_tier == "flex"
        state.observe_ttft(1.0, model="m")
        assert state.streak == 0
        assert state.effective_tier == "flex"

    def test_escalates_only_after_n_consecutive_slow(self):
        state = ServiceTierEscalationState(
            ServiceTierEscalationConfig(enabled=True, ttft_threshold_seconds=8.0, consecutive_slow_requests=2),
            base_tier="flex",
        )
        state.observe_ttft(9.0, model="m")
        assert state.effective_tier == "flex"
        state.observe_ttft(9.0, model="m")
        assert state.effective_tier is None
        assert state.streak == 0

    def test_begin_turn_resets_to_base(self):
        agent = _agent(service_tier="flex")
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        assert agent._service_tier_escalation.effective_tier is None
        begin_escalation_turn(agent)
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert agent._service_tier_escalation.streak == 0

    def test_pin_disables_finish_observation(self):
        agent = _agent(service_tier="flex", pinned=True)
        obs = TtftObservation(clock=lambda: 0.0)
        obs.t_send = 0.0
        obs.t_first = 20.0
        obs.open_count = 1
        finish_request_ttft(agent, obs)
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert agent._service_tier_escalation.streak == 0

    def test_non_streaming_resets_streak_keeps_tier(self):
        agent = _agent(service_tier="flex")
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        assert agent._service_tier_escalation.effective_tier is None
        agent._service_tier_escalation.streak = 4
        note_non_observation(agent)
        assert agent._service_tier_escalation.effective_tier is None
        assert agent._service_tier_escalation.streak == 0

    def test_switch_model_resets_and_adopts_new_base(self):
        agent = _agent(service_tier="flex")
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        agent.service_tier = "priority"
        reset_escalation_for_model_switch(agent)
        assert agent._service_tier_escalation.base_tier == "priority"
        assert agent._service_tier_escalation.effective_tier == "priority"
        assert agent._service_tier_escalation.streak == 0

    def test_base_priority_is_noop(self):
        state = ServiceTierEscalationState(
            ServiceTierEscalationConfig(enabled=True),
            base_tier="priority",
        )
        state.observe_ttft(30.0, model="m")
        assert state.effective_tier == "priority"


class TestServiceTierEscalationApply:
    def test_next_request_carries_escalated_tier_and_baseline_untouched(self):
        agent = _agent(service_tier="flex", request_overrides={"keep": 1})
        canonical_overrides = dict(agent.request_overrides)
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        applied = _effective_request_overrides(agent)
        assert "service_tier" not in applied
        assert applied.get("keep") == 1
        assert agent.service_tier == "flex"
        assert agent.request_overrides == canonical_overrides

        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        applied = _effective_request_overrides(agent)
        assert applied["service_tier"] == "priority"
        assert agent.service_tier == "flex"
        assert agent.request_overrides == canonical_overrides

    def test_flex_to_default_removes_override_key(self):
        agent = _agent(
            service_tier="flex",
            request_overrides={"service_tier": "flex"},
        )
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        applied = _effective_request_overrides(agent)
        assert "service_tier" not in applied
        assert "speed" not in applied
        assert agent.request_overrides == {"service_tier": "flex"}

    def test_non_openrouter_is_not_overlaid(self):
        agent = _agent(
            service_tier="flex",
            provider="openai",
            base_url="https://api.openai.com/v1",
        )
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        applied = apply_escalation_to_overrides(agent, {"service_tier": "flex"})
        assert applied == {"service_tier": "flex"}


class TestServiceTierEscalationIsolation:
    def test_cron_platform_does_not_escalate(self):
        agent = _agent(platform="cron", service_tier="flex")
        obs = TtftObservation(clock=lambda: 0.0)
        obs.t_send = 0.0
        obs.t_first = 20.0
        obs.open_count = 1
        finish_request_ttft(agent, obs)
        assert agent._service_tier_escalation.effective_tier == "flex"
        applied = _effective_request_overrides(agent)
        assert applied.get("service_tier") == "flex"

    def test_subagent_platform_does_not_escalate(self):
        agent = _agent(platform="subagent", service_tier="flex")
        obs = TtftObservation(clock=lambda: 0.0)
        obs.t_send = 0.0
        obs.t_first = 20.0
        obs.open_count = 1
        finish_request_ttft(agent, obs)
        assert agent._service_tier_escalation.effective_tier == "flex"

    def test_background_review_persist_disabled_does_not_escalate(self):
        agent = _agent(persist_disabled=True, service_tier="flex")
        obs = TtftObservation(clock=lambda: 0.0)
        obs.t_send = 0.0
        obs.t_first = 20.0
        obs.open_count = 1
        finish_request_ttft(agent, obs)
        assert agent._service_tier_escalation.effective_tier == "flex"

    def test_unconfigured_agent_is_disabled(self):
        agent = SimpleNamespace(
            service_tier="flex",
            platform="cli",
            _service_tier_session_pinned=False,
            _persist_disabled=False,
            _delegate_depth=0,
            is_subagent=False,
        )
        bind_service_tier_escalation(agent, None)
        assert agent._service_tier_escalation.enabled is False


class TestTtftObservationClock:
    def test_injected_clock_without_sleep(self):
        ticks = iter([10.0, 19.5])
        obs = TtftObservation(clock=lambda: next(ticks))
        obs.mark_send()
        obs.mark_first_delta()
        assert obs.ttft_seconds() == 9.5
        assert obs.was_retried() is False

    def test_retry_open_count(self):
        obs = TtftObservation(clock=lambda: 1.0)
        obs.mark_send()
        obs.mark_send()
        assert obs.was_retried() is True

    def test_stack_is_request_local(self):
        agent = _agent()
        first = begin_request_ttft(agent, clock=lambda: 1.0)
        second = begin_request_ttft(agent, clock=lambda: 2.0)
        assert agent._ttft_obs_stack[-1] is second
        end_request_ttft(agent, second)
        assert agent._ttft_obs_stack[-1] is first
        end_request_ttft(agent, first)
        assert agent._ttft_obs_stack == []


class TestServiceTierEscalationLoaders:
    def test_cli_defaults_include_disabled_section(self):
        import cli as cli_mod
        from hermes_constants import resolve_service_tier_escalation_config

        cfg = resolve_service_tier_escalation_config(
            cli_mod.load_cli_config().get("agent") or {},
        )
        assert cfg.enabled is False
        assert cfg.ttft_threshold_seconds == 8.0
        assert cfg.consecutive_slow_requests == 1

    def test_gateway_loader_reads_section(self, monkeypatch):
        import gateway.run as gateway_run

        monkeypatch.setattr(
            gateway_run,
            "_load_gateway_runtime_config",
            lambda: {
                "agent": {
                    "service_tier_escalation": {
                        "enabled": True,
                        "ttft_threshold_seconds": 2.5,
                        "consecutive_slow_requests": 4,
                    },
                },
            },
        )
        cfg = gateway_run.GatewayRunner._load_service_tier_escalation()
        assert cfg.enabled is True
        assert cfg.ttft_threshold_seconds == 2.5
        assert cfg.consecutive_slow_requests == 4

    def test_gateway_loader_defaults_disabled(self, monkeypatch):
        import gateway.run as gateway_run

        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {})
        cfg = gateway_run.GatewayRunner._load_service_tier_escalation()
        assert cfg.enabled is False

    def test_tui_loader_reads_section(self, monkeypatch):
        import tui_gateway.server as server

        monkeypatch.setattr(
            server,
            "_load_cfg",
            lambda: {
                "agent": {
                    "service_tier_escalation": {
                        "enabled": True,
                        "ttft_threshold_seconds": 1.25,
                        "consecutive_slow_requests": 2,
                    },
                },
            },
        )
        cfg = server._load_service_tier_escalation()
        assert cfg.enabled is True
        assert cfg.ttft_threshold_seconds == 1.25
        assert cfg.consecutive_slow_requests == 2

    def test_tui_loader_defaults_disabled(self, monkeypatch):
        import tui_gateway.server as server

        monkeypatch.setattr(server, "_load_cfg", lambda: {})
        cfg = server._load_service_tier_escalation()
        assert cfg.enabled is False


class TestSwitchModelResetsEscalation:
    def test_real_switch_model_resets_ladder(self, monkeypatch):
        from agent.agent_runtime_helpers import switch_model

        agent = MagicMock()
        agent.model = "openai/gpt-5"
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.api_mode = "chat_completions"
        agent.api_key = "test-key"
        agent._client_kwargs = {
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
        }
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent.reasoning_config = None
        agent.service_tier = "flex"
        agent._service_tier_session_pinned = False
        agent._provider_routing_config = {}
        agent.providers_allowed = None
        agent.providers_ignored = None
        agent.providers_order = None
        agent.provider_sort = None
        agent.provider_require_parameters = False
        agent.provider_data_collection = None
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._fallback_chain = []
        agent._fallback_model = None
        agent._config_context_length = None
        agent._transport_cache = {}
        agent.context_compressor = None
        agent._cached_system_prompt = None
        agent._anthropic_api_key = ""
        agent._anthropic_base_url = None
        agent._is_anthropic_oauth = False
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
        agent._ensure_lmstudio_runtime_loaded = MagicMock(return_value=None)
        agent._lmstudio_load_was_unverified = MagicMock(return_value=False)
        agent._effective_lmstudio_context_length = MagicMock(return_value=None)
        agent.requested_provider = "openrouter"
        bind_service_tier_escalation(agent, _enabled_cfg())
        agent._service_tier_escalation.observe_ttft(20.0, model="m")
        assert agent._service_tier_escalation.effective_tier is None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"agent": {"service_tier": "priority"}, "provider_routing": {}},
        )
        switch_model(
            agent,
            new_model="openai/gpt-5",
            new_provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )
        assert agent._service_tier_escalation.effective_tier == agent.service_tier
        assert agent._service_tier_escalation.streak == 0


class TestOuterRetryHoldsPreAttemptTier:
    def test_slow_ttft_outer_retry_stays_on_base_until_accept(self):
        agent = _agent(service_tier="flex")
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert agent._service_tier_escalation.pending_ttft == 20.0
        retry_overrides = _effective_request_overrides(agent)
        assert retry_overrides.get("service_tier") == "flex"

        begin_logical_request(agent)
        assert agent._service_tier_escalation.pending_ttft is None
        assert agent._service_tier_escalation.streak == 0
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert _effective_request_overrides(agent).get("service_tier") == "flex"

        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier is None
        assert agent._service_tier_escalation.wire_locked is False

        begin_logical_request(agent)
        next_overrides = _effective_request_overrides(agent)
        assert "service_tier" not in next_overrides

    def test_observation_commits_once_per_logical_request(self):
        agent = _agent(service_tier="flex", escalation=_enabled_cfg(consecutive=1))
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier is None
        assert agent._service_tier_escalation.streak == 0
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier is None

    def test_internal_stream_retry_is_not_an_observation(self):
        agent = _agent(service_tier="flex")
        begin_logical_request(agent)
        obs = _slow_obs()
        obs.open_count = 2
        finish_request_ttft(agent, obs)
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert agent._service_tier_escalation.pending_ttft is None


class TestHardGatedSurfaces:
    def test_batch_like_agent_does_not_escalate_with_enabled_config(self):
        agent = _agent(service_tier="flex", block_escalation=True)
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.enabled is True
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert _effective_request_overrides(agent).get("service_tier") == "flex"

    def test_gateway_background_agent_does_not_escalate_with_enabled_config(self):
        agent = _agent(
            service_tier="flex",
            platform="telegram",
            block_escalation=True,
        )
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.enabled is True
        assert agent._service_tier_escalation.effective_tier == "flex"


class TestInjectAndContinueUnlocksWire:
    """Outer-loop inject-and-continue must accept so the next call is not a retry."""

    def test_inject_accept_updates_wire_snapshot_and_allows_later_climb(self):
        agent = _agent(service_tier="flex", escalation=_enabled_cfg(consecutive=1))
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        # * Invalid tool name / invalid-JSON-after-retry inject the response
        # into history and continue — that is a new logical request.
        accept_logical_request(agent)
        assert agent._service_tier_escalation.wire_locked is False
        assert agent._service_tier_escalation.effective_tier is None
        assert agent._service_tier_escalation.pending_ttft is None

        begin_logical_request(agent)
        assert agent._service_tier_escalation.wire_locked is True
        assert agent._service_tier_escalation.wire_tier is None
        assert "service_tier" not in _effective_request_overrides(agent)

        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier == "priority"
        begin_logical_request(agent)
        assert _effective_request_overrides(agent).get("service_tier") == "priority"

    def test_inject_accept_commits_streak_so_next_logical_request_can_escalate(self):
        agent = _agent(service_tier="flex", escalation=_enabled_cfg(consecutive=2))
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.streak == 1
        assert agent._service_tier_escalation.effective_tier == "flex"

        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier is None
        assert agent._service_tier_escalation.wire_locked is False

    def test_skipping_accept_makes_next_begin_a_retry_and_blocks_climb(self):
        agent = _agent(service_tier="flex", escalation=_enabled_cfg(consecutive=2))
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        # * Bug class: inject-and-continue without accept leaves wire_locked.
        begin_logical_request(agent)
        assert agent._service_tier_escalation.pending_ttft is None
        assert agent._service_tier_escalation.streak == 0
        assert agent._service_tier_escalation.wire_tier == "flex"
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert agent._service_tier_escalation.streak == 1

    def test_loop_wrapper_accept_is_idempotent_without_pending(self):
        from agent.conversation_loop import _try_accept_logical_request

        agent = _agent(service_tier="flex")
        begin_logical_request(agent)
        _try_accept_logical_request(agent)
        _try_accept_logical_request(agent)
        assert agent._service_tier_escalation.wire_locked is False
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert agent._service_tier_escalation.pending_ttft is None


class TestModelSwitchClearsDefaultBase:
    def test_flex_to_normal_omits_service_tier_then_escalates_from_none(self):
        agent = _agent(service_tier="flex")
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        agent.service_tier = None
        reset_escalation_for_model_switch(agent)
        assert agent._service_tier_escalation.base_tier is None
        assert agent._service_tier_escalation.effective_tier is None
        applied = apply_escalation_to_overrides(agent, {})
        assert "service_tier" not in applied
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        applied = apply_escalation_to_overrides(agent, {})
        assert applied.get("service_tier") == "priority"
        agent.service_tier = "flex"
        reset_escalation_for_model_switch(agent)
        assert agent._service_tier_escalation.base_tier == "flex"
        assert agent._service_tier_escalation.effective_tier == "flex"

    def test_begin_turn_explicit_none_clears_stale_flex_base(self):
        agent = _agent(service_tier="flex")
        agent.service_tier = None
        begin_escalation_turn(agent)
        assert agent._service_tier_escalation.base_tier is None
        assert "service_tier" not in apply_escalation_to_overrides(agent, {})


class TestNonOpenRouterObservation:
    def test_slow_non_openrouter_stream_does_not_mutate_state(self):
        agent = _agent(
            service_tier="flex",
            provider="openai",
            base_url="https://api.openai.com/v1",
        )
        state = agent._service_tier_escalation
        state.streak = 1
        finish_request_ttft(agent, _slow_obs())
        assert state.pending_ttft is None
        assert state.streak == 1
        assert state.effective_tier == "flex"
        accept_logical_request(agent)
        assert state.effective_tier == "flex"
        assert state.streak == 1
        assert state.pending_ttft is None


class TestRebaseEscalationRuntime:
    def test_rungs_and_streak_transfer_with_priority_cap(self):
        agent = _agent(service_tier="flex")
        state = agent._service_tier_escalation
        state.observe_ttft(12.0, model="m")
        assert state.effective_tier is None
        assert state.climbed_rungs == 1
        state.streak = 2
        begin_logical_request(agent)
        assert state.wire_locked is True
        assert state.wire_tier is None

        rebase_escalation_runtime(agent, "priority")
        assert state.base_tier == "priority"
        assert state.effective_tier == "priority"
        assert state.climbed_rungs == 1
        assert state.streak == 2
        assert state.wire_locked is False
        assert state.wire_tier == "priority"
        assert _effective_request_overrides(agent).get("service_tier") == "priority"

        rebase_escalation_runtime(agent, "flex")
        assert state.base_tier == "flex"
        assert state.effective_tier is None
        assert state.climbed_rungs == 1
        assert state.streak == 2
        assert "service_tier" not in _effective_request_overrides(agent)

    def test_none_base_rungs_zero_omits_wire_tier(self):
        agent = _agent(service_tier="flex")
        state = agent._service_tier_escalation
        rebase_escalation_runtime(agent, None)
        assert state.base_tier is None
        assert state.effective_tier is None
        assert state.climbed_rungs == 0
        applied = apply_escalation_to_overrides(agent, {"service_tier": "flex"})
        assert "service_tier" not in applied
        assert "speed" not in applied

    def test_none_base_with_rungs_climbs_to_priority(self):
        agent = _agent(service_tier="flex")
        state = agent._service_tier_escalation
        state.observe_ttft(12.0, model="m")
        assert state.climbed_rungs == 1
        rebase_escalation_runtime(agent, None)
        assert state.base_tier is None
        assert state.effective_tier == "priority"
        assert state.climbed_rungs == 1
        applied = apply_escalation_to_overrides(agent, {})
        assert applied.get("service_tier") == "priority"

    def test_two_rungs_cap_at_priority_then_restore_keeps_height(self):
        agent = _agent(service_tier="flex")
        state = agent._service_tier_escalation
        state.observe_ttft(12.0, model="m")
        state.observe_ttft(12.0, model="m")
        assert state.effective_tier == "priority"
        assert state.climbed_rungs == 2
        rebase_escalation_runtime(agent, None)
        assert state.effective_tier == "priority"
        assert state.climbed_rungs == 2
        rebase_escalation_runtime(agent, "flex")
        assert state.effective_tier == "priority"
        assert state.climbed_rungs == 2


class TestBoundedAutoColdWindows:
    def test_auto_is_default_ladder_rung_and_keeps_window_overrides(self):
        agent = _agent(service_tier="auto")
        assert agent._service_tier_escalation.base_tier is None
        applied = apply_escalation_to_overrides(
            agent, {"service_tier": "priority"}
        )
        assert applied.get("service_tier") == "priority"

    def test_auto_mode_escalation_can_still_climb_to_priority(self):
        agent = _agent(service_tier="auto")
        agent._service_tier_escalation.observe_ttft(12.0, model="m")
        assert agent._service_tier_escalation.effective_tier == "priority"
        applied = apply_escalation_to_overrides(agent, {})
        assert applied.get("service_tier") == "priority"
