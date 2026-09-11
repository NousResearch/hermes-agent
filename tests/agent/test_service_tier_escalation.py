"""Per-turn OpenRouter service-tier escalation on streaming TTFT."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.fast_mode import effective_request_overrides as _effective_request_overrides
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
    escalation_base_tier,
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


def _validation_agent():
    """Minimal agent for ``validate_tool_calls`` production-home tests."""
    agent = _agent(service_tier="flex", escalation=_enabled_cfg(consecutive=1))
    agent.valid_tool_names = {"web_search"}
    agent._uniquify_tool_call_ids = lambda tcs: tcs
    agent._repair_tool_call = lambda name: None
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._buffer_vprint = lambda *_a, **_k: None
    agent._vprint = lambda *_a, **_k: None
    agent.log_prefix = ""
    agent.quiet_mode = True
    agent._flush_status_buffer = lambda: None
    agent._persist_session = lambda *_a, **_k: None
    agent._cleanup_task_resources = lambda *_a, **_k: None
    agent._build_assistant_message = (
        lambda msg, _fr: {"role": "assistant", "content": getattr(msg, "content", ""), "tool_calls": msg.tool_calls}
    )
    return agent


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

    def test_subagent_depth_does_not_escalate(self):
        agent = _agent(delegate_depth=1, service_tier="flex")
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier == "flex"

    def test_is_subagent_flag_does_not_escalate(self):
        agent = _agent(is_subagent=True, service_tier="flex")
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
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


class TestEscalationBaseTier:
    def test_raw_flex_is_base_when_no_framework_source(self, monkeypatch):
        import hermes_cli.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "load_config_readonly",
            lambda: {"agent": {"service_tier": "", "service_tier_overrides": {}}},
        )
        agent = _agent(
            service_tier=None,
            request_overrides={"service_tier": "flex"},
            pinned=False,
        )
        assert escalation_base_tier(agent) == "flex"

    def test_framework_pin_wins_over_raw(self, monkeypatch):
        import hermes_cli.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "load_config_readonly",
            lambda: {"agent": {"service_tier": "", "service_tier_overrides": {}}},
        )
        agent = _agent(
            service_tier="priority",
            request_overrides={"service_tier": "flex"},
            pinned=True,
        )
        assert escalation_base_tier(agent) == "priority"


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


class TestServiceTierEscalationConfigContracts:
    def test_cli_defaults_match_shipped_default(self):
        import cli as cli_mod
        from hermes_constants import (
            DEFAULT_SERVICE_TIER_ESCALATION,
            resolve_service_tier_escalation_config,
        )

        cfg = resolve_service_tier_escalation_config(
            cli_mod.load_cli_config().get("agent") or {},
        )
        assert cfg == DEFAULT_SERVICE_TIER_ESCALATION
        assert cfg.enabled is False
        assert cfg.ttft_threshold_seconds > 0
        assert cfg.consecutive_slow_requests >= 1


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
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"agent": {"service_tier": "priority"}, "provider_routing": {}},
        )
        agent._block_service_tier_escalation = False
        agent._persist_disabled = False
        agent.is_subagent = False
        agent.platform = "cli"
        agent._delegate_depth = 0
        switch_model(
            agent,
            new_model="openai/gpt-5",
            new_provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )
        assert agent._service_tier_escalation.streak == 0
        # * Base is A1 request-time logical_service_tier (config overlay), not constructor.
        assert agent._service_tier_escalation.effective_tier == "priority"


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

    def test_invalid_json_retry_keeps_wire_locked_until_inject(self):
        """Production ``validate_tool_calls`` retry must not accept."""
        from agent.turn_tool_validation import validate_tool_calls

        agent = _validation_agent()
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        tc = SimpleNamespace(
            function=SimpleNamespace(name="web_search", arguments="{not json}"),
            id="c1",
        )
        assistant = SimpleNamespace(tool_calls=[tc], content="")
        first = validate_tool_calls(
            agent, assistant, "tool_calls",
            messages=[], conversation_history=[], api_call_count=1,
            effective_task_id="t1",
        )
        assert first.action == "continue"
        assert agent._invalid_json_retries == 1
        assert agent._service_tier_escalation.wire_locked is True

        second = validate_tool_calls(
            agent, assistant, "tool_calls",
            messages=[], conversation_history=[], api_call_count=2,
            effective_task_id="t1",
        )
        assert second.action == "continue"
        assert agent._invalid_json_retries == 2
        assert agent._service_tier_escalation.wire_locked is True

        messages = []
        injected = validate_tool_calls(
            agent, assistant, "tool_calls",
            messages=messages, conversation_history=[], api_call_count=3,
            effective_task_id="t1",
        )
        assert injected.action == "continue"
        assert messages, "inject path must mutate history"
        assert agent._service_tier_escalation.wire_locked is False

    def test_invalid_tool_name_inject_accepts_logical_request(self):
        """Unknown-tool inject mutates history — dest dest site ~7878."""
        from agent.turn_tool_validation import validate_tool_calls

        agent = _validation_agent()
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        tc = SimpleNamespace(
            function=SimpleNamespace(name="not_a_real_tool", arguments="{}"),
            id="c1",
        )
        assistant = SimpleNamespace(tool_calls=[tc], content="")
        messages = []
        verdict = validate_tool_calls(
            agent, assistant, "tool_calls",
            messages=messages, conversation_history=[], api_call_count=1,
            effective_task_id="t1",
        )
        assert verdict.action == "continue"
        assert messages, "invalid-name inject must mutate history"
        assert agent._service_tier_escalation.wire_locked is False

    def test_ok_tool_validation_accepts_logical_request(self):
        from agent.turn_tool_validation import validate_tool_calls

        agent = _validation_agent()
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        tc = SimpleNamespace(
            function=SimpleNamespace(name="web_search", arguments="{}"),
            id="c1",
        )
        assistant = SimpleNamespace(tool_calls=[tc], content="")
        verdict = validate_tool_calls(
            agent, assistant, "tool_calls",
            messages=[], conversation_history=[], api_call_count=1,
            effective_task_id="t1",
        )
        assert verdict.action == "ok"
        assert agent._service_tier_escalation.wire_locked is False

    def test_codex_incomplete_accepts_only_on_history_mutation(self):
        from agent.turn_truncation import continue_codex_incomplete

        def _codex_agent():
            agent = _agent(service_tier="flex", escalation=_enabled_cfg(consecutive=1))
            agent._codex_incomplete_retries = 0
            agent.quiet_mode = True
            agent.log_prefix = ""
            agent._vprint = lambda *_a, **_k: None
            agent._emit_wait_notice = lambda *_a, **_k: None
            agent._emit_interim_assistant_message = lambda *_a, **_k: None
            agent._interim_assistant_visible_text = (
                lambda msg: (msg.get("content") or "") if isinstance(msg, dict) else ""
            )
            agent._build_assistant_message = lambda msg, fr: {
                "role": "assistant",
                "content": getattr(msg, "content", "") or "",
                "finish_reason": fr,
                "reasoning": getattr(msg, "reasoning", "") or "",
            }
            agent._persist_session = lambda *_a, **_k: None
            return agent

        mutated = _codex_agent()
        begin_logical_request(mutated)
        finish_request_ttft(mutated, _slow_obs())
        messages = [{"role": "user", "content": "hi"}]
        result = continue_codex_incomplete(
            mutated,
            SimpleNamespace(content="partial thought", reasoning=""),
            "incomplete",
            messages=messages,
            conversation_history=[],
            api_call_count=1,
        )
        assert result is None
        assert any(m.get("role") == "assistant" for m in messages)
        assert mutated._service_tier_escalation.wire_locked is False

        locked = _codex_agent()
        begin_logical_request(locked)
        finish_request_ttft(locked, _slow_obs())
        dup_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "same", "finish_reason": "incomplete"},
        ]
        result = continue_codex_incomplete(
            locked,
            SimpleNamespace(content="same", reasoning=""),
            "incomplete",
            messages=dup_messages,
            conversation_history=[],
            api_call_count=1,
        )
        assert result is None
        assert locked._service_tier_escalation.wire_locked is True


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
    def test_outer_retry_keeps_auto_window_tier_after_deadline(self):
        """Same logical request keeps the first attempt's window tier after the deadline."""
        import time

        agent = _agent(service_tier="auto")
        agent._fast_until = time.monotonic() + 60.0
        begin_logical_request(agent)
        first = _effective_request_overrides(agent)
        assert first.get("service_tier") == "priority"
        agent._fast_until = time.monotonic() - 1.0
        begin_logical_request(agent)
        retry = _effective_request_overrides(agent)
        assert retry.get("service_tier") == "priority"

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


class TestEscalationPreservesRawTierUntilClimb:
    """Enabled escalation must not strip a raw user tier before any observation."""

    def test_first_request_keeps_raw_flex(self):
        agent = _agent(
            service_tier=None,
            request_overrides={"service_tier": "flex"},
        )
        begin_logical_request(agent)
        first = _effective_request_overrides(agent)
        assert first.get("service_tier") == "flex"
        assert agent.request_overrides["service_tier"] == "flex"

    def test_slow_observations_climb_from_raw_flex(self):
        agent = _agent(
            service_tier=None,
            request_overrides={"service_tier": "flex"},
            escalation=_enabled_cfg(consecutive=1),
        )
        begin_logical_request(agent)
        assert _effective_request_overrides(agent).get("service_tier") == "flex"
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier is None
        begin_logical_request(agent)
        climbed = _effective_request_overrides(agent)
        assert "service_tier" not in climbed

    def test_raw_flex_wire_snapshot_stable_across_outer_retry(self):
        """Enabled escalation keeps the raw-flex first-attempt snapshot on an outer retry."""
        agent = _agent(
            service_tier=None,
            request_overrides={"service_tier": "flex"},
            escalation=_enabled_cfg(consecutive=1),
        )
        begin_logical_request(agent)
        first = _effective_request_overrides(agent)
        assert first.get("service_tier") == "flex"
        snapshot = dict(agent._service_tier_escalation.request_wire_snapshot or {})
        assert snapshot.get("service_tier") == "flex"
        finish_request_ttft(agent, _slow_obs())
        begin_logical_request(agent)
        retry = _effective_request_overrides(agent)
        assert retry.get("service_tier") == "flex"
        assert agent._service_tier_escalation.request_wire_snapshot == snapshot
        assert agent.request_overrides["service_tier"] == "flex"


class TestDefaultOffByteIdentity:
    def test_disabled_escalation_matches_unbound_overrides(self):
        """Default OFF must not add keys or mutate canonical request_overrides."""
        unbound = SimpleNamespace(
            service_tier="flex",
            request_overrides={"keep": 1, "extra_body": {"x": 1}},
            model="google/gemini-flash",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            _service_tier_session_pinned=False,
            platform="cli",
            _persist_disabled=False,
            _delegate_depth=0,
            is_subagent=False,
        )
        left = _effective_request_overrides(unbound)
        bind_service_tier_escalation(unbound, {"enabled": False})
        right = _effective_request_overrides(unbound)
        assert left == right
        assert unbound.request_overrides == {"keep": 1, "extra_body": {"x": 1}}
        assert "ttft" not in right
        assert set(right) <= {"keep", "extra_body", "service_tier", "speed"}

    def test_enabled_false_does_not_climb_on_slow_ttft(self):
        agent = _agent(service_tier="flex", escalation={"enabled": False})
        begin_logical_request(agent)
        finish_request_ttft(agent, _slow_obs())
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert _effective_request_overrides(agent).get("service_tier") == "flex"


class TestEscalationConfigResolver:
    def test_missing_section_is_disabled(self):
        from hermes_constants import (
            DEFAULT_SERVICE_TIER_ESCALATION,
            resolve_service_tier_escalation_config,
        )

        cfg = resolve_service_tier_escalation_config({})
        assert cfg == DEFAULT_SERVICE_TIER_ESCALATION
        assert cfg.enabled is False
        assert cfg.ttft_threshold_seconds > 0
        assert cfg.consecutive_slow_requests >= 1

    def test_valid_section_is_parsed(self):
        from hermes_constants import resolve_service_tier_escalation_config

        cfg = resolve_service_tier_escalation_config(
            {
                "service_tier_escalation": {
                    "enabled": True,
                    "ttft_threshold_seconds": 2.5,
                    "consecutive_slow_requests": 3,
                }
            }
        )
        assert cfg.enabled is True
        assert cfg.ttft_threshold_seconds == 2.5
        assert cfg.consecutive_slow_requests == 3

    def test_invalid_values_fall_back_and_warn(self, caplog):
        import logging

        from hermes_constants import (
            DEFAULT_SERVICE_TIER_ESCALATION,
            resolve_service_tier_escalation_config,
        )

        with caplog.at_level(logging.WARNING):
            cfg = resolve_service_tier_escalation_config(
                {
                    "service_tier_escalation": {
                        "enabled": "maybe",
                        "ttft_threshold_seconds": -1,
                        "consecutive_slow_requests": 0,
                    }
                }
            )
        assert cfg == DEFAULT_SERVICE_TIER_ESCALATION
        assert "service_tier_escalation" in caplog.text


class TestNonStreamingNoObservation:
    def test_missing_first_delta_is_not_an_observation(self):
        agent = _agent(service_tier="flex")
        begin_logical_request(agent)
        obs = TtftObservation(clock=lambda: 0.0)
        obs.t_send = 0.0
        obs.open_count = 1
        finish_request_ttft(agent, obs)
        accept_logical_request(agent)
        assert agent._service_tier_escalation.effective_tier == "flex"
        assert agent._service_tier_escalation.pending_ttft is None
        assert agent._service_tier_escalation.streak == 0


def _write_enabled_escalation_config():
    """Put ``agent.service_tier_escalation.enabled: true`` in the isolated home."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    (home / "config.yaml").write_text(
        "agent:\n"
        "  service_tier_escalation:\n"
        "    enabled: true\n"
        "    ttft_threshold_seconds: 8.0\n"
        "    consecutive_slow_requests: 1\n"
        "model:\n"
        "  default: google/gemini-flash\n",
        encoding="utf-8",
    )
    from hermes_cli import config as cfg_mod

    invalidate = getattr(cfg_mod, "_invalidate_load_config_cache", None)
    if callable(invalidate):
        invalidate()


class TestConstructionGates:
    """Real constructors stay inactive even when config enables the ladder."""

    def test_cron_scheduler_construction_blocks_escalation(self):
        from agent.service_tier_escalation import escalation_is_active
        from cron.scheduler import _CronAgentSetup, _construct_cron_agent
        from run_agent import AIAgent

        _write_enabled_escalation_config()
        setup = _CronAgentSetup(
            model="google/gemini-flash",
            runtime={
                "api_key": "k",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "requested_provider": "openrouter",
                "api_mode": "chat_completions",
                "request_overrides": {},
            },
            max_iterations=2,
        )
        agent = _construct_cron_agent(
            AIAgent,
            {"id": "job-gate", "enabled_toolsets": ["file"]},
            {"agent": {"service_tier_escalation": {"enabled": True}}},
            setup,
            workdir=None,
            session_id="cron-gate",
            session_db=None,
        )
        try:
            assert agent.platform == "cron"
            assert agent._block_service_tier_escalation is True
            assert agent._service_tier_escalation.enabled is True
            assert escalation_is_active(agent) is False
        finally:
            agent.close()

    def test_batch_runner_construction_blocks_escalation(self, monkeypatch):
        from agent.service_tier_escalation import escalation_is_active
        import batch_runner

        _write_enabled_escalation_config()
        captured = {}
        real_agent = batch_runner.AIAgent

        def _capturing_agent(*args, **kwargs):
            agent = real_agent(*args, **kwargs)
            captured["agent"] = agent

            def _fake_run(*_a, **_k):
                return {
                    "messages": [],
                    "completed": True,
                    "partial": False,
                    "api_calls": 0,
                }

            agent.run_conversation = _fake_run
            return agent

        monkeypatch.setattr(batch_runner, "AIAgent", _capturing_agent)
        batch_runner._process_single_prompt(
            0,
            {"prompt": "hi"},
            1,
            {
                "model": "google/gemini-flash",
                "max_iterations": 2,
                "distribution": "minimal",
                "verbose": False,
                "api_key": "k",
                "base_url": "https://openrouter.ai/api/v1",
            },
        )
        agent = captured["agent"]
        try:
            assert agent._block_service_tier_escalation is True
            assert escalation_is_active(agent) is False
        finally:
            agent.close()

    def test_delegate_child_construction_blocks_escalation(self, monkeypatch):
        from agent.service_tier_escalation import escalation_is_active
        from run_agent import AIAgent
        from tools import delegate_tool as dt
        import tools.delegate_tool_config as dtc

        _write_enabled_escalation_config()
        monkeypatch.setattr(dt, "_load_config", lambda: {})
        monkeypatch.setattr(dtc, "_load_config", lambda: {})
        kw = dict(
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            model="google/gemini-flash",
            platform="cli",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            enabled_toolsets=["file"],
        )
        parent = AIAgent(session_id="p-gate", **kw)
        child = None
        try:
            assert parent._service_tier_escalation.enabled is True
            assert escalation_is_active(parent) is True
            child = dt._build_child_agent(
                task_index=0,
                goal="goal",
                context=None,
                toolsets=["file"],
                model=None,
                max_iterations=4,
                task_count=1,
                parent_agent=parent,
            )
            assert child.platform == "subagent"
            assert child._delegate_depth > 0
            assert child._service_tier_escalation.enabled is True
            assert escalation_is_active(child) is False
        finally:
            if child is not None:
                child.close()
            parent.close()


    def test_curator_construction_blocks_escalation(self, monkeypatch):
        from agent.service_tier_escalation import escalation_is_active, finish_request_ttft
        from run_agent import AIAgent

        _write_enabled_escalation_config()
        captured = {}
        real_agent = AIAgent

        def _capturing_agent(*args, **kwargs):
            agent = real_agent(*args, **kwargs)
            captured["agent"] = agent

            def _fake_run(*_a, **_k):
                return {"final_response": "ok", "messages": []}

            agent.run_conversation = _fake_run
            return agent

        monkeypatch.setattr("run_agent.AIAgent", _capturing_agent)
        monkeypatch.setattr(
            "agent.curator._resolve_review_provider",
            lambda: (
                {
                    "api_key": "k",
                    "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                },
                "google/gemini-flash",
                "openrouter",
                {},
            ),
        )
        from agent.curator import _run_llm_review

        _run_llm_review("review skills")
        agent = captured["agent"]
        try:
            assert agent.platform == "curator"
            assert agent._block_service_tier_escalation is True
            assert agent._service_tier_escalation.enabled is True
            assert escalation_is_active(agent) is False
            begin_logical_request(agent)
            finish_request_ttft(agent, _slow_obs())
            accept_logical_request(agent)
            assert agent._service_tier_escalation.effective_tier is None
            assert _effective_request_overrides(agent).get("service_tier") != "priority"
        finally:
            agent.close()


class TestEscalationHookHardening:
    def test_accept_hook_skips_when_disabled(self):
        from agent.conversation_loop import _try_accept_logical_request

        agent = _agent(escalation={"enabled": False})
        with patch("agent.service_tier_escalation.accept_logical_request") as accept:
            _try_accept_logical_request(agent)
        accept.assert_not_called()

    def test_accept_hook_logs_failure_without_raising(self, caplog):
        import logging

        from agent.conversation_loop import _try_accept_logical_request

        agent = _agent()
        with patch(
            "agent.service_tier_escalation.accept_logical_request",
            side_effect=RuntimeError("boom"),
        ):
            with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
                _try_accept_logical_request(agent)
        assert "accept_logical_request failed" in caplog.text
        assert "boom" in caplog.text

    def test_streaming_call_skips_marks_when_stack_empty(self):
        from agent.chat_completion_helpers import _StreamingCall

        agent = _agent(escalation={"enabled": False})
        call = _StreamingCall(agent, {}, None)
        assert call._mark_ttft_send is None
        assert call._mark_ttft_first_delta is None
        with patch("agent.service_tier_escalation.mark_ttft_first_delta") as mark:
            call._fire_first_delta()
        mark.assert_not_called()

    def test_streaming_call_binds_marks_when_obs_stacked(self):
        from agent.chat_completion_helpers import _StreamingCall

        agent = _agent()
        obs = begin_request_ttft(agent, clock=lambda: 1.0)
        call = _StreamingCall(agent, {}, None)
        assert call._mark_ttft_send is not None
        assert call._mark_ttft_first_delta is not None
        call._fire_first_delta()
        assert obs.t_first == 1.0
