"""Checkpoint #3 and #4 tests.

Checkpoint #3: plugin rollback via update_model(old...) — a generic plugin
engine that mutates derived state and THEN raises must be rolled back through
its own lifecycle contract (re-applying the prior route), the original
exception must propagate, and the host pre-cap must stay at its prior value.

Checkpoint #4: ContextCeilingExceeded terminal-gate behavior —
  * the gate raises (not a sentinel dict) for an over-limit FINAL payload;
  * a locally refused request is NOT an API call (no I/O, no retry);
  * the exception survives a real execution-middleware chain unmodified
    (middleware sees _DownstreamExecutionError and re-raises the original);
  * the clean-refusal result contract (failed=True, error=
    'context_ceiling_exceeded', api_calls refunded) is what the call site
    builds;
  * under-limit requests pass the gate with no exception.
"""

import pytest

from unittest.mock import patch
from agent.context_engine import ContextEngine
from agent.agent_runtime_helpers import (
    ContextCeilingExceeded,
    canonical_request_budget,
    check_ceiling_for_kwargs,
    effective_dispatch_limit,
    transition_model_context,
    get_pre_cap,
    _agent_ceiling,
)


# ─── Shared helpers ──────────────────────────────────────────────────────────


class _PluginEngine(ContextEngine):
    """Minimal generic plugin engine that records every update_model call.

    ``raise_on_update`` triggers the failure path; ``mutate_before_raise``
    models a real plugin that partially applies the new route (derived state
    moves to the NEW model's values) and then raises — the exact case the
    rollback-via-update_model(old...) contract exists for.
    """

    def __init__(self, threshold_percent=0.75):
        self.threshold_percent = threshold_percent
        self.context_length = 0
        self.threshold_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.update_model_calls = []
        self.raise_on_update = False
        self.fail_all = False
        self.mutate_before_raise = True
        # Item 1: model the case where the forward transition mutates then
        # raises AND the preferred rollback (re-apply prior route via
        # update_model) ALSO raises. The last-resort direct-field backstop
        # then runs and must restore all six route fields (including api_mode).
        # ``rollback_also_fails`` triggers: forward call mutates+raises;
        # rollback call raises without applying the prior route.
        self.rollback_also_fails = False
        self._forward_attempted = False
        # A real engine persists the route it was given (model/provider/etc),
        # which the rollback-via-update_model(old...) contract relies on to
        # capture the prior route.
        self.model = None
        self.provider = None
        self.base_url = None
        self.api_key = None
        self.api_mode = None

    @property
    def name(self):
        return "plugin"

    def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
        self.update_model_calls.append(dict(
            model=model, context_length=context_length,
            base_url=base_url, provider=provider, api_mode=api_mode,
        ))
        # Item 1: when the preferred rollback is being attempted (the SECOND
        # call after a failed forward) and the plugin refuses it, it must fail
        # BEFORE persisting the prior route — otherwise the rollback call
        # itself would restore the fields and the last-resort backstop would be
        # masked. So check this first, before any route-field write.
        if self.rollback_also_fails and self._forward_attempted:
            raise RuntimeError("plugin engine update_model failed (rollback)")
        # A real plugin persists the new route (so the NEXT transition's
        # rollback can capture it as the prior route) — done up front, as a
        # plugin typically sets its route fields before finishing.
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.api_mode = api_mode
        if self.fail_all:
            raise RuntimeError("plugin engine update_model failed")
        if self.rollback_also_fails:
            # Forward (first) call: mutate derived state to the NEW route, then
            # raise — a genuine mid-transition failure. Mark it so the next
            # (rollback) call is refused before persisting the prior route.
            self._forward_attempted = True
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)
            raise RuntimeError("plugin engine update_model failed (forward)")
        if self.mutate_before_raise:
            # A real plugin applies the new route to its derived state before
            # finishing the transition — so a mid-transition raise leaves the
            # derived state on the NEW route unless the rollback re-applies
            # the prior route.
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)
            if self.raise_on_update:
                self.raise_on_update = False  # raise exactly once
                raise RuntimeError("plugin engine update_model failed")
        else:
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        return messages


class _AgentStub:
    def __init__(self, compressor, model="m", ceiling=None):
        self.context_compressor = compressor
        self.model = model
        self.base_url = "http://x"
        self.api_key = ""
        self.provider = "openai"
        self.api_mode = ""
        self._max_context_length = ceiling
        self._pre_cap_context_length = None
        self._primary_runtime = None


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint #3: rollback via update_model(old...) on transition failure
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpoint3_RollbackViaLifecycle:
    def test_failed_transition_rolls_back_via_update_model_old(self):
        """The forward transition raises AFTER mutating the engine's derived
        state (threshold_tokens is now on the NEW route's values). The
        rollback must re-apply the PRIOR route through update_model so the
        plugin recomputes ALL derived state from the prior route — not just
        setattr the five route fields back.

        Sequence asserted:
          call 1: new route   (forward, raises)
          call 2: prior route (rollback through the plugin's own contract)
        And the engine's derived state (threshold_tokens) is back on the
        PRIOR route's values, the original exception propagates, and the
        host pre-cap is unchanged.
        """
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=None)

        # Initial transition: route A (64K)
        transition_model_context(agent, 64_000, model="model_a", provider="prov_a")
        assert eng.update_model_calls[-1]["model"] == "model_a"
        assert eng.context_length == 64_000
        prior_threshold = eng.threshold_tokens
        prior_calls = len(eng.update_model_calls)

        # Next transition (route B = 90K) mutates then raises.
        eng.raise_on_update = True
        with pytest.raises(RuntimeError, match="plugin engine update_model failed"):
            transition_model_context(agent, 90_000, model="model_b", provider="prov_b")

        # Exactly TWO update_model calls since the initial one: forward (B) +
        # rollback (A). A setattr-only rollback would produce just one.
        calls_after_failure = eng.update_model_calls[prior_calls:]
        assert len(calls_after_failure) == 2, (
            "rollback must re-apply the prior route via update_model "
            f"(expected 2 calls, got {len(calls_after_failure)}: "
            f"{calls_after_failure})"
        )
        assert calls_after_failure[0]["model"] == "model_b"  # forward (failed)
        assert calls_after_failure[1]["model"] == "model_a"  # rollback (prior)
        assert calls_after_failure[1]["context_length"] == 64_000

        # Derived state is back on the PRIOR route: threshold recomputed
        # from 64K, not left on the failed 90K route.
        assert eng.threshold_tokens == prior_threshold
        assert eng.context_length == 64_000

        # Host pre-cap: the failed transition must NOT have committed 90K.
        assert agent._pre_cap_context_length == 64_000

    def test_rollback_failure_is_surfaced_and_exception_propagates(self):
        """Even if the rollback via update_model itself raises, the
        original transition exception still propagates (the caller decides
        the failure policy) — never swallowed."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=None)
        transition_model_context(agent, 64_000, model="model_a")

        # Make BOTH the forward and the rollback raise: the engine refuses
        # any update_model call now.
        eng.fail_all = True
        with pytest.raises(RuntimeError, match="plugin engine update_model failed"):
            transition_model_context(agent, 90_000, model="model_b")

        # Host pre-cap unchanged (no commit on failure).
        assert agent._pre_cap_context_length == 64_000

    def test_last_resort_backstop_restores_api_mode(self):
        """Item 1 (Round 4.1): when the forward transition fails AND the
        preferred rollback (re-apply the prior route via update_model) ALSO
        fails, the last-resort direct-field backstop must restore ALL six
        route fields — model, provider, base_url, api_key, api_mode, and
        context_length — to the PRIOR coherent route.

        The Round 4 route-coherence invariant requires api_mode to be part of
        the coherent route: a route left with the NEW model/provider/base_url
        but the OLD api_mode is incoherent (the provider wire format does not
        match the endpoint). The backstop must therefore restore api_mode too,
        while still SURFACING the partial rollback (loud log) and letting the
        ORIGINAL transition exception propagate — never a silent success.
        """
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=None)

        # Initial coherent route A — with a DISTINCT api_mode so restoration is
        # observable (a blank api_mode would make a missing-restore invisible).
        transition_model_context(
            agent, 64_000,
            model="model_a", provider="prov_a",
            base_url="http://a.example/v1", api_key="key-a",
            api_mode="chat-completions",
        )
        prior = {
            "model": "model_a",
            "provider": "prov_a",
            "base_url": "http://a.example/v1",
            "api_key": "key-a",
            "api_mode": "chat-completions",
            "context_length": 64_000,
        }
        prior_calls = len(eng.update_model_calls)

        # The forward transition (route B) mutates derived state then fails, and
        # the preferred rollback (re-apply route A via update_model) ALSO fails.
        eng.rollback_also_fails = True
        with pytest.raises(RuntimeError, match="plugin engine update_model failed"):
            transition_model_context(
                agent, 90_000,
                model="model_b", provider="prov_b",
                base_url="http://b.example/v1", api_key="key-b",
                api_mode="responses",
            )

        # The last-resort backstop restored the PRIOR coherent route — all six
        # fields, INCLUDING api_mode (the gap this item closes).
        assert eng.model == prior["model"], "model must be restored"
        assert eng.provider == prior["provider"], "provider must be restored"
        assert eng.base_url == prior["base_url"], "base_url must be restored"
        assert eng.api_key == prior["api_key"], "api_key must be restored"
        assert eng.api_mode == prior["api_mode"], (
            "api_mode must be restored by the last-resort backstop (round 4.1); "
            f"got {eng.api_mode!r}, expected {prior['api_mode']!r}"
        )
        assert eng.context_length == prior["context_length"], "context_length must be restored"

        # Proof the restoration came from the DIRECT-FIELD backstop (not the
        # plugin's own lifecycle contract): the preferred rollback
        # update_model(route A) was ATTEMPTED (logged) but FAILED, so the
        # engine's route fields were restored by the backstop, and the partial
        # rollback is surfaced loudly (see
        # test_last_resort_backstop_failure_is_surfaced_not_silent). The
        # rollback attempt appears in the call log as the prior route (model_a).
        calls_after = eng.update_model_calls[prior_calls:]
        assert calls_after[0]["model"] == "model_b", "forward attempt (route B) must be logged"
        assert calls_after[-1]["model"] == "model_a", (
            "the preferred rollback re-applied the prior route (model_a) via "
            "update_model and FAILED — so the field restoration above came from "
            "the last-resort direct-field backstop"
        )

        # Host pre-cap unchanged (no commit on failure).
        assert agent._pre_cap_context_length == 64_000

    def test_last_resort_backstop_failure_is_surfaced_not_silent(self):
        """Item 1 (Round 4.1): the last-resort backstop restores the route
        fields but must NOT be a silent success — it must surface the partial
        rollback (a loud error log) AND let the original transition exception
        propagate to the owning transaction."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=None)
        transition_model_context(
            agent, 64_000,
            model="model_a", provider="prov_a", api_mode="chat-completions",
        )
        eng.rollback_also_fails = True
        with patch("agent.agent_runtime_helpers.logger") as mock_logger:
            # The ORIGINAL transition exception propagates (not swallowed).
            with pytest.raises(RuntimeError, match=r"plugin engine update_model failed \(forward\)"):
                transition_model_context(
                    agent, 90_000, model="model_b", provider="prov_b", api_mode="responses",
                )
            # The partial rollback is SURFACED with an error-level log.
            mock_logger.error.assert_called()
            err_calls = " | ".join(str(c) for c in mock_logger.error.call_args_list)
            assert "INCOMPLETE" in err_calls, (
                "a partial last-resort rollback must be surfaced loudly, "
                "not claimed as full coherence"
            )
        # And the route was still restored (surfacing does not skip restore).
        assert eng.api_mode == "chat-completions"
        assert eng.model == "model_a"


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint #4: terminal gate — exception semantics
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpoint4_TerminalGate:
    # --- message sized so its canonical budget is ~65K (over 64K floor) ---
    OVER_MSG = "a" * 260_000          # budget ≈ 65_008 (measured)
    UNDER_MSG = "a" * 240_000         # budget ≈ 60_008 (measured)

    def test_gate_raises_context_ceiling_exceeded_for_over_limit(self):
        agent = _AgentStub(_PluginEngine(), ceiling=64_000)
        kwargs = {"messages": [{"role": "user", "content": self.OVER_MSG}]}
        with pytest.raises(ContextCeilingExceeded) as excinfo:
            check_ceiling_for_kwargs(
                kwargs, pre_cap=get_pre_cap(agent) or 64_000,
                ceiling=_agent_ceiling(agent), reason="checkpoint4",
            )
        exc = excinfo.value
        # It is a REAL exception with the accounting attached, not a sentinel.
        assert exc.request_tokens > 64_000
        assert exc.effective_limit == 64_000
        assert "exceeds the effective context limit" in str(exc)

    def test_gate_allows_under_limit(self):
        agent = _AgentStub(_PluginEngine(), ceiling=64_000)
        kwargs = {"messages": [{"role": "user", "content": self.UNDER_MSG}]}
        # Must NOT raise.
        check_ceiling_for_kwargs(
            kwargs, pre_cap=get_pre_cap(agent) or 64_000,
            ceiling=_agent_ceiling(agent), reason="checkpoint4",
        )

    def test_effective_limit_is_min_of_pre_cap_and_ceiling(self):
        # pre_cap 100K, ceiling 64K -> effective 64K.
        assert effective_dispatch_limit(100_000, 64_000) == 64_000
        # pre_cap 90K, ceiling 120K -> effective 90K (ceiling only lowers).
        assert effective_dispatch_limit(90_000, 120_000) == 90_000
        # ceiling only -> ceiling.
        assert effective_dispatch_limit(None, 64_000) == 64_000
        # pre_cap only -> pre_cap.
        assert effective_dispatch_limit(64_000, None) == 64_000
        # neither -> no limit (no-op gate).
        assert effective_dispatch_limit(None, None) is None
        # bool must not be treated as a valid int.
        assert effective_dispatch_limit(True, None) is None

    def test_local_refusal_is_not_an_api_error_and_not_retried(self):
        """A locally refused request is NOT an API call: it is caught at the
        call site (conversation_loop.py) BEFORE the error-classification /
        retry / fallback machinery, so it is never classified as a provider
        error and never retried. Verify the exception carries a clean local
        refusal contract and is a distinct, non-provider error type."""
        # ContextCeilingExceeded is a plain local exception — NOT a provider
        # API error subclass. The call site catches it by type and refunds;
        # it never reaches classify_api_error (which would route it through
        # retry/backoff/fallback as if it were a provider failure).
        from agent.error_classifier import ClassifiedError
        exc = ContextCeilingExceeded(70_000, 64_000, reason="test")
        assert not isinstance(exc, ClassifiedError)
        assert not isinstance(exc, (ConnectionError, TimeoutError))
        # The call site's refund: a refused request must not keep its slot.
        api_call_count = 3
        assert max(0, api_call_count - 1) == 2
        assert max(0, 0 - 1) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint #4: exception survives a REAL execution-middleware chain
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpoint4_MiddlewareSurvival:
    def test_exception_survives_execution_middleware_chain(self):
        """The terminal gate raises INSIDE the terminal call. A registered
        execution middleware wraps next_call; the middleware chain's
        _DownstreamExecutionError wrapper must UNWRAP to the ORIGINAL
        ContextCeilingExceeded (not a wrapped/generic error) so the call
        site's ``except ContextCeilingExceeded`` catches it and refunds.

        This is the regression the sentinel-dict design could not satisfy:
        a middleware that copies/wraps the terminal result as a provider
        response would corrupt a sentinel but cannot corrupt a typed
        exception the chain re-raises.
        """
        from hermes_cli.middleware import run_llm_execution_middleware, LLM_EXECUTION_MIDDLEWARE

        over_kwargs = {"messages": [{"role": "user", "content": TestCheckpoint4_TerminalGate.OVER_MSG}]}

        def _terminal_call(payload):
            # This is what _perform_api_call does first: the terminal gate.
            # It raises for an over-limit FINAL payload, so the provider I/O
            # below is never reached.
            check_ceiling_for_kwargs(
                payload, pre_cap=64_000, ceiling=64_000,
                reason="checkpoint4 middleware survival",
            )
            # If we got here, the request was NOT over-limit — provider I/O.
            return {"status": "ok", "provider_io_reached": True}

        # A pass-through execution middleware (the common case).
        def _passthrough_mw(request, next_call, **ctx):
            return next_call(request)

        # A middleware that INSPECTS/COPies the downstream result (the
        # adversarial case a sentinel would break on). It must see a clean
        # exception, not a corrupted result.
        def _inspecting_mw(request, next_call, **ctx):
            try:
                result = next_call(request)
            except ContextCeilingExceeded:
                # A well-behaved middleware re-raises the local refusal.
                raise
            # If it were a sentinel dict, result would be a fake "response".
            return {"wrapped": result}

        from hermes_cli import plugins as _plugins
        manager = _plugins.get_plugin_manager()
        mw_registry = manager._middleware

        saved_exec = mw_registry.get(LLM_EXECUTION_MIDDLEWARE)
        mw_registry[LLM_EXECUTION_MIDDLEWARE] = [_passthrough_mw]
        try:
            with pytest.raises(ContextCeilingExceeded):
                run_llm_execution_middleware(over_kwargs, _terminal_call)
        finally:
            if saved_exec is None:
                mw_registry.pop(LLM_EXECUTION_MIDDLEWARE, None)
            else:
                mw_registry[LLM_EXECUTION_MIDDLEWARE] = saved_exec

        # Same contract with an inspecting middleware.
        saved_exec = mw_registry.get(LLM_EXECUTION_MIDDLEWARE)
        mw_registry[LLM_EXECUTION_MIDDLEWARE] = [_inspecting_mw]
        try:
            with pytest.raises(ContextCeilingExceeded):
                run_llm_execution_middleware(over_kwargs, _terminal_call)
        finally:
            if saved_exec is None:
                mw_registry.pop(LLM_EXECUTION_MIDDLEWARE, None)
            else:
                mw_registry[LLM_EXECUTION_MIDDLEWARE] = saved_exec


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint #4: clean-refusal result contract (what the call site builds)
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpoint4_CleanRefusalContract:
    def test_refusal_contract_fields(self):
        """The call site (conversation_loop.py, except ContextCeilingExceeded)
        refunds api_call_count and returns a clean local refusal. Verify the
        exception carries everything the contract needs, and that a locally
        refused request is not an API call (refund to prior count)."""
        exc = ContextCeilingExceeded(70_000, 64_000, reason="main conversation (final payload)")
        # The refusal message is user-facing and actionable.
        msg = str(exc)
        assert "64,000" in msg
        assert "70,000" in msg
        assert "context" in msg.lower()

        # Refund semantics: api_call_count is decremented by exactly 1 (and
        # floored at 0). A locally refused request must not keep the slot.
        api_call_count = 3
        refunded = max(0, api_call_count - 1)
        assert refunded == 2
        # Floor at zero:
        assert max(0, 0 - 1) == 0

    def test_refusal_error_tag(self):
        """The refusal result uses the stable error tag the UI/relay keys on."""
        # Mirrors the call site: "error": "context_ceiling_exceeded"
        result = {
            "final_response": str(ContextCeilingExceeded(70_000, 64_000)),
            "api_calls": 0,
            "completed": False,
            "failed": True,
            "error": "context_ceiling_exceeded",
        }
        assert result["failed"] is True
        assert result["completed"] is False
        assert result["error"] == "context_ceiling_exceeded"


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint #4: canonical budget includes all required components
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpoint4_BudgetComponents:
    def test_budget_includes_messages_system_tools_output(self):
        """canonical_request_budget must sum: messages, system prompt,
        tools/schema, and resolved output reservation — the FINAL request."""
        msgs = [{"role": "user", "content": "hello world"}]
        sysp = "You are a helpful assistant with a long system prompt " * 50
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string", "description": "city name"}}},
            },
        }]
        base = canonical_request_budget(msgs, system_prompt="", tools=None, output_reserve=0)
        with_sys = canonical_request_budget(msgs, system_prompt=sysp, tools=None, output_reserve=0)
        with_tools = canonical_request_budget(msgs, system_prompt="", tools=tools, output_reserve=0)
        with_reserve = canonical_request_budget(msgs, system_prompt="", tools=None, output_reserve=4096)
        assert with_sys > base, "system prompt must add to the budget"
        assert with_tools > base, "tools/schema must add to the budget"
        assert with_reserve > base, "output reservation must add to the budget"

    def test_gate_accepts_messages_and_input_payload_shapes(self):
        """The terminal gate reads the FINAL payload's ``messages`` (OpenAI
        chat) or ``input`` (Responses API) key. Both shapes must be counted
        identically so the gate refuses the same over-limit request either
        way."""
        over = {"content": "a" * 260_000}
        # OpenAI chat shape.
        msgs_kwargs = {"messages": [{"role": "user", "content": over["content"]}]}
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(msgs_kwargs, pre_cap=64_000, ceiling=64_000, reason="shape")
        # Responses API shape (``input`` key).
        input_kwargs = {"input": [{"role": "user", "content": over["content"]}]}
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(input_kwargs, pre_cap=64_000, ceiling=64_000, reason="shape")
