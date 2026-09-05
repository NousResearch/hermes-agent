"""Unit tests for pre_llm_call runtime_override (issue #23739)."""

from __future__ import annotations

import pytest

from agent.runtime_override import (
    RUNTIME_OVERRIDE_KEYS,
    apply_runtime_override,
    validate_runtime_override,
)


# ---------------------------------------------------------------------------
# validate_runtime_override
# ---------------------------------------------------------------------------

class TestValidate:
    def test_full_valid_dict(self):
        ro = validate_runtime_override({"model": "gpt-5.6"})
        assert ro == {"model": "gpt-5.6"}

    def test_system_prompt_rejected(self):
        # system_prompt is intentionally NOT supported (cache-prefix sacred):
        # it must be dropped, not applied.
        ro = validate_runtime_override({
            "model": "gpt-5.6",
            "system_prompt": "You are a test.",
        })
        assert ro == {"model": "gpt-5.6"}

    def test_empty_dict(self):
        assert validate_runtime_override({}) == {}

    def test_not_a_dict(self):
        # Non-dict runtime_override (e.g. 42) -> warning + {} (never crash).
        assert validate_runtime_override(42) == {}

    def test_unsupported_key_ignored(self):
        ro = validate_runtime_override({"model": "m", "temperature": 0.7})
        assert ro == {"model": "m"}

    def test_api_key_and_base_url_rejected(self):
        # The narrowed contract (see agent/runtime_override.py module docstring)
        # removed api_key/base_url: credentials never flow through the hook
        # return and a plugin cannot pick the endpoint.  They must be dropped
        # like any other unsupported key, never applied.
        ro = validate_runtime_override({
            "model": "m",
            "api_key": "sk-test",
            "base_url": "https://api.example.com/v1",
        })
        assert ro == {"model": "m"}

    def test_invalid_value_type_ignored(self):
        ro = validate_runtime_override({"model": 12345})
        assert ro == {}

    def test_empty_string_ignored(self):
        ro = validate_runtime_override({"model": ""})
        assert ro == {}

    def test_whitelist_matches_spec(self):
        assert RUNTIME_OVERRIDE_KEYS == frozenset({"model"})


# ---------------------------------------------------------------------------
# apply_runtime_override (context manager snapshot/restore)
# ---------------------------------------------------------------------------

class _FakeAgent:
    """Minimal stand-in for AIAgent with the attributes the override touches."""

    def __init__(self):
        self.model = "orig-model"
        self.provider = "orig-provider"
        self.api_mode = "chat_completions"
        self._client_kwargs = {"service_tier": "standard"}
        self._is_anthropic_oauth = False
        self.requested_provider = "orig-provider"
        self.request_overrides = {"service_tier": "standard"}
        self.runtime_capabilities = {"native_compaction": False}
        self._transport_cache = {"chat_completions": "warmed-transport"}
        self._fallback_activated = False


class TestApply:
    def test_apply_and_restore(self):
        agent = _FakeAgent()
        with apply_runtime_override(agent, {"model": "new-model"}):
            assert agent.model == "new-model"
        # Restored on exit.
        assert agent.model == "orig-model"

    def test_restore_on_exception(self):
        agent = _FakeAgent()
        with pytest.raises(RuntimeError):
            with apply_runtime_override(agent, {"model": "new-model"}):
                assert agent.model == "new-model"
                raise RuntimeError("boom")
        assert agent.model == "orig-model"

    def test_bare_agent_not_polluted(self):
        # Agent created via __new__ has NO attributes; entering the scope must
        # not manufacture attributes on the agent that survive the exit.
        agent = object.__new__(_FakeAgent)
        with apply_runtime_override(agent, {"model": "m"}):
            assert agent.model == "m"
        assert not hasattr(agent, "model")
        assert not hasattr(agent, "_client_kwargs")

    def test_partial_override_only_changes_given_keys(self):
        agent = _FakeAgent()
        with apply_runtime_override(agent, {"model": "only-model"}):
            assert agent.model == "only-model"
            assert agent.provider == "orig-provider"  # untouched
            assert agent.api_mode == "chat_completions"  # untouched

    def test_empty_value_key_is_dropped_not_applied(self):
        # Direct callers may skip validate_runtime_override; an empty value is
        # warned-and-ignored and must not be applied (the rejected key is
        # dropped from the override in place).
        agent = _FakeAgent()
        overrides = {"model": "   "}
        with apply_runtime_override(agent, overrides):
            assert agent.model == "orig-model"  # empty model ignored
        assert agent.model == "orig-model"
        assert "model" not in overrides  # rejected key dropped in place

    def test_route_change_refreshes_and_restores_derived_state(self):
        # P1-3: a model change refreshes the route-derived state
        # (runtime_capabilities) like the canonical switch, and restores
        # atomically on exit.
        agent = _FakeAgent()
        with apply_runtime_override(agent, {"model": "new-model"}):
            assert agent.model == "new-model"
            assert agent.runtime_capabilities is not None
        assert agent.model == "orig-model"
        assert agent.runtime_capabilities == {"native_compaction": False}

    def test_fallback_supersession_skips_the_restore(self):
        # P1-2 precedence: a proactive override owns the primary attempt; once
        # _try_activate_fallback succeeds mid-scope, the fallback route
        # supersedes the override and the scope must NOT restore the
        # pre-override identity over the fallback.  Supersession is the
        # EXPLICIT consume_runtime_override handoff, never an inference.
        from agent.runtime_override import consume_runtime_override

        agent = _FakeAgent()
        with apply_runtime_override(agent, {"model": "override-model"}):
            # Simulate try_activate_fallback taking ownership of the route,
            # then the fallback call site performing the supersede handoff.
            agent.model = "fallback-model"
            agent.provider = "fallback-provider"
            agent._fallback_activated = True
            consume_runtime_override(agent)
        # The fallback state stands; the pre-override identity is NOT restored.
        assert agent.model == "fallback-model"
        assert agent.provider == "fallback-provider"
        assert agent._fallback_activated is True
        # The handoff also cleared the turn-scoped override.
        assert agent._runtime_override == {}

    def test_supersession_requires_the_explicit_handoff(self):
        # A route change alone (no consume_runtime_override handoff) must NOT
        # be mistaken for supersession — the scope restores normally.
        agent = _FakeAgent()
        with apply_runtime_override(agent, {"model": "override-model"}):
            agent.model = "changed-by-something-else"
            agent._fallback_activated = True
        assert agent.model == "orig-model"

    def test_scope_registers_and_unregisters_itself(self):
        agent = _FakeAgent()
        assert getattr(agent, "_active_runtime_override_scope", None) is None
        with apply_runtime_override(agent, {"model": "m"}):
            assert agent._active_runtime_override_scope is not None
        assert getattr(agent, "_active_runtime_override_scope", None) is None

    def test_nested_scope_outer_wins_registration(self):
        # Scope 2 (wire-time safety net) is created inside Scope 1; it must
        # not steal the registration, or the fallback handoff would find the
        # inner scope after it already exited and miss superseding Scope 1.
        from agent.runtime_override import consume_runtime_override

        agent = _FakeAgent()
        with apply_runtime_override(agent, {"model": "override-model"}):
            outer = agent._active_runtime_override_scope
            with apply_runtime_override(agent, {"model": "inner-model"}):
                # Inner scope did not steal the registration.
                assert agent._active_runtime_override_scope is outer
            # Inner scope exiting did not clear the outer registration either.
            assert agent._active_runtime_override_scope is outer
            agent.model = "fallback-model"
            consume_runtime_override(agent)
        assert agent.model == "fallback-model"  # outer scope was superseded
        assert getattr(agent, "_active_runtime_override_scope", None) is None

    def test_consume_runtime_override_clears_the_turn_override(self):
        from agent.runtime_override import consume_runtime_override

        agent = _FakeAgent()
        agent._runtime_override = {"model": "m"}
        consume_runtime_override(agent)
        assert agent._runtime_override == {}

    def test_consume_runtime_override_none_safe_on_bare_agent(self):
        # A bare agent (created via __new__) has neither the registration
        # attribute nor _runtime_override; the handoff must not raise.
        from agent.runtime_override import consume_runtime_override

        agent = object.__new__(_FakeAgent)
        consume_runtime_override(agent)  # must not raise AttributeError
        consume_runtime_override(None)  # must not raise on None either
