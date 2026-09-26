"""Unit tests for pre_llm_call runtime_override (issue #23739)."""

from __future__ import annotations

import time

import pytest

from agent.context_compressor import (
    PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY,
    ContextCompressor,
)
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


# ---------------------------------------------------------------------------
# P1-1: the scope projects the canonical model-owned state (context compressor,
# prompt-cache flags, reasoning config) and restores it EXACTLY
# ---------------------------------------------------------------------------

_SESSION_MODEL = "orig-model"
_OVERRIDE_MODEL = "override-model"
_FALLBACK_MODEL = "fallback-model"
# Resolution patches keyed by model (context lengths differ within one provider).
_CTX_BY_MODEL = {
    _SESSION_MODEL: 200_000,
    _OVERRIDE_MODEL: 128_000,
    _FALLBACK_MODEL: 96_000,
}


def _build_model_owned_agent(*, model: str = _SESSION_MODEL, reasoning=None):
    """A fake agent carrying a REAL ContextCompressor plus every model-owned
    attribute the canonical switch projection reads/writes."""

    class _ModelOwnedAgent:
        def __init__(self):
            self.model = model
            self.provider = "openai"
            self.api_mode = "chat_completions"
            self.base_url = "https://api.openai.com/v1"
            self.api_key = "sk-test"
            self.requested_provider = "openai"
            self.context_compressor = ContextCompressor(
                model=model, threshold_percent=0.85, quiet_mode=True,
                config_context_length=_CTX_BY_MODEL[model], provider="openai",
                api_mode="chat_completions", base_url="https://api.openai.com/v1",
            )
            self.reasoning_config = reasoning if reasoning is not None else {
                "enabled": False, "effort": "low",
            }
            self._use_prompt_caching = False
            self._use_native_cache_layout = False
            self._config_context_length = _CTX_BY_MODEL[model]
            # A distinguishable pre-override object so the test can prove the
            # reference (not just the value) is restored.
            self._custom_providers = ["session-provider-entry"]
            self.request_overrides = {}
            self.runtime_capabilities = {"native_compaction": False}
            self._client_kwargs = {}
            self._transport_cache = {}
            self._fallback_activated = False
            self._runtime_override = {}
            # Distinguishable pre-override value so the restore test can prove
            # the cached system prompt is invalidated mid-scope and restored on
            # exit (its context-file caps depend on context_length).
            self._cached_system_prompt = "session-system-prompt"

        def _anthropic_prompt_cache_policy(self, *, provider=None, base_url=None,
                                           api_mode=None, model=None):
            # Model-driven flags so the test can assert they follow the model.
            return (
                str(model or "").startswith("override"),
                str(model or "").endswith("-native"),
            )

    return _ModelOwnedAgent()


def _patch_model_owned_resolution(monkeypatch):
    """Make context-length + reasoning resolution deterministic for the fake."""
    monkeypatch.setattr(
        "hermes_cli.config.get_compatible_custom_providers", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda model, **k: _CTX_BY_MODEL.get(str(model or ""), 256_000),
    )
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config",
        lambda cfg, model: {"enabled": model != _SESSION_MODEL, "effort": "high"},
    )


class TestModelOwnedProjection:
    """P1-1: switching model is a route transaction — the scope must project the
    model-owned state ``switch_model`` projects (compressor, prompt-cache flags,
    reasoning config) and restore it exactly on exit."""

    def test_override_projects_model_owned_state(self, monkeypatch):
        _patch_model_owned_resolution(monkeypatch)
        agent = _build_model_owned_agent()
        session_cc = agent.context_compressor

        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            # The compressor now describes the OVERRIDE model (different resolved
            # context length within the same provider), on a scope-owned object —
            # never the session compressor.
            cc = agent.context_compressor
            assert cc is not session_cc
            assert cc.model == _OVERRIDE_MODEL
            assert cc.context_length == 128_000
            assert cc.threshold_tokens == 108_800  # int(128000 * 0.85)
            # Prompt-cache flags follow the override model.
            assert agent._use_prompt_caching is True
            assert agent._use_native_cache_layout is False
            # Reasoning config follows the override model.
            assert agent.reasoning_config == {"enabled": True, "effort": "high"}
            # The session compressor is untouched mid-scope.
            assert session_cc.model == _SESSION_MODEL
            assert session_cc.threshold_tokens == 170_000  # int(200000 * 0.85)
            # The cached system prompt is invalidated so the next build re-scales
            # its context-file caps to the override model's context window.
            assert agent._cached_system_prompt is None

    def test_scope_restore_is_exact(self, monkeypatch):
        _patch_model_owned_resolution(monkeypatch)
        agent = _build_model_owned_agent()
        session_cc = agent.context_compressor
        session_reasoning = agent.reasoning_config
        session_custom_providers = agent._custom_providers

        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            pass

        # context_compressor is the SAME object as before — the pre-override
        # compressor was never mutated in place (no context-length leak).
        assert agent.context_compressor is session_cc
        assert session_cc.model == _SESSION_MODEL
        assert session_cc.context_length == 200_000
        assert session_cc.threshold_tokens == 170_000
        # Plain values are back to the pre-override values (identity included).
        assert agent.model == _SESSION_MODEL
        assert agent._use_prompt_caching is False
        assert agent._use_native_cache_layout is False
        assert agent.reasoning_config is session_reasoning
        # Fields the shared projection writes are restored too.
        assert agent._config_context_length == 200_000
        assert agent._custom_providers is session_custom_providers
        # The invalidated cached system prompt is restored to its pre-override
        # bytes, so the session's prefix-cache reuse survives the override.
        assert agent._cached_system_prompt == "session-system-prompt"

    def test_scope_restore_is_exact_on_exception(self, monkeypatch):
        _patch_model_owned_resolution(monkeypatch)
        agent = _build_model_owned_agent()
        session_cc = agent.context_compressor
        session_reasoning = agent.reasoning_config

        with pytest.raises(RuntimeError):
            with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
                assert agent.context_compressor.model == _OVERRIDE_MODEL
                raise RuntimeError("boom")

        assert agent.context_compressor is session_cc
        assert session_cc.model == _SESSION_MODEL
        assert session_cc.threshold_tokens == 170_000
        assert agent.model == _SESSION_MODEL
        assert agent._use_prompt_caching is False
        assert agent.reasoning_config is session_reasoning

    def test_fallback_supersession_keeps_fallback_compressor_state(self, monkeypatch):
        # P1-2 precedence with the compressor: when the fallback chain takes the
        # route mid-scope, the superseded scope must NOT restore the pre-override
        # compressor over the fallback-owned state.
        from agent.runtime_override import consume_runtime_override

        _patch_model_owned_resolution(monkeypatch)
        agent = _build_model_owned_agent()
        session_cc = agent.context_compressor

        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            scope_cc = agent.context_compressor
            assert scope_cc is not session_cc
            assert scope_cc.model == _OVERRIDE_MODEL
            # try_activate_fallback re-points the compressor to the fallback
            # model, then performs the explicit supersede handoff.
            agent.model = _FALLBACK_MODEL
            agent._fallback_activated = True
            scope_cc.update_model(
                model=_FALLBACK_MODEL, context_length=96_000, provider="openai",
                base_url="https://api.openai.com/v1", api_key="sk-test",
                api_mode="chat_completions",
            )
            consume_runtime_override(agent)

        # The fallback-owned compressor state stands after the scope exits.
        assert agent.context_compressor is scope_cc
        assert agent.context_compressor.model == _FALLBACK_MODEL
        assert agent.context_compressor.threshold_tokens == 81_600  # int(96000 * 0.85)
        assert agent.model == _FALLBACK_MODEL
        # The pre-override session compressor was never touched.
        assert session_cc.model == _SESSION_MODEL
        assert session_cc.threshold_tokens == 170_000


# ---------------------------------------------------------------------------
# P1 (durable leak): a temporary model override must not permanently clear the
# session's persisted compression-protection rows.
#
# ``_apply_model_owned_state`` -> ``ContextCompressor.update_model`` clears the
# durable fallback streak, failure cooldown, ineffective-strike count and
# proactive-prune runway.  For a REAL switch_model that is correct; for the
# ephemeral runtime_override the scope must roll those clears back exactly.
# ---------------------------------------------------------------------------


class _RecordingSessionDB:
    """Minimal in-memory session row recording every durable compression call.

    Implements the subset of the SessionDB compression API a real
    ``ContextCompressor`` reaches, so the production durable-write path is
    exercised without a sqlite file.  Values are stored raw so the test can
    assert byte-identical restore.
    """

    def __init__(self, session_id: str = "S1") -> None:
        self.session_id = session_id
        self.calls = []
        self.fallback_streak = 5
        self.ineffective_count = 3
        self.cooldown_until = time.time() + 600.0
        self.cooldown_error = "summary stall"
        self.model_config = {PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: 4096}

    def _record(self, name, *args):
        self.calls.append((name, args))

    def names(self):
        return [name for name, _ in self.calls]

    # ── fallback streak ────────────────────────────────────────────────
    def get_compression_fallback_streak(self, session_id):
        self._record("get_compression_fallback_streak", session_id)
        return self.fallback_streak

    def set_compression_fallback_streak(self, session_id, streak):
        self._record("set_compression_fallback_streak", session_id, streak)
        self.fallback_streak = streak

    # ── ineffective-compaction strike count ────────────────────────────
    def get_compression_ineffective_count(self, session_id):
        self._record("get_compression_ineffective_count", session_id)
        return self.ineffective_count

    def set_compression_ineffective_count(self, session_id, count):
        self._record("set_compression_ineffective_count", session_id, count)
        self.ineffective_count = count

    # ── failure cooldown ───────────────────────────────────────────────
    def get_compression_failure_cooldown(self, session_id):
        self._record("get_compression_failure_cooldown", session_id)
        if self.cooldown_until is None:
            return None
        remaining = self.cooldown_until - time.time()
        if remaining <= 0:
            return None
        return {
            "cooldown_until": self.cooldown_until,
            "remaining_seconds": remaining,
            "error": self.cooldown_error,
        }

    def get_compression_failure_cooldown_row(self, session_id):
        self._record("get_compression_failure_cooldown_row", session_id)
        return {
            "session_exists": True,
            "cooldown_until": self.cooldown_until,
            "error": self.cooldown_error,
        }

    def clear_compression_failure_cooldown(self, session_id):
        self._record("clear_compression_failure_cooldown", session_id)
        self.cooldown_until = None
        self.cooldown_error = None

    def restore_compression_failure_cooldown_row(self, session_id, snapshot):
        self._record("restore_compression_failure_cooldown_row", session_id, dict(snapshot))
        self.cooldown_until = snapshot.get("cooldown_until")
        self.cooldown_error = snapshot.get("error")

    # ── model_config (proactive-prune runway) ──────────────────────────
    def get_session_model_config_value(self, session_id, key, default=None):
        self._record("get_session_model_config_value", session_id, key, default)
        return self.model_config.get(key, default)

    def patch_session_model_config(self, session_id, patch):
        self._record("patch_session_model_config", session_id, dict(patch))
        for key, value in patch.items():
            if value is None:
                self.model_config.pop(key, None)
            else:
                self.model_config[key] = value


class TestDurableProtectionRestore:
    """P1: one ephemeral model override must leave every durable
    compression-protection row byte-identical after the scope exits."""

    def _agent_with_recording_db(self, monkeypatch):
        _patch_model_owned_resolution(monkeypatch)
        agent = _build_model_owned_agent()
        db = _RecordingSessionDB()
        cc = agent.context_compressor
        cc.bind_session_state(db, "S1")
        # bind_session_state hydrates the in-memory values from the durable rows.
        assert cc._fallback_compression_streak == 5
        assert cc._ineffective_compression_count == 3
        assert cc._proactive_prune_rearm_tokens == 4096
        return agent, db, cc

    @staticmethod
    def _durable_state(db):
        return (
            db.fallback_streak,
            db.ineffective_count,
            db.cooldown_until,
            db.cooldown_error,
            dict(db.model_config),
        )

    def test_override_restores_durable_protection_exactly(self, monkeypatch):
        agent, db, session_cc = self._agent_with_recording_db(monkeypatch)
        baseline = self._durable_state(db)

        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            # The canonical projection clears the durable protection rows
            # mid-scope — this is the leak the override must undo.
            assert db.fallback_streak == 0
            assert db.ineffective_count == 0
            assert db.cooldown_until is None
            assert db.cooldown_error is None
            assert PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY not in db.model_config

        # Scope exit restored every durable value byte-for-byte.
        assert self._durable_state(db) == baseline
        # The pre-override compressor (and its in-memory protection state) is back.
        assert agent.context_compressor is session_cc
        assert session_cc._fallback_compression_streak == 5
        assert session_cc._ineffective_compression_count == 3
        assert session_cc._proactive_prune_rearm_tokens == 4096

    def test_override_restore_is_exact_on_exception(self, monkeypatch):
        agent, db, _session_cc = self._agent_with_recording_db(monkeypatch)
        baseline = self._durable_state(db)

        with pytest.raises(RuntimeError):
            with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
                raise RuntimeError("boom")

        assert self._durable_state(db) == baseline

    def test_fallback_supersession_keeps_durable_writes(self, monkeypatch):
        # A superseded scope must NOT roll back the fallback chain's own durable
        # ``update_model`` clears — the fallback owns the route now.
        from agent.runtime_override import consume_runtime_override

        agent, db, _session_cc = self._agent_with_recording_db(monkeypatch)
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            agent.model = _FALLBACK_MODEL
            agent.context_compressor.update_model(
                model=_FALLBACK_MODEL, context_length=96_000, provider="openai",
                base_url="https://api.openai.com/v1", api_key="sk-test",
                api_mode="chat_completions",
            )
            consume_runtime_override(agent)

        # The fallback's cleared rows stand (the pre-override values are not
        # resurrected over the route the fallback now owns).
        assert db.fallback_streak == 0
        assert db.ineffective_count == 0
        assert db.cooldown_until is None
        assert PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY not in db.model_config
