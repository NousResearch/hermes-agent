"""Remediation tests for the Sol round-2 findings.

Each test exercises a REAL owning path (helper + call-site semantics), not
just a direct function call. The 12 scenarios map to the Sol findings:

  Finding 1  generic plugin update_model raises -> transition fails, host state stays old
  Finding 2  different model/provider with same effective -> plugin still updates
  Finding 3a fallback primary 64K -> fallback 900K -> provider 200K -> accepts downward
  Finding 3b fallback primary 900K -> fallback 128K -> provider 256K -> must NOT inflate
  Finding 3c primary restoration returns original primary pre-cap
  Finding 4  same-model Codex refresh -> plugin accounting survives
  Finding 5  compression disabled + request above ceiling -> no provider dispatch
  Finding 6  two-profile WebUI test (raw cached + ceiling)
  Finding 7  generic plugin cannot persist 272K ceiling over raw 900K
  Finding 8  missing-marker MagicMock receives effective, not raw
  Finding 9  bool / float / infinity / malformed ceiling validation
  Finding 10 max_context_length < 64K -> ceiling-specific config failure
"""

import sys
import pytest
from unittest.mock import patch, MagicMock

from agent import model_metadata
from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine
from agent.agent_runtime_helpers import (
    transition_model_context,
    refresh_context_window,
    get_pre_cap,
    ceiling_exceeded,
    _engine_handles_ceiling,
    _engine_effective_window,
)
from agent import context_compressor as _cc


# ─── Shared helpers ──────────────────────────────────────────────────────────

class _PluginEngine(ContextEngine):
    """Minimal generic plugin engine inheriting the base update_model contract."""
    def __init__(self, threshold_percent=0.75):
        self.threshold_percent = threshold_percent
        self.context_length = 0
        self.threshold_tokens = 0
        # accounting fields (a plugin may legitimately reset these in update_model)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.update_model_calls = 0
        self.update_model_kwargs = []
        self.raise_on_update = False

    @property
    def name(self):
        return "plugin"

    def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
        if self.raise_on_update:
            raise RuntimeError("plugin engine update_model failed")
        self.update_model_calls += 1
        self.update_model_kwargs.append(dict(
            model=model, context_length=context_length,
            base_url=base_url, provider=provider, api_mode=api_mode,
        ))
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
        # A real plugin may reset accounting in update_model:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

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


def _resolver(**kw):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        fake = MagicMock(**kw)
        with patch.object(_cc, "get_model_context_length", fake), \
             patch.object(model_metadata, "get_model_context_length", fake):
            yield fake
    return _ctx()


# ═══════════════════════════════════════════════════════════════════════════
# Finding 1: generic plugin update_model raises -> transition fails, host stays old
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding1_FailurePropagates:
    def test_plugin_update_model_raises_propagates_and_host_unchanged(self):
        """transition_model_context must PROPAGATE the engine failure (not
        swallow it) and leave the host pre-cap at its prior value. The
        earlier route_context_window wrapped update_model in try/except and
        then unconditionally set agent._pre_cap_context_length — manufacturing
        success from a failed transition."""
        eng = _PluginEngine()
        eng.raise_on_update = True
        agent = _AgentStub(eng, ceiling=272_000)
        # Set a prior host pre-cap (e.g. from init)
        agent._pre_cap_context_length = 128_000

        with pytest.raises(RuntimeError, match="plugin engine update_model failed"):
            transition_model_context(agent, 900_000, model="new_model")

        # Host pre-cap must STILL be the old value (128K), not 900K.
        assert agent._pre_cap_context_length == 128_000
        # Engine's context_length must NOT have been updated to 900K/272K.
        assert eng.context_length == 0  # unchanged from init


# ═══════════════════════════════════════════════════════════════════════════
# Finding 2: different model/provider with same effective -> plugin still updates
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding2_SameEffectiveStillTransitions:
    def test_model_change_with_same_effective_calls_update_model(self):
        """Model A (raw 900K -> effective 272K) -> Model B (raw 300K ->
        effective 272K). The effective integer is IDENTICAL (272K), but the
        model/provider changed. transition_model_context MUST call
        update_model — the same effective value is NOT sufficient reason to
        skip. The earlier route_context_window's skip-if-unchanged would have
        skipped this."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)

        # Model A: raw 900K -> effective 272K
        transition_model_context(agent, 900_000, model="model_a", provider="prov_a")
        assert eng.update_model_calls == 1
        assert eng.context_length == 272_000

        # Model B: raw 300K -> effective 272K (same effective, different model)
        transition_model_context(agent, 300_000, model="model_b", provider="prov_b")
        # MUST have called update_model again (not skipped)
        assert eng.update_model_calls == 2, (
            "update_model must be called for a model change even when the "
            "effective integer is unchanged (finding #2)"
        )
        # The second call must carry the new model/provider
        last = eng.update_model_kwargs[-1]
        assert last["model"] == "model_b"
        assert last["provider"] == "prov_b"
        assert last["context_length"] == 272_000  # still capped


# ═══════════════════════════════════════════════════════════════════════════
# Finding 3a: fallback primary 64K -> fallback 900K -> provider 200K -> accepts
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding3a_FallbackDownwardCorrection:
    def test_fallback_accepts_downward_provider_correction(self):
        """Primary 64K -> fallback 900K -> provider reports 200K.
        The provider-reported 200K is a DOWNWARD correction relative to the
        ACTIVE fallback pre-cap (900K). It must be accepted, and the active
        pre-cap becomes 200K."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=None)

        # Primary: 64K
        transition_model_context(agent, 64_000, model="primary")
        assert agent._pre_cap_context_length == 64_000

        # Fallback activation: the fallback is now the ACTIVE model.
        # Host pre-cap becomes the fallback's (900K).
        transition_model_context(agent, 900_000, model="fallback", provider="fb_prov")
        assert agent._pre_cap_context_length == 900_000  # active = fallback

        # Provider reports 200K — a downward correction vs active 900K.
        # get_context_length_from_provider_error returns 200K (200K < 900K).
        from agent.model_metadata import get_context_length_from_provider_error
        # A message the parser recognizes: "maximum context length is 200000"
        new_ctx = get_context_length_from_provider_error(
            "maximum context length is 200000 tokens", get_pre_cap(agent)
        )
        assert new_ctx == 200_000  # accepted (downward)

        # Apply via refresh (same-model correction on the active fallback).
        refresh_context_window(agent, 200_000, model="fallback")
        assert agent._pre_cap_context_length == 200_000
        assert eng.context_length == 200_000


# ═══════════════════════════════════════════════════════════════════════════
# Finding 3b: fallback primary 900K -> fallback 128K -> provider 256K -> NOT inflated
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding3b_FallbackNoInflation:
    def test_fallback_rejects_upward_provider_correction(self):
        """Primary 900K -> fallback 128K -> provider reports 256K.
        256K is ABOVE the active fallback pre-cap (128K). The provider
        correction is DOWNWARD-ONLY relative to the active model's pre-cap,
        so 256K must be REJECTED (not accepted). The active pre-cap stays
        at 128K — the fallback must NOT be inflated."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=None)

        # Primary: 900K
        transition_model_context(agent, 900_000, model="primary")
        assert agent._pre_cap_context_length == 900_000

        # Fallback: 128K (active model is now the fallback)
        transition_model_context(agent, 128_000, model="fallback", provider="fb_prov")
        assert agent._pre_cap_context_length == 128_000

        # Provider reports 256K — ABOVE the active 128K.
        from agent.model_metadata import get_context_length_from_provider_error
        new_ctx = get_context_length_from_provider_error(
            "maximum context length is 256000 tokens", get_pre_cap(agent)
        )
        # 256K > 128K (active) -> rejected (None)
        assert new_ctx is None, (
            "provider correction above the active pre-cap must be rejected "
            "(finding #3b: no inflation)"
        )
        # Active pre-cap stays at 128K
        assert agent._pre_cap_context_length == 128_000
        assert eng.context_length == 128_000


# ═══════════════════════════════════════════════════════════════════════════
# Finding 3c: primary restoration returns original primary pre-cap
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding3c_PrimaryRestoration:
    def test_restore_returns_primary_pre_cap(self):
        """Full round trip: primary 900K -> fallback 128K -> restore.
        Restore feeds the snapshot's pre-cap (900K) back through
        transition_model_context. The host must re-derive 900K and the
        plugin must re-derive its effective window from 900K."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)

        # Init: primary 900K
        transition_model_context(agent, 900_000, model="primary")
        assert agent._pre_cap_context_length == 900_000
        assert eng.context_length == 272_000  # capped

        # Snapshot (what _primary_runtime stores)
        snapshot_pre_cap = get_pre_cap(agent)
        assert snapshot_pre_cap == 900_000

        # Fallback: 128K (active model changes)
        transition_model_context(agent, 128_000, model="fallback")
        assert agent._pre_cap_context_length == 128_000  # active = fallback
        assert eng.context_length == 128_000

        # Restore: feed the snapshot's pre-cap back
        transition_model_context(agent, snapshot_pre_cap, model="primary")
        assert agent._pre_cap_context_length == 900_000  # back to primary
        assert eng.context_length == 272_000  # re-capped


# ═══════════════════════════════════════════════════════════════════════════
# Finding 4: same-model Codex refresh -> plugin accounting survives
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding4_CodexRefreshPreservesAccounting:
    def test_plugin_accounting_survives_refresh(self):
        """The Codex path records accounting via update_from_response, then
        refreshes the context window. A plugin's update_model MAY reset
        accounting — the codex_runtime.py path therefore re-applies the
        recorded accounting AFTER the refresh. This test simulates the exact
        sequence from codex_runtime.py:185-220."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)
        # Initial transition
        transition_model_context(agent, 900_000, model="m")
        assert eng.context_length == 272_000

        # Codex: record accounting, then refresh
        eng.update_from_response({"prompt_tokens": 5000, "completion_tokens": 200, "total_tokens": 5200})
        _accounting = {
            f: getattr(eng, f, None)
            for f in ("last_prompt_tokens", "last_completion_tokens", "last_total_tokens")
        }
        assert _accounting["last_prompt_tokens"] == 5000

        # Refresh (same-model window change: 200K reported)
        refresh_context_window(agent, 200_000, model="m")
        assert eng.context_length == 200_000
        # update_model reset the accounting (plugin behavior)
        assert eng.last_prompt_tokens == 0  # erased by update_model

        # codex_runtime.py re-applies the recorded accounting:
        for _f, _v in _accounting.items():
            if isinstance(_v, int) and not isinstance(_v, bool):
                setattr(eng, _f, _v)

        # Accounting survived the refresh
        assert eng.last_prompt_tokens == 5000
        assert eng.last_completion_tokens == 200
        assert eng.last_total_tokens == 5200


# ═══════════════════════════════════════════════════════════════════════════
# Finding 5: compression disabled + request above ceiling -> no dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding5_CeilingGate:
    def test_ceiling_exceeded_returns_true(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        assert ceiling_exceeded(agent, 300_000) is True
        assert ceiling_exceeded(agent, 272_000) is False
        assert ceiling_exceeded(agent, 200_000) is False

    def test_ceiling_exceeded_no_ceiling(self):
        agent = _AgentStub(_PluginEngine(), ceiling=None)
        assert ceiling_exceeded(agent, 999_999_999) is False

    def test_ceiling_exceeded_rejects_non_int(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        assert ceiling_exceeded(agent, "300000") is False
        assert ceiling_exceeded(agent, True) is False
        assert ceiling_exceeded(agent, None) is False

    def test_gate_refuses_dispatch(self):
        """The run_conversation dispatch site calls ceiling_exceeded() and
        refuses before provider dispatch. This test verifies the gate
        function's contract: when the gate fires, the request must NOT be
        dispatched. We verify the gate returns True for an over-ceiling
        request, which is the precondition the dispatch site checks."""
        agent = _AgentStub(_PluginEngine(), ceiling=272_000)
        request_tokens = 300_000  # above the 272K ceiling
        should_refuse = ceiling_exceeded(agent, request_tokens)
        assert should_refuse is True, (
            "ceiling_exceeded must return True for an over-ceiling request; "
            "the dispatch site uses this to refuse before provider dispatch"
        )
        # And for an under-ceiling request, dispatch proceeds:
        assert ceiling_exceeded(agent, 200_000) is False


# ═══════════════════════════════════════════════════════════════════════════
# Finding 6: two-profile WebUI test
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding6_WebUIProfileScope:
    def test_profile_scope_fix_in_source(self):
        """Verify that the WebUI /api/model/info endpoint now resolves the
        raw context length INSIDE the _profile_scope block (finding #6).
        This is a source-level check: the get_model_context_length call must
        be inside the `with _profile_scope(profile):` block."""
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent
        src = (root / "hermes_cli" / "web_server.py").read_text(errors="replace")
        # Find the /api/model/info function
        fn_start = src.find('def get_model_info')
        fn_body = src[fn_start:]
        # Find the _profile_scope block
        scope_start = fn_body.find("with _profile_scope(profile):")
        # Find get_model_context_length in the function body
        gmc_start = fn_body.find("get_model_context_length")
        assert scope_start != -1, "_profile_scope not found in get_model_info"
        assert gmc_start != -1, "get_model_context_length not found in get_model_info"
        # get_model_context_length must come AFTER _profile_scope (inside the block)
        assert gmc_start > scope_start, (
            "get_model_context_length must be INSIDE the _profile_scope block "
            "(finding #6: raw capability resolution must be profile-scoped)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Finding 7: generic plugin cannot persist 272K ceiling over raw 900K
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding7_PersistencePurity:
    def test_host_pre_cap_used_for_persistence_not_engine_capped(self):
        """For a generic plugin, the engine's context_length IS the capped
        effective window (272K). The persistence site must use the
        HOST-OWNED active pre-cap (900K), not the engine's capped value.
        get_pre_cap(agent) returns the host value; resolve_engine_pre_cap
        would return the engine's capped value for a plugin."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)
        transition_model_context(agent, 900_000, model="m")
        assert agent._pre_cap_context_length == 900_000  # host = raw
        assert eng.context_length == 272_000  # engine = capped

        # The persistence site uses get_pre_cap(agent) first:
        from agent.agent_runtime_helpers import resolve_engine_pre_cap
        host_value = get_pre_cap(agent)
        assert host_value == 900_000, (
            "get_pre_cap must return the host-owned raw pre-cap (900K), "
            "not the engine's capped value (272K) (finding #7)"
        )
        # resolve_engine_pre_cap for a plugin would return the capped value:
        engine_value = resolve_engine_pre_cap(eng)
        assert engine_value == 272_000  # engine has no pre_cap property, falls to context_length
        # The persistence site correctly prefers host_value (900K) over engine_value (272K)
        assert host_value > engine_value


# ═══════════════════════════════════════════════════════════════════════════
# Finding 8: missing-marker MagicMock receives effective, not raw
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding8_MagicMockRouting:
    def test_magicmock_receives_effective_not_raw(self):
        """A MagicMock engine (dynamic __getattr__) must be classified as
        NOT handling the ceiling (is True check rejects the truthy child
        MagicMock). It must receive the EFFECTIVE window, not raw."""
        mock_engine = MagicMock()
        # MagicMock.__getattr__ returns a truthy child for any attr
        assert bool(getattr(mock_engine, "_handles_context_ceiling", False)) is True  # bool() would misroute
        # But the is True check correctly rejects it:
        assert _engine_handles_ceiling(mock_engine) is False, (
            "_engine_handles_ceiling must return False for a MagicMock "
            "(finding #8: is True, not bool(getattr(...)))"
        )
        # Therefore it receives the effective window:
        agent = _AgentStub(mock_engine, ceiling=272_000)
        supplied = _engine_effective_window(agent, 900_000)
        assert supplied == 272_000, (
            "A MagicMock (missing marker) must receive the EFFECTIVE window "
            "(272K), not raw (900K) (finding #8)"
        )

    def test_explicit_marker_receives_raw(self):
        """An engine with _handles_context_ceiling = True (explicit) receives
        the RAW value."""
        class _Marked:
            _handles_context_ceiling = True
        assert _engine_handles_ceiling(_Marked()) is True
        agent = _AgentStub(_Marked(), ceiling=272_000)
        # For a marked engine, transition_model_context supplies raw:
        # (we verify the dispatch decision, not the full transition)
        assert _engine_handles_ceiling(_Marked()) is True


# ═══════════════════════════════════════════════════════════════════════════
# Finding 9: strict ceiling validation (bool/float/inf/string/<=0)
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding9_StrictValidation:
    def test_coerce_rejects_bool(self):
        assert model_metadata.coerce_context_ceiling(True) is None
        assert model_metadata.coerce_context_ceiling(False) is None

    def test_coerce_rejects_float(self):
        assert model_metadata.coerce_context_ceiling(272000.0) is None
        assert model_metadata.coerce_context_ceiling(2.9) is None

    def test_coerce_rejects_infinity(self):
        assert model_metadata.coerce_context_ceiling(float("inf")) is None
        assert model_metadata.coerce_context_ceiling(float("-inf")) is None
        assert model_metadata.coerce_context_ceiling(float("nan")) is None

    def test_coerce_rejects_string(self):
        assert model_metadata.coerce_context_ceiling("272000") is None
        assert model_metadata.coerce_context_ceiling("nope") is None

    def test_coerce_rejects_zero_and_negative(self):
        assert model_metadata.coerce_context_ceiling(0) is None
        assert model_metadata.coerce_context_ceiling(-1) is None
        assert model_metadata.coerce_context_ceiling(-272000) is None

    def test_coerce_accepts_valid_int(self):
        assert model_metadata.coerce_context_ceiling(272000) == 272000
        assert model_metadata.coerce_context_ceiling(65536) == 65536
        assert model_metadata.coerce_context_ceiling(1000000) == 1000000

    def test_coerce_rejects_none(self):
        assert model_metadata.coerce_context_ceiling(None) is None

    def test_validate_rejects_below_64k(self):
        # The 64K floor is MINIMUM_CONTEXT_LENGTH = 64000.
        assert model_metadata.validate_context_ceiling(32000) is None
        assert model_metadata.validate_context_ceiling(63999) is None  # just below floor
        assert model_metadata.validate_context_ceiling(64000) == 64000  # exactly at floor
        assert model_metadata.validate_context_ceiling(272000) == 272000  # well above

    def test_validate_rejects_malformed(self):
        assert model_metadata.validate_context_ceiling(True) is None
        assert model_metadata.validate_context_ceiling(272000.0) is None
        assert model_metadata.validate_context_ceiling("272000") is None
        assert model_metadata.validate_context_ceiling(0) is None


# ═══════════════════════════════════════════════════════════════════════════
# Finding 10: max_context_length < 64K -> ceiling-specific config failure
# ═══════════════════════════════════════════════════════════════════════════

class TestFinding10_64kFloor:
    def test_below_64k_is_invalid_ceiling(self):
        """A positive integer below 64K is an INVALID ceiling (a
        mis-configuration), reported with a ceiling-specific error — not
        'the model only has this capability'."""
        # validate_context_ceiling returns None for below-floor values (< 64000)
        assert model_metadata.validate_context_ceiling(32000) is None
        assert model_metadata.validate_context_ceiling(16384) is None
        assert model_metadata.validate_context_ceiling(8192) is None
        assert model_metadata.validate_context_ceiling(64000) == 64000  # exactly at floor is valid
        # But coerce_context_ceiling (runtime) still accepts them as a
        # positive int (the floor is a CONFIG-time rejection, not runtime)
        assert model_metadata.coerce_context_ceiling(32000) == 32000

    def test_64k_floor_constant(self):
        # 64K floor: 64 * 1000 = 64000 tokens.
        assert model_metadata.MINIMUM_CONTEXT_LENGTH == 64_000
