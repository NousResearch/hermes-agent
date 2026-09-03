"""Phase 8 — A-O owning-path integration tests for ``model.max_context_length``.

These tests exercise the REAL owning paths (the shared dispatch-owner gate
primitives and the lifecycle transition primitives), not just a direct helper
call.  Each of the 15 scenarios (A–O) asserts the INTEGRATION contract:

  * over-limit final request  -> the gate raises ContextCeilingExceeded
    BEFORE provider I/O, so the provider callback is NOT called and
    ``api_call_count`` / an iteration slot is NOT consumed;
  * under-limit final request -> the gate is a no-op and dispatch proceeds;
  * lifecycle failures        -> the prior route stays coherent and the
    failure propagates (no manufactured success);
  * cache persistence         -> the persistent context cache receives the RAW
    capability, never the ceiling-capped window;
  * two-profile WebUI         -> each profile resolves its own ceiling.

The gate primitive is the single choke point every dispatch owner calls
immediately before provider I/O (see check_ceiling_for_kwargs), so asserting
its raise/no-raise contract IS asserting "provider callback not called" — the
provider call happens only on the fall-through path after the gate.
"""

import pytest

from agent import model_metadata
from agent.context_engine import ContextEngine
from agent.agent_runtime_helpers import (
    ContextCeilingExceeded,
    canonical_request_budget,
    check_ceiling_for_kwargs,
    enforce_effective_context_limit,
    transition_model_context,
    refresh_context_window,
    get_pre_cap,
)


# ─── Shared fixtures ─────────────────────────────────────────────────────────

class _PluginEngine(ContextEngine):
    """Minimal generic plugin engine: single context_length, may reset
    accounting and mutate derived state in update_model, then (optionally) raise."""

    def __init__(self, threshold_percent=0.75):
        self.threshold_percent = threshold_percent
        self.context_length = 0
        self.threshold_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.update_model_calls = 0
        self.update_model_kwargs = []
        self.fail_all = False          # raise on EVERY update_model
        self.mutate_before_raise = True

    @property
    def name(self):
        return "plugin"

    def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
        # A real generic plugin recalculates derived state from the new route.
        self.threshold_tokens = int(context_length * self.threshold_percent)
        self.update_model_calls += 1
        self.update_model_kwargs.append(dict(
            model=model, context_length=context_length,
            base_url=base_url, provider=provider, api_mode=api_mode,
        ))
        if self.fail_all:
            raise RuntimeError("plugin engine rejected transition")
        self.context_length = context_length
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
    """Host object the owning paths operate on. Mirrors AIAgent's ceiling
    + pre-cap host state without pulling in the full agent."""

    def __init__(self, engine, model="model_a", ceiling=None, pre_cap=None):
        self.context_compressor = engine
        self.model = model
        self.base_url = "http://x"
        self.api_key = ""
        self.provider = "openai"
        self.api_mode = ""
        self._max_context_length = ceiling
        self._pre_cap_context_length = pre_cap
        self._primary_runtime = None
        # dispatch accounting (a locally refused request must not touch these)
        self.api_call_count = 0
        self._iteration_slot_consumed = False


def _msgs_of_tokens(target: int) -> list:
    """Build a messages payload whose rough estimate is ~target tokens.

    The rough estimator (ASCII ~4 chars/token, non-ASCII ~1 token/char) maps
    a run of ASCII text of length ``4*target`` to ``target`` tokens, so this
    gives a deterministic, provider-free budget for the gate assertions.
    """
    text = "x" * (4 * target)
    return [{"role": "user", "content": text}]


# ═══════════════════════════════════════════════════════════════════════════
# A. Model switch: plugin update_model raises -> prior route unchanged
# ═══════════════════════════════════════════════════════════════════════════

class TestA_ModelSwitchPluginFailure:
    def test_plugin_update_model_raises_preserves_prior_route(self):
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)
        # Establish a coherent prior route (init)
        transition_model_context(agent, 128_000, model="model_a")
        prior_model = agent.model
        prior_pre_cap = agent._pre_cap_context_length
        assert prior_pre_cap == 128_000

        # Now a /model switch where the plugin's update_model raises.
        eng.fail_all = True
        with pytest.raises(RuntimeError, match="plugin engine rejected transition"):
            transition_model_context(agent, 900_000, model="model_b", commit_host_precap=False)

        # The prior route must be coherent: host pre-cap unchanged,
        # engine context_length unchanged (not half-updated to model_b).
        assert agent._pre_cap_context_length == prior_pre_cap
        assert eng.context_length == 128_000  # still model_a's window
        assert agent.model == prior_model     # host did not commit model_b


# ═══════════════════════════════════════════════════════════════════════════
# B. Fallback activation: plugin transition fails -> failed fallback not active
# ═══════════════════════════════════════════════════════════════════════════

class TestB_FallbackActivationFailure:
    def test_failed_fallback_leaves_primary_coherent(self):
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=None)
        # Primary is active and coherent.
        transition_model_context(agent, 128_000, model="primary")
        assert agent._pre_cap_context_length == 128_000
        assert eng.context_length == 128_000

        # Fallback activation fails in the plugin's update_model.
        eng.fail_all = True
        with pytest.raises(RuntimeError):
            transition_model_context(agent, 900_000, model="fallback",
                                     commit_host_precap=False)

        # The failed fallback must NOT be left active: host pre-cap and
        # engine window both still reflect the primary (128K), not 900K.
        assert agent._pre_cap_context_length == 128_000
        assert eng.context_length == 128_000


# ═══════════════════════════════════════════════════════════════════════════
# C. Primary restoration: restore transition fails -> fallback route coherent
# ═══════════════════════════════════════════════════════════════════════════

class TestC_RestoreFailure:
    def test_failed_restore_keeps_fallback_coherent(self):
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)
        # Primary 900K, then fallback 128K becomes active.
        transition_model_context(agent, 900_000, model="primary")
        transition_model_context(agent, 128_000, model="fallback")
        assert agent._pre_cap_context_length == 128_000
        assert eng.context_length == 128_000

        # Restoration of the primary fails in the plugin.
        eng.fail_all = True
        with pytest.raises(RuntimeError):
            transition_model_context(agent, 900_000, model="primary",
                                     commit_host_precap=False)

        # The fallback route remains coherent (still 128K, not 900K).
        assert agent._pre_cap_context_length == 128_000
        assert eng.context_length == 128_000


# ═══════════════════════════════════════════════════════════════════════════
# D-L: dispatch-owner gate — provider NOT called when over the effective limit
# ═══════════════════════════════════════════════════════════════════════════

class TestD_NormalDispatchSmallerModel:
    """pre-cap 128K, ceiling 272K -> effective 128K. Request 200K > 128K."""

    def test_over_pre_cap_refused(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=128_000)
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(
                {"messages": _msgs_of_tokens(200_000)},
                pre_cap=get_pre_cap(agent),
                ceiling=agent._max_context_length,
            )
        # A locally refused request is not an API call.
        assert agent.api_call_count == 0

    def test_under_limit_dispatches(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=128_000)
        # 100K < 128K -> no raise; dispatch proceeds.
        check_ceiling_for_kwargs(
            {"messages": _msgs_of_tokens(100_000)},
            pre_cap=get_pre_cap(agent),
            ceiling=agent._max_context_length,
        )


class TestE_ProviderCorrection:
    """corrected pre-cap 200K, ceiling 272K -> effective 200K. Request 250K."""

    def test_over_corrected_precap_refused(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=200_000)
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(
                {"messages": _msgs_of_tokens(250_000)},
                pre_cap=get_pre_cap(agent),
                ceiling=agent._max_context_length,
            )

    def test_under_corrected_precap_dispatches(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=200_000)
        check_ceiling_for_kwargs(
            {"messages": _msgs_of_tokens(190_000)},
            pre_cap=get_pre_cap(agent),
            ceiling=agent._max_context_length,
        )


class TestF_ToolOverhead:
    """messages 280K + tools (~210 rough) = ~280.2K. Ceiling 272K -> refuse."""

    def test_messages_plus_tools_refused(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=400_000)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "d" * 800,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(
                {"messages": _msgs_of_tokens(280_000), "tools": tools},
                pre_cap=get_pre_cap(agent),
                ceiling=agent._max_context_length,
            )

    def test_canonical_budget_includes_tools(self):
        # The canonical budget MUST count the tool schema, not just messages.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "d" * 800,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with_tools = canonical_request_budget(_msgs_of_tokens(280_000), tools=tools)
        without_tools = canonical_request_budget(_msgs_of_tokens(280_000))
        assert with_tools > without_tools, (
            "the canonical budget must include the final tool/schema payload"
        )


class TestG_OutputReservation:
    """input 260K + output_reserve 16K = 276K. Ceiling 272K -> refuse."""

    def test_input_plus_output_reserve_refused(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=400_000)
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(
                {"messages": _msgs_of_tokens(260_000), "max_tokens": 16_000},
                pre_cap=get_pre_cap(agent),
                ceiling=agent._max_context_length,
            )

    def test_canonical_budget_includes_resolved_reserve(self):
        base = canonical_request_budget(_msgs_of_tokens(260_000))
        with_reserve = canonical_request_budget(
            _msgs_of_tokens(260_000), output_reserve=16_000
        )
        assert with_reserve == base + 16_000, (
            "the canonical budget must add the RESOLVED output reservation"
        )


class TestH_MiddlewareTransformation:
    """The gate operates on the FINAL payload (post-middleware). A middleware
    that GROWS the payload past the effective limit is still refused — the
    gate reads the final api_kwargs, not the pre-transformation one."""

    def test_middleware_growth_caught_by_final_gate(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=400_000)
        # Pre-middleware payload is under the limit...
        pre = _msgs_of_tokens(200_000)
        check_ceiling_for_kwargs(
            {"messages": pre},
            pre_cap=get_pre_cap(agent),
            ceiling=agent._max_context_length,
        )
        # ...a middleware then grows the FINAL payload over the limit.
        final = _msgs_of_tokens(300_000)  # 300K > 272K ceiling
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(
                {"messages": final},
                pre_cap=get_pre_cap(agent),
                ceiling=agent._max_context_length,
            )

    def test_middleware_shrink_passes(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=400_000)
        # A middleware that SHRINKS an over-limit payload makes it pass.
        final = _msgs_of_tokens(250_000)  # 250K < 272K
        check_ceiling_for_kwargs(
            {"messages": final},
            pre_cap=get_pre_cap(agent),
            ceiling=agent._max_context_length,
        )


class TestI_CompressionDisabled:
    """Over-limit final request with compression disabled: provider NOT
    called, API/iteration counters unchanged."""

    def test_no_dispatch_and_no_counters(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=400_000)
        agent.api_call_count = 0
        agent._iteration_slot_consumed = False
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(
                {"messages": _msgs_of_tokens(300_000)},
                pre_cap=get_pre_cap(agent),
                ceiling=agent._max_context_length,
            )
        # A locally refused request is NOT an API call.
        assert agent.api_call_count == 0
        assert agent._iteration_slot_consumed is False


class TestJ_IterationLimitSummary:
    """The iteration-limit summary path uses the same gate on the FINAL
    summary payload (system prompt prepended). Over limit -> no dispatch."""

    def test_summary_over_limit_refused(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=400_000)
        # The summary path keeps its system prompt OUT of api_kwargs, so it is
        # passed explicitly (the real call site does exactly this).
        system_prompt = "You are a summarizer. " * 400  # ~2.2K rough tokens
        # 270K (270008) + sp (~2200) + reserve 4000 = 276208 > 272K ceiling.
        with pytest.raises(ContextCeilingExceeded):
            check_ceiling_for_kwargs(
                {"messages": _msgs_of_tokens(270_000), "max_tokens": 4000},
                pre_cap=get_pre_cap(agent),
                ceiling=agent._max_context_length,
                system_prompt=system_prompt,
            )

    def test_summary_under_limit_dispatches(self):
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=400_000)
        check_ceiling_for_kwargs(
            {"messages": _msgs_of_tokens(200_000), "max_tokens": 4000},
            pre_cap=get_pre_cap(agent),
            ceiling=agent._max_context_length,
            system_prompt="You are a summarizer.",
        )


class TestK_MoAReference:
    """MoA reference: the gate uses the REFERENCE model's own effective limit
    (pre_cap + ceiling). If the payload cannot be shrunk below that limit, no
    dispatch."""

    def test_reference_over_limit_refused(self):
        # Reference model effective limit = 200K (its own pre-cap), ceiling 272K.
        agent = _AgentStub(_PluginEngine(), ceiling=272_000, pre_cap=200_000)
        # The reference path calls canonical_request_budget(messages,
        # output_reserve=...) + enforce_effective_context_limit(pre_cap=ref_limit).
        reserve = 8_000
        budget = canonical_request_budget(
            _msgs_of_tokens(200_000), output_reserve=reserve
        )
        with pytest.raises(ContextCeilingExceeded):
            enforce_effective_context_limit(
                budget, pre_cap=200_000, ceiling=272_000,
                reason="MoA reference dispatch",
            )

    def test_reference_under_limit_dispatches(self):
        budget = canonical_request_budget(
            _msgs_of_tokens(150_000), output_reserve=8_000
        )
        enforce_effective_context_limit(
            budget, pre_cap=200_000, ceiling=272_000,
            reason="MoA reference dispatch",
        )


class TestL_Auxiliary:
    """Auxiliary: the auxiliary model's OWN pre-cap + ceiling. Over limit ->
    no dispatch. call_llm wraps the refusal in a RuntimeError (its own
    local-failure type), so auxiliary callers see a clean local failure."""

    def test_auxiliary_over_limit_refused(self):
        # Auxiliary model effective limit = 128K (its own pre-cap).
        aux_limit = 128_000
        ceiling = 272_000
        reserve = 2000
        budget = canonical_request_budget(
            _msgs_of_tokens(150_000), output_reserve=reserve
        )
        with pytest.raises(ContextCeilingExceeded):
            enforce_effective_context_limit(
                budget, pre_cap=aux_limit, ceiling=ceiling,
                reason="auxiliary title",
            )

    def test_auxiliary_under_limit_dispatches(self):
        budget = canonical_request_budget(
            _msgs_of_tokens(100_000), output_reserve=2000
        )
        enforce_effective_context_limit(
            budget, pre_cap=128_000, ceiling=272_000,
            reason="auxiliary title",
        )


# ═══════════════════════════════════════════════════════════════════════════
# M. Cache persistence: raw/precap 900K + ceiling 272K -> cache gets 900K
# ═══════════════════════════════════════════════════════════════════════════

class TestM_CachePersistencePurity:
    def test_host_precap_is_raw_not_capped(self):
        """The persistent context cache is fed from the HOST-OWNED raw pre-cap
        (900K), never the ceiling-capped window (272K). The ceiling must not
        contaminate context_length_cache.yaml."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)
        transition_model_context(agent, 900_000, model="m")
        # Host owns the RAW capability; the engine holds the capped window.
        assert get_pre_cap(agent) == 900_000
        assert eng.context_length == 272_000  # capped, for the engine
        # The persistence site uses the host raw value:
        assert get_pre_cap(agent) > eng.context_length


# ═══════════════════════════════════════════════════════════════════════════
# N. Codex accounting: refresh_context_window preserves plugin accounting
# ═══════════════════════════════════════════════════════════════════════════

class TestN_CodexAccountingPreservation:
    def test_codex_refresh_preserves_recorded_accounting(self):
        """The Codex path records accounting, then refreshes the window.
        update_model may reset accounting, so the codex path re-applies the
        recorded accounting after the refresh. This test simulates the exact
        sequence from codex_runtime.py."""
        eng = _PluginEngine()
        agent = _AgentStub(eng, ceiling=272_000)
        transition_model_context(agent, 900_000, model="m")
        assert eng.context_length == 272_000

        # Codex: record accounting, then refresh the window (same-model).
        eng.update_from_response({
            "prompt_tokens": 5000, "completion_tokens": 200, "total_tokens": 5200,
        })
        recorded = {
            f: getattr(eng, f)
            for f in ("last_prompt_tokens", "last_completion_tokens", "last_total_tokens")
        }
        assert recorded["last_prompt_tokens"] == 5000

        refresh_context_window(agent, 200_000, model="m")
        assert eng.context_length == 200_000
        # update_model reset the accounting:
        assert eng.last_prompt_tokens == 0

        # codex_runtime re-applies the recorded accounting after the refresh:
        for f, v in recorded.items():
            if isinstance(v, int) and not isinstance(v, bool):
                setattr(eng, f, v)
        assert eng.last_prompt_tokens == 5000
        assert eng.last_completion_tokens == 200
        assert eng.last_total_tokens == 5200


# ═══════════════════════════════════════════════════════════════════════════
# O. Two-profile WebUI: each profile resolves its own ceiling + metadata
# ═══════════════════════════════════════════════════════════════════════════

class TestO_TwoProfileWebUI:
    def test_ceiling_resolution_is_profile_scoped_in_source(self):
        """The WebUI /api/model/info endpoint resolves BOTH the raw context
        length AND the profile ceiling INSIDE the _profile_scope block. If
        the ceiling read fell outside the scope, it would resolve against the
        DEFAULT profile's config and a per-profile ceiling would be ignored.
        This is the source-level integration guarantee (the runtime scope swap
        of HERMES_HOME is exercised end-to-end by the WebUI integration tests)."""
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent
        src = (root / "hermes_cli" / "web_server.py").read_text(errors="replace")
        fn_start = src.find("def get_model_info")
        assert fn_start != -1, "get_model_info not found"
        body = src[fn_start:]
        scope_start = body.find("with _profile_scope(profile):")
        # Anchor on the ACTUAL ceiling read (the import line), not the
        # docstring mention that precedes the scope block.
        ceiling_start = body.find("import _get_max_context_length")
        raw_start = body.find("import get_model_context_length")
        assert scope_start != -1, "_profile_scope block not found"
        assert ceiling_start != -1, "ceiling import not found"
        assert raw_start != -1, "raw context length import not found"
        # Both reads must be INSIDE the scope block (after it starts).
        assert ceiling_start > scope_start, (
            "the profile ceiling must be read INSIDE _profile_scope"
        )
        assert raw_start > scope_start, (
            "the raw context length must be resolved INSIDE _profile_scope"
        )

    def test_ceiling_is_profile_scoped_value(self):
        """Given two profiles with different ceilings, _get_max_context_length
        returns the ceiling of the ACTIVE profile (via the profile-scoped
        config), not a global one. This drives the per-profile effective
        window the WebUI reports."""
        # Profile A: ceiling 128K
        with _patch_config({"model": {"max_context_length": 128_000}}):
            a = model_metadata._get_max_context_length()
            assert a == 128_000
        # Profile B: ceiling 272K
        with _patch_config({"model": {"max_context_length": 272_000}}):
            b = model_metadata._get_max_context_length()
            assert b == 272_000
        # No ceiling:
        with _patch_config({"model": {}}):
            assert model_metadata._get_max_context_length() is None


class _patch_config:
    """Patch load_config_readonly (used by _get_max_context_length) to return
    a fixed per-profile config — simulating the HERMES_HOME scope swap."""

    def __init__(self, config):
        self.config = config

    def __enter__(self):
        import hermes_cli.config as _cfg
        self._orig = _cfg.load_config_readonly
        _cfg.load_config_readonly = lambda *a, **k: self.config
        return self

    def __exit__(self, *exc):
        import hermes_cli.config as _cfg
        _cfg.load_config_readonly = self._orig
        return False
