"""Context-fill warning (compression.warning_*) + pre_llm_call usage payload.

Covers:
  - config defaults expose the warning keys under ``compression``
  - the warning fires at/above the threshold, not below
  - the warning fires once per compression cycle and re-arms after a
    compaction (compression_count bump + last_prompt_tokens sentinel)
  - reset_session_state() re-arms the warning for a fresh session
  - the enriched pre_llm_call payload stays backward compatible with
    hooks written before the context-usage fields existed
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Stub out optional heavy dependencies not installed in the test environment
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.conversation_loop import (
    _DEFAULT_CONTEXT_WARNING_MESSAGE,
    _context_usage_snapshot,
    _maybe_emit_context_fill_warning,
)


def _make_agent(
    tokens=0,
    length=200_000,
    compressions=0,
    enabled=True,
    threshold=0.7,
    message="",
    with_engine=True,
):
    """Lightweight stand-in for AIAgent with just the attrs the helpers read."""
    statuses = []
    engine = None
    if with_engine:
        engine = SimpleNamespace(
            last_prompt_tokens=tokens,
            context_length=length,
            compression_count=compressions,
            threshold_percent=0.8,
        )
    agent = SimpleNamespace(
        context_compressor=engine,
        _context_warning_enabled=enabled,
        _context_warning_threshold=threshold,
        _context_warning_message=message,
        _context_fill_warning_cycle=None,
        _emit_status=lambda msg: statuses.append(msg),
    )
    agent._statuses = statuses
    return agent


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_warning_keys_present_under_compression(self):
        from hermes_cli.config import DEFAULT_CONFIG

        comp = DEFAULT_CONFIG["compression"]
        assert comp["warning_enabled"] is True
        assert comp["warning_threshold"] == 0.7
        assert (
            comp["warning_message"]
            == "Context is getting high. Save important progress before continuing."
        )

    def test_default_message_constant_matches_config(self):
        """The in-code fallback and the config default must not drift apart."""
        from hermes_cli.config import DEFAULT_CONFIG

        assert (
            _DEFAULT_CONTEXT_WARNING_MESSAGE
            == DEFAULT_CONFIG["compression"]["warning_message"]
        )

    def test_compression_trigger_keys_unchanged(self):
        """Adding warning keys must not disturb the compression trigger keys."""
        from hermes_cli.config import DEFAULT_CONFIG

        comp = DEFAULT_CONFIG["compression"]
        assert comp["enabled"] is True
        assert 0 < comp["threshold"] <= 1


# ---------------------------------------------------------------------------
# _context_usage_snapshot (pre_llm_call payload fields)
# ---------------------------------------------------------------------------

class TestContextUsageSnapshot:
    def test_fields_from_engine(self):
        agent = _make_agent(tokens=50_000, length=200_000, compressions=2)
        snap = _context_usage_snapshot(agent)
        assert snap == {
            "context_tokens": 50_000,
            "context_length": 200_000,
            "context_percent": 25.0,
            "compression_count": 2,
            "compression_threshold": 0.8,
            "context_warning_threshold": 0.7,
        }

    def test_post_compression_sentinel_clamped(self):
        """last_prompt_tokens is parked at -1 right after a compaction."""
        agent = _make_agent(tokens=-1, length=200_000)
        snap = _context_usage_snapshot(agent)
        assert snap["context_tokens"] == 0
        assert snap["context_percent"] == 0.0

    def test_no_engine_is_safe(self):
        agent = _make_agent(with_engine=False)
        snap = _context_usage_snapshot(agent)
        assert snap["context_tokens"] == 0
        assert snap["context_length"] == 0
        assert snap["context_percent"] == 0.0
        assert snap["compression_count"] == 0
        assert snap["compression_threshold"] is None

    def test_zero_context_length_no_division(self):
        agent = _make_agent(tokens=10_000, length=0)
        assert _context_usage_snapshot(agent)["context_percent"] == 0.0


# ---------------------------------------------------------------------------
# _maybe_emit_context_fill_warning
# ---------------------------------------------------------------------------

class TestContextFillWarning:
    def test_fires_at_exact_threshold(self):
        agent = _make_agent(tokens=140_000, length=200_000)  # exactly 70%
        assert _maybe_emit_context_fill_warning(agent) is True
        assert len(agent._statuses) == 1
        assert agent._statuses[0].startswith("⚠️ CONTEXT 70% FULL\n\n")
        assert _DEFAULT_CONTEXT_WARNING_MESSAGE in agent._statuses[0]

    def test_fires_above_threshold(self):
        agent = _make_agent(tokens=180_000, length=200_000)  # 90%
        assert _maybe_emit_context_fill_warning(agent) is True

    def test_does_not_fire_below_threshold(self):
        agent = _make_agent(tokens=139_999, length=200_000)  # just under 70%
        assert _maybe_emit_context_fill_warning(agent) is False
        assert agent._statuses == []
        assert agent._context_fill_warning_cycle is None

    def test_fires_once_per_cycle(self):
        agent = _make_agent(tokens=150_000, length=200_000)
        assert _maybe_emit_context_fill_warning(agent) is True
        # Same cycle, even higher usage — must stay quiet.
        agent.context_compressor.last_prompt_tokens = 158_000
        assert _maybe_emit_context_fill_warning(agent) is False
        assert len(agent._statuses) == 1

    def test_rearms_after_compression(self):
        agent = _make_agent(tokens=150_000, length=200_000, compressions=0)
        assert _maybe_emit_context_fill_warning(agent) is True

        # Simulate what compress_context() does: bump the cycle counter and
        # park last_prompt_tokens at the -1 sentinel until real usage arrives.
        agent.context_compressor.compression_count = 1
        agent.context_compressor.last_prompt_tokens = -1
        assert _maybe_emit_context_fill_warning(agent) is False  # 0% — fresh cycle

        # Context builds back up past the threshold in the new cycle.
        agent.context_compressor.last_prompt_tokens = 150_000
        assert _maybe_emit_context_fill_warning(agent) is True
        assert len(agent._statuses) == 2

    def test_disabled_never_fires(self):
        agent = _make_agent(tokens=190_000, length=200_000, enabled=False)
        assert _maybe_emit_context_fill_warning(agent) is False
        assert agent._statuses == []

    def test_no_engine_is_safe(self):
        agent = _make_agent(tokens=190_000, with_engine=False)
        assert _maybe_emit_context_fill_warning(agent) is False

    def test_zero_context_length_never_fires(self):
        agent = _make_agent(tokens=190_000, length=0)
        assert _maybe_emit_context_fill_warning(agent) is False

    def test_nonpositive_threshold_never_fires(self):
        agent = _make_agent(tokens=190_000, length=200_000, threshold=0)
        assert _maybe_emit_context_fill_warning(agent) is False

    def test_custom_message_used(self):
        agent = _make_agent(
            tokens=150_000, length=200_000, message="Save your work now."
        )
        assert _maybe_emit_context_fill_warning(agent) is True
        assert "Save your work now." in agent._statuses[0]

    def test_empty_message_falls_back_to_default(self):
        agent = _make_agent(tokens=150_000, length=200_000, message="   ")
        assert _maybe_emit_context_fill_warning(agent) is True
        assert _DEFAULT_CONTEXT_WARNING_MESSAGE in agent._statuses[0]

    def test_works_with_real_compressor_attrs(self):
        """Guard the attribute coupling against ContextCompressor renames."""
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model="test", quiet_mode=True, config_context_length=200_000
        )
        compressor.last_prompt_tokens = 150_000  # 75% of 200K
        agent = _make_agent()
        agent.context_compressor = compressor

        snap = _context_usage_snapshot(agent)
        assert snap["context_tokens"] == 150_000
        assert snap["context_length"] == 200_000
        assert snap["compression_threshold"] == compressor.threshold_percent

        assert _maybe_emit_context_fill_warning(agent) is True
        assert len(agent._statuses) == 1


# ---------------------------------------------------------------------------
# reset_session_state re-arms the warning
# ---------------------------------------------------------------------------

class TestResetSessionStateRearm:
    def test_reset_clears_fired_cycle(self):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)  # skip __init__ entirely
        # Seed the attributes reset_session_state() writes
        agent.session_total_tokens = 0
        agent.session_input_tokens = 0
        agent.session_output_tokens = 0
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_cache_read_tokens = 0
        agent.session_cache_write_tokens = 0
        agent.session_reasoning_tokens = 0
        agent.session_api_calls = 0
        agent.session_estimated_cost_usd = 0.0
        agent.session_cost_status = "unknown"
        agent.session_cost_source = "none"
        agent._user_turn_count = 3
        agent.context_compressor = None

        agent._context_fill_warning_cycle = 2
        agent.reset_session_state()
        assert agent._context_fill_warning_cycle is None


# ---------------------------------------------------------------------------
# pre_llm_call payload backward compatibility
# ---------------------------------------------------------------------------

def _enriched_payload(agent):
    """The exact kwargs shape run_conversation() now sends to pre_llm_call."""
    return dict(
        session_id="s1",
        task_id="t1",
        turn_id="turn-1",
        user_message="hi",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
        sender_id="",
        **_context_usage_snapshot(agent),
    )


class TestPreLlmCallPayloadBackCompat:
    def test_legacy_named_signature_still_works(self):
        """Hooks written against the documented pre-enrichment signature
        (named params + **kwargs) must keep working unchanged."""
        from hermes_cli.plugins import PluginManager

        def legacy_hook(
            session_id, user_message, conversation_history,
            is_first_turn, model, platform, **kwargs,
        ):
            return {"context": f"recalled for {session_id}"}

        mgr = PluginManager()
        mgr._hooks.setdefault("pre_llm_call", []).append(legacy_hook)

        agent = _make_agent(tokens=50_000, length=200_000)
        results = mgr.invoke_hook("pre_llm_call", **_enriched_payload(agent))
        assert results == [{"context": "recalled for s1"}]

    def test_new_fields_reach_kwargs_only_hooks(self):
        from hermes_cli.plugins import PluginManager

        seen = {}

        def observer(**kwargs):
            seen.update(kwargs)
            return None

        mgr = PluginManager()
        mgr._hooks.setdefault("pre_llm_call", []).append(observer)

        agent = _make_agent(tokens=150_000, length=200_000, compressions=1)
        mgr.invoke_hook("pre_llm_call", **_enriched_payload(agent))

        assert seen["context_tokens"] == 150_000
        assert seen["context_length"] == 200_000
        assert seen["context_percent"] == 75.0
        assert seen["compression_count"] == 1
        assert seen["compression_threshold"] == 0.8
        assert seen["context_warning_threshold"] == 0.7
        # Pre-existing fields are untouched
        assert seen["user_message"] == "hi"
        assert seen["is_first_turn"] is True
