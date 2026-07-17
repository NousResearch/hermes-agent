"""Plugin-facing contract for background compression preparation.

Backwards compatibility is the core requirement: the new ContextEngine hooks
(``can_prepare_compression`` / ``prepare_compression`` / ``adopt_prepared_state``)
must default to no-ops so existing third-party engines keep working unchanged,
while the built-in ContextCompressor gains a real implementation that runs on a
DEDICATED clone — ``compress()`` mutates counters and iterative-summary state,
so the live instance must never be driven by a background worker.
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

from agent.async_context_compression import (
    BackgroundCompressionConfig,
    PreparedCompressionCandidate,
    PrepareResult,
    maybe_prepare_background_compression,
)
from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor
from agent.context_engine import ContextEngine


class MinimalEngine(ContextEngine):
    """A plugin engine written before the async-compression hooks existed."""

    @property
    def name(self):
        return "minimal"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages


def _make_compressor(**overrides) -> ContextCompressor:
    kwargs = dict(model="test/model", quiet_mode=True, config_context_length=200_000)
    kwargs.update(overrides)
    return ContextCompressor(**kwargs)


def _make_candidate(prepared, **overrides) -> PreparedCompressionCandidate:
    fields = dict(
        session_id="sess-1",
        generation=1,
        prefix_message_count=10,
        prefix_digest="0" * 64,
        prepared_messages=tuple(prepared),
        source_prompt_tokens=180_000,
        created_at_monotonic=0.0,
        created_at_turn=1,
        used_fallback=False,
        summary_error=None,
    )
    fields.update(overrides)
    return PreparedCompressionCandidate(**fields)


class TestContextEngineDefaultHooks:
    def test_defaults_are_noop(self):
        engine = MinimalEngine()
        msgs = [{"role": "user", "content": "hi"}]
        assert engine.can_prepare_compression(msgs) is False
        assert engine.can_prepare_compression(msgs, current_tokens=100_000) is False
        assert engine.prepare_compression(msgs) is None
        assert engine.prepare_compression(
            msgs, current_tokens=100_000, focus_topic="x"
        ) is None
        candidate = _make_candidate(
            [{"role": "user", "content": f"{SUMMARY_PREFIX}\nsummary"}]
        )
        assert engine.adopt_prepared_state(candidate) is None

    def test_legacy_engine_disables_background_preparation(self):
        """An engine that never opted in must keep the feature inert even when
        the config block enables it."""
        agent = SimpleNamespace(
            background_compression_config=BackgroundCompressionConfig.from_dict(
                {"enabled": True, "shadow_only": True, "min_delta_tokens": 0,
                 "min_frozen_messages": 2}
            ),
            background_compression=None,
            context_compressor=MinimalEngine(),
            session_id="sess-1",
            compression_enabled=True,
        )
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        assert maybe_prepare_background_compression(
            agent, msgs, current_tokens=250_000, current_turn=1
        ) is False
        assert agent.background_compression is None


class TestBuiltinCompressorPreparation:
    def test_builtin_compressor_opts_in(self):
        comp = _make_compressor()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
        assert comp.can_prepare_compression(msgs, current_tokens=150_000) is True

    def test_builtin_compressor_declines_under_active_cooldown(self):
        comp = _make_compressor()
        comp._record_compression_failure_cooldown(60.0, "summary failed")
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
        assert comp.can_prepare_compression(msgs, current_tokens=150_000) is False

    def test_clone_for_background_copies_config_and_summary_state(self):
        comp = _make_compressor()
        comp._previous_summary = "PREV SUMMARY"
        comp.threshold_tokens = 123_456
        comp.protect_first_n = 2
        comp.protect_last_n = 7
        comp.tail_token_budget = 999

        clone = comp.clone_for_background()
        assert clone is not comp
        assert clone._previous_summary == "PREV SUMMARY"
        assert clone.threshold_tokens == 123_456
        assert clone.protect_first_n == 2
        assert clone.protect_last_n == 7
        assert clone.tail_token_budget == 999
        assert clone.context_length == comp.context_length

        # Mutating the clone must never reach the live compressor.
        clone._previous_summary = "MUTATED"
        clone.compression_count = 99
        clone._ineffective_compression_count = 5
        assert comp._previous_summary == "PREV SUMMARY"
        assert comp.compression_count == 0

    def test_prepare_compression_runs_on_clone_and_never_mutates_live_state(self):
        comp = _make_compressor()
        # Shrink the protected head/tail so a small transcript has a real
        # compressable middle window.
        comp.protect_first_n = 1
        comp.protect_last_n = 2
        comp.tail_token_budget = 80

        msgs = [{"role": "system", "content": "sys"}]
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"turn {i}: " + ("x" * 400)})
        snapshot = copy.deepcopy(msgs)

        # No aux provider: the clone's compress() falls back to the static
        # summary and still compacts the middle window.
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=RuntimeError("no provider"),
        ):
            prepared = comp.prepare_compression(msgs, current_tokens=150_000)

        assert isinstance(prepared, PrepareResult)
        # No aux provider means the deterministic fallback summary was used —
        # surfaced as candidate metadata, not hidden.
        assert prepared.used_fallback is True
        assert len(prepared.messages) < len(msgs)
        # The live input list and its dicts are untouched.
        assert msgs == snapshot
        # The live compressor never ran compress(): every mutable field is
        # still pristine.
        assert comp.compression_count == 0
        assert comp._previous_summary is None
        assert comp._last_summary_fallback_used is False
        assert comp._last_compress_aborted is False
        assert comp._ineffective_compression_count == 0
        assert comp.awaiting_real_usage_after_compression is False

    def test_adopt_prepared_state_updates_live_counters(self):
        comp = _make_compressor()
        summary_body = "key facts preserved across compaction"
        candidate = _make_candidate(
            [
                {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
                {"role": "user", "content": "tail message"},
            ]
        )
        comp.adopt_prepared_state(candidate)
        assert comp.compression_count == 1
        assert summary_body in (comp._previous_summary or "")
        # Mirrors the synchronous post-compression bookkeeping: real usage is
        # awaited before the next threshold decision.
        assert comp.awaiting_real_usage_after_compression is True
        assert comp.last_prompt_tokens == -1
