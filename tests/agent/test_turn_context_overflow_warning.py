"""Tests for the silent-context-overflow warning (the fix for the bug where a
session crosses the compression threshold but compression is blocked — by the
summary-LLM cooldown (#11529) or anti-thrashing (#40803) — and the model then
silently stops answering because nothing tells the user why.

The fix surfaces a deduped ``_emit_warning`` from ``build_turn_context`` and
exposes ``ContextCompressor.should_compress_info`` (a ``(bool, reason)`` tuple)
so callers can tell *why* compression was skipped while still over threshold.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.turn_context import build_turn_context
from tests.agent.test_turn_context import _FakeAgent


# ---------------------------------------------------------------------------
# Unit tests for ContextCompressor.should_compress_info
# ---------------------------------------------------------------------------

def _make_compressor(**kwargs) -> ContextCompressor:
    defaults = dict(
        model="test-model",
        threshold_percent=0.65,
        protect_first_n=2,
        protect_last_n=3,
        quiet_mode=True,
    )
    defaults.update(kwargs)
    # 96K context -> small-context floor raises threshold_percent to 0.75,
    # so threshold_tokens = 72_000. 73_000 is "over threshold".
    with patch("agent.context_compressor.get_model_context_length", return_value=96000):
        return ContextCompressor(**defaults)


class TestShouldCompressInfo:
    def test_below_threshold_is_clear(self):
        comp = _make_compressor()
        comp.last_prompt_tokens = 10_000
        should, reason = comp.should_compress_info(10_000)
        assert should is False
        assert reason is None

    def test_over_threshold_runs(self):
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        should, reason = comp.should_compress_info(73_000)
        assert should is True
        assert reason is None

    def test_cooldown_reports_reason(self):
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 60
        should, reason = comp.should_compress_info(73_000)
        assert should is False
        assert reason is not None
        assert reason.startswith("cooldown:")

    def test_cooldown_reason_has_seconds(self):
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 42
        _should, reason = comp.should_compress_info(73_000)
        assert reason == f"cooldown:{42:.0f}"

    def test_ineffective_reports_reason(self):
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._ineffective_compression_count = 2
        should, reason = comp.should_compress_info(73_000)
        assert should is False
        assert reason == "ineffective"

    def test_should_compress_bool_shim_unchanged(self):
        """should_compress() must still return a bare bool for existing
        callers in conversation_loop.py (and/or chains)."""
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 60
        result = comp.should_compress(73_000)
        assert result is False
        assert not isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Integration tests: build_turn_context surfaces the warning
# ---------------------------------------------------------------------------

class _WarnAgent(_FakeAgent):
    """_FakeAgent already covers the prologue; we just enable compression and
    record _emit_warning calls (the base class now has a MagicMock for it)."""

    def __init__(self):
        super().__init__()
        self.compression_enabled = True
        self._warnings = []
        self._compress_calls = 0
        # Replace the MagicMock with a recorder so we can assert contents.
        self._emit_warning = lambda message: self._warnings.append(message)

    def _compress_context(self, messages, *a, **k):
        self._compress_calls += 1
        return messages, "SYSTEM"


def _build_warn_agent(compressor: ContextCompressor) -> _WarnAgent:
    agent = _WarnAgent()
    agent.context_compressor = compressor
    return agent


def _run_build(agent):
    """Run build_turn_context with the prologue-side effects stubbed."""
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
         patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=999_999):
        return build_turn_context(
            agent=agent,
            user_message="hello",
            system_message=None,
            conversation_history=None,
            task_id=None,
            stream_callback=None,
            persist_user_message=None,
            restore_or_build_system_prompt=lambda *a, **k: None,
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda s: s,
            summarize_user_message_for_log=lambda s: s,
            set_session_context=lambda _sid: None,
            set_current_write_origin=lambda _o: None,
            ra=lambda: type("R", (), {"_set_interrupt": lambda *a, **k: None})(),
        )


class TestTurnContextOverflowWarning:
    def test_warns_on_cooldown_block(self):
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        agent = _build_warn_agent(comp)
        _run_build(agent)
        assert len(agent._warnings) == 1
        assert "over the compression threshold" in agent._warnings[0]
        assert "blocked (cooldown:" in agent._warnings[0]

    def test_warns_on_ineffective_block(self):
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._ineffective_compression_count = 2
        agent = _build_warn_agent(comp)
        _run_build(agent)
        assert len(agent._warnings) == 1
        assert "blocked (ineffective)" in agent._warnings[0]

    def test_no_warning_when_compression_runs(self):
        """When compression actually runs, no overflow warning is emitted."""
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000  # over threshold, no block
        agent = _build_warn_agent(comp)
        _run_build(agent)
        assert agent._warnings == []
        # compression was triggered instead
        assert agent._compress_calls > 0

    def test_dedup_does_not_spam(self):
        """Two turns with the same block kind fire the warning only once."""
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        agent = _build_warn_agent(comp)
        _run_build(agent)
        _run_build(agent)  # second turn, same cooldown kind
        assert len(agent._warnings) == 1

    def test_warning_refires_after_block_clears(self):
        """Once the block clears, a later block of the same kind warns again."""
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        agent = _build_warn_agent(comp)
        _run_build(agent)
        assert len(agent._warnings) == 1
        # Block clears: simulate the cooldown expiring.
        comp._summary_failure_cooldown_until = 0.0
        agent._last_ctx_overflow_warn = None
        # Re-arm the same block kind.
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        _run_build(agent)
        assert len(agent._warnings) == 2

    def test_warning_kind_switch_refires(self):
        """Switching block kind (cooldown -> ineffective) re-warns."""
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        agent = _build_warn_agent(comp)
        _run_build(agent)
        assert len(agent._warnings) == 1
        # Now ineffective instead of cooldown.
        comp._summary_failure_cooldown_until = 0.0
        comp._ineffective_compression_count = 2
        _run_build(agent)
        assert len(agent._warnings) == 2
        assert "blocked (ineffective)" in agent._warnings[1]

    def test_dedup_resets_when_block_clears_while_over_threshold(self):
        """The dedup reset must fire when the block clears while pressure is
        still high (the sweeper-review gap): execution enters the compression
        branch — not the ``else`` reset — so the reset must live on the
        compression path itself. No manual state clearing here; only the
        cooldown timer moves.
        """
        comp = _make_compressor()
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        agent = _build_warn_agent(comp)
        # Turn 1: over threshold + cooldown -> warn.
        _run_build(agent)
        assert len(agent._warnings) == 1
        # Turn 2: cooldown expires while STILL over threshold -> compression
        # branch runs; the reset must happen there (not in the else branch).
        comp._summary_failure_cooldown_until = 0.0
        _run_build(agent)
        assert agent._compress_calls > 0
        assert agent._last_ctx_overflow_warn is None
        # Turn 3: cooldown re-arms -> the warning must re-fire.
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        _run_build(agent)
        assert len(agent._warnings) == 2

    def test_no_warning_below_threshold_with_persisted_cooldown(self):
        """A live cooldown with the context BELOW threshold must not warn —
        there is no overflow to warn about (the cooldown branch is reached
        via the cheap preflight pre-check, which is not a threshold
        guarantee)."""
        comp = _make_compressor()
        comp.last_prompt_tokens = 10_000
        comp._summary_failure_cooldown_until = time.monotonic() + 30
        agent = _build_warn_agent(comp)
        with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
             patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
             patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000):
            build_turn_context(
                agent=agent,
                user_message="hello",
                system_message=None,
                conversation_history=None,
                task_id=None,
                stream_callback=None,
                persist_user_message=None,
                restore_or_build_system_prompt=lambda *a, **k: None,
                install_safe_stdio=lambda: None,
                sanitize_surrogates=lambda s: s,
                summarize_user_message_for_log=lambda s: s,
                set_session_context=lambda _sid: None,
                set_current_write_origin=lambda _o: None,
                ra=lambda: type("R", (), {"_set_interrupt": lambda *a, **k: None})(),
            )
        assert agent._warnings == []


# ---------------------------------------------------------------------------
# Plugin context engines: backward-compatible should_compress_info default
# ---------------------------------------------------------------------------

class TestPluginEngineDefault:
    def test_abc_default_returns_tuple(self):
        """Engines that only implement should_compress() get the tuple shape
        for free from the ContextEngine base class — the call sites in
        turn_context.py / conversation_loop.py must not raise
        AttributeError on plugin engines (sweeper review, #62625)."""
        from tests.run_agent.test_plugin_context_engine_init import _StubEngine

        engine = _StubEngine()
        result = engine.should_compress_info(123_456)
        assert result == (False, None)

    def test_abc_default_delegates_to_should_compress(self):
        from agent.context_engine import ContextEngine

        class _TrueEngine(ContextEngine):
            @property
            def name(self):
                return "true-stub"

            def update_from_response(self, usage):
                pass

            def should_compress(self, prompt_tokens=None):
                return True

            def compress(self, messages, current_tokens=None, focus_topic=None):
                return messages

        assert _TrueEngine().should_compress_info(1) == (True, None)


# ---------------------------------------------------------------------------
# Gateway noise filter: the warning is FAILURE-CLASS and must survive
# ---------------------------------------------------------------------------

class TestWarningSurvivesNoiseFilter:
    """The blocked-overflow warning is a deliberate carve-out from
    routine-compression silence (#16775 class). The gateway noise regex
    (#69550 just widened it) must NOT swallow it, or the fix is dead on
    every chat platform. Executes the REAL compiled regex — never eyeball
    a regex (noise-regex salvage rule).
    """

    def _emitted_warning(self, reason: str) -> str:
        from agent.conversation_compression import (
            CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
        )

        return CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
            tokens=85_000, threshold=72_000, reason=reason
        )

    def test_cooldown_warning_not_matched_by_noise_regex(self):
        from gateway.run import _TELEGRAM_NOISY_STATUS_RE

        assert not _TELEGRAM_NOISY_STATUS_RE.search(
            self._emitted_warning("cooldown:30")
        )

    def test_ineffective_warning_not_matched_by_noise_regex(self):
        from gateway.run import _TELEGRAM_NOISY_STATUS_RE

        assert not _TELEGRAM_NOISY_STATUS_RE.search(
            self._emitted_warning("ineffective")
        )

    def test_warning_delivered_on_chat_platform(self):
        """End-to-end through the fail-closed gateway status preparer."""
        from gateway.config import Platform
        from gateway.run import _prepare_gateway_status_message

        message = self._emitted_warning("cooldown:30")
        assert (
            _prepare_gateway_status_message(Platform.TELEGRAM, "warn", message)
            == message
        )
