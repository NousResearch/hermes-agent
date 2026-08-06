"""Regression for #23975: context compression must survive a mid-flight
gateway interrupt.

While the compression summary LLM call is in flight, an incoming gateway
message sets the thread interrupt flag. The Codex Responses aux stream polls
that flag and used to raise InterruptedError unconditionally — aborting the
summary, which then fell back to a degraded static "summary unavailable"
marker (losing the real handoff). Compression now runs its summary call
under aux_interrupt_protection(), so the interrupt poll is masked for the
compression task only (timeouts and other aux tasks stay interruptible).
"""

from __future__ import annotations

from unittest.mock import patch

import agent.auxiliary_client as aux


class TestAuxInterruptProtection:
    def test_protected_flag_defaults_false(self):
        # Fresh thread-local state.
        assert aux._aux_interrupt_protected() is False

    def test_context_manager_sets_and_restores(self):
        assert aux._aux_interrupt_protected() is False
        with aux.aux_interrupt_protection():
            assert aux._aux_interrupt_protected() is True
        assert aux._aux_interrupt_protected() is False

    def test_context_manager_is_reentrant(self):
        with aux.aux_interrupt_protection():
            assert aux._aux_interrupt_protected() is True
            with aux.aux_interrupt_protection():
                assert aux._aux_interrupt_protected() is True
            # inner exit must NOT clear protection while still inside outer
            assert aux._aux_interrupt_protected() is True
        assert aux._aux_interrupt_protected() is False

    def test_restores_on_exception(self):
        try:
            with aux.aux_interrupt_protection():
                raise ValueError("boom")
        except ValueError:
            pass
        assert aux._aux_interrupt_protected() is False

    def test_explicit_inactive_is_noop(self):
        with aux.aux_interrupt_protection(active=False):
            assert aux._aux_interrupt_protected() is False


class TestCompressionProtectsSummaryCall:
    """The compressor must wrap its summary call_llm in aux_interrupt_protection
    so a mid-flight interrupt doesn't abort it (#23975)."""

    def test_compressor_call_site_uses_protection(self):
        # The summary call must run inside aux_interrupt_protection. We assert
        # the protection flag is ACTIVE at the moment call_llm is invoked.
        from agent.context_compressor import ContextCompressor

        seen = {}

        class _Resp:
            class _Choice:
                class _Msg:
                    content = "[CONTEXT SUMMARY]: ok"
                message = _Msg()
            choices = [_Choice()]

        def fake_call_llm(**kwargs):
            # Capture whether protection was active during the call.
            seen["protected"] = aux._aux_interrupt_protected()
            seen["task"] = kwargs.get("task")
            return _Resp()

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        msgs = [
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "done"},
        ]
        with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
            summary = c._generate_summary(msgs)

        assert summary is not None
        assert seen.get("task") == "compression"
        assert seen.get("protected") is True, (
            "compression summary call must run under aux_interrupt_protection"
        )
        # Protection must be cleared after the call returns.
        assert aux._aux_interrupt_protected() is False
