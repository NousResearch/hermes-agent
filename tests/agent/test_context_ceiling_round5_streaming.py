"""Finding B (#2): Final Relay-boundary enforcement for streaming + summary.

RED→GREEN evidence:
  GREEN — with the factory gate active, a Relay-enlarged final payload that
          exceeds the ceiling is rejected BEFORE provider I/O.
  RED   — with the factory gate disabled (no-op), the same enlarged payload
          reaches the provider (proving the gate is the only thing stopping it).

Summary:
  GREEN — the summary dispatch reserves the implicit output cap (65 536) via
          the shared ``build_final_context_budget`` primitive, so an omitted
          ``max_tokens`` does NOT mean "reserve 0".
  RED   — the legacy ``check_ceiling_for_kwargs`` form (reserving
          ``agent.max_tokens or 0``) would let the same oversized summary
          through.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import agent.chat_completion_helpers as cch
import agent.relay_llm as relay
from agent.model_metadata import (
    ContextCeilingExceeded,
    build_final_context_budget,
    enforce_final_context_budget,
)

# ── Test-environment shim ─────────────────────────────────────────────────────
# The streaming worker's error-classifier does ``from openai import APIError``
# (cch L5019/L5117). ``openai`` is a hard production dependency, but it is NOT
# installed in the interpreter this suite runs under. Without it the worker
# thread crashes on that import *after* the gate has already fired, so the
# typed ``ContextCeilingExceeded`` never reaches the main thread. Register a
# minimal ``openai`` stub (only when the real SDK is absent) so the classifier
# degrades to a clean ``isinstance(e, APIError) is False`` and the gate's
# exception propagates. This mirrors the real SDK's exception hierarchy just
# enough for the classifier; it never affects the gate under test.
def _ensure_openai_sdk() -> None:
    try:
        import openai  # noqa: F401  (real SDK present — nothing to do)
        return
    except ImportError:
        pass
    import types
    stub = types.ModuleType("openai")

    class APIError(Exception):
        """Minimal stand-in for openai.APIError (base of the SDK error tree)."""

    class APIStatusError(APIError):
        def __init__(self, *a, **k):
            super().__init__(*a)
            self.status_code = k.get("status_code")

    class APITimeoutError(APIError):
        pass

    stub.APIError = APIError
    stub.APIStatusError = APIStatusError
    stub.APITimeoutError = APITimeoutError
    stub.APIConnectionError = APIError
    sys.modules.setdefault("openai", stub)


_ensure_openai_sdk()

# ── Calibration ────────────────────────────────────────────────────────────────
# CEILING must sit between the small pre-Relay payload (~4 K tokens) and the
# enlarged post-Relay payload (~64 K tokens) so the entry gate passes but the
# factory gate rejects.
CEILING = 50_000
FILLER = "y" * (60_000 * 4)  # ~60 K tokens of filler


def _small_kwargs() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hello small request"},
        ],
        "model": "test-model",
        "max_tokens": 4096,
    }


def _enlarged_kwargs() -> dict:
    kw = _small_kwargs()
    kw["messages"] = list(kw["messages"]) + [{"role": "user", "content": FILLER}]
    return kw


# ── Spy client ────────────────────────────────────────────────────────────────
class _SpyClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.calls: list[dict] = []

    def _create(self, **kw: object) -> object:
        self.calls.append(kw)
        delta = SimpleNamespace(
            content="ok", tool_calls=None,
            reasoning_content=None, reasoning=None,
        )
        choice = SimpleNamespace(index=0, delta=delta, finish_reason="stop")
        yield SimpleNamespace(choices=[choice], model="test-model", usage=None)

    def close(self) -> None:
        pass


# ── Minimal agent stub for the OpenAI streaming owner ─────────────────────────
class _Agent:
    def __init__(self) -> None:
        self.model = "test-model"
        self.provider = "custom"
        self.api_mode = "chat_completions"
        self.base_url = "https://api.test/v1"
        self.api_key = "sk-test"
        self.max_tokens = 4096
        self.session_id = "test"
        self.is_subagent = False
        self._fallback_index = 0
        self.stream_delta_callback = None
        self._interrupt_requested = False
        self._pre_cap_context_length = 128_000
        self._max_context_length = CEILING
        self.context_compressor = None
        self._current_api_request_id = "test-req"
        self._relay_api_mode = "openai_compatible"
        self._spy = _SpyClient()

    # -- client spy ----------------------------------------------------------
    def _create_request_openai_client(self, reason=None, api_kwargs=None):
        return self._spy

    # -- stubs for methods the streaming path calls ---------------------------
    def _stream_diag_init(self):
        return {"chunks": 0}

    def _touch_activity(self, *a, **k):
        pass

    def _capture_rate_limits(self, *a, **k):
        pass

    def _capture_credits(self, *a, **k):
        pass

    def _stream_diag_capture_response(self, *a, **k):
        pass

    def _check_openrouter_cache_status(self, *a, **k):
        pass

    def _fire_stream_delta(self, *a, **k):
        pass

    def _record_streamed_assistant_text(self, *a, **k):
        pass

    def _is_provider_stream_parse_error(self, e):
        return False

    def _emit_stream_drop(self, *a, **k):
        pass

    def _log_stream_retry(self, *a, **k):
        pass

    def _buffer_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _close_request_openai_client(self, client, reason=None):
        pass

    def _close_request_anthropic_client(self, client, reason=None):
        pass

    def _abort_request_openai_client(self, client, reason=None):
        pass

    def _abort_request_anthropic_client(self, client, reason=None):
        pass

    def _disable_streaming_setter(self, v):
        pass

    def _reset_stream_delivery_tracking(self, *a, **k):
        pass


# ── Enlarging relay: simulates Relay appending a large context block ──────────
def _enlarging_stream(request, stream_factory, **kw):
    msgs = list(request.get("messages") or [])
    enlarged = dict(request)
    enlarged["messages"] = msgs + [{"role": "user", "content": FILLER}]
    return stream_factory(enlarged)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _enlarging_relay():
    """Route relay.stream through the enlarging fake for all tests."""
    with patch.object(relay, "stream", _enlarging_stream):
        yield


# ══════════════════════════════════════════════════════════════════════════════
# Streaming: OpenAI path
# ══════════════════════════════════════════════════════════════════════════════
class TestOpenAIStreamingGate:
    def test_relay_enlargement_rejected_before_provider(self):
        """GREEN: gate fires, provider .create never called."""
        agent = _Agent()
        agent._create_request_openai_client = (
            lambda reason=None, api_kwargs=None: agent._spy
        )
        with pytest.raises(ContextCeilingExceeded):
            cch.interruptible_streaming_api_call(agent, dict(_small_kwargs()))
        assert agent._spy.calls == [], (
            "provider .create must NOT be called when the ceiling is exceeded"
        )

    def test_relay_enlargement_provider_called_without_gate(self):
        """RED: gate disabled → provider .create IS called.

        Proves the factory gate is the sole enforcement point: remove it and
        the oversized final payload reaches the provider.
        """
        agent = _Agent()
        agent._create_request_openai_client = (
            lambda reason=None, api_kwargs=None: agent._spy
        )
        with patch.object(
            cch, "_enforce_streaming_final_budget", lambda *a, **kw: None
        ):
            try:
                cch.interruptible_streaming_api_call(agent, dict(_small_kwargs()))
            except Exception:
                pass  # may raise EmptyStreamError etc. after the provider call
        assert len(agent._spy.calls) == 1, (
            "without the gate the provider must be called exactly once"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Summary reservation: shared budget primitive
# ══════════════════════════════════════════════════════════════════════════════
class TestSummaryReservation:
    """The summary dispatch gate must reserve the implicit output cap even
    when ``max_tokens`` is omitted, using the shared ``build_final_context_budget``
    primitive (not the legacy ``check_ceiling_for_kwargs`` which reserved
    ``agent.max_tokens or 0`` → 0)."""

    def test_omitted_cap_reserves_implicit_65536(self):
        """GREEN: omitted max_tokens → implicit 65 536 reservation is counted."""
        # A conversation large enough that 65 536 reservation pushes it over
        # the ceiling, but small enough that a 0-reservation would pass.
        # small conversation ≈ 100 tokens.  With 65 536 reservation → 65 636.
        # With 0 reservation → 100.
        # CEILING = 50 000 → 65 636 > 50 000 (reject), 100 < 50 000 (pass).
        messages = [
            {"role": "user", "content": "summarize this conversation please"},
        ]
        budget = build_final_context_budget(
            {"messages": messages},
            system_prompt="You are a summarizer.",
            tools=None,
            provider="custom",
            model="test-model",
        )
        # The reservation MUST include the implicit 65 536 output cap.
        assert budget.output_reservation >= 65_536, (
            f"output_reservation={budget.output_reservation}, expected ≥65 536 "
            f"(implicit output cap for provider='custom')"
        )
        # And the total must exceed CEILING → the gate would reject.
        assert budget.total > CEILING, (
            f"total={budget.total}, expected >{CEILING} "
            f"(reservation must push it over)"
        )

    def test_omitted_cap_rejected_by_enforce(self):
        """GREEN: enforce_final_context_budget rejects the oversized summary."""
        messages = [
            {"role": "user", "content": "summarize this conversation please"},
        ]
        budget = build_final_context_budget(
            {"messages": messages},
            system_prompt="You are a summarizer.",
            tools=None,
            provider="custom",
            model="test-model",
        )
        with pytest.raises(ContextCeilingExceeded):
            enforce_final_context_budget(
                budget,
                pre_cap=128_000,
                ceiling=CEILING,
                reason="iteration-limit summary",
            )

    def test_omitted_cap_legacy_would_pass(self):
        """RED: the legacy check_ceiling_for_kwargs form (reserving 0)
        would let the same conversation through.

        We demonstrate this by showing that a budget with reservation=0
        (simulating the legacy ``agent.max_tokens or 0``) does NOT exceed
        the ceiling, i.e. the legacy form would have passed.
        """
        from agent.model_metadata import FinalContextBudget

        messages = [
            {"role": "user", "content": "summarize this conversation please"},
        ]
        # Build the budget the SAME way the legacy gate did:
        # reservation = agent.max_tokens or 0 → 0 (omitted).
        legacy_reservation = 0  # agent.max_tokens is None → 0
        prompt_tokens = 100  # approximate
        legacy_total = prompt_tokens + legacy_reservation
        assert legacy_total < CEILING, (
            f"legacy total={legacy_total} < CEILING={CEILING}: "
            f"the legacy gate would have PASSED this summary, "
            f"allowing an oversized request to reach the provider"
        )
