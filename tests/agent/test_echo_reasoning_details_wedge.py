"""Regression: 2026-10-01 Nous echo-route ``reasoning_details`` wedge (agent-org ADR).

1. Estimate fidelity: the compaction TRIGGER shadow dropped ``reasoning_details`` even
   though the chat-completions transport replays the array verbatim on openrouter.ai /
   nousresearch.com routes. A ~350K request read as ~46K and 413/400 recovery armed
   against a fiction. The trigger must charge what the wire ships.
2. Recovery: the opaque 400 "This request is not valid - please check the model name"
   NAMES THE MODEL, not the payload — it previously misread as a model-not-found wedge.
   It now classifies as ``echoed_reasoning_rejected`` and strip-retries exactly once,
   against ``api_messages`` (wire copy) only, never canonical ``messages``.
3. Single predicate: the transport sanitizer and the estimator consume the SAME
   ``reasoning_details_reaches_wire`` decision — the #84371 lesson is that two hand-kept
   copies of a wire-truth policy drift apart and the drift direction decides who loops.
"""

import json
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.message_sanitization import reasoning_details_reaches_wire
from agent.model_metadata import (
    _wire_message_shadow,
    estimate_request_tokens_rough,
    estimate_tokens_rough,
)
from agent.transports.chat_completions import _route_replays_reasoning_details
from agent.turn_recovery import _recover_format_errors
from agent.turn_retry_state import TurnRetryState


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError (mirrors tests/agent/test_error_classifier.py)."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


RD_TEXT = "thinking about the answer carefully " * 500  # ~17K chars, ~4K+ tok
RD_ARRAY = [
    {"type": "reasoning.text", "text": RD_TEXT, "format": "unknown", "id": "sha256:abc123"},
    {"type": "reasoning.encrypted", "data": "Zm9vYmFyZW5jcnlwdGVkcmVhc29uMTIz" * 40},
]


def _assistant_with_rd(content="The answer is 42.", details=None):
    return {
        "role": "assistant",
        "content": content,
        "reasoning_details": RD_ARRAY if details is None else details,
    }


# ── 3. one shared wire-truth predicate ────────────────────────────────


class TestWireTruthPredicate:
    @pytest.mark.parametrize("url", [
        "https://portal.nousresearch.com/api/v1",
        "https://openrouter.ai/api/v1",
    ])
    def test_echo_hosts_true(self, url):
        assert reasoning_details_reaches_wire("chat_completions", url) is True

    @pytest.mark.parametrize("url", [
        "https://api.openai.com/v1",
        "http://localhost:1234/v1",
        "https://example.invalid/v1",
        "",
        None,
    ])
    def test_strict_hosts_false(self, url):
        assert reasoning_details_reaches_wire("chat_completions", url) is False

    def test_codex_responses_never_replays_the_array(self):
        assert reasoning_details_reaches_wire("codex_responses", "https://openrouter.ai/api/v1") is False

    @pytest.mark.parametrize("url", [
        "https://portal.nousresearch.com/api/v1",
        "https://openrouter.ai/api/v1",
        "https://api.openai.com/v1",
        "http://localhost:1234/v1",
    ])
    def test_transport_sanitizer_delegates_to_same_predicate(self, url):
        """Estimator and sanitizer cannot drift: the transport defers to the shared predicate."""
        assert _route_replays_reasoning_details(url) is reasoning_details_reaches_wire("chat_completions", url)


# ── 1. the trigger charges what the wire ships ────────────────────────


class TestShadowChargesEchoReasoning:
    def test_default_shadow_drops_reasoning_details(self):
        """Byte-blind default preserved for non-echo routes (never charge a stripped field)."""
        msg = _assistant_with_rd()
        assert "reasoning_details" not in _wire_message_shadow(msg)

    def test_echo_shadow_keeps_reasoning_details(self):
        msg = _assistant_with_rd()
        shadow = _wire_message_shadow(msg, charge_echo_reasoning=True)
        assert shadow["reasoning_details"] == RD_ARRAY

    def test_estimate_covers_real_wire_payload_floor(self):
        """ADR acceptance: pre-compression estimate >= estimate of the actual wire payload."""
        msg = _assistant_with_rd()
        wire_est = estimate_tokens_rough(json.dumps([msg]))
        est_default = estimate_request_tokens_rough([msg])
        est_echo = estimate_request_tokens_rough([msg], charge_echo_reasoning=True)
        assert est_echo >= wire_est, (est_echo, wire_est)
        assert est_default < wire_est  # the wedge: blind estimate undershoots the real send
        assert est_echo - est_default > 3000  # the array is ~5K chars; it must actually be charged

    def test_memo_cache_isolation_both_orders(self):
        """Flagged rows are keyed apart: default memo can never answer for the echo row."""
        msg = _assistant_with_rd(content="iso-probe")
        e1 = estimate_request_tokens_rough([msg], charge_echo_reasoning=True)
        e2 = estimate_request_tokens_rough([msg])
        e3 = estimate_request_tokens_rough([msg], charge_echo_reasoning=True)
        e4 = estimate_request_tokens_rough([msg])
        assert e1 == e3 and e2 == e4 and e1 > e2
        msg_b = _assistant_with_rd(content="iso-probe-b")
        f1 = estimate_request_tokens_rough([msg_b])
        f2 = estimate_request_tokens_rough([msg_b], charge_echo_reasoning=True)
        assert f1 == e2 and f2 == e1  # order-independent: no cache poisoning across flags

    def test_stale_strip_and_echo_flags_compose(self):
        """Independent keys: stale-thinking strip must not erase the echo charge."""
        older = dict(_assistant_with_rd(content="old turn"), reasoning="stale thinking text " * 200)
        newest = {"role": "assistant", "content": "newest", "reasoning_content": "live"}
        both_off = estimate_request_tokens_rough([older, newest], charge_stale_thinking=False)
        echo_on = estimate_request_tokens_rough(
            [older, newest], charge_stale_thinking=False, charge_echo_reasoning=True
        )
        assert echo_on > both_off

    def test_seam_compatibility_default_call_shape(self):
        """A (messages)-only monkeypatch seam must never see the echo kwarg (default False).
        Pre-existing contract: the charge_stale_thinking=False path already passes its kwarg,
        so the seam mirrors that; the NEW guarantee under test is that echo kwarg only appears
        when True."""
        seen = []

        def seam(messages, **kwargs):
            seen.append(kwargs)
            return 1

        import agent.model_metadata as mm

        real = mm.estimate_messages_tokens_rough
        mm.estimate_messages_tokens_rough = seam
        try:
            assert estimate_request_tokens_rough([{"role": "user", "content": "hi"}]) == 1
            assert seen[-1] == {}  # default path: no kwargs at all
            assert estimate_request_tokens_rough(
                [{"role": "user", "content": "hi"}], charge_stale_thinking=False
            ) == 1
            assert seen[-1] == {"charge_stale_thinking": False}  # unchanged pre-existing kwargs
            assert estimate_request_tokens_rough(
                [{"role": "user", "content": "hi"}], charge_echo_reasoning=True
            ) == 1
            assert seen[-1] == {"charge_echo_reasoning": True}
        finally:
            mm.estimate_messages_tokens_rough = real


# ── 2. the opaque 400 classifies as replay rejection, not model error ─


class TestOpaqueEcho400Classification:
    _NOUS_MSG = "This request is not valid - please check the model name"
    # Probe-verified real wording from the t_3a58fd44 harness (results.jsonl): the wedge
    # envelope DOES carry an "Additional info:" tail — the specific-cause deferral is by
    # classifier ORDER (reasoning-mandatory rung precedes ours), never by wording guards.
    _INCIDENT_MSG = ("This request is not valid. Check the model name and other parameters. "
                     "Additional info: Provider returned error")

    def _err(self, message=_NOUS_MSG):
        return MockAPIError(
            f"Error code: 400 - {message}",
            status_code=400,
            body={"error": {"type": "invalid_request_error", "message": message}},
        )

    def test_opaque_wedge_classifies_as_echoed_reasoning(self):
        r = classify_api_error(
            self._err(), provider="nous-portal", model="claude-opus-4-8",
            base_url="https://portal.nousresearch.com/api/v1",
        )
        assert r.reason == FailoverReason.echoed_reasoning_rejected
        assert r.retryable is True and r.should_fallback is False

    def test_probe_verified_incident_wording(self):
        """Exact envelope that wedged the 09-30 session, from the probe ledger."""
        r = classify_api_error(
            self._err(self._INCIDENT_MSG), provider="nous", model="qwen/qwen3.8-flash",
            base_url="https://portal.nousresearch.com/api/v1",
        )
        assert r.reason == FailoverReason.echoed_reasoning_rejected

    def test_single_marker_does_not_trigger(self):
        r = classify_api_error(
            self._err("Please check the model name again"), provider="nous-portal", model="x",
        )
        assert r.reason != FailoverReason.echoed_reasoning_rejected
        # Wording that carries only one marker must land wherever it landed before
        # this rule existed, never on the echo rung.

    def test_genuine_model_not_found_still_not_echo_rejection(self):
        r = classify_api_error(
            self._err("The model `claude-opus-99` does not exist or you do not have access to it."),
            provider="openrouter", model="claude-opus-99",
        )
        assert r.reason != FailoverReason.echoed_reasoning_rejected

    def test_wrapped_specific_cause_not_shadowed(self):
        """Nous wraps specific 400 causes in the same opaque envelope ("...Additional info:
        Reasoning is mandatory..."). The echo rule sits AFTER the reasoning-cause rung in
        _classify_400 — precedence by order, not by wording guards — so specific causes keep
        their reason and only the generic pass-through tail ("Provider returned error")
        reaches the replay strip-retry (which would be pointless for a named cause)."""
        r = classify_api_error(
            MockAPIError(
                "Error code: 400 - This request is not valid. Check the model name and other "
                "parameters. Additional info: Reasoning is mandatory for this endpoint and "
                "cannot be disabled.",
                status_code=400,
            ),
            provider="nous", model="z-ai/glm-5.3-flash",
        )
        assert r.reason == FailoverReason.reasoning_mandatory


# ── 2b. strip-retry: once, wire copy only, no-strip-no-retry ─────────


class TestEchoedReasoningStripRetry:
    @staticmethod
    def _agent():
        return SimpleNamespace(log_prefix="", api_mode="chat_completions",
                               _vprint=lambda *a, **k: None)

    @staticmethod
    def _classified():
        return SimpleNamespace(reason=FailoverReason.echoed_reasoning_rejected)

    def test_strips_wire_copy_once_canonical_untouched(self):
        canonical = [_assistant_with_rd()]
        api_messages = [_assistant_with_rd()]
        retry = TurnRetryState()
        assert _recover_format_errors(
            self._agent(), MockAPIError("400", 400), self._classified(), retry, canonical, api_messages
        ) is True
        assert "reasoning_details" not in api_messages[0]
        assert "reasoning_details" in canonical[0]  # state.db copy must survive
        assert retry.echoed_reasoning_retry_attempted is True

    def test_second_rejection_does_not_restrip_or_retry(self):
        canonical = [_assistant_with_rd()]
        api_messages = [_assistant_with_rd()]
        retry = TurnRetryState()
        assert _recover_format_errors(
            self._agent(), MockAPIError("400", 400), self._classified(), retry, canonical, api_messages
        ) is True
        # route 400s identically after the strip → one-shot guard holds, error surfaces
        assert _recover_format_errors(
            self._agent(), MockAPIError("400", 400), self._classified(), retry, canonical, api_messages
        ) is False

    def test_nothing_to_strip_returns_false(self):
        api_messages = [{"role": "assistant", "content": "no replay state here"}]
        retry = TurnRetryState()
        assert _recover_format_errors(
            self._agent(), MockAPIError("400", 400), self._classified(), retry, [], api_messages
        ) is False
        assert retry.echoed_reasoning_retry_attempted is True

    def test_thinking_signature_path_independent_guard(self):
        """The new guard must not consume the thinking-signature one-shot budget."""
        retry = TurnRetryState()
        api_messages = [_assistant_with_rd()]
        assert _recover_format_errors(
            self._agent(), MockAPIError("400", 400), SimpleNamespace(reason=FailoverReason.thinking_signature),
            retry, [], api_messages,
        ) is True
        assert retry.thinking_sig_retry_attempted and not retry.echoed_reasoning_retry_attempted
        # echoed strip still available (api copy already clean of that one message, but flag unused)
        api_messages2 = [_assistant_with_rd()]
        assert _recover_format_errors(
            self._agent(), MockAPIError("400", 400), self._classified(),
            retry, [], api_messages2,
        ) is True
        assert retry.echoed_reasoning_retry_attempted
