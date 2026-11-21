"""A failed turn's provider diagnostic is runtime copy, not the model's answer.

The shape gate in ``_looks_like_gateway_provider_error`` caps envelopes at 400 chars / 4 lines so
prose citing a status code is left alone, and ``_normalize_empty_agent_response`` only rewrote the
*empty* failure case. A turn that terminally failed (quota, billing, 5xx body) therefore delivered
its multi-line provider diagnostic verbatim into the chat — a Heartbeat posting into a normal
conversation included. The turn result already says ``failed``; that is the proof the text is
runtime copy, so the same anchored gate applies without the prose cap.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import (
    GatewayRunner,
    _gateway_provider_error_reply,
    _normalize_empty_agent_response,
    _sanitize_gateway_final_response,
)

DIAGNOSTIC = (
    "API call failed: insufficient_quota\n"
    + "\n".join("provider diagnostic detail line" for _ in range(20))
)
FAILED_RESULT = {
    "failed": True,
    "completed": False,
    "failure_reason": "billing",
    "failure_retryable": False,
    "error": "API call failed: insufficient_quota",
    "provider": "synthetic-provider",
    "model": "synthetic-model",
}


def test_long_diagnostic_on_a_failed_turn_becomes_the_category_reply():
    assert len(DIAGNOSTIC) > 400 and DIAGNOSTIC.count("\n") > 4

    reply = _sanitize_gateway_final_response(
        Platform.DISCORD, DIAGNOSTIC, agent_result=FAILED_RESULT)

    assert reply == _gateway_provider_error_reply(DIAGNOSTIC)
    assert "provider diagnostic detail line" not in reply
    assert "/retry" in reply


def test_the_same_text_without_a_failed_result_is_left_alone():
    """The cap still protects prose: only a proven terminal failure unlocks the long form."""
    assert _sanitize_gateway_final_response(Platform.DISCORD, DIAGNOSTIC) == DIAGNOSTIC
    assert _sanitize_gateway_final_response(
        Platform.DISCORD, DIAGNOSTIC, agent_result={"failed": False}) == DIAGNOSTIC


def test_ordinary_prose_about_a_provider_failure_is_untouched_even_when_the_turn_failed():
    prose = (
        "The billing error insufficient_quota means the account needs attention. "
        + "Here is what the last run showed: " * 20
    )

    assert _sanitize_gateway_final_response(
        Platform.DISCORD, prose, agent_result=FAILED_RESULT) == prose


def test_curated_failure_copy_survives_a_failed_turn():
    """The billing path's own copy (label + guidance) is what the user needs to read."""
    copy = (
        "Billing or credits exhausted: API call failed: insufficient_quota\n\n"
        "Synthetic provider reported that billing, credits, or account entitlement is exhausted "
        "for synthetic-model.\nAdd credits or update billing with that provider, then retry.\n"
        "You can switch providers temporarily with /model <model> --provider <provider>."
    )

    assert _sanitize_gateway_final_response(
        Platform.DISCORD, copy, agent_result=FAILED_RESULT) == copy


def test_normalizer_replaces_a_failed_turns_raw_envelope_and_keeps_the_facts():
    before = dict(FAILED_RESULT)

    reply = _normalize_empty_agent_response(FAILED_RESULT, DIAGNOSTIC)

    assert reply == _gateway_provider_error_reply(DIAGNOSTIC)
    assert FAILED_RESULT == before  # failed/failure_reason/provider survive: the record is untouched


def test_gateway_shape_delivers_the_category_not_the_diagnostic():
    result = {**FAILED_RESULT, "final_response": DIAGNOSTIC, "messages": [], "api_calls": 1}
    runner = SimpleNamespace(_is_intentional_silence=lambda *a, **kw: False)
    source = SimpleNamespace(platform=Platform.DISCORD, chat_id="synthetic-chat")
    entry = SimpleNamespace(session_id="synthetic-session")

    text, silent, _messages = asyncio.run(GatewayRunner._hmwa_shape_agent_response(
        runner, result, source, [], entry, "synthetic-key", "synthetic-key", 1,
        "synthetic-session", "discord", time.time(), persist_user_display_kind="heartbeat",
    ))

    assert text == _gateway_provider_error_reply(DIAGNOSTIC)
    assert "provider diagnostic detail line" not in text
    assert silent is False
    assert result["failed"] is True and result["failure_reason"] == "billing"


def test_context_overflow_failure_keeps_its_own_reply():
    from gateway.run import _context_overflow_reply

    overflow = {**FAILED_RESULT, "failure_reason": "context_overflow",
                "error": "prompt is too long: context length exceeded", "compression_exhausted": True}

    assert _normalize_empty_agent_response(overflow, DIAGNOSTIC, history_len=400) == \
        _context_overflow_reply()
