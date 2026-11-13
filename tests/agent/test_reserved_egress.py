from types import SimpleNamespace
import logging

import pytest

from agent.reserved_egress import assert_no_unauthorized_reserved_egress
from agent.turn_finalizer import _resolve_budget_fallback


def test_unclassified_payload_is_allowed():
    assert_no_unauthorized_reserved_egress(SimpleNamespace(), {"messages": [{"content": "ordinary"}]})


def test_reserved_payload_requires_explicit_authorization():
    with pytest.raises(PermissionError, match="Blocked outbound LLM request"):
        assert_no_unauthorized_reserved_egress(
            SimpleNamespace(reserved_content_authorized=False),
            {"extra_body": {"classification": "restricted", "content": "internal"}},
        )


def test_serialized_reserved_payload_is_blocked_even_with_legacy_flag():
    with pytest.raises(PermissionError, match="Blocked outbound LLM request"):
        assert_no_unauthorized_reserved_egress(
            SimpleNamespace(reserved_content_authorized=True),
            '{"classification":"restricted","content":"internal"}',
        )


def test_summary_guard_becomes_a_nonretryable_turn_failure():
    agent = SimpleNamespace(
        max_iterations=1,
        iteration_budget=SimpleNamespace(remaining=0),
        quiet_mode=True,
        provider="nous",
        _emit_diagnostic_status=lambda *_args: None,
        _handle_max_iterations=lambda *_args: (_ for _ in ()).throw(
            PermissionError("Blocked outbound LLM request: reserved content")
        ),
    )

    final_response, exit_reason, _, _ = _resolve_budget_fallback(
        agent,
        final_response=None,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[],
        _turn_exit_reason="unknown",
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
        logger=logging.getLogger(__name__),
    )

    assert "safety filter refused" in final_response
    assert exit_reason == "content_policy_blocked"
