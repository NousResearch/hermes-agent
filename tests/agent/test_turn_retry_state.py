"""Unit tests for TurnRetryState (god-file Phase 1b).

The dataclass holds the inner-retry-loop's one-shot recovery guards + restart
signals. These tests pin its shape and default semantics — the behavioral
guarantee for the loop itself is the existing recovery-branch tests in
tests/run_agent/ which now exercise these fields via `_retry.<flag>`.
"""

from __future__ import annotations

from dataclasses import fields

from agent.turn_retry_state import TurnRetryState


EXPECTED_FIELDS = {
    "codex_auth_retry_attempted",
    "anthropic_auth_retry_attempted",
    "nous_auth_retry_attempted",
    "nous_paid_entitlement_refresh_attempted",
    "copilot_auth_retry_attempted",
    "thinking_sig_retry_attempted",
    "invalid_encrypted_content_retry_attempted",
    "image_shrink_retry_attempted",
    "payload_image_shrink_attempted",
    "multimodal_tool_content_retry_attempted",
    "oauth_1m_beta_retry_attempted",
    "llama_cpp_grammar_retry_attempted",
    "primary_recovery_attempted",
    "has_retried_429",
    "restart_with_compressed_messages",
    "restart_with_length_continuation",
}


def test_all_guards_default_false():
    s = TurnRetryState()
    for name, value in s:
        assert value is False, f"{name} should default to False"


def test_field_set_matches_contract():
    names = {f.name for f in fields(TurnRetryState)}
    assert names == EXPECTED_FIELDS, (
        f"unexpected drift: missing={EXPECTED_FIELDS - names} extra={names - EXPECTED_FIELDS}"
    )


def test_loop_control_vars_are_not_on_state():
    # retry_count / max_retries / max_compression_attempts stay as loop locals,
    # NOT on the state object (they are while-mechanics, not recovery bookkeeping).
    names = {f.name for f in fields(TurnRetryState)}
    for loop_local in ("retry_count", "max_retries", "max_compression_attempts"):
        assert loop_local not in names


def test_guards_are_independently_mutable():
    s = TurnRetryState()
    s.codex_auth_retry_attempted = True
    s.restart_with_compressed_messages = True
    assert s.codex_auth_retry_attempted is True
    assert s.restart_with_compressed_messages is True
    # untouched guards stay False
    assert s.has_retried_429 is False
    assert s.anthropic_auth_retry_attempted is False
