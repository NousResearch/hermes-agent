import json
from pathlib import Path

from agent.context_health_compact import build_compact_exhaustion_hold


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERSATION_LOOP = REPO_ROOT / "agent" / "conversation_loop.py"


def _source() -> str:
    return CONVERSATION_LOOP.read_text(encoding="utf-8")


def _between(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_payload_too_large_cannot_compress_further_routes_to_safe_hold_not_new_or_compress_advice():
    """Phase 7: exhausted compact fallback must become continuity/HOLD, not manual reset advice."""
    branch = _between(
        _source(),
        "if is_payload_too_large:",
        "# Check for context-length errors BEFORE generic 4xx handler.",
    )

    assert "Cannot compress further" in branch, "test must target the legacy exhausted compact path"
    assert "compression_exhausted" in branch, "test must target compact exhaustion behavior"

    assert any(
        marker in branch
        for marker in (
            "context_health_compact",
            "continuity_packet",
            "safe HOLD",
            "safe_hold",
            "rehydrate",
        )
    ), "Phase 7 should route exhausted compact failure to a continuity packet + HOLD/rehydrate path"
    assert "Try /new" not in branch
    assert "start a fresh conversation" not in branch
    assert "Try /compress" not in branch


def test_payload_too_large_safe_hold_contract_excludes_raw_cross_task_and_secret_material():
    """Phase 7 HOLD/continuity output must encode redaction boundaries, not raw transcript leakage."""
    branch = _between(
        _source(),
        "if is_payload_too_large:",
        "# Check for context-length errors BEFORE generic 4xx handler.",
    )

    required_safety_terms = (
        "raw transcript",
        "unrelated",
        "secret",
        "token",
        "password",
        "private",
    )
    missing_terms = [term for term in required_safety_terms if term not in branch.lower()]
    assert not missing_terms, (
        "Phase 7 compact fallback must make the safe HOLD/continuity contract explicit; "
        f"missing safety terms: {missing_terms}"
    )

    forbidden_needles = (
        "CLOSED_TASK_A_NEEDLE_DO_NOT_LEAK",
        "UNRELATED_TASK_B_NEEDLE_DO_NOT_MIX",
        "SYNTHETIC_TOKEN_NEEDLE_SHOULD_NOT_APPEAR",
        "SYNTHETIC_PASSWORD_NEEDLE_SHOULD_NOT_APPEAR",
        "PRIVATE_RAW_BODY_DO_NOT_STORE",
    )
    for needle in forbidden_needles:
        assert needle not in branch


def test_disabled_auto_compaction_preserves_existing_manual_pass_through_guidance():
    """Disabled Phase 7 compact fallback must not break compression.enabled:false legacy behavior."""
    branch = _between(
        _source(),
        "# ── Respect disabled auto-compaction on overflow",
        "# ── Anthropic Sonnet long-context tier gate",
    )

    assert "compression.enabled: false" in branch
    assert "compaction_disabled" in branch
    assert "Run /compress" in branch
    assert "/new to start fresh" in branch


def _payload_too_large_branch() -> str:
    return _between(
        _source(),
        "if is_payload_too_large:",
        "# Check for context-length errors BEFORE generic 4xx handler.",
    )


def _context_overflow_branch() -> str:
    return _between(
        _source(),
        "if is_context_length_error:",
        "# Check for non-retryable client errors.",
    )


def test_payload_too_large_hold_result_does_not_return_raw_messages_transcript():
    """Safe HOLD result assembly must not reattach the raw live transcript list."""
    branch = _payload_too_large_branch()

    assert "build_compact_exhaustion_hold" in branch
    assert "continuity_packet" in branch
    assert '"messages": messages' not in branch, (
        "Phase 7 compact HOLD currently reattaches raw messages; expected "
        "sanitized assistant-only messages or another safe non-transcript shape."
    )


def test_context_overflow_max_attempts_routes_to_compact_safe_hold_not_legacy_guidance():
    """Context-overflow max attempts must use compact HOLD, not /new or /compress advice."""
    branch = _between(
        _context_overflow_branch(),
        "if available_out is not None:",
        "# The error is output-cap-shaped",
    )

    assert "compression_exhausted" in branch
    assert any(
        marker in branch
        for marker in (
            "build_compact_exhaustion_hold",
            "context_health_compact",
            "continuity_packet",
            "safe_hold",
        )
    ), "context-overflow max-attempts must route to compact safe HOLD"
    assert "Try /new" not in branch
    assert "/compress" not in branch
    assert '"messages": messages' not in branch


def test_context_overflow_cannot_compress_further_routes_to_compact_safe_hold():
    """Context-overflow cannot-compress-further path must not keep legacy fallback result."""
    overflow_branch = _context_overflow_branch()
    start_index = overflow_branch.index("# Can't compress further and already at minimum tier")
    branch = overflow_branch[start_index:]

    assert "Cannot compress further" in branch
    assert "compression_exhausted" in branch
    assert any(
        marker in branch
        for marker in (
            "build_compact_exhaustion_hold",
            "context_health_compact",
            "continuity_packet",
            "safe_hold",
        )
    ), "context-overflow cannot-compress-further must route to compact safe HOLD"
    assert "Try /new" not in branch
    assert "/compress" not in branch
    assert '"messages": messages' not in branch


def test_compact_hold_result_repr_excludes_raw_needles_and_uses_safe_message_shape():
    """A complete compact HOLD result shape must be safe across repr(result), not final_response only."""
    hold = build_compact_exhaustion_hold(
        reason="test_constant_reason",
        session_id="session-for-red-contract",
        task_id="task-for-red-contract",
        message_count=5,
        approx_tokens=12345,
    )
    safe_result = {
        **hold,
        "messages": [{"role": "assistant", "content": hold["final_response"]}],
        "api_calls": 1,
    }

    assert safe_result["messages"] == [
        {"role": "assistant", "content": hold["final_response"]}
    ]
    result_repr = json.dumps(safe_result, sort_keys=True)
    forbidden_needles = (
        "CLOSED_TASK_A_NEEDLE_DO_NOT_LEAK",
        "UNRELATED_TASK_B_NEEDLE_DO_NOT_MIX",
        "SYNTHETIC_TOKEN_NEEDLE_SHOULD_NOT_APPEAR",
        "SYNTHETIC_PASSWORD_NEEDLE_SHOULD_NOT_APPEAR",
        "PRIVATE_RAW_BODY_DO_NOT_STORE",
    )
    for needle in forbidden_needles:
        assert needle not in result_repr


def test_payload_too_large_current_callsite_would_leak_raw_messages_in_result_repr():
    """RED guard: current payload-too-large callsite must stop assembling raw transcript results."""
    branch = _payload_too_large_branch()
    raw_messages = [
        {"role": "user", "content": "CLOSED_TASK_A_NEEDLE_DO_NOT_LEAK"},
        {"role": "tool", "content": "SYNTHETIC_TOKEN_NEEDLE_SHOULD_NOT_APPEAR"},
        {"role": "assistant", "content": "PRIVATE_RAW_BODY_DO_NOT_STORE"},
    ]
    hold = build_compact_exhaustion_hold(
        reason="payload_too_large_cannot_compress_further",
        session_id="session-for-red-contract",
        task_id="task-for-red-contract",
        message_count=len(raw_messages),
        approx_tokens=999999,
    )

    if '"messages": messages' in branch:
        current_shape = {**hold, "messages": raw_messages, "api_calls": 2}
        result_repr = repr(current_shape)
    else:
        current_shape = {
            **hold,
            "messages": [{"role": "assistant", "content": hold["final_response"]}],
            "api_calls": 2,
        }
        result_repr = repr(current_shape)

    assert "CLOSED_TASK_A_NEEDLE_DO_NOT_LEAK" not in result_repr
    assert "SYNTHETIC_TOKEN_NEEDLE_SHOULD_NOT_APPEAR" not in result_repr
    assert "PRIVATE_RAW_BODY_DO_NOT_STORE" not in result_repr
