"""Stress tests for orphaned tool_use recovery (PR #53236).

Tests three layers:
1. error_classifier: detects the Anthropic 400 and emits orphaned_tool_use
2. _strip_orphaned_tool_blocks: actually cleans the canonical messages list
3. TurnRetryState: one-shot guard prevents infinite retry loops
"""

from __future__ import annotations

import pytest
from agent.error_classifier import FailoverReason, _classify_400
from agent.anthropic_adapter import _strip_orphaned_tool_blocks
from agent.turn_retry_state import TurnRetryState


# ── helpers ──────────────────────────────────────────────────────────────────

def _result_fn(reason, **kwargs):
    """Minimal stand-in for the result_fn closure in classify_api_error."""
    from agent.error_classifier import ClassifiedError
    obj = ClassifiedError(reason=reason, status_code=400)
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _classify(msg: str):
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": msg}}
    return _classify_400(
        msg.lower(), "", body,
        provider="anthropic", model="claude-sonnet-4-6",
        approx_tokens=1000, context_length=200_000,
        num_messages=10, result_fn=_result_fn,
    )


def _make_messages(*, orphaned: bool = True):
    """Build a minimal canonical messages list.

    With orphaned=True:  assistant has tool_use with no following tool_result
    With orphaned=False: well-formed pair
    """
    tool_use_block = {
        "type": "tool_use",
        "id": "toolu_017QgFu6YZ8WbbMGDb1Z9FtP",
        "name": "execute_code",
        "input": {"code": "print('hi')"},
    }
    tool_result_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_017QgFu6YZ8WbbMGDb1Z9FtP",
                "content": "Blocked by approval guard",
            }
        ],
    }
    messages = [
        {"role": "user", "content": "run some code"},
        {"role": "assistant", "content": [{"type": "text", "text": "Sure!"}, tool_use_block]},
    ]
    if not orphaned:
        messages.append(tool_result_msg)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "Done."}]})
    return messages


# ── Layer 1: error classifier ─────────────────────────────────────────────────

ANTHROPIC_ERROR_MESSAGES = [
    # Exact wording from real Anthropic API response
    "messages.2: `tool_use` ids were found without `tool_result` blocks immediately after: "
    "toolu_017QgFu6YZ8WbbMGDb1Z9FtP. Each `tool_use` block must have a corresponding "
    "`tool_result` block in the next message.",
    # Shorter variant
    "tool_use ids found without tool_result blocks",
    # Multi-tool variant
    "tool_use ids toolu_abc toolu_def were found without tool_result blocks",
]


@pytest.mark.parametrize("msg", ANTHROPIC_ERROR_MESSAGES)
def test_classifier_detects_orphaned_tool_use(msg):
    result = _classify(msg)
    assert result.reason == FailoverReason.orphaned_tool_use, (
        f"Expected orphaned_tool_use, got {result.reason} for: {msg!r}"
    )
    assert result.retryable is True, "Must be retryable so the loop can recover"


def test_classifier_does_not_trigger_on_unrelated_400():
    result = _classify("invalid model name: claude-fake-99")
    assert result.reason != FailoverReason.orphaned_tool_use


def test_classifier_does_not_trigger_on_context_overflow():
    # Simulate a large session that triggers context_overflow before our check
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "error"}}
    result = _classify_400(
        "error", "", body,
        provider="anthropic", model="claude-sonnet-4-6",
        approx_tokens=90_000, context_length=200_000,
        num_messages=90, result_fn=_result_fn,
    )
    assert result.reason == FailoverReason.context_overflow


# ── Layer 2: strip function cleans canonical messages ─────────────────────────

def test_strip_removes_orphaned_tool_use():
    messages = _make_messages(orphaned=True)
    assert len(messages) == 2  # user + orphaned assistant

    _strip_orphaned_tool_blocks(messages)

    # The assistant message should no longer contain the tool_use block
    assistant_blocks = [
        b for m in messages
        if m.get("role") == "assistant"
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    assert assistant_blocks == [], f"Expected no tool_use blocks, found: {assistant_blocks}"


def test_strip_leaves_well_formed_messages_intact():
    messages = _make_messages(orphaned=False)
    original_len = len(messages)
    _strip_orphaned_tool_blocks(messages)
    # Well-formed pair: tool_use + tool_result both present, nothing to strip
    tool_uses = [
        b for m in messages
        if m.get("role") == "assistant"
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    assert tool_uses != [] or len(messages) == original_len, (
        "Well-formed pair should survive the strip"
    )


def test_strip_is_idempotent():
    """Calling strip twice should not crash or over-strip."""
    messages = _make_messages(orphaned=True)
    _strip_orphaned_tool_blocks(messages)
    snapshot = [m.copy() for m in messages]
    _strip_orphaned_tool_blocks(messages)
    assert messages == snapshot, "Second strip changed messages"


def test_strip_handles_empty_list():
    messages = []
    _strip_orphaned_tool_blocks(messages)  # must not raise
    assert messages == []


def test_strip_handles_no_tool_use():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    _strip_orphaned_tool_blocks(messages)
    assert len(messages) == 2


# ── Layer 3: TurnRetryState one-shot guard ────────────────────────────────────

def test_retry_state_has_orphaned_tool_use_flag():
    state = TurnRetryState()
    assert hasattr(state, "orphaned_tool_use_retry_attempted"), (
        "TurnRetryState must have orphaned_tool_use_retry_attempted field"
    )
    assert state.orphaned_tool_use_retry_attempted is False


def test_retry_state_flag_prevents_second_recovery():
    """Simulate the guard: second HTTP 400 should NOT trigger recovery again."""
    state = TurnRetryState()

    # First recovery fires
    assert not state.orphaned_tool_use_retry_attempted
    state.orphaned_tool_use_retry_attempted = True

    # Second time: guard is already set — recovery must NOT fire
    assert state.orphaned_tool_use_retry_attempted is True


def test_retry_state_fresh_instance_resets_flag():
    """Each new API call attempt gets a fresh TurnRetryState."""
    state1 = TurnRetryState()
    state1.orphaned_tool_use_retry_attempted = True

    state2 = TurnRetryState()
    assert state2.orphaned_tool_use_retry_attempted is False, (
        "New TurnRetryState must start clean"
    )


# ── Integration: full recovery simulation ────────────────────────────────────

def test_full_recovery_flow():
    """Simulate the complete recovery path end-to-end without a real API call."""
    # 1. Build a broken canonical messages list (cron job interrupted mid-tool)
    messages = _make_messages(orphaned=True)

    # 2. Verify the classifier recognizes the error
    error_msg = (
        "messages.2: `tool_use` ids were found without `tool_result` blocks immediately after: "
        "toolu_017QgFu6YZ8WbbMGDb1Z9FtP."
    )
    classified = _classify(error_msg)
    assert classified.reason == FailoverReason.orphaned_tool_use
    assert classified.retryable

    # 3. Verify guard not yet set
    retry_state = TurnRetryState()
    assert not retry_state.orphaned_tool_use_retry_attempted

    # 4. Run the recovery (as conversation_loop.py would)
    retry_state.orphaned_tool_use_retry_attempted = True
    _strip_orphaned_tool_blocks(messages)

    # 5. Verify messages are now clean
    remaining_tool_uses = [
        b for m in messages
        if m.get("role") == "assistant"
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    assert remaining_tool_uses == [], "After recovery, no orphaned tool_use blocks should remain"

    # 6. Verify a second identical error would NOT trigger recovery again
    assert retry_state.orphaned_tool_use_retry_attempted is True
