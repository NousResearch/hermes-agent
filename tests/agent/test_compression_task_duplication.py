"""Regression coverage for #106864: the first compression of a single-user
task must not keep two full copies of the still-unfinished request.

The protected head retains the original user turn verbatim, while
``_reappend_inflight_user_task`` (#100818) restates the same instruction
after the handoff so SUMMARY_PREFIX's "latest user message" pointer resolves
to it. Both halves are needed — but keeping the original payload *and* the
restatement doubles the input exactly when compression should be reclaiming
space, so a large request risks the context overflow the compressor was
trying to avoid.

The fix displaces the superseded head copy with a short stub: the row (and
its role) survives so alternation/user-leading contracts hold, but the
restatement after the summary becomes the only full copy.
"""

from typing import Any, Dict, List
from unittest.mock import patch

from agent.context_compressor import (
    _INFLIGHT_TASK_DISPLACED_STUB,
    _SUMMARY_END_MARKER,
    SUMMARY_PREFIX,
    ContextCompressor,
)


TASK_SENTINEL = "AUDIT_TASK_sentinel_review_the_synthetic_policy_document"
TASK = (TASK_SENTINEL + "\n" + "Synthetic policy material. " * 3000).rstrip()


def _tool_pairs(count: int) -> List[Dict[str, Any]]:
    """``count`` assistant(tool_calls) + tool result pairs — an unfinished run."""
    turns: List[Dict[str, Any]] = []
    for i in range(count):
        turns.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"c{i}", "function": {"name": "terminal", "arguments": "{}"}}
                ],
            }
        )
        turns.append({"role": "tool", "tool_call_id": f"c{i}", "content": "accepted"})
    return turns


def _compress(messages: List[Dict[str, Any]], protect_first_n: int = 3):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test",
            quiet_mode=True,
            protect_first_n=protect_first_n,
            protect_last_n=20,
        )
    compressor.tail_token_budget = 2_000
    with patch.object(
        compressor,
        "_generate_summary",
        return_value=SUMMARY_PREFIX + "\nSynthetic handoff.",
    ) as summary:
        result = compressor.compress(messages, current_tokens=180_000, force=True)
    assert summary.call_count == 1, "fixture must cross a summary boundary"
    return result


def _full_copies(compressed: List[Dict[str, Any]]) -> int:
    """Count full copies of the task across ALL rows (any role), like #106864."""
    return sum(str(m.get("content")).count(TASK) for m in compressed)


def _text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else str(content)


def _handoff_idx(compressed: List[Dict[str, Any]]) -> int:
    for idx in range(len(compressed) - 1, -1, -1):
        if _SUMMARY_END_MARKER in _text(compressed[idx]):
            return idx
    return -1


def test_first_compaction_keeps_one_actionable_task_copy():
    """Protected head + unfinished task: the restatement after the handoff
    must be the ONLY full copy (#106864) while staying actionable (#100818)."""
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": TASK},
        *_tool_pairs(10),
    ]
    compressed = _compress(messages)

    idx = _handoff_idx(compressed)
    assert idx >= 0, "expected a compaction handoff"

    after_handoff = [_text(compressed[idx]).split(_SUMMARY_END_MARKER, 1)[1], *map(_text, compressed[idx + 1:])]
    assert any(TASK in text for text in after_handoff), (
        "the unfinished task must remain actionable after the handoff (#100818)"
    )
    assert _full_copies(compressed) == 1, (
        f"full task copies after compression={_full_copies(compressed)}; "
        "the superseded head copy must be displaced (#106864)"
    )


def test_displaced_head_row_keeps_its_role_and_position():
    """The stub replaces the payload, not the row: role layout and head size
    must survive so alternation / Anthropic user-leading contracts hold."""
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": TASK},
        *_tool_pairs(10),
    ]
    compressed = _compress(messages)

    head_user_rows = [
        m for m in compressed[:_handoff_idx(compressed)]
        if m.get("role") == "user"
        and ContextCompressor._is_actionable_user_turn(m)
    ]
    assert head_user_rows, "the head user row must survive displacement"
    assert all(TASK not in _text(m) for m in compressed[:_handoff_idx(compressed)])


def test_completed_task_is_never_duplicated():
    """Control: a completed exchange is not re-appended, so the head original
    is the one copy and must NOT be displaced."""
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": TASK},
        *_tool_pairs(10),
        {"role": "assistant", "content": "Finished."},
    ]
    compressed = _compress(messages)

    assert _full_copies(compressed) == 1
    assert any(
        TASK in _text(m)
        for m in compressed
        if m.get("role") == "user" and ContextCompressor._is_actionable_user_turn(m)
    ), "a completed task's original copy must stay intact"


def test_zero_protected_head_keeps_one_copy():
    """Control: without head protection the original is summarised away, so
    the restatement is already the only copy."""
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": TASK},
        *_tool_pairs(10),
    ]
    compressed = _compress(messages, protect_first_n=0)

    assert _full_copies(compressed) == 1


def _head_user_rows(compressed: List[Dict[str, Any]]):
    idx = _handoff_idx(compressed)
    assert idx >= 0, "expected a compaction handoff"
    return [
        m
        for m in compressed[:idx]
        if m.get("role") == "user"
        and ContextCompressor._is_actionable_user_turn(m)
        and not ContextCompressor._is_synthetic_compression_user_turn(m)
    ]


def test_completed_same_text_turn_is_not_displaced():
    """Equal old/active requests: an earlier COMPLETED turn whose text equals
    the re-stated in-flight request keeps its payload verbatim — displacement
    must rewrite only the row derived from the active in-flight turn."""
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": TASK},  # earlier completed turn (history)
        {"role": "assistant", "content": "Done previously."},
        {"role": "user", "content": TASK},  # re-stated request, still in flight
        *_tool_pairs(15),
    ]
    compressed = _compress(messages, protect_first_n=5)

    user_rows = _head_user_rows(compressed)
    assert len(user_rows) == 2, "both head user rows must survive"
    assert TASK_SENTINEL not in _text(user_rows[-1]), (
        "the in-flight original (newest head user row) must be displaced"
    )
    assert TASK_SENTINEL in _text(user_rows[0]), (
        "the earlier completed turn with identical text must keep its payload"
    )
    assert _INFLIGHT_TASK_DISPLACED_STUB in _text(user_rows[-1])
    assert _full_copies(compressed) == 2, "history copy + restatement"


def test_inflight_summarized_away_leaves_head_history_verbatim():
    """The in-flight row itself is summarized away: the same-text row left in
    the head is an earlier completed turn, so nothing may be displaced."""
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": TASK},  # earlier completed turn (history)
        {"role": "assistant", "content": "Done previously."},
        {"role": "user", "content": TASK},  # in flight, but inside the summary window
        *_tool_pairs(15),
    ]
    compressed = _compress(messages, protect_first_n=2)

    user_rows = _head_user_rows(compressed)
    assert len(user_rows) == 1, "only the history row survives in the head"
    assert TASK_SENTINEL in _text(user_rows[0]), (
        "a summarized-away in-flight task must not displace the history row"
    )
    assert _INFLIGHT_TASK_DISPLACED_STUB not in _text(user_rows[0])
    idx = _handoff_idx(compressed)
    after_handoff = [
        _text(compressed[idx]).split(_SUMMARY_END_MARKER, 1)[1],
        *map(_text, compressed[idx + 1:]),
    ]
    assert any(
        TASK_SENTINEL in t for t in after_handoff
    ), "the restatement after the handoff must stay actionable (#100818)"
    assert _full_copies(compressed) == 2, "history copy + restatement"
