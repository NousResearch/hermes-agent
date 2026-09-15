"""Regression coverage for busy-steer preservation across compaction (#100053).

With ``display.busy_input_mode: steer`` the follow-up rides inside the latest
``role=tool`` result (``apply_pending_steer_to_tool_results``), never as a
``role=user`` row. ``_ensure_compressed_has_user_turn`` must treat that marker
as live user intent — and must pick whichever intent-bearing row is LAST in
the original transcript, so an older steer never outranks a newer real user
request.
"""

import pytest

from agent.context_compressor import (
    COMPRESSION_CONTINUATION_USER_CONTENT,
    SUMMARY_PREFIX,
)
from agent.conversation_compression import (
    _compressed_has_busy_steer,
    _ensure_compressed_has_user_turn,
)
from agent.prompt_builder import STEER_MARKER_OPEN, format_steer_marker

REQUEST_A = "Historical request A: audit the auth module."
STEER_B = "Steer B: stop, switch to fixing the login bug instead."
REQUEST_C = "Newer real user request C: now write the release notes."


def _tool_turns(start: int, count: int, *, steer_at: int | None = None) -> list[dict]:
    turns: list[dict] = []
    for idx in range(start, start + count):
        turns.append(
            {
                "role": "assistant",
                "content": "Working.",
                "tool_calls": [
                    {
                        "id": f"call-{idx}",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            }
        )
        content = f"tool output {idx}"
        if steer_at == idx:
            content += format_steer_marker(STEER_B)
        turns.append({"role": "tool", "tool_call_id": f"call-{idx}", "content": content})
    return turns


def _summary_row() -> dict:
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\n\nEarlier work summarized."}


def _assert_alternation(messages: list[dict]) -> None:
    roles = [m.get("role") for m in messages]
    for left, right in zip(roles, roles[1:]):
        assert not (left == right == "user"), f"user/user adjacency in {roles}"
        assert not (left == right == "assistant"), f"assistant/assistant adjacency in {roles}"


def _user_rows(messages: list[dict]) -> list[str]:
    return [str(m.get("content")) for m in messages if m.get("role") == "user"]


def test_s1_steer_summarized_away_becomes_anchor_not_historical_request():
    """S1: the steer lived in a tool row that compaction dropped; the only
    ``role=user`` row in history is the already-consumed request A. The steer
    must be restored as the anchor, and A must not be replayed."""
    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 6, steer_at=2)
    compressed = [_summary_row(), *_tool_turns(5, 1)]

    outcome = _ensure_compressed_has_user_turn(original, compressed)

    assert outcome == "inserted"
    _assert_alternation(compressed)
    users = _user_rows(compressed)
    assert STEER_B in users, users
    assert REQUEST_A not in users, "historical request replayed as new input"
    assert COMPRESSION_CONTINUATION_USER_CONTENT not in users
    # Steer text is used exactly once across the whole compressed transcript.
    assert sum(str(m.get("content")).count(STEER_B) for m in compressed) == 1


def test_s2_steer_surviving_in_tail_tool_row_counts_as_present():
    """S2: the steer-bearing tool row survived into the tail. No anchor may be
    inserted (the intent is already there) and A must not be cloned."""
    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 6, steer_at=5)
    compressed = [_summary_row(), *_tool_turns(5, 1, steer_at=5)]
    before = [dict(m) for m in compressed]

    outcome = _ensure_compressed_has_user_turn(original, compressed)

    assert outcome == "already_present"
    assert compressed == before, "transcript mutated despite live steer present"
    assert REQUEST_A not in _user_rows(compressed)
    assert sum(str(m.get("content")).count(STEER_B) for m in compressed) == 1


def test_s3_newer_real_user_turn_outranks_older_steer():
    """S3: ``[user A, tool(steer B), ..., user C]`` — C is the newest intent.
    A steer-first scan would anchor the consumed steer B and replay it."""
    original = (
        [{"role": "user", "content": REQUEST_A}]
        + _tool_turns(0, 3, steer_at=1)
        + [{"role": "user", "content": REQUEST_C}]
        + _tool_turns(3, 4)
    )
    compressed = [_summary_row(), *_tool_turns(6, 1)]

    outcome = _ensure_compressed_has_user_turn(original, compressed)

    assert outcome == "inserted"
    _assert_alternation(compressed)
    users = _user_rows(compressed)
    assert REQUEST_C in users, users
    assert STEER_B not in users, "older consumed steer replayed over newer user turn"
    assert REQUEST_A not in users
    assert not any(STEER_B in u for u in users)


def test_newer_steer_outranks_older_real_user_turn():
    """Mirror of S3: ``[user A, ..., tool(steer B)]`` — the steer is newest."""
    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 4, steer_at=3)
    compressed = [_summary_row(), *_tool_turns(4, 1)]

    outcome = _ensure_compressed_has_user_turn(original, compressed)

    assert outcome == "inserted"
    _assert_alternation(compressed)
    users = _user_rows(compressed)
    assert STEER_B in users
    assert REQUEST_A not in users


@pytest.mark.parametrize(
    "role",
    ["user", "assistant"],
)
def test_compressed_steer_presence_only_counts_tool_rows(role):
    """A summary or assistant row that merely quotes the marker text is not a
    live steer delivery — only ``role=tool`` rows carry real steers."""
    quoted = {"role": role, "content": f"{SUMMARY_PREFIX}\n{format_steer_marker(STEER_B)}"}
    assert _compressed_has_busy_steer([quoted]) is False
    assert STEER_MARKER_OPEN in quoted["content"]
    live = {"role": "tool", "tool_call_id": "c", "content": f"ok{format_steer_marker(STEER_B)}"}
    assert _compressed_has_busy_steer([live]) is True
