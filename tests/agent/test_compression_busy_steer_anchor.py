"""Regression coverage for busy-steer preservation across compaction (#100053).

With ``display.busy_input_mode: steer`` the follow-up rides inside the latest
``role=tool`` result (``apply_pending_steer_to_tool_results``), never as a
``role=user`` row. ``_ensure_compressed_has_user_turn`` must treat that marker
as live user intent — and must pick whichever intent-bearing row is LAST in
the original transcript, so an older steer never outranks a newer real user
request.
"""

import pytest
import unittest.mock

# When the checkout lives inside ~/.hermes (default install layout), the lazy
# agent.conversation_loop import inside _is_synthetic_compression_user_turn trips the
# per-test home-IO guard via pm's payload-manifest probe (it reads <home>/manifest.json,
# outside the checkout allow-list). Importing at collection time — before the guard is
# armed — caches the module so every test in this file behaves the same on every layout.
try:
    import agent.conversation_loop
except Exception:
    pass

from agent.context_compressor import (
    COMPRESSION_CONTINUATION_USER_CONTENT,
    SUMMARY_PREFIX,
)
from agent.conversation_compression import (
    _compressed_has_busy_steer,
    _ensure_compressed_has_user_turn,
)
from agent.prompt_builder import STEER_MARKER_OPEN, format_steer_marker
import itertools

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
    for left, right in itertools.pairwise(roles):
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
    # The anchor is a typed steer row: its content is the marker-wrapped text,
    # so membership is a substring check, not exact equality.
    assert any(STEER_B in u for u in users), users
    assert not any(REQUEST_A in u for u in users), "historical request replayed as new input"
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
    assert any(REQUEST_C in u for u in users), users
    assert not any(STEER_B in u for u in users), "older consumed steer replayed over newer user turn"
    assert not any(REQUEST_A in u for u in users)


def test_newer_steer_outranks_older_real_user_turn():
    """Mirror of S3: ``[user A, ..., tool(steer B)]`` — the steer is newest."""
    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 4, steer_at=3)
    compressed = [_summary_row(), *_tool_turns(4, 1)]

    outcome = _ensure_compressed_has_user_turn(original, compressed)

    assert outcome == "inserted"
    _assert_alternation(compressed)
    users = _user_rows(compressed)
    assert any(STEER_B in u for u in users)
    assert not any(REQUEST_A in u for u in users)


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


# --- #154763: the anchored steer must be a typed delivery, and it must retire
# the buffered copy so the next drain cannot inject a second carrier.


def test_anchor_from_tool_borne_steer_is_a_typed_steer_row():
    """The compaction anchor is a live delivery: it must carry the steer type and
    the self-describing marker. A bare dict loses both on replay and gets merged
    into the user's next prompt by alternation repair."""
    from agent.prompt_builder import STEER_DISPLAY_KIND

    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 6, steer_at=2)
    compressed = [_summary_row(), *_tool_turns(5, 1)]

    outcome = _ensure_compressed_has_user_turn(original, compressed)

    assert outcome == "inserted"
    anchor = next(
        m for m in compressed
        if m.get("role") == "user" and STEER_B in str(m.get("content"))
    )
    assert anchor.get("display_kind") == STEER_DISPLAY_KIND
    assert STEER_MARKER_OPEN in str(anchor["content"])


def test_buffered_steer_consumed_by_anchor_is_not_reinjected():
    """A steer still buffered when compaction anchors its text (tool-borne shape)
    must leave the buffer: the anchor carries it, the next drain must find
    nothing, and settlement must read the transfer as consumed (not withdrawable
    back to the queue)."""
    from agent.interrupt_control import InterruptControlMixin

    agent = InterruptControlMixin()
    agent.steer(STEER_B, ticket="adm-1")

    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 6, steer_at=2)
    compressed = [_summary_row(), *_tool_turns(5, 1)]

    outcome = _ensure_compressed_has_user_turn(original, compressed, agent=agent)

    assert outcome == "inserted"
    assert sum(str(m.get("content")).count(STEER_B) for m in compressed) == 1
    # withdraw BEFORE any drain: only the anchor's consume can have stamped the ticket.
    assert agent.withdraw_steer("adm-1") is False, "settlement would re-queue text the anchor already carries"
    assert agent._drain_pending_steer() is None, "buffered copy survived the anchor — second inject incoming"


def test_already_present_tool_borne_steer_retires_buffered_copy():
    """Steer marker survived into the compressed tail AND the part is still
    buffered: already_present must not leave two carriers — the context has it,
    so the buffer must not inject it again."""
    from agent.interrupt_control import InterruptControlMixin

    agent = InterruptControlMixin()
    agent.steer(STEER_B, ticket="adm-1")

    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 6, steer_at=5)
    compressed = [_summary_row(), *_tool_turns(5, 1, steer_at=5)]

    outcome = _ensure_compressed_has_user_turn(original, compressed, agent=agent)

    assert outcome == "already_present"
    assert agent.withdraw_steer("adm-1") is False
    assert agent._drain_pending_steer() is None


def test_typed_steer_user_row_anchor_retires_buffered_copy():
    """Every live steer is a typed user row now: when the anchor is a copy of
    that row, the identically-texted buffered part must be retired too."""
    from agent.interrupt_control import InterruptControlMixin
    from agent.prompt_builder import steer_user_row

    agent = InterruptControlMixin()
    agent.steer(STEER_B, ticket="adm-1")

    original = [{"role": "user", "content": REQUEST_A}, steer_user_row(STEER_B)] + _tool_turns(1, 4)
    compressed = [_summary_row(), *_tool_turns(5, 1)]

    outcome = _ensure_compressed_has_user_turn(original, compressed, agent=agent)

    assert outcome == "inserted"
    assert sum(str(m.get("content")).count(STEER_B) for m in compressed) == 1
    assert agent.withdraw_steer("adm-1") is False
    assert agent._drain_pending_steer() is None


def test_consume_matches_legacy_marker_shape():
    """Text extracted through the legacy-marker fallback carries a lead-in line
    the buffered part never had; shape-aware matching must still retire it, or
    duplication silently survives and settlement re-queues."""
    import agent.conversation_compression as cc
    from agent.interrupt_control import InterruptControlMixin, consume_buffered_steer

    agent = InterruptControlMixin()
    agent.steer(STEER_B, ticket="adm-1")

    legacy_open = "[OUT-OF-BAND USER MESSAGE (legacy)]\nThe user sent a new message while Hermes was working:"
    legacy_close = "[/OUT-OF-BAND USER MESSAGE (legacy)]"
    legacy_row = {"role": "tool", "tool_call_id": "c",
                  "content": f"ok\n\n{legacy_open}\n{STEER_B}\n{legacy_close}"}
    with (
        unittest.mock.patch.object(cc, "_STEER_FALLBACK_OPEN", legacy_open),
        unittest.mock.patch.object(cc, "_STEER_FALLBACK_CLOSE", legacy_close),
    ):
        legacy_extracted = cc._extract_steer_text_from_message(legacy_row)
        assert legacy_extracted == f"The user sent a new message while Hermes was working:\n{STEER_B}"
        assert consume_buffered_steer(agent, legacy_extracted) is True
    assert agent._drain_pending_steer() is None
    assert agent.withdraw_steer("adm-1") is False


def test_merged_anchor_keeps_steer_display_kind():
    """A merged anchor must carry the steer type onto the composite row, or
    alternation repair welds the user's NEXT prompt into the steer text."""
    from agent.prompt_builder import STEER_DISPLAY_KIND

    original = [{"role": "user", "content": REQUEST_A}] + _tool_turns(0, 6, steer_at=2)
    # Trailing user scaffolding (no assistant follows) forces the merge branch.
    compressed = [_summary_row(), {"role": "assistant", "content": "Working."},
                  {"role": "user", "content": "todo snapshot", "_todo_snapshot_synthetic": True}]

    outcome = _ensure_compressed_has_user_turn(original, compressed)

    assert outcome == "merged"
    assert compressed[-1].get("display_kind") == STEER_DISPLAY_KIND
    assert "_todo_snapshot_synthetic" not in compressed[-1]


def test_consume_retires_one_copy_of_two_identical_buffered_steers():
    """Two legitimate identical steers are representable; the anchor retires at
    most one part, and the second steer still reaches the model."""
    from agent.interrupt_control import InterruptControlMixin, consume_buffered_steer

    agent = InterruptControlMixin()
    agent.steer(STEER_B, ticket="adm-1")
    agent.steer(STEER_B, ticket="adm-2")

    assert consume_buffered_steer(agent, STEER_B) is True
    assert consume_buffered_steer(agent, "unrelated text") is False
    assert agent._drain_pending_steer() == STEER_B
