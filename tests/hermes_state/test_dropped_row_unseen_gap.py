"""A row the alternation repair drops is named at commit, so the rest of the held history keeps exact coverage.

Falling back to the watermark instead archives rows another surface appended below it, which the
compressor never saw: they leave the model's history without being in the summary.
"""

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.conversation_compression_archive import coverage_for_commit
from hermes_state import SessionDB
from tests.agent.test_in_place_preflight_rewind import _turn, session  # noqa: F401

_CALL = {"tool_calls": [{"id": "call_killed", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]}
DROPPED = {
    "tool_result": [("tool", "ORPHAN_RESULT", {"tool_call_id": "call_killed"})],
    "tool_call": [("assistant", "", _CALL)],
    "two_replies": [("assistant", "A2b a second reply", {})],
}


@pytest.mark.parametrize("row_ids", [False, True])
@pytest.mark.parametrize("left", list(DROPPED))
def test_an_unseen_gap_survives_a_dropped_row(tmp_path, left, row_ids):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="test")
        db.append_message("sid", "user", "U1")
        db.append_message("sid", "assistant", "A1")
        held = db.get_messages_as_conversation("sid", include_row_ids=row_ids)
        # Another surface writes after this process loaded its prefix; its live list never loads the gap.
        db.append_message("sid", "user", "UNSEEN_USER")
        db.append_message("sid", "assistant", "UNSEEN_REPLY")
        db.append_message("sid", "user", "U2")
        db.append_message("sid", "assistant", "A2")
        for role, content, fields in DROPPED[left]:
            db.append_message("sid", role, content, **fields)
        db.append_message("sid", "user", "U3")
        db.append_message("sid", "assistant", "A3")
        held.extend(db.get_messages_as_conversation("sid", include_row_ids=row_ids)[4:])
        repair_message_sequence(None, held)
        covered, unresolved = coverage_for_commit(db, "sid", held)

        db.archive_and_compact(
            "sid", [{"role": "user", "content": "summary of the held rows"}],
            watermark=db.get_active_message_watermark("sid"), covered_ids=covered, unresolved_held=unresolved)

        live = db.get_messages("sid")
        assert [row["content"] for row in live] == ["summary of the held rows", "UNSEEN_USER", "UNSEEN_REPLY"]
        assert not any(row["tool_calls"] or row["role"] == "tool" for row in live)
    finally:
        db.close()


def test_the_named_rows_stay_off_the_request_copy(session):
    from agent.conversation_compression_archive import RETIRED_DURABLE_ROWS
    from agent.turn_context import build_api_messages
    from tests.agent.test_in_place_merged_user_rows import KILLED_TURN_LEFT

    db, agent = session
    _turn(db, agent, SimpleNamespace(conversation_history=[]), "cli", 1, 5_000)
    for role, content, fields in KILLED_TURN_LEFT["two_prompts_tool_call"]:
        db.append_message("sid", role, content, **fields)
    history = db.get_messages_as_conversation("sid", repair_alternation=True)
    assert RETIRED_DURABLE_ROWS in history[-1]

    request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="")

    assert not any(RETIRED_DURABLE_ROWS in message for message in request)
