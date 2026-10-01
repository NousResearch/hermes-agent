"""Composed repairs must not archive context the compressor never held (#129123)."""

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.conversation_compression_archive import coverage_for_commit
from hermes_state import SessionDB


def _call(call_id):
    return {"tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "terminal", "arguments": "{}",
    }}]}


@pytest.mark.parametrize("row_ids", [False, True])
@pytest.mark.parametrize("shape", ["chained_calls", "users_split_by_drop"])
def test_composed_repairs_preserve_unseen_context(tmp_path, row_ids, shape):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="test")
        db.append_message("sid", "user", "U1")
        db.append_message("sid", "assistant", "A1")
        held = db.get_messages_as_conversation("sid", include_row_ids=row_ids)
        # Deterministic interleaving: these rows never enter the held snapshot.
        db.append_message("sid", "user", "UNSEEN_USER")
        db.append_message("sid", "assistant", "UNSEEN_REPLY")
        db.append_message("sid", "user", "U2")
        if shape == "chained_calls":
            db.append_message("sid", "assistant", "tool setup", **_call("z"))
            db.append_message("sid", "tool", "answer z", tool_call_id="z")
            db.append_message("sid", "assistant", "", **_call("a"))
            db.append_message("sid", "assistant", "", **_call("b"))
        else:
            db.append_message("sid", "tool", "orphan", tool_call_id="orphan")
        db.append_message("sid", "user", "U3")
        db.append_message("sid", "assistant", "A3")
        held.extend(db.get_messages_as_conversation("sid", include_row_ids=row_ids)[4:])
        repair_message_sequence(None, held)
        covered, unresolved = coverage_for_commit(db, "sid", held)
        db.archive_and_compact(
            "sid", [{"role": "user", "content": "summary HELD only"}],
            watermark=db.get_active_message_watermark("sid"),
            covered_ids=covered, unresolved_held=unresolved,
        )
        assert [m["content"] for m in db.get_messages("sid")] == [
            "summary HELD only", "UNSEEN_USER", "UNSEEN_REPLY",
        ]
    finally:
        db.close()


@pytest.mark.parametrize("row_ids", [False, True])
@pytest.mark.parametrize("layout", ["gap", "after_watermark"])
def test_folded_assistant_does_not_claim_unseen_row_with_its_text(tmp_path, row_ids, layout):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="test")
        db.append_message("sid", "user", "U1")
        db.append_message("sid", "assistant", "A1")
        held = db.get_messages_as_conversation("sid", include_row_ids=row_ids)
        # Another surface's reply whose text equals the fold of A2 and A2b, below or above the watermark.
        if layout == "gap":
            db.append_message("sid", "user", "UNSEEN_USER")
            db.append_message("sid", "assistant", "A2\nA2b")
        db.append_message("sid", "user", "U2")
        db.append_message("sid", "assistant", "A2")
        db.append_message("sid", "assistant", "A2b")
        db.append_message("sid", "user", "U3")
        db.append_message("sid", "assistant", "A3")
        loaded = db.get_messages_as_conversation("sid", include_row_ids=row_ids)
        held.extend(loaded[4:] if layout == "gap" else loaded[2:])
        watermark = db.get_active_message_watermark("sid")
        repair_message_sequence(None, held)
        if layout == "after_watermark":
            db.append_message("sid", "user", "UNSEEN_USER")
            db.append_message("sid", "assistant", "A2\nA2b")
        covered, unresolved = coverage_for_commit(db, "sid", held)
        db.archive_and_compact(
            "sid", [{"role": "user", "content": "summary HELD only"}],
            watermark=watermark, covered_ids=covered, unresolved_held=unresolved,
        )
        assert [m["content"] for m in db.get_messages("sid")] == [
            "summary HELD only", "UNSEEN_USER", "A2\nA2b",
        ]
    finally:
        db.close()
