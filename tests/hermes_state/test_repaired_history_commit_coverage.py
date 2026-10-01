"""The alternation repair's provenance keeps compaction from archiving rows the compressor never held.

Another surface appends rows after this process loaded its prefix. The held history is repaired
(dropped tool rows, merged users, folded assistants) and reloaded without row ids, so the commit has
to name each held row from the repair's records. Falling back to the watermark, or matching by text,
archives those unseen rows: they leave the model's history without being in the summary.
"""

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.conversation_compression_archive import coverage_for_commit
from hermes_state import SessionDB

SUMMARY = "summary of the held rows"
UNSEEN = [("user", "UNSEEN_USER", {}), ("assistant", "UNSEEN_REPLY", {})]


def _call(call_id):
    return {"tool_calls": [{"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}


# behaviour -> (rows another surface writes before the held suffix, the held suffix, render the held
# suffix with gateway timestamps, rows written after the snapshot, the active rows expected after commit)
CASES = {
    "timestamp_rendered_merged_user_names_its_run": (
        UNSEEN, [("user", "U2", {}), ("user", "U3", {}), ("assistant", "A3", {})], True, [],
        [SUMMARY, "UNSEEN_USER", "UNSEEN_REPLY"]),
    "dropped_row_is_named_not_a_watermark_fallback": (
        UNSEEN, [("user", "U2", {}), ("assistant", "A2", {}), ("tool", "ORPHAN", {"tool_call_id": "killed"}),
                 ("user", "U3", {}), ("assistant", "A3", {})], False, [],
        [SUMMARY, "UNSEEN_USER", "UNSEEN_REPLY"]),
    "composed_drops_record_the_original_row_once": (
        UNSEEN, [("user", "U2", {}), ("assistant", "tool setup", _call("z")),
                 ("tool", "answer z", {"tool_call_id": "z"}), ("assistant", "", _call("a")),
                 ("assistant", "", _call("b")), ("user", "U3", {}), ("assistant", "A3", {})], False, [],
        [SUMMARY, "UNSEEN_USER", "UNSEEN_REPLY"]),
    "users_adjacent_only_after_a_drop_skip_the_dropped_row": (
        UNSEEN, [("user", "U2", {}), ("tool", "orphan", {"tool_call_id": "orphan"}), ("user", "U3", {}),
                 ("assistant", "A3", {})], False, [],
        [SUMMARY, "UNSEEN_USER", "UNSEEN_REPLY"]),
    "folded_assistant_text_never_claims_a_later_row": (
        [], [("user", "U2", {}), ("assistant", "A2", {}), ("assistant", "A2b", {}), ("user", "U3", {}),
             ("assistant", "A3", {})], False, [("user", "UNSEEN_USER", {}), ("assistant", "A2\nA2b", {})],
        [SUMMARY, "UNSEEN_USER", "A2\nA2b"]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_repaired_reload_commit_keeps_unseen_rows_live(tmp_path, case):
    from gateway.run import _build_gateway_agent_history

    unseen, suffix, render, late, expected = CASES[case]
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="test")
        db.append_message("sid", "user", "U1")
        db.append_message("sid", "assistant", "A1")
        held = db.get_messages_as_conversation("sid", include_row_ids=True)
        for role, content, fields in [*unseen, *suffix]:
            db.append_message("sid", role, content, **fields)
        held.extend(db.get_messages_as_conversation("sid", include_row_ids=False)[2 + len(unseen):])
        watermark = db.get_active_message_watermark("sid")
        repair_message_sequence(None, held)
        if render:
            held[2:], _ = _build_gateway_agent_history(held[2:], inject_timestamps=True)
        for role, content, fields in late:
            db.append_message("sid", role, content, **fields)

        covered, unresolved = coverage_for_commit(db, "sid", held)
        db.archive_and_compact(
            "sid", [{"role": "user", "content": SUMMARY}],
            watermark=watermark, covered_ids=covered, unresolved_held=unresolved)

        assert [row["content"] for row in db.get_messages("sid")] == expected
    finally:
        db.close()
