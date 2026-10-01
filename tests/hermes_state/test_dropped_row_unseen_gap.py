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


@pytest.mark.parametrize("trigger", ["ambiguous", "rendered", "orphan", "folded"])
def test_unresolved_repair_provenance_never_archives_unseen_rows(tmp_path, trigger):
    from gateway.run import _build_gateway_agent_history

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="test")
        db.append_message("sid", "user", "U1")
        db.append_message("sid", "assistant", "A1")
        held = db.get_messages_as_conversation("sid", include_row_ids=True)
        pair = ["continue", "please continue"] if trigger == "ambiguous" else ["U2", "U3"]
        unseen = [("user", pair[0], {}), ("user", pair[1], {}), ("assistant", "UNSEEN_REPLY", {})]
        if trigger == "rendered":
            unseen = [("user", "UNSEEN_USER", {}), ("assistant", "UNSEEN_REPLY", {})]
        if trigger in {"orphan", "folded"}:
            extra = DROPPED["tool_result" if trigger == "orphan" else "two_replies"]
            unseen = [("user", "UNSEEN_USER", {}), *extra, ("assistant", "UNSEEN_REPLY", {})]
            suffix = [("user", "U2", {}), ("assistant", "A2", {}), *extra,
                      ("user", "U3", {}), ("assistant", "A3", {})]
        else:
            suffix = [("user", pair[0], {}), ("user", pair[1], {}), ("assistant", "A3", {})]
        for role, content, fields in [*unseen, *suffix]:
            db.append_message("sid", role, content, **fields)
        held.extend(db.get_messages_as_conversation("sid", include_row_ids=False)[2 + len(unseen):])
        repair_message_sequence(None, held)
        if trigger == "rendered":
            held[2:], _ = _build_gateway_agent_history(held[2:], inject_timestamps=True)
        covered, unresolved = coverage_for_commit(db, "sid", held)
        db.archive_and_compact(
            "sid", [{"role": "user", "content": "summary of the held rows"}],
            watermark=db.get_active_message_watermark("sid"), covered_ids=covered, unresolved_held=unresolved)
        active = [row["content"] for row in db.get_messages("sid")]
        assert "UNSEEN_REPLY" in active
        assert all(content in active for role, content, fields in unseen)
        if trigger == "rendered":
            assert active == ["summary of the held rows", *[content for role, content, fields in unseen]]
    finally:
        db.close()


def test_idless_tool_pairs_with_repeated_content_have_exact_coverage(tmp_path):
    """Call identity distinguishes held empty bodies / results from an unseen pair."""
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("sid", source="test")
        db.append_message("sid", "user", "U1")
        held = db.get_messages_as_conversation("sid")
        for call_id in ("held_1", "unseen", "held_2"):
            db.append_message("sid", "assistant", "", tool_calls=[{
                "id": call_id, "type": "function",
                "function": {"name": "terminal", "arguments": "{}"}}])
            db.append_message("sid", "tool", "ok", tool_call_id=call_id)
            if call_id != "unseen":
                held.extend(db.get_messages_as_conversation("sid")[-2:])
        covered, unresolved = coverage_for_commit(db, "sid", held)
        db.archive_and_compact(
            "sid", [{"role": "user", "content": "summary of the held rows"}],
            watermark=db.get_active_message_watermark("sid"),
            covered_ids=covered, unresolved_held=unresolved)
        active = db.get_messages("sid")
        assert [row["role"] for row in active] == ["user", "assistant", "tool"]
        assert active[-1]["tool_call_id"] == "unseen"
        assert active[-2]["tool_calls"][0]["id"] == "unseen"
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
