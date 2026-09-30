"""In-place compaction over a turn that was killed before it finished.

A crash between turn start and the reply leaves a durable ``user;user`` pair. The alternation repair
hands the compressor one dict for both rows, so the commit has to account for two originals behind it.
A turn killed later leaves a tool call with no result, or a result the repair cannot pair. The repair
drops those rows, and they are still behind the dict before them. Two assistant rows in a row are
folded into one turn the same way.
"""

from types import SimpleNamespace

import pytest

from tests.agent.test_in_place_preflight_rewind import _replies_displayed, _turn, session  # noqa: F401


def _call(call_id):
    return ("assistant", "", {"tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})


_UNANSWERED = ("user", "U12x this prompt never got a reply", {})
_RETRIED = ("user", "U12y the same request again", {})
_RESULT = ("tool", "partial output", {"tool_call_id": "call_killed"})
KILLED_TURN_LEFT = {
    "prompt": [_UNANSWERED],
    "two_prompts": [_UNANSWERED, _RETRIED],
    "tool_call": [_UNANSWERED, _call("call_killed")],
    "tool_result": [_UNANSWERED, _RESULT],
    "two_prompts_tool_call": [_UNANSWERED, _RETRIED, _call("call_killed")],
    "two_tool_calls": [_UNANSWERED, _call("call_killed"), _call("call_killed_again")],
    "result_after_a_reply": [_RESULT],
    "two_replies": [("assistant", "A12b a second reply to the same prompt", {})],
}


@pytest.mark.parametrize("surface", ["cli", "resume", "gateway"])
@pytest.mark.parametrize("left", list(KILLED_TURN_LEFT))
def test_compaction_over_an_unanswered_prompt_keeps_one_copy_of_every_row(session, surface, left):
    db, agent = session
    cli = SimpleNamespace(conversation_history=[])
    for n in range(1, 13):
        _turn(db, agent, cli, surface, n, 5_000)
    for role, content, fields in KILLED_TURN_LEFT[left]:
        db.append_message("sid", role, content, **fields)
    if surface == "cli":  # ACP and the classic CLI restore start from the repaired reload
        cli.conversation_history = db.get_messages_as_conversation("sid", repair_alternation=True)
    elif surface == "resume":  # --resume and the TUI load the same history with row ids
        cli.conversation_history = db.get_resume_conversations("sid")[0]
    _turn(db, agent, cli, surface, 13, 5_000)
    _turn(db, agent, cli, surface, 14, 200_000)  # real usage over the threshold: the next turn compacts first

    _turn(db, agent, cli, surface, 15, 20_000)

    assert getattr(agent, "_last_compaction_in_place", None) is True
    assert {f"A{n}" for n in range(1, 16)} <= _replies_displayed(db)
    live = [m["content"] for m in db.get_messages_as_conversation("sid") if isinstance(m.get("content"), str)]
    # The prompts the killed turn left are in exactly one live row, the merged carried copy.
    unanswered = [c for role, c, _ in KILLED_TURN_LEFT[left] if role == "user"]
    merged = "\n\n".join([*unanswered, "U13 please continue with the next step"])
    assert [c for c in live if "U12x" in c or "U12y" in c] == ([merged] if unanswered else [])
    recalled = [row["content"] for row in db._conn.execute(
        "SELECT content FROM messages WHERE session_id = 'sid' AND (active = 1 OR compacted = 1)").fetchall()]
    carried = live[next(i for i, c in enumerate(live) if "Numbered steps" in c) + 1:]
    assert [recalled.count(content) for content in carried] == [1] * len(carried)
    # No turn here uses a tool, so a live tool row is the killed turn's, appended behind the running turn.
    assert db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'sid' AND active = 1"
        " AND (role = 'tool' OR tool_calls IS NOT NULL)").fetchone()[0] == 0
    assert [c.split(" ")[0] for c in live[-2:]] == ["U15", "A15"]


@pytest.mark.parametrize("row_ids", [True, False])
@pytest.mark.parametrize("left", ["tool_result", "tool_call"])
def test_a_row_dropped_ahead_of_the_first_survivor_is_archived_with_it(tmp_path, left, row_ids):
    """Nothing is kept before a row the repair drops at the head of the history, so the first
    survivor stands for it."""
    from agent.conversation_compression_archive import coverage_for_commit
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sid", source="test")
    for role, content, fields in (KILLED_TURN_LEFT[left][-1], ("user", "U1", {}), ("assistant", "A1", {})):
        db.append_message("sid", role, content, **fields)
    watermark = db.get_active_message_watermark("sid")
    held = db.get_messages_as_conversation("sid", repair_alternation=True, include_row_ids=row_ids)
    assert [m["content"] for m in held] == ["U1", "A1"]
    covered_ids, unresolved = coverage_for_commit(db, "sid", held)

    db.archive_and_compact(
        "sid", [{"role": "user", "content": "summary"}], watermark=watermark,
        covered_ids=covered_ids, unresolved_held=unresolved)

    assert [row["content"] for row in db.get_messages("sid")] == ["summary"]
    db.close()


def test_the_repair_row_counts_stay_off_the_request_copy(session):
    """The commit reads the counts off the live dict. A transport is handed the request copy, and
    one that forwards unknown keys must not find them there."""
    from agent.conversation_compression_archive import MERGED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS
    from agent.turn_context import build_api_messages

    db, agent = session
    _turn(db, agent, SimpleNamespace(conversation_history=[]), "cli", 1, 5_000)
    for role, content, fields in KILLED_TURN_LEFT["two_prompts_tool_call"]:
        db.append_message("sid", role, content, **fields)
    history = db.get_messages_as_conversation("sid", repair_alternation=True)
    counts = {MERGED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS}
    assert counts <= set(history[-1])  # two prompts merged, and the tool call dropped behind them

    request, _ = build_api_messages(
        agent, history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="")

    assert not any(counts & set(message) for message in request)
    assert counts <= set(history[-1])
