"""In-place compaction over a turn that was killed before it finished.

A crash between turn start and the reply leaves a durable ``user;user`` pair. The alternation repair
hands the compressor one dict for both rows, so the commit has to account for two originals behind it.
A turn killed later leaves a tool call with no result, or a result the repair cannot pair. The repair
drops those rows, and they are still behind the dict before them.
"""

from types import SimpleNamespace

import pytest

from tests.agent.test_in_place_preflight_rewind import _replies_displayed, _turn, session  # noqa: F401

_UNANSWERED = ("user", "U12x this prompt never got a reply", {})
_RETRIED = ("user", "U12y the same request again", {})
_CALL = ("assistant", "", {"tool_calls": [
    {"id": "call_killed", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
_RESULT = ("tool", "partial output", {"tool_call_id": "call_killed"})
KILLED_TURN_LEFT = {
    "prompt": [_UNANSWERED],
    "two_prompts": [_UNANSWERED, _RETRIED],
    "tool_call": [_UNANSWERED, _CALL],
    "tool_result": [_UNANSWERED, _RESULT],
    "two_prompts_tool_call": [_UNANSWERED, _RETRIED, _CALL],
    "result_after_a_reply": [_RESULT],
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
