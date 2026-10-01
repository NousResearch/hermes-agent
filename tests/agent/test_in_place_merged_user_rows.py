"""In-place compaction over a turn that was killed before it finished.

A crash between turn start and the reply leaves a durable ``user;user`` pair. The alternation repair
hands the compressor one dict for both rows, so the commit has to account for two originals behind it.
A turn killed later leaves a tool call with no result; the repair drops that row, and it is still
behind the dict before it.
"""

from types import SimpleNamespace

import pytest

from tests.agent import test_in_place_preflight_rewind as _rewind
from tests.agent.test_in_place_preflight_rewind import _replies_displayed, _turn

session = _rewind.session  # shared fixture


def _call(call_id):
    return ("assistant", "", {"tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})


_UNANSWERED = ("user", "U12x this prompt never got a reply", {})
_MERGED = "\n\n".join([_UNANSWERED[1], "U13 please continue with the next step"])
_REPEATED = [("user", "U12x continue", {}), ("assistant", "U12x ok", {})] * 2
# rows the killed turn left -> the live rows that mention them afterwards
KILLED_TURN_LEFT = {
    "prompt": ([_UNANSWERED], [_MERGED]),  # #125564: user;user, the unanswered prompt must not be sent twice
    "tool_call": ([_UNANSWERED, _call("call_killed")], [_MERGED]),  # #129123: the dropped tool row is accounted for
    # Identical durable rows on an id-less reload cannot be named: the watermark path, never a re-clone behind U15.
    "repeated": (_REPEATED, [c for _, c, _ in _REPEATED]),
    "benign": ([], []),  # plain alternating history: compaction is unchanged
}


@pytest.mark.parametrize("surface", ["cli", "resume", "gateway"])
@pytest.mark.parametrize("left", list(KILLED_TURN_LEFT))
def test_compaction_over_an_unanswered_prompt_keeps_one_copy_of_every_row(session, surface, left):
    db, agent = session
    cli = SimpleNamespace(conversation_history=[])
    for n in range(1, 13):
        _turn(db, agent, cli, surface, n, 5_000)
    rows, expected = KILLED_TURN_LEFT[left]
    for role, content, fields in rows:
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
    # The rows the killed turn left are live exactly once (an unanswered prompt only as the merged carried copy).
    assert [c for c in live if "U12x" in c] == expected
    recalled = [row["content"] for row in db._conn.execute(
        "SELECT content FROM messages WHERE session_id = 'sid' AND (active = 1 OR compacted = 1)").fetchall()]
    carried = live[next(i for i, c in enumerate(live) if "Numbered steps" in c) + 1:]
    assert [recalled.count(content) for content in carried] == [carried.count(content) for content in carried]
    # No turn here uses a tool, so a live tool row is the killed turn's, appended behind the running turn.
    assert db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'sid' AND active = 1"
        " AND (role = 'tool' OR tool_calls IS NOT NULL)").fetchone()[0] == 0
    assert [c.split(" ")[0] for c in live[-2:]] == ["U15", "A15"]
