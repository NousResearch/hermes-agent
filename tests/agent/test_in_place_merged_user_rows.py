"""In-place compaction over a prompt that never got its reply.

A crash between turn start and the reply leaves a durable ``user;user`` pair. The alternation repair
hands the compressor one dict for both rows, so the commit has to account for two originals behind it.
"""

from types import SimpleNamespace

import pytest

from tests.agent.test_in_place_preflight_rewind import _replies_displayed, _turn, session  # noqa: F401

UNANSWERED = "U12x this prompt never got a reply"


@pytest.mark.parametrize("surface", ["cli", "gateway"])
def test_compaction_over_an_unanswered_prompt_keeps_one_copy_of_every_row(session, surface):
    db, agent = session
    cli = SimpleNamespace(conversation_history=[])
    for n in range(1, 13):
        _turn(db, agent, cli, surface, n, 5_000)
    db.append_message("sid", "user", UNANSWERED)
    if surface == "cli":  # a resumed CLI session starts from the repaired reload
        cli.conversation_history = db.get_messages_as_conversation("sid", repair_alternation=True)
    _turn(db, agent, cli, surface, 13, 5_000)
    _turn(db, agent, cli, surface, 14, 200_000)  # real usage over the threshold: the next turn compacts first

    _turn(db, agent, cli, surface, 15, 20_000)

    assert getattr(agent, "_last_compaction_in_place", None) is True
    assert {f"A{n}" for n in range(1, 16)} <= _replies_displayed(db)
    live = [m["content"] for m in db.get_messages_as_conversation("sid") if isinstance(m.get("content"), str)]
    assert [c for c in live if UNANSWERED in c] == [f"{UNANSWERED}\n\nU13 please continue with the next step"]
    recalled = [row["content"] for row in db._conn.execute(
        "SELECT content FROM messages WHERE session_id = 'sid' AND (active = 1 OR compacted = 1)").fetchall()]
    carried = live[next(i for i, c in enumerate(live) if "Numbered steps" in c) + 1:]
    assert [recalled.count(content) for content in carried] == [1] * len(carried)
