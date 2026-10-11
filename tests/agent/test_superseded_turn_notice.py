"""A turn superseded by the user's newer message is not a failed turn (#136146).

``_close_durable_failed_turn`` appends the ``FAILED_TURN_NOTICE`` boundary when a turn ends
with ``user`` as the durable tail. An interrupt that carries the user's next message
(``interrupt_message``) means the request was superseded, not refused, so no
"not processed, send it again" row may be written; genuine failures keep it.
"""

from __future__ import annotations

from agent.conversation_loop import _close_durable_failed_turn
from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, FAILED_TURN_NOTICE


class _FakeDB:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def latest_conversation_role(self, session_id):
        return self.rows[-1]["role"] if self.rows else None


class _FakeAgent:
    provider = "fixture"
    model = "fixture-model"
    session_id = "s1"
    _persist_disabled = False

    def __init__(self, db: _FakeDB) -> None:
        self._session_db = db

    def _flush_messages_to_session_db(self, messages):
        self._session_db.rows[:] = [dict(m) for m in messages]


def _open_user_tail():
    db = _FakeDB()
    messages = [{"role": "user", "content": "build the report"}]
    db.rows = [dict(m) for m in messages]
    return _FakeAgent(db), db, messages


def test_superseded_turn_gets_no_failed_turn_notice():
    agent, db, messages = _open_user_tail()
    result = {
        "messages": messages, "completed": False, "interrupted": True,
        "interrupt_message": "actually, use the March numbers",
        "final_response": "Operation interrupted: waiting for model response (1.2s elapsed).",
    }

    _close_durable_failed_turn(agent, result)

    assert [m["role"] for m in messages] == ["user"]
    assert not any(r.get("display_kind") == FAILED_TURN_DISPLAY_KIND for r in db.rows)


def test_genuine_failure_still_gets_failed_turn_notice():
    agent, db, messages = _open_user_tail()
    result = {
        "messages": messages, "completed": False, "failed": True,
        "error": "provider returned 503",
    }

    _close_durable_failed_turn(agent, result)

    assert messages[-1]["content"] == FAILED_TURN_NOTICE
    assert db.rows[-1]["display_kind"] == FAILED_TURN_DISPLAY_KIND
