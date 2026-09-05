"""Persist wire context without changing another turn's cached prefix."""

from types import SimpleNamespace

import pytest

from agent.turn_context import _stamp_api_content_sidecar, compose_user_api_content
from hermes_state import SessionDB


@pytest.mark.parametrize("compacted", [False, True])
def test_early_persist_replays_exact_context_without_rewriting_identical_turn(
    tmp_path, compacted
):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session("session", source="cli")
        old = "ok\n\nprevious context"
        db.append_message("session", "user", content="ok", api_content=old)
        db.append_message("session", "assistant", content="done")
        current = {"role": "user", "content": "ok"}
        if compacted:
            db.archive_and_compact(
                "session",
                [
                    {"role": "user", "content": "ok", "api_content": old},
                    {"role": "assistant", "content": "done"},
                    current,
                ],
            )
        else:
            current["_row_id"] = db.append_message("session", "user", content="ok")
        current["_db_persisted"] = True
        agent = SimpleNamespace(
            _session_db=db, session_id="session", _last_compaction_in_place=compacted
        )
        _stamp_api_content_sidecar(
            agent,
            [current],
            0,
            "remember this",
            "plugin context",
            preflight_compressed=compacted,
        )
    finally:
        db.close()
    reopened = SessionDB(db_path=path)
    try:
        users = [
            m
            for m in reopened.get_messages_as_conversation("session")
            if m["role"] == "user"
        ]
        assert users[0]["api_content"] == old
        assert users[-1]["content"] == "ok"
        assert users[-1]["api_content"] == compose_user_api_content(
            "ok", "remember this", "plugin context"
        )
    finally:
        reopened.close()


def test_unpersisted_turn_cannot_backfill_a_previous_identical_message(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session", source="cli")
        old = "ok\n\nprevious context"
        db.append_message("session", "user", content="ok", api_content=old)
        agent = SimpleNamespace(_session_db=db, session_id="session")
        for marker in ({}, {"_db_persisted": True}, {"_row_id": True}, {"_row_id": -1}):
            current = {"role": "user", "content": "ok", **marker}
            _stamp_api_content_sidecar(
                agent, [current], 0, "new context", "", preflight_compressed=False
            )
            assert db.get_messages_as_conversation("session")[0]["api_content"] == old
    finally:
        db.close()
