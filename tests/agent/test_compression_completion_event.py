"""Completion hooks describe accepted context, not discarded SQL candidates."""
import copy
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from tests.agent.test_compression_concurrent_fork import _build_agent_with_db


SUMMARY = [
    {"role": "user", "content": "Summary of prior work."},
    {"role": "assistant", "content": "Ready."},
    {"role": "user", "content": "Continue."},
]


# Rotation publishes its prompt in the transcript transaction, so only
# in-place publication has a separate post-publication prompt-failure case.
@pytest.mark.parametrize("in_place,failure,observer_raises", [
    pytest.param(True, None, False, id="success-in-place"),
    pytest.param(False, None, False, id="success-rotation"),
    pytest.param(True, "insert", False, id="rollback-in-place"),
    pytest.param(False, "insert", False, id="rollback-rotation"),
    pytest.param(True, "prompt", False, id="post-publication-in-place"),
    pytest.param(True, None, True, id="observer-error-in-place"),
    pytest.param(False, None, True, id="observer-error-rotation"),
    pytest.param(True, "prompt", True, id="observer-error-post-publication"),
])
def test_completion_event_tracks_sql_publication(tmp_path, in_place, failure, observer_raises):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = "completion-test"
        db.create_session(sid, source="test")
        for i in range(12):
            db.append_message(sid, role="user" if i % 2 == 0 else "assistant",
                              content=f"turn-{i} " + "evidence " * 120, timestamp=1000 + i)
        agent = _build_agent_with_db(db, sid)
        agent.compression_in_place = in_place
        agent._build_system_prompt = lambda *a, **kw: "synthetic system"
        agent._cached_system_prompt = None
        agent.tools = []
        agent._memory_manager = MagicMock()
        agent.context_compressor.compress.side_effect = lambda *a, **kw: copy.deepcopy(SUMMARY)
        messages = db.get_messages_as_conversation(sid)
        for message in messages:
            message["_db_persisted"] = True
        agent._session_messages = messages
        agent._persist_user_message_idx = len(messages)
        original = copy.deepcopy(messages)
        before_rows = [dict(r) for r in db._conn.execute("SELECT * FROM messages ORDER BY id")]
        events = []
        def observe(event, payload):
            events.append((event, payload, db.get_messages_as_conversation(agent.session_id)))
            if observer_raises:
                raise RuntimeError("test observer failure after persisted snapshot")
        agent.event_callback = observe
        if failure == "insert":
            db._conn.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON messages "
                             "BEGIN SELECT RAISE(ABORT, 'test insert rollback'); END")
        elif failure == "prompt":
            db._conn.execute("CREATE TRIGGER fail_prompt BEFORE UPDATE OF system_prompt ON sessions "
                             "BEGIN SELECT RAISE(ABORT, 'test prompt rollback'); END")
        db._conn.commit()
        sql = []
        db._conn.set_trace_callback(sql.append)
        returned, _ = agent._compress_context(messages, "synthetic system", force=True, approx_tokens=120000)
        completions = [e for e in events if e[0] == "session:compress"]
        assert db.get_compression_lock_holder(sid) is None
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if failure == "insert":
            assert "ROLLBACK" in sql
            assert [dict(r) for r in db._conn.execute("SELECT * FROM messages ORDER BY id")] == before_rows
            assert returned == messages == original
            assert agent.session_id == sid
            assert db._conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
            assert completions == []
        else:
            projection = lambda rows: [(m["role"], m["content"]) for m in rows]
            assert projection(returned) == projection(SUMMARY)
            assert len(completions) == 1
            assert projection(completions[0][2]) == projection(returned)
            assert all(m["_db_persisted"] for m in returned)
            after_rows = [dict(r) for r in db._conn.execute("SELECT * FROM messages ORDER BY id")]
            by_id = {r["id"]: r for r in after_rows}
            for old_row in before_rows:
                expected = dict(old_row)
                if in_place:
                    expected.update(active=0, compacted=1)
                assert by_id[old_row["id"]] == expected
            assert completions[0][1]["in_place"] is in_place
            assert completions[0][1]["old_session_id"] == ("" if in_place else sid)
            if failure == "prompt":
                archive = next(i for i, statement in enumerate(sql) if "UPDATE messages" in statement)
                commit = sql.index("COMMIT", archive)
                prompt = next(i for i, statement in enumerate(sql) if statement.lstrip().startswith("UPDATE sessions SET system_prompt"))
                assert archive < commit < prompt < sql.index("ROLLBACK", prompt)
    finally:
        db.close()


@pytest.mark.parametrize("outcome", ["accepted", "abort", "unchanged", "empty", "denied", "superseded", "callback-error"])
def test_no_store_completion_requires_an_accepted_candidate(monkeypatch, outcome):
    from types import SimpleNamespace
    from agent.conversation_compression import CompressionCommitFence

    agent = _build_agent_with_db(None, "no-store")
    agent.compression_in_place = True
    agent._build_system_prompt = lambda *a, **kw: "synthetic system"
    agent._cached_system_prompt = None
    agent.tools = []
    # A plugin-shaped compressor with no built-in private progress latch.
    compressor = SimpleNamespace(compression_count=1, last_prompt_tokens=0,
                                 last_completion_tokens=0)
    def compress(messages, *a, **kw):
        if outcome == "abort":
            compressor._last_compress_aborted = True
        if outcome == "superseded":
            compressor._compression_attempt_generation += 1
        return copy.deepcopy(messages if outcome == "unchanged" else [] if outcome == "empty" else SUMMARY)
    compressor.compress = compress
    agent.context_compressor = compressor
    if outcome == "denied":
        monkeypatch.setattr(CompressionCommitFence, "begin_commit", lambda *a, **kw: False)
    events = []
    def observe(event, payload):
        events.append((event, payload))
        if outcome == "callback-error":
            raise RuntimeError("test observer error")
    agent.event_callback = observe
    original = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} " + "evidence " * 120}
                for i in range(12)]
    returned, _ = agent._compress_context(copy.deepcopy(original), "synthetic system", force=True, approx_tokens=120000)
    assert agent._session_db is None
    completions = [e for e in events if e[0] == "session:compress"]
    if outcome in ("accepted", "callback-error"):
        assert returned == SUMMARY
        assert len(completions) == 1
        assert not hasattr(compressor, "_last_compression_made_progress")
    else:
        assert returned == original
        assert completions == []
