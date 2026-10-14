import sqlite3

import pytest


from hermes_durability import (Journal, TXN_COMMIT, USER_MESSAGE,
                            ASSISTANT_MESSAGE, DurableRuntime)


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "j.db")


def test_commit_is_atomic_and_chained(db):
    j = Journal(db)
    with j.begin("s1") as txn:
        txn.record(USER_MESSAGE, {"text": "hi"})
        txn.record(ASSISTANT_MESSAGE, {"text": "hello"})
    recs = list(j.records("s1"))
    assert [r.record_type for r in recs] == [USER_MESSAGE, ASSISTANT_MESSAGE,
                                             TXN_COMMIT]
    ok, bad = j.verify_chain()
    assert ok and bad == -1
    j.close()


def test_rollback_writes_nothing(db):
    j = Journal(db)
    with pytest.raises(RuntimeError):
        with j.begin("s1") as txn:
            txn.record(USER_MESSAGE, {"text": "hi"})
            raise RuntimeError("boom")
    assert list(j.records("s1")) == []
    j.close()


def test_uncommitted_buffer_never_touches_disk(db):
    j = Journal(db)
    txn = j.begin("s1")
    txn.record(USER_MESSAGE, {"text": "hi"})
    # simulate crash: just drop the txn, reopen db
    j.close()
    j2 = Journal(db)
    assert list(j2.records("s1")) == []
    j2.close()


def test_torn_tail_detected_and_repaired(db):
    j = Journal(db)
    with j.begin("s1") as txn:
        txn.record(USER_MESSAGE, {"text": "one"})
    with j.begin("s1") as txn:
        txn.record(USER_MESSAGE, {"text": "two"})
    j.close()
    # corrupt the last record's payload directly
    conn = sqlite3.connect(db)
    conn.execute("UPDATE journal SET payload = ? WHERE seq ="
                 " (SELECT MAX(seq) FROM journal)", (b'{"text":"EVIL"}',))
    conn.commit()
    conn.close()
    j2 = Journal(db)
    ok, bad = j2.verify_chain(repair=True)
    assert not ok
    ok2, _ = j2.verify_chain()
    assert ok2
    # "two"'s TransactionCommit was the corrupted record: after truncation
    # its records survive raw but the transaction is uncommitted, so replay
    # must discard it.
    committed = j2.committed_transactions("s1")
    texts = [r.payload.get("text") for r in j2.records("s1")
             if r.record_type == USER_MESSAGE
             and r.transaction_id in committed]
    assert texts == ["one"]
    j2.close()


def test_snapshot_and_replay(db):
    rt = DurableRuntime(db, start_worker=False)
    for i in range(5):
        with rt.transaction("s1") as txn:
            txn.record(USER_MESSAGE, {"text": f"m{i}"})
    state = rt.replay_state("s1")
    assert len(state["messages"]) == 5
    rt.journal.compact("s1", state)
    state2 = rt.replay_state("s1")
    assert state2 == state  # snapshot equivalent to full replay
    with rt.transaction("s1") as txn:
        txn.record(USER_MESSAGE, {"text": "after-compact"})
    state3 = rt.replay_state("s1")
    assert state3["messages"][-1]["text"] == "after-compact"
    assert len(state3["messages"]) == 6
    rt.close()


def test_incomplete_snapshot_ignored(db):
    rt = DurableRuntime(db, start_worker=False)
    with rt.transaction("s1") as txn:
        txn.record(USER_MESSAGE, {"text": "m"})
    # write snapshot row but leave complete=0 (simulate crash mid-compaction)
    conn = rt.journal._conn
    conn.execute("INSERT INTO snapshot (snapshot_id, session_id, base_seq,"
                 " state, checksum, complete, created_at)"
                 " VALUES ('x','s1',999,'{}',x'00',0,0)")
    assert rt.journal.latest_snapshot("s1") is None
    state = rt.replay_state("s1")
    assert len(state["messages"]) == 1
    rt.close()
