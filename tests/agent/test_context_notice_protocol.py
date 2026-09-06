"""Durable state ordering through real SQLite, replay and gateway wire callbacks."""
import json
import multiprocessing
import threading
from types import SimpleNamespace

import pytest

from agent.context_notices import record_compression_outcome
from hermes_state import SessionDB
from tui_gateway import event_replay, server


def _recover_in_process(path, output):
    # A compute worker has an independent process, DB connection and callback set.
    db = SessionDB(db_path=path)
    server._emit = lambda kind, sid, payload=None: output.put((kind, sid, payload))
    callbacks = server._agent_cbs("protocol-runtime")
    agent = SimpleNamespace(_session_db=db, **callbacks)
    try:
        record_compression_outcome(agent, {
            "session_id": "conversation", "attempt_id": "healthy",
            "commit_status": "committed", "fallback_used": False,
        })
    finally:
        db.close()


@pytest.mark.parametrize("isolated", [False, True])
def test_reconnect_snapshot_carries_state_order_not_publication_order(tmp_path, monkeypatch, isolated):
    frames = []

    def emit(kind, sid, payload=None):
        frame = {"method": "event", "params": {"type": kind, "session_id": sid, "payload": payload}}
        event_replay._stamp_event(frame)
        # Real wire serialization must not lose the revision on string-like keys.
        frames.append(json.loads(json.dumps(frame["params"])))

    monkeypatch.setattr(server, "_emit", emit)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("conversation", source="gui")
    callbacks = server._agent_cbs("protocol-runtime")
    agent = SimpleNamespace(_session_db=db, **callbacks)
    for attempt in ("bad-1", "bad-2"):
        record_compression_outcome(agent, {
            "session_id": "conversation", "attempt_id": attempt,
            "commit_status": "aborted", "failure_class": "summary_network_failure",
        })
    read, release = threading.Event(), threading.Event()
    getter = db.get_context_notice_state

    def paused_snapshot(session_id):
        snapshot = getter(session_id)
        read.set()
        assert release.wait(15)
        return snapshot

    monkeypatch.setattr(db, "get_context_notice_state", paused_snapshot)
    monkeypatch.setattr(server, "_sessions", {"protocol-runtime": {"session_key": "conversation", "agent": None}})
    replay = threading.Thread(target=server._replay_context_notice, args=("protocol-runtime",), kwargs={"db": db})
    replay.start()
    try:
        assert read.wait(5)
        if isolated:
            ctx = multiprocessing.get_context("spawn")
            output = ctx.Queue()
            worker = ctx.Process(target=_recover_in_process, args=(db.db_path, output))
            worker.start()
            try:
                for _ in range(2):
                    emit(*output.get(timeout=15))
                worker.join(10)
                assert worker.exitcode == 0
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(5)
                output.close()
        else:
            record_compression_outcome(agent, {
                "session_id": "conversation", "attempt_id": "healthy",
                "commit_status": "committed", "fallback_used": False,
            })
        release.set()
        replay.join(5)
        assert not replay.is_alive()
        assert getter("conversation")["failures"] == 0
        clear = next(frame for frame in frames if frame["type"] == "notification.clear")
        stale = frames[-1]
        assert stale["type"] == "notification.show" and stale["payload"]["kind"] == "sticky"
        assert stale["seq"] > clear["seq"], "publication seq alone reproduces the stale replay bug"
        assert "state_revision" in clear["payload"], "clear must carry durable state order across the worker wire"
        assert stale["payload"]["state_revision"] < clear["payload"]["state_revision"]
        assert stale["payload"]["state_key"] == clear["payload"]["state_key"]
        # A delayed recovery TTL from the compute wire must also lose to newer
        # failures, despite its different toast key and later publication seq.
        recovery = next(frame for frame in frames if frame["payload"].get("kind") == "ttl")
        for attempt in ("bad-3", "bad-4"):
            record_compression_outcome(agent, {
                "session_id": "conversation", "attempt_id": attempt,
                "commit_status": "aborted", "failure_class": "summary_network_failure",
            })
            if attempt == "bad-3":
                assert frames[-1]["type"] == "notification.clear", "a new fault must invalidate an old recovery TTL"
                assert frames[-1]["payload"]["state_revision"] == getter("conversation")["revision"]
        latest_warning = frames[-1]
        emit(recovery["type"], recovery["session_id"], recovery["payload"])
        assert frames[-1]["seq"] > latest_warning["seq"]
        assert recovery["payload"]["state_revision"] < latest_warning["payload"]["state_revision"]
        assert recovery["payload"]["state_key"] == latest_warning["payload"]["state_key"]
        print("CONTEXT_NOTICE_WIRE=" + json.dumps(frames))
    finally:
        release.set()
        replay.join(5)
        db.close()
