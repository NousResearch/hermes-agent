"""Tests for Hermes Code Mode realtime event bus."""

from __future__ import annotations

from hermes_cli.code.event_bus import CodeEventBus, build_event_filters_from_query
from hermes_state import SessionDB


def _bus(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    return CodeEventBus(db_path=db_path), db_path


def _seed_workspace(db_path, workspace_id: str):
    import time

    db = SessionDB(db_path=db_path)
    try:
        now = time.time()
        db._conn.execute(
            """
            INSERT OR REPLACE INTO code_workspaces
                (id, name, owner, repo, path, git_remote, repo_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                workspace_id,
                "owner",
                "repo",
                f"/tmp/{workspace_id}",
                "",
                "",
                now,
                now,
            ),
        )
        db._conn.commit()
    finally:
        db.close()


def test_publish_persists_event(tmp_path):
    bus, db_path = _bus(tmp_path)
    _seed_workspace(db_path, "ws-1")
    event = bus.publish(
        "code.approval.created",
        payload={"approval_id": "a1"},
        workspace_id="ws-1",
        code_session_id="cs-1",
        approval_id="a1",
    )
    assert event["type"] == "code.approval.created"
    assert event["approval_id"] == "a1"

    db = SessionDB(db_path=db_path)
    try:
        rows = db.list_code_events(workspace_id="ws-1", session_id="cs-1", limit=10)
    finally:
        db.close()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "code.approval.created"


def test_publish_broadcasts_to_subscribers(tmp_path):
    bus, _db_path = _bus(tmp_path)
    sub_id, sub_queue = bus.subscribe(filters={"type": "code.approval.approved"})
    try:
        bus.publish("code.approval.approved", payload={"status": "approved"})
        event = sub_queue.get(timeout=1.0)
    finally:
        bus.unsubscribe(sub_id)
    assert event["type"] == "code.approval.approved"


def test_filters_type_workspace_session_approval_repo(tmp_path):
    bus, _db_path = _bus(tmp_path)
    _seed_workspace(_db_path, "ws-1")
    _seed_workspace(_db_path, "ws-2")
    bus.publish(
        "github.write.executed",
        payload={"ok": True},
        workspace_id="ws-1",
        code_session_id="cs-1",
        approval_id="ap-1",
        github_repo_full_name="acme/repo",
    )
    bus.publish(
        "github.write.executed",
        payload={"ok": True},
        workspace_id="ws-2",
        code_session_id="cs-2",
        approval_id="ap-2",
        github_repo_full_name="acme/other",
    )
    items = bus.fetch_events(
        filters=build_event_filters_from_query(
            event_type="github.write.executed",
            workspace_id="ws-1",
            code_session_id="cs-1",
            approval_id="ap-1",
            github_repo_full_name="acme/repo",
        ),
        limit=50,
    )
    assert len(items) == 1
    assert items[0]["workspace_id"] == "ws-1"
    assert items[0]["approval_id"] == "ap-1"


def test_replay_since_id_uses_persisted_order(tmp_path):
    bus, _db_path = _bus(tmp_path)
    first = bus.publish("code.test.a", payload={"n": 1})
    second = bus.publish("code.test.b", payload={"n": 2})
    third = bus.publish("code.test.c", payload={"n": 3})

    replay = bus.replay(since_id=second["id"], limit=10)
    assert [event["id"] for event in replay] == [third["id"]]
    assert replay[0]["payload"]["n"] == 3
    assert first["id"] != third["id"]


def test_redaction_recursive_for_persist_and_replay(tmp_path):
    bus, _db_path = _bus(tmp_path)
    bus.publish(
        "code.secret.test",
        payload={
            "token": "abc123",
            "nested": {"authorization": "Bearer super-secret", "safe": "ok"},
            "list": [{"password": "hidden"}, {"value": "ok"}],
        },
    )
    events = bus.fetch_events(filters={"type": "code.secret.test"}, limit=10)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["token"] == "[REDACTED]"
    assert payload["nested"]["authorization"] == "[REDACTED]"
    assert payload["nested"]["safe"] == "ok"
    assert payload["list"][0]["password"] == "[REDACTED]"


def test_no_subscriber_safety(tmp_path):
    bus, _db_path = _bus(tmp_path)
    event = bus.publish("code.events.safe", payload={"x": 1})
    assert event["type"] == "code.events.safe"


def test_stale_subscriber_cleanup(tmp_path):
    bus, _db_path = _bus(tmp_path)
    sub_id, _sub_queue = bus.subscribe(filters={"type": "code.queue.test"}, max_queue_size=1)
    bus.publish("code.queue.test", payload={"n": 1})
    # second push overflows the queue and should evict stale subscriber
    bus.publish("code.queue.test", payload={"n": 2})
    stats = bus.subscription_stats()
    assert stats["active_subscribers"] == 0
    # unsubscribe remains safe even if already evicted
    bus.unsubscribe(sub_id)
