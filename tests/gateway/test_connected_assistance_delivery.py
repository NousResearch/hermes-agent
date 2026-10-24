"""Private canonical event-stream delivery; never call a messaging send fallback."""
import pytest

from gateway.connected_assistance_store import AssistanceStore
from tests.gateway.test_connected_assistance_store import finding, scan


async def local_owner(tmp_path, monkeypatch):
    from tests.gateway.test_local_receipt_untracked_compression import _owner
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session

    authority, old = await _owner(tmp_path, monkeypatch, "assistance")
    await old.close()
    runner = authority.runner
    runner._running = True
    runner.session_authority = authority
    connection = AuthorityConnection(authority, object(), {"user_id": "owner", "provider": "local",
                                      "profile_id": authority.profile_id, "instance_id": authority.instance_id})
    ref = create_local_session(authority, connection.actor, {"request_id": "inbox", "source": "tui",
                                "cwd": str(tmp_path), "model": "fixture", "toolsets": []}, client_local=True)
    live = authority.sessions[ref.session_id]
    live.subscribers["fixture"] = connection.actor
    return runner, authority, connection, ref, live


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["group", "foreign_source", "remote_viewer", "paused", "retiring", "bad_receipt"])
async def test_private_delivery_fails_closed(tmp_path, monkeypatch, refusal):
    from gateway.connected_assistance_delivery import publish_private_findings, notification_receipts
    from gateway.session_contract import Principal
    from gateway.config import Platform
    from gateway.session import SessionSource

    _runner, authority, connection, _ref, live = await local_owner(tmp_path, monkeypatch)
    store = AssistanceStore(tmp_path)
    rows = store.ingest_scan(scan(finding()))
    frames = []
    live.event_stream.observers.add(frames.append)
    try:
        mutations = {
            "group": lambda: setattr(live.source, "chat_type", "group"),
            "foreign_source": lambda: setattr(live, "source", SessionSource(
                platform=Platform.SLACK, chat_id="public", chat_type="dm")),
            "remote_viewer": lambda: live.subscribers.update(remote=Principal(
                "other", authority.profile_id, frozenset({"session:read"}), "remote")),
            "paused": lambda: store.set_paused(True),
            "retiring": lambda: setattr(authority, "retiring", True),
            "bad_receipt": lambda: setattr(authority, "profile_id", str(tmp_path / "other")),
        }
        mutations[refusal]()
        assert publish_private_findings(authority, rows) == 0
        assert not frames
        assert not notification_receipts(store)
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_unknown_notification_is_not_retried_even_after_reopen(tmp_path, monkeypatch):
    from gateway.connected_assistance_delivery import publish_private_findings, notification_receipts

    _runner, authority, connection, _ref, live = await local_owner(tmp_path, monkeypatch)
    try:
        store = AssistanceStore(tmp_path)
        rows = store.ingest_scan(scan(finding()))
        assert publish_private_findings(authority, rows) == 0
        assert notification_receipts(AssistanceStore(tmp_path))[0]["status"] == "unknown"
        frames = []
        live.event_stream.observers.add(frames.append)
        assert publish_private_findings(authority, rows) == 0
        assert not frames
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_only_owned_local_notification_and_durable_no_retry(tmp_path, monkeypatch):
    from gateway.connected_assistance_delivery import publish_private_findings, notification_receipts

    _runner, authority, connection, ref, live = await local_owner(tmp_path, monkeypatch)
    try:
        store = AssistanceStore(tmp_path)
        rows = store.ingest_scan(scan(finding()))
        frames = []
        live.event_stream.observers.add(frames.append)
        assert publish_private_findings(authority, rows) == 1
        assert len(frames) == 1
        event = frames[0]["params"]
        assert event["type"] == "notification.show" and event["session_id"] == ref.session_id
        assert "hermes connectors findings" in event["payload"]["text"]
        assert "Review deadline" not in str(frames)  # no sensitive content in replay/toasts
        assert notification_receipts(store)[0]["status"] == "shown"
        assert publish_private_findings(authority, rows) == 0
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [6, 25, 110])
async def test_notifications_eventually_consume_entire_backlog_and_updated_old_row(tmp_path, monkeypatch, count):
    from gateway.connected_assistance_delivery import publish_private_findings, notification_receipts
    from tests.gateway.test_connected_assistance_store import observed_scan

    _runner, authority, connection, _ref, live = await local_owner(tmp_path, monkeypatch)
    try:
        store = AssistanceStore(tmp_path)
        oldest = None
        for i in range(count):
            created = store.ingest_scan(observed_scan(entity=str(i), record=str(i)))[0]
            if i == 0:
                oldest = created
        assert len(store.list_findings()) == min(count, 20)  # UI contract is unchanged.
        # A full failed publication batch is unknown; it spends five claims, not zero.
        pending = store.list_findings(status="open", pending_notification=True)
        assert publish_private_findings(authority, pending) == 0
        assert len(notification_receipts(store)) == 5
        frames = []
        live.event_stream.observers.add(frames.append)
        for _ in range(count):
            pending = AssistanceStore(tmp_path).list_findings(status="open", pending_notification=True)
            if not pending:
                break
            assert 1 <= publish_private_findings(authority, pending) <= 5
        receipts = notification_receipts(store)
        assert len(receipts) == count
        assert sum(r["status"] == "unknown" for r in receipts) == 5
        assert len(frames) == count - 5
        assert publish_private_findings(authority, store.list_findings(status="open")) == 0
        assert len(frames) == count - 5
        assert oldest is not None
        store.ingest_scan(observed_scan(entity="0", record="0", text="Please review Atlas today before dinner."))
        pending = store.list_findings(status="open", pending_notification=True)
        assert [r["id"] for r in pending] == [oldest["id"]]
        assert publish_private_findings(authority, pending) == 1
        assert not store.list_findings(status="open", pending_notification=True)
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_delivery_limit_counts_new_claims_not_repeated_candidates(tmp_path, monkeypatch):
    from gateway.connected_assistance_delivery import publish_private_findings, notification_receipts

    _runner, authority, connection, _ref, live = await local_owner(tmp_path, monkeypatch)
    try:
        store = AssistanceStore(tmp_path)
        for i in range(6):
            row = finding(entity_key=str(i))
            row["evidence"][0]["record_id"] = str(i)
            store.ingest_scan(scan(row))
        frames = []
        live.event_stream.observers.add(frames.append)
        rows = store.list_findings(status="open")
        assert publish_private_findings(authority, rows) == 5
        assert publish_private_findings(authority, rows) == 1
        assert len(notification_receipts(store)) == len(frames) == 6
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_notification_identity_uses_observed_version_not_model_quote(tmp_path, monkeypatch):
    from gateway.connected_assistance_delivery import publish_private_findings, notification_receipts
    from tests.gateway.test_connected_assistance_store import observed_scan

    _runner, authority, connection, _ref, live = await local_owner(tmp_path, monkeypatch)
    try:
        store = AssistanceStore(tmp_path)
        original = store.ingest_scan(observed_scan())[0]
        frames = []
        live.event_stream.observers.add(frames.append)
        assert publish_private_findings(authority, [original]) == 1
        alternate = original | {"entity_key": "Atlas review", "evidence": [original["evidence"][0] | {
            "excerpt": "Please review Atlas today before lunch.", "url": "https://mail.test/m1"}]}
        assert publish_private_findings(authority, [alternate]) == 0
        changed = store.ingest_scan(observed_scan(text="Please review Atlas today before dinner."))
        assert publish_private_findings(authority, changed) == 1
        assert len(notification_receipts(store)) == len(frames) == 2
    finally:
        await connection.close()
        authority.db.close()
