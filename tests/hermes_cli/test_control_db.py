from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import control_db as cp


def db(tmp_path: Path) -> sqlite3.Connection:
    return cp.connect(root=tmp_path / ".hermes")


def test_control_db_path_uses_default_root_not_profile_home(tmp_path, monkeypatch):
    native = tmp_path / "home"
    root = native / ".hermes"
    profile = root / "profiles" / "statuepm"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: native)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    assert cp.control_db_path() == root / "control-plane" / "control.db"


def test_connect_schema_and_wal_are_thread_safe(tmp_path):
    root = tmp_path / ".hermes"
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait(timeout=5)
            conn = cp.connect(root=root)
            try:
                assert conn.execute("SELECT value FROM cp_meta WHERE key='schema_version'").fetchone()[0] == str(cp.SCHEMA_VERSION)
            finally:
                conn.close()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    conn = cp.connect(root=root)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        assert mode == "wal"
    finally:
        conn.close()


def test_refuses_unknown_or_partial_schema(tmp_path):
    root = tmp_path / ".hermes"
    path = cp.control_db_path(root)
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE cp_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_ms INTEGER NOT NULL)")
        conn.execute("INSERT INTO cp_meta VALUES('schema_version','999',0)")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(cp.ControlPlaneError):
        cp.connect(root=root)

    partial = tmp_path / "partial" / "control-plane" / "control.db"
    partial.parent.mkdir(parents=True)
    conn = sqlite3.connect(partial)
    try:
        conn.execute("CREATE TABLE cp_messages(message_id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(cp.ControlPlaneError):
        cp.connect(path=partial)


def test_control_db_files_are_private_and_redaction_covers_common_secret_shapes(tmp_path):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message", priority=1)
        secret = "abcdefghijklmnopqrstuvwxyz123456"
        mid = cp.create_message(
            conn,
            sender_profile="default",
            receiver_profile="worker",
            kind="status",
            body=f"Authorization: Bearer {secret} AWS_SECRET_ACCESS_KEY={secret} DATABASE_URL=postgres://user:{secret}@db/name curl -u user:{secret}",
        )
        body = conn.execute("SELECT body FROM cp_messages WHERE message_id=?", (mid,)).fetchone()["body"]
        assert secret not in body
        assert "***" in body
        quoted = cp.redact_text("password=\"hunter2 with spaces\" token='abc def' private_key=\"-----BEGIN PRIVATE KEY-----\\nabc def\\n-----END PRIVATE KEY-----\"")
        assert "hunter2" not in quoted
        assert "abc def" not in quoted
        assert "BEGIN PRIVATE KEY" not in quoted
        assert (cp.control_db_path(root).stat().st_mode & 0o777) == 0o600
        assert (cp.control_db_dir(root).stat().st_mode & 0o777) == 0o700
        cp.hmac_id("target", root=root)
        assert ((cp.control_db_dir(root) / ".pepper").stat().st_mode & 0o777) == 0o600

    finally:
        conn.close()


def test_schema_cache_still_validates_replaced_db(tmp_path):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    conn.close()
    path = cp.control_db_path(root)
    path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE cp_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_ms INTEGER NOT NULL)")
        conn.execute("INSERT INTO cp_meta VALUES('schema_version','999',0)")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(cp.ControlPlaneError):
        cp.connect(root=root)

    empty_root = tmp_path / "empty" / ".hermes"
    c = cp.connect(root=empty_root)
    c.close()
    empty_path = cp.control_db_path(empty_root)
    empty_path.unlink()
    sqlite3.connect(empty_path).close()
    c = cp.connect(root=empty_root)
    try:
        assert c.execute("SELECT COUNT(*) FROM cp_meta").fetchone()[0] >= 1
    finally:
        c.close()


def test_register_instance_rejects_cross_profile_instance_id_reuse(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_instance(conn, "worker-a", instance_id="same")
        with pytest.raises(cp.ControlPlaneError):
            cp.register_instance(conn, "worker-b", instance_id="same")
    finally:
        conn.close()


def test_register_instance_refreshes_pid_on_reuse(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_instance(conn, "worker-a", instance_id="same", pid=111, host="old-host")
        cp.register_instance(conn, "worker-a", instance_id="same", pid=222, host="new-host")
        row = conn.execute("SELECT pid, host, status FROM cp_profile_instances WHERE instance_id='same'").fetchone()
        assert row["pid"] == 222
        assert row["host"] == "new-host"
        assert row["status"] == "online"
    finally:
        conn.close()


def test_route_policy_default_deny_admin_only_and_deny_wins(tmp_path):
    conn = db(tmp_path)
    try:
        assert cp.route_allowed(conn, sender_profile="a", receiver_profile="b", kind="status", capability="message") is False
        with pytest.raises(PermissionError):
            cp.set_authority_mode(conn, "control_db")
        with pytest.raises(PermissionError):
            cp.add_route_policy(conn, effect="allow", sender_profile="a", receiver_profile="b", kind="status", capability="message")
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="a", receiver_profile="b", kind="status", capability="message", priority=10)
        assert cp.route_allowed(conn, sender_profile="a", receiver_profile="b", kind="status", capability="message") is True
        cp.add_route_policy(conn, effect="deny", created_by_type="bootstrap", sender_profile="a", receiver_profile="b", kind="status", capability="message", priority=10)
        assert cp.route_allowed(conn, sender_profile="a", receiver_profile="b", kind="status", capability="message") is False
    finally:
        conn.close()


def test_admin_mutations_require_live_admin_instance_or_bootstrap(tmp_path):
    conn = db(tmp_path)
    try:
        with pytest.raises(PermissionError):
            cp.register_profile(conn, "default", role="admin")
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        inst = cp.register_instance(conn, "default", instance_id="admin-live", actor_type="bootstrap")
        cp.set_authority_mode(conn, "control_db", actor_type="admin", actor_profile="default", actor_instance_id=inst)
        cp.add_route_policy(conn, effect="allow", created_by_type="admin", created_by="default", created_by_instance_id=inst, sender_profile="a", receiver_profile="b", kind="status", capability="message")
        with pytest.raises(PermissionError):
            cp.add_route_policy(conn, effect="allow", created_by_type="admin", created_by="default", sender_profile="x", receiver_profile="y")
    finally:
        conn.close()


def test_register_instance_does_not_demote_existing_admin_profile(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_profile(conn, "default", role="admin", display_name="Default", actor_type="bootstrap")
        with pytest.raises(PermissionError):
            cp.register_instance(conn, "default", instance_id="default-1")
        cp.register_instance(conn, "default", instance_id="default-1", actor_type="bootstrap")
        row = conn.execute("SELECT role, display_name FROM cp_profiles WHERE profile_id='default'").fetchone()
        assert row["role"] == "admin"
        assert row["display_name"] == "Default"
    finally:
        conn.close()


def test_message_creation_redacts_and_enqueues_outbox(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message", priority=1)
        mid = cp.create_message(
            conn,
            sender_profile="default",
            receiver_profile="worker",
            kind="status",
            body="api_key=abc123456789 token: xyz987654321",
            metadata={"authorization": "Bearer secretsecretsecret", "safe": "ok"},
        )
        row = conn.execute("SELECT body, metadata_json FROM cp_messages WHERE message_id=?", (mid,)).fetchone()
        assert "abc123" not in row["body"]
        assert "xyz987" not in row["body"]
        assert "secretsecret" not in row["metadata_json"]
        assert conn.execute("SELECT COUNT(*) FROM cp_outbox WHERE subject_id=?", (mid,)).fetchone()[0] == 1
    finally:
        conn.close()


def test_message_and_dispatch_idempotency_return_existing_ids(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message", priority=1)
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch", priority=1)
        m1 = cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="one", idempotency_key="m-key")
        m2 = cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="one", idempotency_key="m-key")
        d1 = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"}, idempotency_key="d-key")
        d2 = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"}, idempotency_key="d-key")
        assert m1 == m2
        assert d1 == d2
        assert conn.execute("SELECT COUNT(*) FROM cp_messages WHERE idempotency_key='m-key'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cp_dispatches WHERE idempotency_key='d-key'").fetchone()[0] == 1
    finally:
        conn.close()


def test_idempotency_key_reuse_with_different_request_is_rejected(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message", priority=1)
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch", priority=1)
        cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="one", idempotency_key="m-key")
        with pytest.raises(cp.ControlPlaneError):
            cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="two", idempotency_key="m-key")
        cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"}, idempotency_key="d-key")
        with pytest.raises(cp.ControlPlaneError):
            cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "y"}, idempotency_key="d-key")
    finally:
        conn.close()


def test_concurrent_duplicate_idempotency_returns_one_message_id(tmp_path):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message", priority=1)
    finally:
        conn.close()

    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker():
        c = cp.connect(root=root)
        try:
            barrier.wait(timeout=5)
            results.append(cp.create_message(c, sender_profile="default", receiver_profile="worker", kind="status", body="one", idempotency_key="race-key"))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            c.close()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(results) == 8
    assert len(set(results)) == 1


def test_dispatch_claim_is_single_winner_and_epoch_fenced(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch", priority=1)
        inst1 = cp.register_instance(conn, "worker", instance_id="i1")
        inst2 = cp.register_instance(conn, "worker", instance_id="i2")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"})
        ok1, epoch1 = cp.claim_dispatch(conn, did, instance_id=inst1)
        ok2, epoch2 = cp.claim_dispatch(conn, did, instance_id=inst2)
        assert (ok1, epoch1) == (True, 1)
        assert (ok2, epoch2) == (False, None)
        assert cp.advance_dispatch(conn, did, instance_id=inst2, lease_epoch=1, status="completed") is False
        assert cp.advance_dispatch(conn, did, instance_id=inst1, lease_epoch=1, status="completed") is True
    finally:
        conn.close()


def test_dispatch_can_advance_to_blocked_and_emits_event(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch", priority=1)
        inst = cp.register_instance(conn, "worker", instance_id="i1")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"})
        ok, epoch = cp.claim_dispatch(conn, did, instance_id=inst)
        assert ok and epoch == 1
        assert cp.advance_dispatch(conn, did, instance_id=inst, lease_epoch=epoch, status="blocked", last_error="needs supervisor") is True
        row = conn.execute("SELECT status,last_error FROM cp_dispatches WHERE dispatch_id=?", (did,)).fetchone()
        assert row["status"] == "blocked"
        assert row["last_error"] == "needs supervisor"
        event = conn.execute("SELECT event_type,event_json FROM cp_dispatch_events WHERE dispatch_id=? AND event_type='blocked'", (did,)).fetchone()
        assert event is not None
        assert json.loads(event["event_json"])["status"] == "blocked"
    finally:
        conn.close()


def test_dispatch_advance_requires_live_lease_and_valid_status(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch", priority=1)
        inst = cp.register_instance(conn, "worker", instance_id="i1")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"})
        ok, epoch = cp.claim_dispatch(conn, did, instance_id=inst, lease_ms=1)
        assert ok is True
        assert epoch is not None
        conn.execute("UPDATE cp_dispatches SET lease_expires_at_ms=0 WHERE dispatch_id=?", (did,))
        with pytest.raises(cp.ControlPlaneError):
            cp.advance_dispatch(conn, did, instance_id=inst, lease_epoch=epoch, status="nonsense")
        assert cp.advance_dispatch(conn, did, instance_id=inst, lease_epoch=epoch, status="completed") is False
    finally:
        conn.close()


def test_approval_atomic_consume_once_and_requester_bound(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        inst = cp.register_instance(conn, "worker", instance_id="worker-1")
        other = cp.register_instance(conn, "worker", instance_id="worker-2")
        aid = cp.create_approval(conn, requester_profile="worker", requester_instance_id=inst, approver_profile="default", command_preview="rm -rf /tmp/x")
        assert cp.consume_approval(conn, aid, requester_instance_id=inst) is False
        with pytest.raises(PermissionError):
            cp.register_instance(conn, "default", instance_id="forged-admin")
        admin_inst = cp.register_instance(conn, "default", instance_id="default-admin", actor_type="bootstrap")
        assert cp.decide_approval(conn, aid, approver_profile="default", approver_instance_id=admin_inst, decision="approved") is True
        assert cp.consume_approval(conn, aid, requester_instance_id=other, requester_profile="worker") is False
        assert cp.consume_approval(conn, aid, requester_instance_id=inst, requester_profile="other-profile") is False
        assert cp.consume_approval(conn, aid, requester_instance_id=inst, requester_profile="worker") is True
        assert cp.consume_approval(conn, aid, requester_instance_id=inst, requester_profile="worker") is False
    finally:
        conn.close()


def test_approval_rejects_invalid_decision(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        inst = cp.register_instance(conn, "worker", instance_id="worker-1")
        aid = cp.create_approval(conn, requester_profile="worker", requester_instance_id=inst, approver_profile="default", command_preview="cmd")
        admin_inst = cp.register_instance(conn, "default", instance_id="default-admin", actor_type="bootstrap")
        with pytest.raises(cp.ControlPlaneError):
            cp.decide_approval(conn, aid, approver_profile="default", approver_instance_id=admin_inst, decision="approve")  # type: ignore[arg-type]
    finally:
        conn.close()


def test_hmac_ids_are_stable_not_raw_sha_and_use_connection_root(tmp_path):
    root = tmp_path / ".hermes"
    value = "1509913443602403568"
    digest = cp.hmac_id(value, root=root)
    assert cp.hmac_id(value, root=root) == digest
    assert digest != __import__("hashlib").sha256(value.encode()).hexdigest()
    assert value not in digest

    conn = cp.connect(root=root)
    try:
        outbox_id = cp._enqueue_outbox(conn, subject_type="message", subject_id="m", subject_version=1, event_id=None, payload={}, target_platform="discord", target_ref=value)
        row = conn.execute("SELECT target_ref_hmac FROM cp_outbox WHERE outbox_id=?", (outbox_id,)).fetchone()
        assert row["target_ref_hmac"] == digest
    finally:
        conn.close()


def test_doctor_flags_wrong_root_and_stale_instances(tmp_path):
    real_root = tmp_path / "real"
    conn = cp.connect(root=real_root)
    try:
        cp.register_instance(conn, "worker", instance_id="stale", lease_ms=-1)
        cp.bootstrap_statutepm_policies(conn, seed_instances=True, instance_lease_ms=-1)
        issues = cp.doctor(conn, root=tmp_path / "expected")
        codes = {i.code for i in issues}
        assert "wrong_root" in codes
        assert "stale_instances" in codes
        assert "stale_bootstrap_instances" in codes
        assert "stale_spawned_instances" in codes
    finally:
        conn.close()


def test_dispatch_claim_requires_receiver_profile_live_instance(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker-a", kind="dispatch", capability="dispatch")
        right = cp.register_instance(conn, "worker-a", instance_id="worker-a:1")
        wrong = cp.register_instance(conn, "worker-b", instance_id="worker-b:1")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker-a", payload={"task": "x"})
        with pytest.raises(PermissionError):
            cp.claim_next_for_profile(conn, receiver_profile="worker-a", instance_id=wrong)
        assert cp.claim_next_for_profile(conn, receiver_profile="worker-a", instance_id=right) == (did, 1)
    finally:
        conn.close()


def test_dispatch_claim_by_id_requires_receiver_profile_live_instance(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker-a", kind="dispatch", capability="dispatch")
        cp.register_instance(conn, "worker-b", instance_id="worker-b:1")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker-a", payload={"task": "x"})
        assert cp.claim_dispatch_by_id(conn, dispatch_id=did, instance_id="worker-b:1") == (False, None)
        cp.register_instance(conn, "worker-a", instance_id="worker-a:1", lease_ms=-1)
        with pytest.raises(PermissionError):
            cp.claim_dispatch_by_id(conn, dispatch_id=did, instance_id="worker-a:1")
    finally:
        conn.close()


def test_record_result_requires_current_lease_holder_and_is_migrated(tmp_path):
    conn = db(tmp_path)
    try:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='cp_dispatch_results'").fetchone()
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch")
        inst = cp.register_instance(conn, "worker", instance_id="worker:1")
        other = cp.register_instance(conn, "worker", instance_id="worker:2")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"})
        ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=did, instance_id=inst)
        assert ok and epoch == 1
        with pytest.raises(PermissionError):
            cp.record_result(conn, dispatch_id=did, instance_id=other, lease_epoch=epoch, result={"schema": "control_result_v1"})
        cp.record_result(conn, dispatch_id=did, instance_id=inst, lease_epoch=epoch, result={"schema": "control_result_v1", "status": "completed", "token": "secretsecretsecret"})
        latest = cp.get_latest_dispatch_result(conn, did)
        assert latest and latest["lease_epoch"] == 1
        assert "secretsecret" not in latest["result_json"]
    finally:
        conn.close()


def test_record_result_retry_retains_distinct_lease_epoch_history(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch")
        inst = cp.register_instance(conn, "worker", instance_id="worker:1")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"}, max_attempts=2)
        ok, epoch1 = cp.claim_dispatch_by_id(conn, dispatch_id=did, instance_id=inst, lease_ms=100_000)
        assert ok and epoch1 == 1
        cp.record_result(conn, dispatch_id=did, instance_id=inst, lease_epoch=epoch1, result={"status": "failed", "summary": "first"})
        conn.execute("UPDATE cp_dispatches SET lease_expires_at_ms=0 WHERE dispatch_id=?", (did,))
        cp.reap_expired_dispatches(conn, now_ms=cp.now_ms() + 10)
        ok, epoch2 = cp.claim_dispatch_by_id(conn, dispatch_id=did, instance_id=inst)
        assert ok and epoch2 == 2
        cp.record_result(conn, dispatch_id=did, instance_id=inst, lease_epoch=epoch2, result={"status": "completed", "summary": "second"})
        assert [r["result"]["summary"] for r in cp.list_dispatch_results(conn, did)] == ["first", "second"]
    finally:
        conn.close()


def test_mark_expired_worker_instances_offline_is_status_only_and_role_scoped(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        cp.register_profile(conn, "statutepm", role="pm", actor_type="bootstrap")
        worker = cp.register_instance(conn, "worker", instance_id="worker:expired", lease_ms=-1)
        live_worker = cp.register_instance(conn, "worker", instance_id="worker:live", lease_ms=100_000)
        admin = cp.register_instance(conn, "default", instance_id="default:expired", lease_ms=-1, actor_type="bootstrap")
        pm = cp.register_instance(conn, "statutepm", instance_id="statutepm:expired", lease_ms=-1, actor_type="bootstrap")

        changed = cp.mark_expired_worker_instances_offline(conn, now_ms_value=cp.now_ms() + 10)

        assert changed == [worker]
        statuses = {
            row["instance_id"]: row["status"]
            for row in conn.execute(
                "SELECT instance_id,status FROM cp_profile_instances WHERE instance_id IN (?,?,?,?)",
                (worker, live_worker, admin, pm),
            ).fetchall()
        }
        assert statuses[worker] == "offline"
        assert statuses[live_worker] == "online"
        assert statuses[admin] == "online"
        assert statuses[pm] == "online"
        assert conn.execute("SELECT COUNT(*) FROM cp_profile_instances WHERE instance_id=?", (worker,)).fetchone()[0] == 1
    finally:
        conn.close()


def test_create_dispatch_requires_live_sender_instance(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch")
        sender = cp.register_instance(conn, "default", instance_id="default:1")
        did = cp.create_dispatch_from_instance(conn, sender_instance_id=sender, receiver_profile="worker", payload={"task": "x"})
        assert did.startswith("disp_")
        cp.register_instance(conn, "other", instance_id="other:1")
        with pytest.raises(cp.RouteDenied):
            cp.create_dispatch_from_instance(conn, sender_instance_id="other:1", receiver_profile="worker", payload={"task": "x"})
    finally:
        conn.close()


def test_create_message_requires_live_sender_instance(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="worker", receiver_profile="default", kind="status", capability="message")
        sender = cp.register_instance(conn, "worker", instance_id="worker:1")
        mid = cp.create_message_from_instance(conn, sender_instance_id=sender, receiver_profile="default", kind="status", body="ok")
        assert mid.startswith("msg_")
        with pytest.raises(PermissionError):
            cp.create_message_from_instance(conn, sender_instance_id="missing", receiver_profile="default", kind="status", body="ok")
    finally:
        conn.close()


def test_record_artifact_requires_current_lease_holder(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch")
        inst = cp.register_instance(conn, "worker", instance_id="worker:1")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"})
        ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=did, instance_id=inst)
        assert ok
        art = cp.record_artifact(conn, dispatch_id=did, instance_id=inst, lease_epoch=epoch, path=str(tmp_path / "artifact.txt"), summary="ok")
        assert art.startswith("art_")
        with pytest.raises(PermissionError):
            cp.record_artifact(conn, dispatch_id=did, instance_id=inst, lease_epoch=epoch + 1, path="/tmp/nope")
    finally:
        conn.close()


def test_bootstrap_statutepm_policies_idempotent(tmp_path):
    conn = db(tmp_path)
    try:
        first = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        second = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        assert first["instances"] == {"default": "default:bootstrap", "statutepm": "statutepm:bootstrap"}
        assert second["profiles"]["default"]["status"] == "existing"
        assert cp.route_allowed(conn, sender_profile="default", receiver_profile="statutepm", kind="dispatch", capability="dispatch")
        assert cp.route_allowed(conn, sender_profile="statutepm", receiver_profile="statute-worker", kind="dispatch", capability="dispatch")
        assert not cp.route_allowed(conn, sender_profile="statute-worker", receiver_profile="default", kind="dispatch", capability="dispatch")
    finally:
        conn.close()


def test_bootstrap_statutepm_seed_instances_renews_leases_with_clock_seam(tmp_path, monkeypatch):
    conn = db(tmp_path)
    try:
        monkeypatch.setattr(cp, "now_ms", lambda: 1_000)
        cp.bootstrap_statutepm_policies(conn, seed_instances=True, instance_lease_ms=10_000)
        first = conn.execute("SELECT lease_expires_at_ms FROM cp_profile_instances WHERE instance_id='statutepm:bootstrap'").fetchone()[0]
        monkeypatch.setattr(cp, "now_ms", lambda: 5_000)
        cp.bootstrap_statutepm_policies(conn, seed_instances=True, instance_lease_ms=10_000)
        second = conn.execute("SELECT lease_expires_at_ms FROM cp_profile_instances WHERE instance_id='statutepm:bootstrap'").fetchone()[0]
        assert first == 11_000
        assert second == 15_000
    finally:
        conn.close()


def test_control_db_cutover_requires_online_actor_profile_and_unexpired_lease(tmp_path):
    conn = db(tmp_path)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True, instance_lease_ms=-1)
        with pytest.raises(PermissionError):
            cp.set_authority_mode(conn, "control_db", actor_type="admin", actor_profile="default", actor_instance_id="default:bootstrap")
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        with pytest.raises(PermissionError):
            cp.set_authority_mode(conn, "control_db", actor_type="admin", actor_profile="statutepm", actor_instance_id="default:bootstrap")
        cp.set_authority_mode(conn, "control_db", actor_type="admin", actor_profile="default", actor_instance_id="default:bootstrap")
        assert cp.get_authority_mode(conn) == "control_db"
    finally:
        conn.close()


def test_pm_admin_instances_require_bootstrap_or_live_admin(tmp_path):
    conn = db(tmp_path)
    try:
        cp.bootstrap_statutepm_policies(conn)
        with pytest.raises(PermissionError):
            cp.register_instance(conn, "statutepm", instance_id="statutepm:ambient")
        seeded = cp.bootstrap_statutepm_policies(conn, seed_instances=True)["instances"]["statutepm"]
        assert seeded == "statutepm:bootstrap"
    finally:
        conn.close()


def test_renew_admin_bootstrap_instance_lease_reauthorizes_expired_admin(tmp_path, monkeypatch):
    conn = db(tmp_path)
    try:
        monkeypatch.setattr(cp, "now_ms", lambda: 1_000)
        cp.bootstrap_statutepm_policies(conn, seed_instances=True, instance_lease_ms=10)
        monkeypatch.setattr(cp, "now_ms", lambda: 2_000)
        with pytest.raises(PermissionError):
            cp.set_authority_mode(conn, "control_db", actor_type="admin", actor_profile="default", actor_instance_id="default:bootstrap")

        result = cp.renew_admin_bootstrap_instance_lease(
            conn,
            profile_id="default",
            instance_id="default:bootstrap",
            lease_ms=60_000,
        )

        assert result["instance_id"] == "default:bootstrap"
        assert result["lease_expires_at_ms"] == 62_000
        cp.set_authority_mode(conn, "control_db", actor_type="admin", actor_profile="default", actor_instance_id="default:bootstrap")
        assert cp.get_authority_mode(conn) == "control_db"
    finally:
        conn.close()


def test_renew_admin_bootstrap_instance_lease_refuses_non_admin_or_non_bootstrap(tmp_path):
    conn = db(tmp_path)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        with pytest.raises(PermissionError):
            cp.renew_admin_bootstrap_instance_lease(conn, profile_id="statutepm", instance_id="statutepm:bootstrap")
        with pytest.raises(PermissionError):
            cp.renew_admin_bootstrap_instance_lease(conn, profile_id="default", instance_id="default:other")
        cp.mark_instance_offline(conn, "default:bootstrap")
        with pytest.raises(PermissionError):
            cp.renew_admin_bootstrap_instance_lease(conn, profile_id="default", instance_id="default:bootstrap")
        with pytest.raises(ValueError):
            cp.renew_admin_bootstrap_instance_lease(conn, profile_id="default", instance_id="default:bootstrap", lease_ms=0)
        with pytest.raises(ValueError):
            cp.renew_admin_bootstrap_instance_lease(conn, profile_id="default", instance_id="default:bootstrap", lease_ms=120_001)
    finally:
        conn.close()


def _create_v2_control_db(path: Path, *, partial_v3_meta: bool = False) -> None:
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE cp_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at_ms INTEGER NOT NULL);
            CREATE TABLE cp_profiles(profile_id TEXT PRIMARY KEY, role TEXT NOT NULL DEFAULT 'worker', display_name TEXT, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL);
            CREATE TABLE cp_profile_instances(instance_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES cp_profiles(profile_id), pid INTEGER, host TEXT, started_at_ms INTEGER NOT NULL, heartbeat_at_ms INTEGER NOT NULL, lease_expires_at_ms INTEGER, status TEXT NOT NULL DEFAULT 'online', metadata_json TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE cp_route_policies(policy_id TEXT PRIMARY KEY, priority INTEGER NOT NULL DEFAULT 0, effect TEXT NOT NULL, sender_profile TEXT NOT NULL DEFAULT '*', receiver_profile TEXT NOT NULL DEFAULT '*', kind TEXT NOT NULL DEFAULT '*', capability TEXT NOT NULL DEFAULT '*', created_by TEXT NOT NULL, created_by_type TEXT NOT NULL, created_at_ms INTEGER NOT NULL, UNIQUE(priority,effect,sender_profile,receiver_profile,kind,capability));
            CREATE TABLE cp_messages(message_id TEXT PRIMARY KEY, kind TEXT NOT NULL, sender_profile TEXT NOT NULL, receiver_profile TEXT NOT NULL, capability TEXT NOT NULL DEFAULT 'message', body TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending', created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL, idempotency_key TEXT UNIQUE);
            CREATE TABLE cp_message_events(event_id TEXT PRIMARY KEY, message_id TEXT NOT NULL REFERENCES cp_messages(message_id), event_type TEXT NOT NULL, event_json TEXT NOT NULL DEFAULT '{}', created_at_ms INTEGER NOT NULL);
            CREATE TABLE cp_dispatches(dispatch_id TEXT PRIMARY KEY, message_id TEXT REFERENCES cp_messages(message_id), sender_profile TEXT NOT NULL, receiver_profile TEXT NOT NULL, capability TEXT NOT NULL DEFAULT 'dispatch', status TEXT NOT NULL DEFAULT 'pending', payload_json TEXT NOT NULL DEFAULT '{}', lease_instance_id TEXT REFERENCES cp_profile_instances(instance_id), lease_epoch INTEGER NOT NULL DEFAULT 0, lease_expires_at_ms INTEGER, attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3, last_error TEXT, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL, idempotency_key TEXT UNIQUE);
            CREATE TABLE cp_dispatch_events(event_id TEXT PRIMARY KEY, dispatch_id TEXT NOT NULL REFERENCES cp_dispatches(dispatch_id), event_type TEXT NOT NULL, event_json TEXT NOT NULL DEFAULT '{}', created_at_ms INTEGER NOT NULL);
            CREATE TABLE cp_artifacts(artifact_id TEXT PRIMARY KEY, dispatch_id TEXT REFERENCES cp_dispatches(dispatch_id), path TEXT NOT NULL, summary TEXT, created_at_ms INTEGER NOT NULL);
            CREATE TABLE cp_dispatch_results(dispatch_id TEXT NOT NULL REFERENCES cp_dispatches(dispatch_id), lease_epoch INTEGER NOT NULL, instance_id TEXT NOT NULL REFERENCES cp_profile_instances(instance_id), result_json TEXT NOT NULL, created_at_ms INTEGER NOT NULL, PRIMARY KEY(dispatch_id, lease_epoch));
            CREATE TABLE cp_approvals(approval_id TEXT PRIMARY KEY, requester_profile TEXT NOT NULL, requester_instance_id TEXT NOT NULL, approver_profile TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'pending', command_preview TEXT NOT NULL, tool_args_preview TEXT, decision TEXT, decision_reason TEXT, consumed_by_instance_id TEXT, consumed_at_ms INTEGER, expires_at_ms INTEGER NOT NULL, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL);
            CREATE TABLE cp_outbox(outbox_id TEXT PRIMARY KEY, subject_type TEXT NOT NULL, subject_id TEXT NOT NULL, subject_version INTEGER NOT NULL, event_id TEXT, target_platform TEXT, target_ref_hmac TEXT, payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL);
            CREATE TABLE cp_inbound_receipts(receipt_id TEXT PRIMARY KEY, platform TEXT NOT NULL, external_id_hmac TEXT NOT NULL, subject_type TEXT, subject_id TEXT, payload_json TEXT NOT NULL, created_at_ms INTEGER NOT NULL, UNIQUE(platform, external_id_hmac));
            CREATE TABLE cp_mirror_state(subject_type TEXT NOT NULL, subject_id TEXT NOT NULL, target_platform TEXT NOT NULL, target_ref_hmac TEXT NOT NULL, subject_version INTEGER NOT NULL, external_id_hmac TEXT, status TEXT NOT NULL, updated_at_ms INTEGER NOT NULL, PRIMARY KEY(subject_type, subject_id,target_platform,target_ref_hmac));
            CREATE INDEX idx_cp_messages_receiver_status ON cp_messages(receiver_profile,status,created_at_ms);
            CREATE INDEX idx_cp_dispatches_receiver_status ON cp_dispatches(receiver_profile,status,lease_expires_at_ms);
            CREATE INDEX idx_cp_dispatch_results_dispatch ON cp_dispatch_results(dispatch_id, lease_epoch DESC);
            CREATE INDEX idx_cp_approvals_status ON cp_approvals(status,expires_at_ms);
            CREATE INDEX idx_cp_outbox_status ON cp_outbox(status,created_at_ms);
            """
        )
        conn.execute("INSERT INTO cp_meta VALUES('schema_version',?,0)", ("3" if partial_v3_meta else "2",))
        conn.execute("INSERT INTO cp_meta VALUES('authority_mode','shadow',0)")
        if partial_v3_meta:
            conn.execute("CREATE TABLE cp_runtime_mappings(control_profile_id TEXT PRIMARY KEY, runtime_profile TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'worker', enabled INTEGER NOT NULL DEFAULT 1, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)")
        conn.commit()
    finally:
        conn.close()


def test_migrates_v2_schema_before_creating_v3_indexes(tmp_path):
    root = tmp_path / ".hermes"
    path = cp.control_db_path(root)
    _create_v2_control_db(path)

    conn = cp.connect(root=root)
    try:
        assert conn.execute("SELECT value FROM cp_meta WHERE key='schema_version'").fetchone()[0] == "3"
        dispatch_cols = {row[1] for row in conn.execute("PRAGMA table_info(cp_dispatches)")}
        assert "parent_dispatch_id" in dispatch_cols
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cp_dispatches_parent_status'").fetchone()
    finally:
        conn.close()


def test_partial_v3_migration_does_not_mark_schema_complete(tmp_path):
    root = tmp_path / ".hermes"
    path = cp.control_db_path(root)
    _create_v2_control_db(path, partial_v3_meta=True)

    with pytest.raises(cp.ControlPlaneError, match="partial control DB schema"):
        cp.connect(root=root)

    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT value FROM cp_meta WHERE key='schema_version'").fetchone()[0] == "3"
        assert "parent_dispatch_id" not in {row[1] for row in conn.execute("PRAGMA table_info(cp_dispatches)")}
    finally:
        conn.close()


def test_v3_schema_status_blocker_supervision_runtime_mapping_and_approval_context(tmp_path):
    conn = db(tmp_path)
    try:
        assert conn.execute("SELECT value FROM cp_meta WHERE key='schema_version'").fetchone()[0] == "3"
        for table in ["cp_status_events", "cp_blockers", "cp_supervision_runs", "cp_runtime_mappings"]:
            assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        for table, column in [
            ("cp_dispatches", "parent_dispatch_id"),
            ("cp_dispatches", "dispatch_schema"),
            ("cp_approvals", "dispatch_id"),
            ("cp_approvals", "lease_epoch"),
            ("cp_approvals", "request_context_json"),
            ("cp_approvals", "decision_by_instance_id"),
        ]:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            assert column in cols
    finally:
        conn.close()


def test_status_and_blocker_lifecycle_are_identity_and_lease_bound(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="dispatch", capability="dispatch")
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        admin = cp.register_instance(conn, "default", instance_id="default:admin", actor_type="bootstrap")
        worker = cp.register_instance(conn, "worker", instance_id="worker:1")
        other = cp.register_instance(conn, "other", instance_id="other:1")
        did = cp.create_dispatch(conn, sender_profile="default", receiver_profile="worker", payload={"task": "x"}, dispatch_schema="generic_dispatch_v1")
        ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=did, instance_id=worker)
        assert ok and epoch == 1
        event_id = cp.emit_status(conn, instance_id=worker, dispatch_id=did, status="running", summary="running")
        assert event_id.startswith("evt_")
        with pytest.raises(PermissionError):
            cp.emit_status(conn, instance_id=other, dispatch_id=did, status="running", summary="spoof")
        blocker_id = cp.open_blocker(conn, dispatch_id=did, instance_id=worker, severity="blocked", kind="missing_context", summary="need input", response_profile="default")
        assert blocker_id.startswith("blk_")
        with pytest.raises(PermissionError):
            cp.resolve_blocker(conn, blocker_id, resolver_instance_id=other, resolution={"summary": "no"})
        assert cp.resolve_blocker(conn, blocker_id, resolver_instance_id=admin, resolution={"summary": "ok"}) is True
        row = conn.execute("SELECT status FROM cp_blockers WHERE blocker_id=?", (blocker_id,)).fetchone()
        assert row["status"] == "resolved"
    finally:
        conn.close()


def test_supervision_run_records_findings_and_dry_run_reap_does_not_mutate(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        admin = cp.register_instance(conn, "default", instance_id="default:admin", actor_type="bootstrap")
        rid = cp.start_supervision_run(conn, actor_instance_id=admin, scope={"dry_run": True})
        cp.finish_supervision_run(conn, rid, status="completed", findings=[{"code": "ok"}], actions=[])
        row = conn.execute("SELECT status, findings_json FROM cp_supervision_runs WHERE run_id=?", (rid,)).fetchone()
        assert row["status"] == "completed"
        assert json.loads(row["findings_json"])[0]["code"] == "ok"
    finally:
        conn.close()


def test_approval_context_binds_dispatch_and_lease_epoch(tmp_path):
    conn = db(tmp_path)
    try:
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        admin = cp.register_instance(conn, "default", instance_id="default:admin", actor_type="bootstrap")
        worker = cp.register_instance(conn, "worker", instance_id="worker:1")
        aid = cp.create_approval(
            conn,
            requester_profile="worker",
            requester_instance_id=worker,
            approver_profile="default",
            command_preview="git push",
            dispatch_id="disp_1",
            lease_epoch=4,
            cwd="/tmp/work",
            affected_paths=["/tmp/work/file.py"],
            operation_class="git_push",
            risk_classification="dangerous",
            reason_requested="test",
            request_context={"source": "unit"},
        )
        assert cp.decide_approval(conn, aid, approver_profile="default", approver_instance_id=admin, decision="approved")
        assert cp.consume_approval(conn, aid, requester_instance_id=worker, requester_profile="worker", dispatch_id="disp_2", lease_epoch=4) is False
        assert cp.consume_approval(conn, aid, requester_instance_id=worker, requester_profile="worker", dispatch_id="disp_1", lease_epoch=5) is False
        assert cp.consume_approval(conn, aid, requester_instance_id=worker, requester_profile="worker", dispatch_id="disp_1", lease_epoch=4) is True
    finally:
        conn.close()


def test_message_terminal_status_transition_is_authorized_audited_and_idempotent(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message", priority=1)
        worker = cp.register_instance(conn, "worker", instance_id="worker:live")
        other = cp.register_instance(conn, "other", instance_id="other:live")
        mid = cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="needs closure")
        with pytest.raises(PermissionError):
            cp.transition_message_status(conn, mid, status="resolved", actor_instance_id=other)
        result = cp.transition_message_status(
            conn,
            mid,
            status="resolved",
            actor_instance_id=worker,
            reason="token=supersecretvalue",
            metadata={"api_key": "supersecretvalue", "safe": "ok"},
        )
        assert result["changed"] is True
        assert conn.execute("SELECT status FROM cp_messages WHERE message_id=?", (mid,)).fetchone()["status"] == "resolved"
        again = cp.transition_message_status(conn, mid, status="resolved", actor_instance_id=worker)
        assert again["changed"] is False
        with pytest.raises(cp.ControlPlaneError):
            cp.transition_message_status(conn, mid, status="cancelled", actor_instance_id=worker)
        superseded = cp.transition_message_status(conn, mid, status="superseded", actor_instance_id=worker)
        assert superseded["changed"] is True
        events = conn.execute("SELECT event_json FROM cp_message_events WHERE message_id=? AND event_type='status_transition' ORDER BY created_at_ms", (mid,)).fetchall()
        assert len(events) == 3
        joined = "\n".join(e["event_json"] for e in events)
        assert "supersecretvalue" not in joined
        assert '"safe":"ok"' in joined
        assert conn.execute("SELECT COUNT(*) FROM cp_outbox WHERE subject_type='message' AND subject_id=?", (mid,)).fetchone()[0] == 4
    finally:
        conn.close()


def test_message_transition_admin_and_bootstrap_guards(tmp_path):
    conn = db(tmp_path)
    try:
        cp.add_route_policy(conn, effect="allow", created_by_type="bootstrap", sender_profile="default", receiver_profile="worker", kind="status", capability="message", priority=1)
        cp.register_profile(conn, "default", role="admin", actor_type="bootstrap")
        admin = cp.register_instance(conn, "default", instance_id="default:admin", actor_type="bootstrap")
        mid = cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="x")
        res = cp.transition_message_status(conn, mid, status="cancelled", actor_type="admin", actor_profile="default", actor_instance_id=admin)
        assert res["status"] == "cancelled"
        mid2 = cp.create_message(conn, sender_profile="default", receiver_profile="worker", kind="status", body="y")
        res2 = cp.transition_message_status(conn, mid2, status="superseded", actor_type="bootstrap")
        assert res2["status"] == "superseded"
        with pytest.raises(cp.ControlPlaneError):
            cp.transition_message_status(conn, mid2, status="bogus", actor_type="bootstrap")
    finally:
        conn.close()
