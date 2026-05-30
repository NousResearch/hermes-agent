from __future__ import annotations

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
        inst = cp.register_instance(conn, "default", instance_id="admin-live")
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
        cp.register_instance(conn, "default", instance_id="default-1")
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
        admin_inst = cp.register_instance(conn, "default", instance_id="default-admin")
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
        with pytest.raises(cp.ControlPlaneError):
            cp.decide_approval(conn, aid, approver_profile="default", approver_instance_id=cp.register_instance(conn, "default", instance_id="default-admin"), decision="approve")  # type: ignore[arg-type]
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
        issues = cp.doctor(conn, root=tmp_path / "expected")
        codes = {i.code for i in issues}
        assert "wrong_root" in codes
        assert "stale_instances" in codes
    finally:
        conn.close()
