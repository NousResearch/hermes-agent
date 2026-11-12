from __future__ import annotations

import multiprocessing
import os
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import (
    SessionIdOwnedElsewhereError,
    SessionLineageAmbiguousError,
    SessionLineageInventoryError,
)


@pytest.fixture
def profile_stores(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root, profile


def _seed_session(db: SessionDB, session_id: str, profile: str, key: str, *, ended_at=None, reason=None):
    db._conn.execute(
        "INSERT INTO sessions (id, source, started_at, profile_name, session_key, ended_at, end_reason, message_count) "
        "VALUES (?, 'tui', ?, ?, ?, ?, ?, 0)",
        (session_id, time.time(), profile, key, ended_at, reason),
    )
    db._conn.commit()


def _local_receipt(root: Path, session_id: str):
    from dataclasses import asdict

    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_policy import build_policy

    source = SessionSource(platform=Platform.LOCAL, chat_id=session_id, user_id="human", chat_type="dm")
    now = _now()
    entry = SessionEntry("local:" + session_id, session_id, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({"source": "cli", "cwd": str(root), "model": "m", "toolsets": []},
                          {"platform_toolsets": {"cli": []}}, private_secrets={})
    return {
        "profile_id": "profile", "principal_id": "human", "request_id": "r", "session_id": session_id,
        "route": entry.session_key, "entry": entry.to_dict(), "policy": asdict(policy),
    }


def test_create_session_rejects_an_id_owned_by_another_profile(profile_stores):
    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    try:
        default_db.create_session("same-id", "cli", session_key="agent:main:local:dm:same-id")
        from hermes_state_lineage import session_lineage_guard
        with session_lineage_guard(default_db.db_path, "same-id"):
            with session_lineage_guard(default_db.db_path, "same-id"):
                assert default_db.get_session("same-id") is not None
        with pytest.raises(SessionIdOwnedElsewhereError) as exc:
            profile_db.create_session("same-id", "tui", session_key="agent:science:local:dm:same-id")
        assert exc.value.code == "session_id_owned_elsewhere"
        assert profile_db.get_session("same-id") is None
        assert default_db.get_session("same-id") is not None
    finally:
        profile_db.close()
        default_db.close()


def test_profile_home_uses_the_shared_hermes_root(profile_stores, monkeypatch):
    root, profile = profile_stores
    monkeypatch.setenv("HERMES_HOME", str(profile))
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    try:
        default_db.create_session("profile-root-id", "cli")
        with pytest.raises(SessionIdOwnedElsewhereError):
            profile_db.create_session("profile-root-id", "tui")
    finally:
        profile_db.close()
        default_db.close()


def test_standalone_database_outside_hermes_home_is_not_lineage_managed(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(tmp_path / "standalone.db")
    try:
        db.create_session("standalone-id", "cli")
        assert db.get_session("standalone-id") is not None
    finally:
        db.close()


def test_gateway_peer_recovery_refuses_an_id_owned_by_another_profile(profile_stores):
    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    session_id = "peer-recovery-id"
    try:
        _seed_session(profile_db, session_id, "science", f"agent:science:local:dm:{session_id}")
        with pytest.raises(SessionIdOwnedElsewhereError):
            default_db.record_gateway_session_peer(
                session_id, source="cli", session_key=f"agent:main:local:dm:{session_id}")
        assert default_db.get_session(session_id) is None
        assert profile_db.get_session(session_id) is not None
    finally:
        profile_db.close()
        default_db.close()


def test_local_session_commit_refuses_an_id_owned_by_another_profile(profile_stores):
    from hermes_state_local import commit_local_session
    from hermes_state_runtime import begin_runtime_epoch

    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    session_id = "local-commit-id"
    try:
        _seed_session(profile_db, session_id, "science", f"agent:science:local:dm:{session_id}")
        epoch = begin_runtime_epoch(default_db, instance_id="default")
        from hermes_state_runtime import RuntimeStoreError

        with pytest.raises(RuntimeStoreError) as refused:
            commit_local_session(default_db, epoch=epoch, receipt=_local_receipt(root, session_id))
        assert refused.value.reason == "admission_conflict"
        assert default_db.get_session(session_id) is None
        assert profile_db.get_session(session_id) is not None
    finally:
        profile_db.close()
        default_db.close()


def test_resume_uses_unique_live_owner_and_blocks_stale_copy(profile_stores):
    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    try:
        _seed_session(default_db, "resume-id", "default", "agent:main:local:dm:resume-id",
                      ended_at=time.time(), reason="agent_close")
        _seed_session(profile_db, "resume-id", "science", "agent:science:local:dm:resume-id")
        with pytest.raises(SessionIdOwnedElsewhereError):
            default_db.reopen_session("resume-id")
        profile_db.reopen_session("resume-id")
        assert profile_db.get_session("resume-id")["ended_at"] is None
        assert default_db.get_session("resume-id")["ended_at"] is not None
    finally:
        profile_db.close()
        default_db.close()


def test_multiple_live_copies_fail_closed(profile_stores):
    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    try:
        _seed_session(default_db, "ambiguous-id", "default", "agent:main:local:dm:ambiguous-id")
        _seed_session(profile_db, "ambiguous-id", "science", "agent:science:local:dm:ambiguous-id")
        with pytest.raises(SessionLineageAmbiguousError):
            default_db.reopen_session("ambiguous-id")
        with pytest.raises(SessionLineageAmbiguousError):
            profile_db.create_session("ambiguous-id", "tui")
    finally:
        profile_db.close()
        default_db.close()


def test_message_writes_refuse_foreign_or_ambiguous_lineages(profile_stores):
    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    session_id = "message-owner-id"
    try:
        _seed_session(profile_db, session_id, "science", f"agent:science:local:dm:{session_id}")
        with pytest.raises(SessionIdOwnedElsewhereError):
            default_db.append_message(session_id, "user", "must not land in a second store")

        _seed_session(default_db, session_id, "default", f"agent:main:local:dm:{session_id}")
        for db in (default_db, profile_db):
            with pytest.raises(SessionLineageAmbiguousError):
                db.append_message(session_id, "user", "ambiguous ownership must fail closed")
            with pytest.raises(SessionLineageAmbiguousError):
                db.append_messages_batch(session_id, [{"role": "user", "content": "also fenced"}])
            with pytest.raises(SessionLineageAmbiguousError):
                db.end_session(session_id, "must not pick an owner")
            with pytest.raises(SessionLineageAmbiguousError):
                db.clear_messages(session_id)
            with db._read_ctx() as conn:
                assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()[0] == 0
                assert conn.execute("SELECT ended_at FROM sessions WHERE id=?", (session_id,)).fetchone()[0] is None
    finally:
        profile_db.close()
        default_db.close()


def test_unreadable_sibling_store_blocks_session_creation(profile_stores):
    root, _profile = profile_stores
    default_db = SessionDB(root / "state.db")
    corrupt = root / "profiles" / "unreadable"
    corrupt.mkdir()
    (corrupt / "state.db").write_bytes(b"not a sqlite database")
    try:
        with pytest.raises(SessionLineageInventoryError):
            default_db.create_session("must-not-be-claimed", "cli")
        assert default_db.get_session("must-not-be-claimed") is None
    finally:
        default_db.close()


def test_sibling_store_without_sessions_schema_blocks_creation(profile_stores):
    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    default_db._TRANSCRIPT_WRITE_PATIENCE_S = 0.0
    incomplete = profile / "state.db"
    with sqlite3.connect(incomplete) as conn:
        conn.execute("CREATE TABLE unrelated (id TEXT)")
    try:
        with pytest.raises(SessionLineageInventoryError, match="lacks sessions table"):
            default_db.create_session("must-not-be-claimed", "cli")
        assert default_db.get_session("must-not-be-claimed") is None
    finally:
        default_db.close()


def _concurrent_create_worker(role, db_path, root, session_id, lock_held, contender_ready, scan_started,
                              release_owner, outcomes):
    os.environ["HERMES_HOME"] = str(root)
    db = SessionDB(db_path)
    try:
        if role == "owner":
            from hermes_state_lineage import session_lineage_guard

            with session_lineage_guard(db_path, session_id):
                def write(conn):
                    conn.execute(
                        "INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', ?)",
                        (session_id, time.time()),
                    )
                    lock_held.set()
                    if not release_owner.wait(20):
                        raise TimeoutError("test owner was not released")

                db._execute_write(write)
            outcomes.put(("owner", "created", None))
            return

        import hermes_state_lineage

        original = hermes_state_lineage._rows_for_ids

        def observe_scan(*args, **kwargs):
            scan_started.set()
            return original(*args, **kwargs)

        hermes_state_lineage._rows_for_ids = observe_scan
        contender_ready.set()
        db.create_session(session_id, "cli")
        outcomes.put(("contender", "created", None))
    except Exception as exc:
        outcomes.put((role, type(exc).__name__, getattr(exc, "code", None)))
    finally:
        db.close()


@pytest.mark.platforms("linux")
def test_concurrent_cross_profile_create_has_one_winner(profile_stores):
    root, profile = profile_stores
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("this cross-process file-lock test requires fork")
    context = multiprocessing.get_context("fork")
    sid = "concurrent-id"
    lock_held, contender_ready = context.Event(), context.Event()
    scan_started, release_owner = context.Event(), context.Event()
    outcomes = context.Queue()
    owner = context.Process(target=_concurrent_create_worker, args=(
        "owner", root / "state.db", root, sid, lock_held, contender_ready, scan_started, release_owner, outcomes))
    contender = context.Process(target=_concurrent_create_worker, args=(
        "contender", profile / "state.db", root, sid, lock_held, contender_ready, scan_started,
        release_owner, outcomes))
    started = []
    scanned_before_owner_commit = False
    try:
        owner.start()
        started.append(owner)
        assert lock_held.wait(10), "owner did not reach its held write transaction"
        contender.start()
        started.append(contender)
        assert contender_ready.wait(10), "contender did not reach the create call"
        # A same-id ownership lock blocks the contender before inventory until the owner's
        # transaction commits. Without it, the contender inventories the uncommitted twin.
        scanned_before_owner_commit = scan_started.wait(2)
    finally:
        release_owner.set()
        for process in started:
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join(5)
    assert not scanned_before_owner_commit, "contender scanned while another profile held ownership lock"
    assert [process.exitcode for process in started] == [0, 0]
    observed = [outcomes.get(timeout=5), outcomes.get(timeout=5)]
    assert sorted((kind, outcome) for _role, kind, outcome in observed) == sorted((
        ("created", None), ("SessionIdOwnedElsewhereError", "session_id_owned_elsewhere")))
    verify = [SessionDB(root / "state.db"), SessionDB(profile / "state.db")]
    try:
        assert sum(db.get_session(sid) is not None for db in verify) == 1
    finally:
        for db in verify:
            db.close()


@pytest.mark.platforms("linux")
def test_different_session_ids_do_not_share_the_ownership_lock(profile_stores):
    root, profile = profile_stores
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("this cross-process file-lock test requires fork")
    context = multiprocessing.get_context("fork")
    owner_id, contender_id = "held-session-id", "independent-session-id"
    lock_held, contender_ready, scan_started, release_owner = (
        context.Event(), context.Event(), context.Event(), context.Event())
    outcomes = context.Queue()
    owner = context.Process(target=_concurrent_create_worker, args=(
        "owner", root / "state.db", root, owner_id, lock_held, contender_ready, scan_started,
        release_owner, outcomes))
    contender = context.Process(target=_concurrent_create_worker, args=(
        "contender", profile / "state.db", root, contender_id, lock_held, contender_ready, scan_started,
        release_owner, outcomes))
    started = []
    scanned_while_other_id_locked = False
    try:
        owner.start()
        started.append(owner)
        assert lock_held.wait(10), "owner did not reach its held write transaction"
        contender.start()
        started.append(contender)
        assert contender_ready.wait(10), "contender did not reach the create call"
        scanned_while_other_id_locked = scan_started.wait(2)
    finally:
        release_owner.set()
        for process in started:
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join(5)
    assert scanned_while_other_id_locked, "unrelated session id was serialized behind another session's lock"
    assert [process.exitcode for process in started] == [0, 0]
    observed = [outcomes.get(timeout=5), outcomes.get(timeout=5)]
    assert sorted((kind, outcome) for _role, kind, outcome in observed) == [("created", None), ("created", None)]
    verify = [SessionDB(root / "state.db"), SessionDB(profile / "state.db")]
    try:
        assert sum(db.get_session(owner_id) is not None for db in verify) == 1
        assert sum(db.get_session(contender_id) is not None for db in verify) == 1
    finally:
        for db in verify:
            db.close()


def test_repair_profiles_reports_dual_home_as_nonrepairable(profile_stores, monkeypatch):
    root, profile = profile_stores
    default_db = SessionDB(root / "state.db")
    profile_db = SessionDB(profile / "state.db")
    session_id = "repair-report-id"
    try:
        _seed_session(default_db, session_id, "default", f"agent:science:local:dm:{session_id}")
        _seed_session(profile_db, session_id, "science", f"agent:science:local:dm:{session_id}")
    finally:
        profile_db.close()
        default_db.close()

    from hermes_cli import sessions_repair_profiles as repair

    monkeypatch.setattr(repair, "_gateway_multiplexes", lambda _root: False)
    stores = [repair.Store("default", root, routing=True), repair.Store("science", profile)]
    session = repair._Session(read_only=True)
    try:
        plan = repair.RepairPlan(stores).scan(session)
    finally:
        session.close()
    findings = [finding for finding in plan.findings if finding.kind == "dual_home" and finding.subject == session_id]
    assert len(findings) == 1
    assert findings[0].action is None
    assert findings[0].repairable is False
    assert "canonical-owner" in findings[0].reason
    assert not [finding for finding in plan.repairable if finding.subject == session_id]
