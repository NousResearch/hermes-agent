"""Durable Idempotency-Key contract for POST /api/sessions/{id}/chat.

Behavior under test (see gateway/platforms/api_idempotency.py):

- the key is durably reserved (``running``) before any agent execution
- concurrent duplicates coalesce onto a single agent run
- completed turns replay byte-identically with zero agent invocations
- terminal evidence never expires: after replay-payload retention (or the
  persistence byte cap) the receipt survives as a tombstone and retries
  return ``idempotency_response_expired`` — never a second execution
- receipts bind the session *incarnation*, so a deleted-and-recreated
  session ID can never replay the old session's response
- ``complete()``/``release()`` are compare-and-swap on the reservation's
  fingerprint + owner, so stale executions/releases cannot touch a
  replacement receipt
- a ``running`` receipt from a dead/other process is 409 uncertain and is
  never re-executed automatically (at-most-once across restart)
- store open/write/corruption/permission/capacity failures fail closed for
  keyed requests only
- unkeyed requests keep the exact legacy behavior and never touch the store

The autouse ``_hermetic_environment`` fixture points HERMES_HOME at a
per-test tempdir, so the adapter's lazily-opened store writes a real SQLite
file there — storage/config integration is exercised for real, only the
agent itself is faked.
"""

import asyncio
import json
import os
import sqlite3
import stat
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.api_idempotency as api_idempotency
from gateway.config import PlatformConfig
from gateway.platforms.api_idempotency import (
    SESSION_CHAT_SCOPE,
    DurableIdempotencyStore,
    IdempotencyStoreUnavailable,
)
from gateway.platforms.api_server import (
    APIServerAdapter,
    _session_chat_fingerprint,
    _session_incarnation,
)
from hermes_constants import get_hermes_home
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


AUTH = {"Authorization": "Bearer sk-test"}


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


def _store_path():
    return get_hermes_home() / "api_idempotency.db"


def _mock_turn(session_id, text="fresh answer"):
    return AsyncMock(return_value=({"final_response": text, "session_id": session_id}, {"total_tokens": 3}))


def _incarnation(session_db, session_id):
    return _session_incarnation(session_db.get_session(session_id))


def _receipt(adapter, session_id, key, scope=SESSION_CHAT_SCOPE):
    """Read one receipt through an independent store instance (real file)."""
    store = DurableIdempotencyStore(_store_path())
    try:
        return store.get_receipt(
            scope=scope,
            principal=adapter._idempotency_principal_scope(),
            session_id=session_id,
            idempotency_key=key,
        )
    finally:
        store.close()


class _FailingCommitConn:
    """Wrap a sqlite connection so commits fail like a dying disk."""

    def __init__(self, real):
        self._real = real

    def execute(self, *a, **kw):
        return self._real.execute(*a, **kw)

    def commit(self):
        raise sqlite3.OperationalError("simulated disk I/O error")

    def rollback(self):
        return self._real.rollback()

    def close(self):
        return self._real.close()


# ---------------------------------------------------------------------------
# Store-level: durable state machine, restart semantics, tombstones, CAS,
# capacity, privacy gate
# ---------------------------------------------------------------------------


class TestDurableIdempotencyStore:
    def _reserve(self, store, key="key-1", fp="fp-1", session_id="sess",
                 principal="prin", incarnation="inc-1"):
        return store.reserve(
            scope=SESSION_CHAT_SCOPE,
            principal=principal,
            session_id=session_id,
            session_incarnation=incarnation,
            idempotency_key=key,
            fingerprint=fp,
        )

    def _complete(self, store, key="key-1", fp="fp-1", body='{"ok": true}',
                  session_id="sess", principal="prin"):
        return store.complete(
            scope=SESSION_CHAT_SCOPE,
            principal=principal,
            session_id=session_id,
            idempotency_key=key,
            fingerprint=fp,
            response_body=body,
            response_headers={"X-Hermes-Session-Id": session_id},
            response_status=200,
        )

    def test_reserve_complete_replay_roundtrip(self, tmp_path):
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        assert self._reserve(store).kind == "reserved"
        assert self._complete(store) is True
        decision = self._reserve(store)
        assert decision.kind == "replay"
        assert decision.response_body == '{"ok": true}'
        assert decision.response_headers == {"X-Hermes-Session-Id": "sess"}
        assert decision.response_status == 200

    def test_same_instance_duplicate_is_in_progress_local(self, tmp_path):
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        assert self._reserve(store).kind == "reserved"
        assert self._reserve(store).kind == "in_progress_local"

    def test_different_fingerprint_conflicts_in_any_state(self, tmp_path):
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(store, fp="fp-a")
        assert self._reserve(store, fp="fp-b").kind == "conflict"
        self._complete(store, fp="fp-a")
        assert self._reserve(store, fp="fp-b").kind == "conflict"

    def test_different_session_incarnation_conflicts(self, tmp_path):
        """Same textual session ID, new incarnation → never classified as
        the old receipt, even with an identical fingerprint string."""
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(store, incarnation="inc-1")
        self._complete(store)
        assert self._reserve(store, incarnation="inc-2").kind == "conflict"

    def test_restart_running_is_uncertain_and_completed_replays(self, tmp_path):
        """A new store instance (process restart) must fail closed on
        ``running`` receipts it did not create, while ``completed`` receipts
        replay normally."""
        first = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(first, key="crashed-mid-turn")
        self._reserve(first, key="finished")
        self._complete(first, key="finished", body='{"answer": 42}')
        first.close()  # process death

        restarted = DurableIdempotencyStore(tmp_path / "idem.db")
        assert self._reserve(restarted, key="crashed-mid-turn").kind == "uncertain"
        replay = self._reserve(restarted, key="finished")
        assert replay.kind == "replay"
        assert replay.response_body == '{"answer": 42}'

    # -- terminal-evidence permanence (review finding 1) -------------------

    def test_expired_replay_payload_becomes_tombstone_never_fresh(self, tmp_path):
        """The at-most-once regression: aging a completed receipt past
        retention must expire the *payload only* — a retry with the same
        key/fingerprint gets ``response_expired``, never a fresh reservation
        (which would execute the turn a second time)."""
        store = DurableIdempotencyStore(tmp_path / "idem.db", retention_hours=1)
        self._reserve(store, key="aged")
        self._complete(store, key="aged")
        store._conn.execute(
            "UPDATE idempotency_receipts SET completed_at=? WHERE idempotency_key='aged'",
            (time.time() - 25 * 3600,),
        )
        store._conn.commit()

        decision = self._reserve(store, key="aged")

        assert decision.kind == "response_expired"
        receipt = store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="aged"
        )
        assert receipt is not None  # evidence survives
        assert receipt["state"] == "completed"
        assert receipt["response_body"] is None  # payload expired
        assert receipt["fingerprint"] == "fp-1"
        # And it stays that way on further retries.
        assert self._reserve(store, key="aged").kind == "response_expired"
        assert self._reserve(store, key="aged", fp="fp-other").kind == "conflict"

    def test_retention_never_touches_running_receipts(self, tmp_path):
        store = DurableIdempotencyStore(tmp_path / "idem.db", retention_hours=1)
        self._reserve(store, key="ancient-running")
        store._conn.execute(
            "UPDATE idempotency_receipts SET created_at=? WHERE idempotency_key='ancient-running'",
            (time.time() - 365 * 24 * 3600,),
        )
        store._conn.commit()

        self._reserve(store, key="trigger-expiry-pass")

        assert self._reserve(store, key="ancient-running").kind == "in_progress_local"
        receipt = store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess",
            idempotency_key="ancient-running",
        )
        assert receipt["state"] == "running"

    def test_capacity_fails_closed_for_new_keys_only(self, tmp_path, monkeypatch):
        """At capacity, NEW keys are refused (no eviction of evidence);
        keys that already have receipts are still classified normally."""
        monkeypatch.setattr(api_idempotency, "MAX_TOTAL_RECEIPTS", 2)
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(store, key="existing-1")
        self._complete(store, key="existing-1", body='{"kept": 1}')
        self._reserve(store, key="existing-2")

        with pytest.raises(IdempotencyStoreUnavailable):
            self._reserve(store, key="brand-new")

        assert store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess",
            idempotency_key="brand-new",
        ) is None
        # Evidence intact and still served at capacity.
        assert store.count_receipts() == 2
        assert self._reserve(store, key="existing-1").kind == "replay"
        assert self._reserve(store, key="existing-2").kind == "in_progress_local"

    def test_running_capacity_guard_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api_idempotency, "MAX_ACTIVE_RUNNING_RECEIPTS", 2)
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(store, key="r1")
        self._reserve(store, key="r2")
        with pytest.raises(IdempotencyStoreUnavailable):
            self._reserve(store, key="r3")
        assert store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="r3"
        ) is None

    def test_purge_completed_is_explicit_and_spares_running_and_fresh(self, tmp_path):
        store = DurableIdempotencyStore(tmp_path / "idem.db", retention_hours=1)
        self._reserve(store, key="old-done")
        self._complete(store, key="old-done")
        self._reserve(store, key="fresh-done")
        self._complete(store, key="fresh-done")
        self._reserve(store, key="old-running")
        ancient = time.time() - 100 * 3600
        store._conn.execute(
            "UPDATE idempotency_receipts SET completed_at=? WHERE idempotency_key='old-done'",
            (ancient,),
        )
        store._conn.execute(
            "UPDATE idempotency_receipts SET created_at=? WHERE idempotency_key='old-running'",
            (ancient,),
        )
        store._conn.commit()

        with pytest.raises(ValueError):
            store.purge_completed(older_than_hours=0.5)  # below replay retention

        removed = store.purge_completed(older_than_hours=48)

        assert removed == 1
        assert store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="old-done"
        ) is None
        assert self._reserve(store, key="fresh-done").kind == "replay"
        assert self._reserve(store, key="old-running").kind == "in_progress_local"

    # -- compare-and-complete / compare-and-release (review finding 3) -----

    def test_stale_complete_cannot_poison_replacement_receipt(self, tmp_path):
        """Exact review repro: owner A reserves fp_old; the reconciler
        releases; owner B reserves fp_new; A's late completion must NOT
        write the old reply into B's receipt."""
        path = tmp_path / "idem.db"
        owner_a = DurableIdempotencyStore(path)
        assert self._reserve(owner_a, fp="fp-old").kind == "reserved"

        # Reconciler (a different process) inspects and releases A's
        # dead-owner receipt via CAS on the inspected values.
        reconciler = DurableIdempotencyStore(path)
        inspected = reconciler.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert reconciler.release(
            scope=SESSION_CHAT_SCOPE,
            principal="prin",
            session_id="sess",
            idempotency_key="key-1",
            fingerprint=inspected["fingerprint"],
            owner_instance_id=inspected["owner_instance_id"],
        ) is True

        owner_b = DurableIdempotencyStore(path)
        assert self._reserve(owner_b, fp="fp-new").kind == "reserved"

        # A finishes late and tries to complete its long-gone reservation.
        assert self._complete(owner_a, fp="fp-old", body='{"stale": "OLD"}') is False

        poisoned = owner_b.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert poisoned["state"] == "running"  # B's reservation untouched
        assert poisoned["fingerprint"] == "fp-new"
        assert poisoned["owner_instance_id"] == owner_b.instance_id
        assert poisoned["response_body"] is None

        # B completes normally and its own reply replays.
        assert self._complete(owner_b, fp="fp-new", body='{"fresh": "NEW"}') is True
        replay = self._reserve(owner_b, fp="fp-new")
        assert replay.kind == "replay"
        assert replay.response_body == '{"fresh": "NEW"}'

    def test_stale_release_returns_false(self, tmp_path):
        """A release() computed from an outdated inspection must not delete
        the key's replacement receipt."""
        path = tmp_path / "idem.db"
        owner_a = DurableIdempotencyStore(path)
        self._reserve(owner_a, fp="fp-old")
        stale_inspection = owner_a.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        # Legitimate reconciliation + re-reservation by a new owner.
        assert owner_a.release(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess",
            idempotency_key="key-1",
            fingerprint=stale_inspection["fingerprint"],
            owner_instance_id=stale_inspection["owner_instance_id"],
        ) is True
        owner_b = DurableIdempotencyStore(path)
        self._reserve(owner_b, fp="fp-new")

        # Replaying the OLD inspection against the NEW receipt: no match.
        assert owner_a.release(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess",
            idempotency_key="key-1",
            fingerprint=stale_inspection["fingerprint"],
            owner_instance_id=stale_inspection["owner_instance_id"],
        ) is False
        survivor = owner_b.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert survivor is not None
        assert survivor["fingerprint"] == "fp-new"

    def test_release_then_reserve_after_reconciliation(self, tmp_path):
        first = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(first)
        first.close()
        restarted = DurableIdempotencyStore(tmp_path / "idem.db")
        assert self._reserve(restarted).kind == "uncertain"
        inspected = restarted.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert restarted.release(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess",
            idempotency_key="key-1",
            fingerprint=inspected["fingerprint"],
            owner_instance_id=inspected["owner_instance_id"],
        ) is True
        assert self._reserve(restarted).kind == "reserved"

    # -- persisted-response byte cap (review finding 5) ---------------------

    def test_oversized_response_terminalizes_without_replay(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api_idempotency, "MAX_PERSISTED_RESPONSE_BYTES", 64)
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(store)
        big_body = json.dumps({"content": "x" * 500})

        assert self._complete(store, body=big_body) is False  # no replay persisted

        receipt = store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert receipt["state"] == "completed"  # terminal evidence
        assert receipt["response_body"] is None  # no oversized bytes stored
        assert self._reserve(store).kind == "response_expired"  # never re-executes

    # -- privacy enforcement gate (review finding 4) ------------------------

    @pytest.mark.skipif(os.name != "posix", reason="POSIX mode enforcement only")
    def test_chmod_denied_fails_closed_on_open(self, tmp_path, monkeypatch):
        """Exact review repro: permission enforcement failure must raise,
        not proceed with a non-private response-bearing DB."""
        def _denied(self, mode):
            raise PermissionError("chmod denied by policy")

        monkeypatch.setattr(Path, "chmod", _denied)
        with pytest.raises(IdempotencyStoreUnavailable):
            DurableIdempotencyStore(tmp_path / "idem.db")

    @pytest.mark.skipif(os.name != "posix", reason="POSIX mode enforcement only")
    def test_chmod_denied_after_open_fails_closed_on_reserve(self, tmp_path, monkeypatch):
        store = DurableIdempotencyStore(tmp_path / "idem.db")

        def _denied(self, mode):
            raise PermissionError("chmod denied by policy")

        monkeypatch.setattr(Path, "chmod", _denied)
        with pytest.raises(IdempotencyStoreUnavailable):
            self._reserve(store)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX mode enforcement only")
    def test_privacy_failure_at_completion_drops_payload_not_evidence(self, tmp_path, monkeypatch):
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(store)

        def _broken(self=store):
            raise IdempotencyStoreUnavailable("simulated unenforceable permissions")

        monkeypatch.setattr(store, "_enforce_owner_only_permissions", _broken)

        assert self._complete(store, body='{"secret": "reply"}') is False

        receipt = store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert receipt["state"] == "completed"
        assert receipt["response_body"] is None  # reply bytes never written

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX file modes are not enforced on NTFS")
    def test_store_and_sidecars_owner_only_under_permissive_umask(self, tmp_path):
        old_umask = os.umask(0o000)
        try:
            path = tmp_path / "idem.db"
            store = DurableIdempotencyStore(path)
            self._reserve(store)
            self._complete(store)
        finally:
            os.umask(old_umask)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        for sidecar in (Path(f"{path}-wal"), Path(f"{path}-shm")):
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600, sidecar

    # -- failure modes -------------------------------------------------------

    def test_open_failure_raises_unavailable(self, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("file where a directory must be")
        with pytest.raises(IdempotencyStoreUnavailable):
            DurableIdempotencyStore(blocker / "idem.db")

    def test_corrupt_database_raises_unavailable(self, tmp_path):
        path = tmp_path / "idem.db"
        path.write_bytes(b"this is not a sqlite database, sorry")
        with pytest.raises(IdempotencyStoreUnavailable):
            DurableIdempotencyStore(path)

    def test_reserve_commit_failure_raises_unavailable(self, tmp_path):
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        real_conn = store._conn
        store._conn = _FailingCommitConn(real_conn)
        with pytest.raises(IdempotencyStoreUnavailable):
            self._reserve(store)
        store._conn = real_conn
        assert store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        ) is None

    def test_complete_failure_keeps_receipt_running(self, tmp_path):
        store = DurableIdempotencyStore(tmp_path / "idem.db")
        self._reserve(store)
        real_conn = store._conn
        store._conn = _FailingCommitConn(real_conn)
        assert self._complete(store) is False
        store._conn = real_conn
        receipt = store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert receipt["state"] == "running"

    # -- schema migration ----------------------------------------------------

    def test_pre_incarnation_schema_migrates_and_old_rows_never_replay(self, tmp_path):
        """Stores created before the incarnation column are ALTERed in place;
        their rows (incarnation='') can never match a real incarnation, so
        the worst case for a pre-migration receipt is a safe conflict."""
        path = tmp_path / "idem.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            """CREATE TABLE idempotency_receipts (
                scope TEXT NOT NULL,
                principal_hash TEXT NOT NULL,
                session_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('running', 'completed')),
                owner_instance_id TEXT NOT NULL,
                response_body TEXT,
                response_headers TEXT,
                response_status INTEGER,
                created_at REAL NOT NULL,
                completed_at REAL,
                PRIMARY KEY (scope, principal_hash, session_id, idempotency_key)
            )"""
        )
        conn.execute(
            "INSERT INTO idempotency_receipts VALUES"
            " (?, 'prin', 'sess', 'key-1', 'fp-1', 'completed', 'dead-owner',"
            "  '{\"old\": true}', '{}', 200, ?, ?)",
            (SESSION_CHAT_SCOPE, time.time(), time.time()),
        )
        conn.commit()
        conn.close()

        store = DurableIdempotencyStore(path)
        receipt = store.get_receipt(
            scope=SESSION_CHAT_SCOPE, principal="prin", session_id="sess", idempotency_key="key-1"
        )
        assert receipt["session_incarnation"] == ""  # row preserved
        assert self._reserve(store, fp="fp-1", incarnation="inc-real").kind == "conflict"


# ---------------------------------------------------------------------------
# Endpoint-level: contract through the real HTTP surface + real store file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_keyed_request_records_durable_receipt(auth_adapter, session_db):
    session_id = session_db.create_session("idem-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={**AUTH, "Idempotency-Key": "founder-msg-1"},
            )
            assert resp.status == 200, await resp.text()
            assert resp.headers["X-Hermes-Idempotency"] == "recorded"
            assert resp.headers["X-Hermes-Session-Id"] == session_id
            payload = await resp.json()

    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["message"]["content"] == "fresh answer"
    mock_run.assert_awaited_once()
    receipt = _receipt(auth_adapter, session_id, "founder-msg-1")
    assert receipt is not None
    assert receipt["state"] == "completed"
    assert json.loads(receipt["response_body"]) == payload
    assert receipt["session_incarnation"] == _incarnation(session_db, session_id)


@pytest.mark.asyncio
async def test_completed_replay_invokes_agent_zero_times(auth_adapter, session_db):
    session_id = session_db.create_session("replay-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={**AUTH, "Idempotency-Key": "founder-msg-2"},
            )
            first_body = await first.text()
            second = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={**AUTH, "Idempotency-Key": "founder-msg-2"},
            )
            second_body = await second.text()

    assert first.headers["X-Hermes-Idempotency"] == "recorded"
    assert second.status == 200
    assert second.headers["X-Hermes-Idempotency"] == "replayed"
    assert second.headers["X-Hermes-Session-Id"] == session_id
    assert second_body == first_body  # byte-identical replay
    mock_run.assert_awaited_once()  # zero additional agent invocations


@pytest.mark.asyncio
async def test_concurrent_duplicates_invoke_agent_once(auth_adapter, session_db):
    session_id = session_db.create_session("concurrent-session", "api_server")
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_run(**kwargs):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"final_response": "single run", "session_id": session_id}, {"total_tokens": 1}

    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:

            async def post():
                return await cli.post(
                    f"/api/sessions/{session_id}/chat",
                    json={"message": "run once"},
                    headers={**AUTH, "Idempotency-Key": "dup-key"},
                )

            store = auth_adapter._ensure_idempotency_store()
            reserve_kinds = []
            real_reserve = store.reserve

            def spy_reserve(**kwargs):
                decision = real_reserve(**kwargs)
                reserve_kinds.append(decision.kind)
                return decision

            with patch.object(store, "reserve", side_effect=spy_reserve):
                t1 = asyncio.create_task(post())
                await started.wait()  # turn is executing and registered in-flight
                t2 = asyncio.create_task(post())
                while len(reserve_kinds) < 2:
                    await asyncio.sleep(0.01)
                release.set()
                r1, r2 = await asyncio.gather(t1, t2)
                bodies = [await r1.text(), await r2.text()]

    assert calls == 1
    assert reserve_kinds == ["reserved", "in_progress_local"]
    assert r1.status == r2.status == 200
    assert bodies[0] == bodies[1]
    assert {r1.headers["X-Hermes-Idempotency"], r2.headers["X-Hermes-Idempotency"]} == {
        "recorded",
        "coalesced",
    }


@pytest.mark.asyncio
async def test_same_key_different_payload_conflicts(auth_adapter, session_db):
    session_id = session_db.create_session("conflict-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "original message"},
                headers={**AUTH, "Idempotency-Key": "reused-key"},
            )
            assert first.status == 200
            conflict = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "DIFFERENT message"},
                headers={**AUTH, "Idempotency-Key": "reused-key"},
            )
            body = await conflict.json()

    assert conflict.status == 409
    assert body["error"]["code"] == "idempotency_conflict"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_key_header_is_part_of_fingerprint(auth_adapter, session_db):
    session_id = session_db.create_session("scope-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={**AUTH, "Idempotency-Key": "scoped-key", "X-Hermes-Session-Key": "client-A"},
            )
            assert first.status == 200
            other_scope = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={**AUTH, "Idempotency-Key": "scoped-key", "X-Hermes-Session-Key": "client-B"},
            )
            body = await other_scope.json()

    assert other_scope.status == 409
    assert body["error"]["code"] == "idempotency_conflict"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_normalized_multimodal_fingerprint_is_stable(auth_adapter, session_db):
    """Two wire encodings that normalize identically must replay, not conflict."""
    session_id = session_db.create_session("normalize-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": [{"type": "input_text", "text": "same content"}]},
                headers={**AUTH, "Idempotency-Key": "normalized-key"},
            )
            assert first.status == 200
            second = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": [{"type": "text", "text": "same content"}]},
                headers={**AUTH, "Idempotency-Key": "normalized-key"},
            )

    assert second.status == 200
    assert second.headers["X-Hermes-Idempotency"] == "replayed"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_deleted_and_recreated_session_never_replays_old_receipt(auth_adapter, session_db):
    """Exact review repro, via the real endpoints: complete a keyed turn,
    DELETE the session, recreate the same textual ID, resend the same
    key/payload — the recreated session must never receive the old
    session's stored reply (and the agent must not silently run either)."""
    app = _create_session_app(auth_adapter)
    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs["session_id"])
        return (
            {"final_response": f"execution-{len(calls)}", "session_id": kwargs["session_id"]},
            {"total_tokens": 1},
        )

    with patch.object(auth_adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            create = await cli.post("/api/sessions", json={"id": "reused-id"}, headers=AUTH)
            assert create.status == 201, await create.text()
            first_incarnation = _incarnation(session_db, "reused-id")

            first = await cli.post(
                "/api/sessions/reused-id/chat",
                json={"message": "same payload"},
                headers={**AUTH, "Idempotency-Key": "same-key"},
            )
            assert first.status == 200
            first_payload = await first.json()
            assert first_payload["message"]["content"] == "execution-1"

            # Sanity: same incarnation still replays.
            replay = await cli.post(
                "/api/sessions/reused-id/chat",
                json={"message": "same payload"},
                headers={**AUTH, "Idempotency-Key": "same-key"},
            )
            assert replay.headers["X-Hermes-Idempotency"] == "replayed"

            deleted = await cli.delete("/api/sessions/reused-id", headers=AUTH)
            assert deleted.status == 200
            recreated = await cli.post("/api/sessions", json={"id": "reused-id"}, headers=AUTH)
            assert recreated.status == 201, await recreated.text()
            second_incarnation = _incarnation(session_db, "reused-id")
            assert second_incarnation != first_incarnation  # new incarnation

            retry = await cli.post(
                "/api/sessions/reused-id/chat",
                json={"message": "same payload"},
                headers={**AUTH, "Idempotency-Key": "same-key"},
            )
            retry_body = await retry.json()

    assert retry.status == 409
    assert retry_body["error"]["code"] == "idempotency_conflict"
    assert "execution-1" not in json.dumps(retry_body)  # no stale replay
    assert calls == ["reused-id"]  # exactly one execution ever


@pytest.mark.asyncio
async def test_expired_retention_retry_never_executes_again(auth_adapter, session_db):
    """Exact review repro: age completed_at past the retention window and
    resend the identical request — pre-fix this reserved fresh and ran the
    agent a second time; now it must be a labeled 409 with zero reruns."""
    session_id = session_db.create_session("aged-session", "api_server")
    app = _create_session_app(auth_adapter)
    calls = []

    async def fake_run(**kwargs):
        calls.append(1)
        return (
            {"final_response": f"execution-{len(calls)}", "session_id": session_id},
            {"total_tokens": 1},
        )

    with patch.object(auth_adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "age me"},
                headers={**AUTH, "Idempotency-Key": "aged-key"},
            )
            assert first.status == 200
            assert (await first.json())["message"]["content"] == "execution-1"

            store = auth_adapter._ensure_idempotency_store()
            store._conn.execute(
                "UPDATE idempotency_receipts SET completed_at=? WHERE idempotency_key='aged-key'",
                (time.time() - 25 * 3600,),  # past the default 24h retention
            )
            store._conn.commit()

            retry = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "age me"},
                headers={**AUTH, "Idempotency-Key": "aged-key"},
            )
            retry_body = await retry.json()

    assert retry.status == 409
    assert retry_body["error"]["code"] == "idempotency_response_expired"
    assert len(calls) == 1  # the turn never executed a second time
    receipt = _receipt(auth_adapter, session_id, "aged-key")
    assert receipt["state"] == "completed"  # tombstone evidence survives
    assert receipt["response_body"] is None


@pytest.mark.asyncio
async def test_oversized_response_returns_live_but_never_replays(auth_adapter, session_db, monkeypatch):
    monkeypatch.setattr(api_idempotency, "MAX_PERSISTED_RESPONSE_BYTES", 128)
    session_id = session_db.create_session("big-session", "api_server")
    big_text = "B" * 1000
    mock_run = _mock_turn(session_id, text=big_text)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "make it big"},
                headers={**AUTH, "Idempotency-Key": "big-key"},
            )
            first_payload = await first.json()
            retry = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "make it big"},
                headers={**AUTH, "Idempotency-Key": "big-key"},
            )
            retry_body = await retry.json()

    assert first.status == 200  # the original caller still gets the reply
    assert first.headers["X-Hermes-Idempotency"] == "recorded"
    assert first_payload["message"]["content"] == big_text
    assert retry.status == 409
    assert retry_body["error"]["code"] == "idempotency_response_expired"
    mock_run.assert_awaited_once()  # never executed twice
    receipt = _receipt(auth_adapter, session_id, "big-key")
    assert receipt["state"] == "completed"
    assert receipt["response_body"] is None  # oversized bytes never persisted


@pytest.mark.asyncio
async def test_capacity_fails_closed_for_new_keys_but_serves_existing(auth_adapter, session_db, monkeypatch):
    monkeypatch.setattr(api_idempotency, "MAX_TOTAL_RECEIPTS", 1)
    session_id = session_db.create_session("cap-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "fills the store"},
                headers={**AUTH, "Idempotency-Key": "cap-key-1"},
            )
            assert first.status == 200
            new_key = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "another"},
                headers={**AUTH, "Idempotency-Key": "cap-key-2"},
            )
            new_key_body = await new_key.json()
            replay = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "fills the store"},
                headers={**AUTH, "Idempotency-Key": "cap-key-1"},
            )

    assert new_key.status == 503
    assert new_key_body["error"]["code"] == "idempotency_store_unavailable"
    # Never suggest bypassing the contract to get past the failure.
    assert "without idempotency-key" not in new_key_body["error"]["message"].lower()
    assert "bypass" not in new_key_body["error"]["message"].lower()
    assert replay.status == 200  # evidence still served at capacity
    assert replay.headers["X-Hermes-Idempotency"] == "replayed"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode enforcement only")
async def test_chmod_denied_fails_closed_with_zero_executions(auth_adapter, session_db, monkeypatch):
    """Exact review repro: under a permissive umask with chmod denied, the
    keyed path must refuse to run rather than store replies in a
    group/world-readable DB."""
    session_id = session_db.create_session("privacy-session", "api_server")

    def _denied(self, mode):
        raise PermissionError("chmod denied by policy")

    monkeypatch.setattr(Path, "chmod", _denied)
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            keyed = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "private"},
                headers={**AUTH, "Idempotency-Key": "privacy-key"},
            )
            keyed_body = await keyed.json()
            unkeyed = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "legacy"},
                headers=AUTH,
            )

    assert keyed.status == 503
    assert keyed_body["error"]["code"] == "idempotency_store_unavailable"
    assert unkeyed.status == 200  # unkeyed traffic unaffected
    mock_run.assert_awaited_once()  # only the unkeyed request ran


@pytest.mark.asyncio
async def test_restart_with_running_receipt_is_uncertain_and_never_reruns(auth_adapter, session_db):
    """A ``running`` receipt persisted by a dead process must hold the key
    closed: deterministic 409 idempotency_state_uncertain, zero agent calls."""
    session_id = session_db.create_session("restart-session", "api_server")
    incarnation = _incarnation(session_db, session_id)
    fingerprint = _session_chat_fingerprint(
        session_id=session_id,
        session_incarnation=incarnation,
        gateway_session_key=None,
        user_message="retry me",
        system_prompt=None,
    )
    dead_process_store = DurableIdempotencyStore(_store_path())
    decision = dead_process_store.reserve(
        scope=SESSION_CHAT_SCOPE,
        principal=auth_adapter._idempotency_principal_scope(),
        session_id=session_id,
        session_incarnation=incarnation,
        idempotency_key="orphaned-key",
        fingerprint=fingerprint,
    )
    assert decision.kind == "reserved"
    dead_process_store.close()

    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "retry me"},
                headers={**AUTH, "Idempotency-Key": "orphaned-key"},
            )
            body = await resp.json()

    assert resp.status == 409
    assert body["error"]["code"] == "idempotency_state_uncertain"
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_restart_with_completed_receipt_replays(auth_adapter, session_db):
    session_id = session_db.create_session("restart-replay-session", "api_server")
    incarnation = _incarnation(session_db, session_id)
    fingerprint = _session_chat_fingerprint(
        session_id=session_id,
        session_incarnation=incarnation,
        gateway_session_key=None,
        user_message="what happened?",
        system_prompt=None,
    )
    principal = auth_adapter._idempotency_principal_scope()
    stored_body = json.dumps({
        "object": "hermes.session.chat.completion",
        "session_id": session_id,
        "message": {"role": "assistant", "content": "answer from before the restart"},
        "usage": {"total_tokens": 9},
    })
    dead_process_store = DurableIdempotencyStore(_store_path())
    dead_process_store.reserve(
        scope=SESSION_CHAT_SCOPE,
        principal=principal,
        session_id=session_id,
        session_incarnation=incarnation,
        idempotency_key="pre-restart-key",
        fingerprint=fingerprint,
    )
    dead_process_store.complete(
        scope=SESSION_CHAT_SCOPE,
        principal=principal,
        session_id=session_id,
        idempotency_key="pre-restart-key",
        fingerprint=fingerprint,
        response_body=stored_body,
        response_headers={"X-Hermes-Session-Id": session_id},
        response_status=200,
    )
    dead_process_store.close()

    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "what happened?"},
                headers={**AUTH, "Idempotency-Key": "pre-restart-key"},
            )
            text = await resp.text()

    assert resp.status == 200
    assert resp.headers["X-Hermes-Idempotency"] == "replayed"
    assert text == stored_body
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_reservation_write_failure_prevents_agent_invocation(auth_adapter, session_db):
    session_id = session_db.create_session("commit-fail-session", "api_server")
    store = auth_adapter._ensure_idempotency_store()
    store._conn = _FailingCommitConn(store._conn)
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "must not run"},
                headers={**AUTH, "Idempotency-Key": "unwritable-key"},
            )
            body = await resp.json()

    assert resp.status == 503
    assert body["error"]["code"] == "idempotency_store_unavailable"
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_store_fails_closed_for_keyed_and_leaves_unkeyed_green(auth_adapter, session_db):
    session_id = session_db.create_session("corrupt-session", "api_server")
    _store_path().write_bytes(b"garbage bytes, definitely not sqlite")

    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            keyed = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "keyed"},
                headers={**AUTH, "Idempotency-Key": "any-key"},
            )
            keyed_body = await keyed.json()
            unkeyed = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "legacy"},
                headers=AUTH,
            )

    assert keyed.status == 503
    assert keyed_body["error"]["code"] == "idempotency_store_unavailable"
    assert "without idempotency-key" not in keyed_body["error"]["message"].lower()
    assert unkeyed.status == 200  # legacy path untouched by the broken store
    mock_run.assert_awaited_once()  # only the unkeyed request ran


@pytest.mark.asyncio
async def test_invalid_and_oversized_keys_rejected_before_any_state(auth_adapter, session_db):
    session_id = session_db.create_session("badkey-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    bad_keys = ["", "has space", "x" * 201, "tab\tchar"]
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            for bad in bad_keys:
                resp = await cli.post(
                    f"/api/sessions/{session_id}/chat",
                    json={"message": "hello"},
                    headers={**AUTH, "Idempotency-Key": bad},
                )
                body = await resp.json()
                assert resp.status == 400, bad
                assert body["error"]["code"] == "invalid_idempotency_key"

    mock_run.assert_not_awaited()
    assert not _store_path().exists()  # rejected keys never open the store


@pytest.mark.asyncio
async def test_unauthenticated_requests_cannot_create_or_probe_receipts(auth_adapter, session_db):
    session_id = session_db.create_session("unauth-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            create_attempt = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={"Idempotency-Key": "probe-key"},
            )
            assert create_attempt.status == 401
            assert not _store_path().exists()

            # Plant a completed receipt, then verify an unauthenticated
            # request cannot replay (probe) it.
            incarnation = _incarnation(session_db, session_id)
            fingerprint = _session_chat_fingerprint(
                session_id=session_id,
                session_incarnation=incarnation,
                gateway_session_key=None,
                user_message="hello",
                system_prompt=None,
            )
            principal = auth_adapter._idempotency_principal_scope()
            planted = DurableIdempotencyStore(_store_path())
            planted.reserve(
                scope=SESSION_CHAT_SCOPE,
                principal=principal,
                session_id=session_id,
                session_incarnation=incarnation,
                idempotency_key="probe-key",
                fingerprint=fingerprint,
            )
            planted.complete(
                scope=SESSION_CHAT_SCOPE,
                principal=principal,
                session_id=session_id,
                idempotency_key="probe-key",
                fingerprint=fingerprint,
                response_body='{"secret": "stored response"}',
                response_headers={},
                response_status=200,
            )
            planted.close()

            probe = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={"Authorization": "Bearer wrong-key", "Idempotency-Key": "probe-key"},
            )
            probe_text = await probe.text()

    assert probe.status == 401
    assert "stored response" not in probe_text
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_idempotency_key_requires_configured_api_key(adapter, session_db):
    """Mirrors X-Hermes-Session-Key: without API_SERVER_KEY there is no
    authenticated principal to scope receipts to, so keys are refused."""
    session_id = session_db.create_session("nokey-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hello"},
                headers={"Idempotency-Key": "some-key"},
            )
            body = await resp.json()

    assert resp.status == 403
    assert "Idempotency-Key requires API key" in body["error"]["message"]
    mock_run.assert_not_awaited()
    assert not _store_path().exists()


@pytest.mark.asyncio
async def test_unkeyed_requests_keep_legacy_behavior_and_skip_the_store(auth_adapter, session_db):
    session_id = session_db.create_session("legacy-session", "api_server")
    session_db.append_message(session_id, "user", "earlier")
    session_db.append_message(session_id, "assistant", "prior answer")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "next", "system_message": "stay focused"},
                headers={**AUTH, "X-Hermes-Session-Key": "client-42"},
            )
            payload = await resp.json()

    assert resp.status == 200
    assert "X-Hermes-Idempotency" not in resp.headers
    assert resp.headers["X-Hermes-Session-Id"] == session_id
    assert resp.headers["X-Hermes-Session-Key"] == "client-42"
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["message"]["content"] == "fresh answer"
    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == session_id
    assert kwargs["gateway_session_key"] == "client-42"
    assert kwargs["ephemeral_system_prompt"] == "stay focused"
    assert [m["role"] for m in kwargs["conversation_history"]] == ["user", "assistant"]
    assert not _store_path().exists()  # unkeyed traffic never touches the store


@pytest.mark.asyncio
async def test_streaming_endpoint_rejects_idempotency_keys_explicitly(auth_adapter, session_db):
    session_id = session_db.create_session("stream-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "stream this"},
                headers={**AUTH, "Idempotency-Key": "stream-key"},
            )
            body = await resp.json()

    assert resp.status == 400
    assert body["error"]["code"] == "idempotency_not_supported"
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_failure_holds_receipt_and_retry_fails_closed(auth_adapter, session_db):
    """If the reserved turn dies mid-execution, tools may already have run:
    the receipt must stay ``running``, a retry must not re-execute, and the
    HTTP error must not reflect internal exception text."""
    session_id = session_db.create_session("failed-turn-session", "api_server")
    mock_run = AsyncMock(side_effect=RuntimeError("provider exploded sk-secret-123"))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "dangerous turn"},
                headers={**AUTH, "Idempotency-Key": "failed-key"},
            )
            first_body = await first.json()
            retry = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "dangerous turn"},
                headers={**AUTH, "Idempotency-Key": "failed-key"},
            )
            retry_body = await retry.json()

    assert first.status == 500
    assert first_body["error"]["code"] == "agent_error"
    # Generic body: internal exception text is not reflected to the caller.
    assert "provider exploded" not in json.dumps(first_body)
    assert "sk-secret-123" not in json.dumps(first_body)
    assert retry.status == 409
    assert retry_body["error"]["code"] == "idempotency_state_uncertain"
    assert mock_run.await_count == 1
    receipt = _receipt(auth_adapter, session_id, "failed-key")
    assert receipt["state"] == "running"


@pytest.mark.asyncio
async def test_disconnect_closes_idempotency_store(auth_adapter, session_db):
    """Same fd-leak contract as the ResponseStore (#37011): the reconnect
    loop builds a fresh adapter per retry, so disconnect() must release the
    receipt store's SQLite connection."""
    session_id = session_db.create_session("teardown-session", "api_server")
    mock_run = _mock_turn(session_id)
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "open the store"},
                headers={**AUTH, "Idempotency-Key": "teardown-key"},
            )
            assert resp.status == 200

    store = auth_adapter._idempotency_store
    assert store is not None
    await auth_adapter.disconnect()
    with pytest.raises(sqlite3.ProgrammingError):
        store._conn.execute("SELECT 1").fetchone()
    assert auth_adapter._idempotency_store is None


@pytest.mark.asyncio
async def test_capabilities_advertise_session_chat_idempotency(auth_adapter):
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities", headers=AUTH)
        assert resp.status == 200
        data = await resp.json()

    assert data["features"]["session_chat_idempotency"] is True
    assert data["features"]["idempotency_key_header"] == "Idempotency-Key"


@pytest.mark.asyncio
async def test_e2e_real_store_real_executor_roundtrip(auth_adapter, session_db, monkeypatch):
    """End-to-end through the real ``_run_agent`` executor, the real receipt
    store file under the (temp) HERMES_HOME, and real config resolution —
    only the AIAgent itself is faked."""
    session_id = session_db.create_session("e2e-session", "api_server")
    turns = []

    class FakeAgent:
        session_prompt_tokens = 5
        session_completion_tokens = 7
        session_total_tokens = 12

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            turns.append(task_id)
            return {"final_response": f"echo:{user_message}"}

    monkeypatch.setattr(
        auth_adapter, "_create_agent", lambda **kwargs: FakeAgent(kwargs["session_id"])
    )

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "real path"},
            headers={**AUTH, "Idempotency-Key": "e2e-key"},
        )
        first_body = await first.text()
        replay = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "real path"},
            headers={**AUTH, "Idempotency-Key": "e2e-key"},
        )
        replay_body = await replay.text()

    assert first.status == 200
    assert json.loads(first_body)["message"]["content"] == "echo:real path"
    assert json.loads(first_body)["usage"]["total_tokens"] == 12
    assert replay.status == 200
    assert replay.headers["X-Hermes-Idempotency"] == "replayed"
    assert replay_body == first_body
    assert turns == [session_id]  # exactly one real executor run

    # The receipt landed in the profile-aware store file and survives a
    # fresh store instance (restart) as a replayable completed receipt.
    assert _store_path().exists()
    if not sys.platform.startswith("win"):
        assert stat.S_IMODE(_store_path().stat().st_mode) == 0o600
    receipt = _receipt(auth_adapter, session_id, "e2e-key")
    assert receipt["state"] == "completed"
    assert receipt["response_body"] == first_body
    assert receipt["session_incarnation"] == _incarnation(session_db, session_id)
