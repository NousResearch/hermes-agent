from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").strip().splitlines()) if path.read_text(encoding="utf-8").strip() else 0


def _valid_event(event_id: str, *, value: str = "dark") -> dict:
    return {
        "schema_version": 1,
        "event_id": event_id,
        "occurred_at": "2026-07-17T20:00:00Z",
        "operation": "assert",
        "fact_id": f"fact_{event_id.removeprefix('evt_')}",
        "supersedes": None,
        "fact": {
            "scope": "user", "kind": "preference", "subject": "u1",
            "key": "pref", "value": value,
        },
        "evidence": {
            "type": "user_stated", "profile": "default", "platform": "cli",
            "session_id": "s1", "turn_id": "t1", "task_id": None,
            "speaker_id": "u1", "conversation_id": None, "thread_id": None,
        },
        "extraction": {
            "schema_name": "truth-ledger.fact-candidates.v1",
            "provider": "test", "model": "test", "prompt_version": 1,
        },
    }


def test_append_event_rejects_schema_invalid_event(tmp_path, ledger_mod):
    store = ledger_mod.LedgerStore(tmp_path)
    out = store.append_event(
        event={"event_id": "evt_invalid", "occurred_at": "2026-07-17T20:00:00Z"},
        event_key="invalid",
    )
    assert out["status"] == "rejected"
    assert out["reason"] == "invalid_ledger_event"


def test_append_event_is_idempotent_by_event_key(tmp_path, ledger_mod):
    store = ledger_mod.LedgerStore(tmp_path)
    event = _valid_event("evt_1")

    first = store.append_event(event=event, event_key="k1")
    second = store.append_event(event=event, event_key="k1")

    assert first["status"] == "indexed"
    assert second["status"] == "duplicate"

    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    assert _line_count(ledger_file) == 1


def test_append_event_rejects_oversized_record(tmp_path, ledger_mod):
    store = ledger_mod.LedgerStore(tmp_path, record_hard_bytes=200)
    event = _valid_event("evt_big", value="x" * 500)

    out = store.append_event(event=event, event_key="big-key")
    assert out["status"] == "rejected"
    assert out["reason"] == "record_hard_cap"


def test_scan_quarantines_partial_tail(tmp_path, ledger_mod):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True)
    f = ledger_dir / "2026-07.jsonl"
    f.write_text('{"event_id":"1","event":"assert"}\n{"event_id":"2","event":"assert"}\n{"event_id":"3"', encoding="utf-8")

    parsed = ledger_mod.scan_jsonl_with_tail_quarantine(
        f,
        tmp_path / "errors",
    )

    assert len(parsed) == 2
    quarantined = list((tmp_path / "errors").glob("corrupt-tail-*.jsonl"))
    assert len(quarantined) == 1
    qtxt = quarantined[0].read_text(encoding="utf-8")
    assert "event_id" in qtxt


def test_append_event_returns_retry_when_index_db_is_locked(tmp_path, ledger_mod):
    store = ledger_mod.LedgerStore(tmp_path)
    locker = sqlite3.connect(store.db_path)
    try:
        locker.execute("BEGIN EXCLUSIVE")
        event = _valid_event("evt_locked")
        out = store.append_event(event=event, event_key="locked-key")
    finally:
        locker.rollback()
        locker.close()

    assert out["status"] == "retry"
    assert out["reason"] in {"index_db_locked", "index_db_busy"}


def test_append_event_is_idempotent_when_interrupted_after_append_before_index_commit(tmp_path, ledger_mod, monkeypatch):
    store = ledger_mod.LedgerStore(tmp_path)
    event = _valid_event("evt_intent_retry")

    real_connect = store._connect
    connect_calls = {"count": 0}

    class _ConnProxy:
        def __init__(self, conn, *, fail_update: bool) -> None:
            self._conn = conn
            self._fail_update = fail_update

        def execute(self, sql, params=()):
            if self._fail_update and sql.strip().upper().startswith("UPDATE EVENT_JOURNAL"):
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, params)

        def commit(self):
            return self._conn.commit()

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

    def _connect_with_one_update_lock():
        connect_calls["count"] += 1
        return _ConnProxy(real_connect(), fail_update=connect_calls["count"] == 3)

    monkeypatch.setattr(store, "_connect", _connect_with_one_update_lock)

    first = store.append_event(event=event, event_key="intent-key")
    assert first["status"] == "retry"
    assert first["reason"] == "index_db_locked"

    second = store.append_event(event=event, event_key="intent-key")
    assert second["status"] in {"indexed", "duplicate"}

    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    lines = [line for line in ledger_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1


def test_append_event_retries_after_post_intent_append_lock_timeout(tmp_path, ledger_mod, monkeypatch):
    store = ledger_mod.LedgerStore(tmp_path)
    event = _valid_event("evt_post_intent_timeout")

    real_file_lock = ledger_mod._FileLock
    lock_calls = {"count": 0}

    class _TimeoutFirstAppendLock(real_file_lock):
        def __enter__(self):
            lock_calls["count"] += 1
            if lock_calls["count"] == 1:
                raise TimeoutError("lock timeout")
            return super().__enter__()

    monkeypatch.setattr(ledger_mod, "_FileLock", _TimeoutFirstAppendLock)

    first = store.append_event(event=event, event_key="post-intent-timeout-key")
    assert first == {
        "status": "retry",
        "reason": "append_lock_timeout",
        "event_id": "evt_post_intent_timeout",
    }

    second = store.append_event(event=event, event_key="post-intent-timeout-key")
    assert second["status"] == "indexed"

    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    lines = [line for line in ledger_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
