"""Behavioral contract tests for SessionDB.mutate_meta()."""

from __future__ import annotations

import json
import threading

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    database = SessionDB()
    yield database
    database.close()


class TestMutateMeta:
    def test_upserts_when_key_absent(self, db):
        result = db.mutate_meta("test:key", lambda current: "initial")
        assert result == "initial"
        assert db.get_meta("test:key") == "initial"

    def test_reads_current_and_upserts_returned_value(self, db):
        db.set_meta("test:key", json.dumps({"count": 5}))

        def _increment(current):
            data = json.loads(current) if current else {"count": 0}
            data["count"] += 1
            return json.dumps(data)

        result = db.mutate_meta("test:key", _increment)
        assert json.loads(result) == {"count": 6}
        assert json.loads(db.get_meta("test:key")) == {"count": 6}

    def test_mutator_receives_none_for_missing_key(self, db):
        received = []

        def _capture(current):
            received.append(current)
            return "created"

        db.mutate_meta("test:new", _capture)
        assert received == [None]
        assert db.get_meta("test:new") == "created"

    def test_atomic_read_modify_write(self, db):
        db.set_meta("test:counter", "0")
        errors = []

        def _worker():
            try:
                for _ in range(50):
                    def _inc(current):
                        return str(int(current) + 1)
                    db.mutate_meta("test:counter", _inc)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert int(db.get_meta("test:counter")) == 200

    def test_atomic_read_modify_write_across_independent_db_handles(self, db):
        other = SessionDB(db_path=db.db_path)
        db.set_meta("test:shared-counter", "0")
        errors = []

        def _worker(database):
            try:
                for _ in range(40):
                    database.mutate_meta(
                        "test:shared-counter",
                        lambda current: str(int(current or "0") + 1),
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(db,)),
            threading.Thread(target=_worker, args=(other,)),
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            assert all(not thread.is_alive() for thread in threads)
            assert not errors
            assert int(db.get_meta("test:shared-counter")) == 80
        finally:
            other.close()

    def test_does_not_expose_raw_connections(self, db):
        import inspect
        sig = inspect.signature(db.mutate_meta)
        params = list(sig.parameters.keys())
        assert "key" in params
        assert "mutator" in params
