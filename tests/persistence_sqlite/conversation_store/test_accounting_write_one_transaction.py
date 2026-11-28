"""A per-API-call accounting write takes one state.db transaction, not two.

``update_token_counts`` ensured the session row existed and then updated the counters as two
separate write transactions. Each one is a BEGIN IMMEDIATE plus a commit and fsync, so the
accounting path paid two commits per API call. The guard and the counter update now share a
single transaction, and the row still lands with the caller's real surface.
"""
import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.sessions import get_session
from persistence.sqlite.conversation._store.usage import update_token_counts


def _count_writes(db):
    calls = []
    real = db._execute_write

    def counting(fn, *args, **kwargs):
        calls.append(fn)
        return real(fn, *args, **kwargs)

    db._execute_write = counting
    return calls


def test_an_accounting_write_takes_one_write_transaction(tmp_path):
    db = StoreHolder(db_path=tmp_path / "state.db")
    try:
        calls = _count_writes(db)
        update_token_counts(
            db, "s", input_tokens=5, output_tokens=1, model="m", api_call_count=1, source="cli")
        assert len(calls) == 1, "the row guard and the counter update must share one transaction"
    finally:
        db.close()


def test_the_guard_still_creates_the_row_with_the_real_surface(tmp_path):
    db = StoreHolder(db_path=tmp_path / "state.db")
    try:
        update_token_counts(
            db, "s", input_tokens=5, output_tokens=1, model="m", api_call_count=1, source="cli")
        row = get_session(db, "s")
        assert row is not None
        assert row["created_source"] == "cli"
    finally:
        db.close()
