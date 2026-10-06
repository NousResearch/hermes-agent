"""``rebuild_fts()`` must catch the corruption class it exists to recover from (#133375).

SQLITE_CORRUPT — ``sqlite3.DatabaseError("database disk image is malformed")`` — is not an
``OperationalError`` (its subclass); it is a sibling. The in-place FTS rebuild is the recovery
path for a corrupt index, so a failing ``'rebuild'`` command raises precisely the class the
except arm must cover. Catching only ``OperationalError`` let the error escape the per-index
loop un-rolled-back, and callers could not distinguish "rebuild attempted and hit structural
corruption" from deferral. These tests pin the caught-and-rolled-back behavior.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    if not d._fts_enabled:
        d.close()
        pytest.skip("FTS5 unavailable in this build")
    d.create_session("s1", source="test")
    d.append_message("s1", "user", "hello world")
    yield d
    d.close()


def _corrupting_execute(match, command="rebuild"):
    """Wrap ``execute`` so FTS *command* statements whose SQL satisfies *match* raise the
    corruption-class error; every other statement passes through."""

    def wrap(real_execute):
        def execute(sql, *args, **kwargs):
            if f"VALUES('{command}')" in sql and match(sql):
                raise sqlite3.DatabaseError("database disk image is malformed")
            return real_execute(sql, *args, **kwargs)

        return execute

    return wrap


def test_corruption_class_error_is_caught_rolled_back_and_reported(
    db, monkeypatch, caplog
):
    """The exact production failure (#133375): every index rebuild raises DatabaseError.
    The call must return 0 (no progress), roll the connection back, and say the offline
    repair path is needed — not propagate the error out of the loop."""
    monkeypatch.setattr(
        db._conn, "execute", _corrupting_execute(lambda sql: True)(db._conn.execute)
    )
    with caplog.at_level("ERROR"):
        assert db.rebuild_fts() == 0
    assert any("offline repair" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("method,command", [("rebuild_fts", "rebuild"), ("optimize_fts", "optimize")])
def test_one_corrupt_index_does_not_stop_the_remaining_indexes(db, monkeypatch, method, command):
    """Only messages_fts is corrupt; trigram/cjk must still be processed — the loop survives
    a corruption-class failure on one index. The trailing ``(`` keeps the match off
    ``messages_fts_trigram``/``_cjk``. optimize_fts() shares the per-index loop shape."""
    monkeypatch.setattr(
        db._conn,
        "execute",
        _corrupting_execute(lambda sql: sql.startswith("INSERT INTO messages_fts("), command)(
            db._conn.execute
        ),
    )
    assert getattr(db, method)() >= 1
