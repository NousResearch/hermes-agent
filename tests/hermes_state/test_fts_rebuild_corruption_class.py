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


_MALFORMED = sqlite3.DatabaseError("database disk image is malformed")


def _corrupting_execute(match, command="rebuild", error=_MALFORMED):
    """Wrap ``execute`` so FTS *command* statements whose SQL satisfies *match* raise *error*
    (default: the corruption-class error); every other statement passes through."""

    def wrap(real_execute):
        def execute(sql, *args, **kwargs):
            if f"VALUES('{command}')" in sql and match(sql):
                raise error
            return real_execute(sql, *args, **kwargs)

        return execute

    return wrap


@pytest.mark.parametrize(
    "error,repair_hint",
    [(_MALFORMED, True), (sqlite3.IntegrityError("UNIQUE constraint failed"), False)],
)
def test_corruption_class_error_is_caught_rolled_back_and_reported(
    db, monkeypatch, caplog, error, repair_hint
):
    """The exact production failure (#133375): every index rebuild raises DatabaseError.
    The call must return 0 (no progress) rather than propagate, and only a malformed-image
    error earns the offline-repair hint — an IntegrityError is not corruption."""
    monkeypatch.setattr(
        db._conn, "execute", _corrupting_execute(lambda sql: True, error=error)(db._conn.execute)
    )
    with caplog.at_level("WARNING"):
        assert db.rebuild_fts() == 0
    assert any("hermes sessions repair" in rec.message for rec in caplog.records) is repair_hint


@pytest.mark.parametrize("method,command", [("rebuild_fts", "rebuild"), ("optimize_fts", "optimize")])
def test_one_corrupt_index_does_not_stop_the_remaining_indexes(db, monkeypatch, method, command):
    """Only messages_fts is corrupt; trigram/cjk must still be processed — the loop survives
    a corruption-class failure on one index. The trailing ``(`` keeps the match off
    ``messages_fts_trigram``/``_cjk``. optimize_fts() shares the per-index loop shape."""
    expected = len(db._present_fts_tables()) - 1
    monkeypatch.setattr(
        db._conn,
        "execute",
        _corrupting_execute(lambda sql: sql.startswith("INSERT INTO messages_fts("), command)(
            db._conn.execute
        ),
    )
    assert getattr(db, method)() == expected
