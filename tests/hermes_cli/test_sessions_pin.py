"""CLI pin / unpin / pinned subcommands (issue #52955).

Pin state is the durable "keep" flag in state.db that the Desktop sidebar
writes; these tests pin the CLI's read/write access to the SAME store
(``sessions.set_pinned`` / ``sessions.list_sessions`` with ``include_pinned``),
not a client-local list.
"""

import contextlib
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def existing_store(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    (tmp_path / "state.db").touch()


class _FakeSessions:
    def __init__(self, rows, known):
        self.rows = rows
        self.known = set(known)
        self.pin_calls = []
        self.list_query = None

    def resolve_id(self, session_id):
        for k in self.known:
            if k.startswith(session_id):
                return k
        return None

    def set_pinned(self, session_id, pinned):
        self.pin_calls.append((session_id, pinned))
        return session_id in self.known

    def get_title(self, session_id):
        return "Alpha Work" if session_id in self.known else None

    def list_sessions(self, query, window=None, **_kwargs):
        self.list_query = query
        return tuple(SimpleNamespace(
            session=SimpleNamespace(session_id=row["id"], title=row["title"], source=row["source"],
                                    message_count=row["message_count"], pinned=row["pinned"]),
            last_active=row["last_active"], preview="") for row in self.rows)


class _FakeDB:
    """The storage interface handle ``hermes sessions`` opens."""

    def __init__(self, rows=None, known=("20260315_092437_c9a6ff",)):
        self.sessions = _FakeSessions(rows or [], known)
        self.closed = False

    @property
    def pin_calls(self):
        return self.sessions.pin_calls

    def ready(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def _run(monkeypatch, capsys, argv_tail, db):
    import hermes_cli.main as main_mod
    import persistence_assembly

    def open_writer(scope, eager=False):
        assert argv_tail[0] != "pinned"
        return db

    def open_reader(target):
        assert argv_tail[0] == "pinned"
        return contextlib.nullcontext(db)

    monkeypatch.setattr(persistence_assembly, "open_persistence", open_writer)
    monkeypatch.setattr(persistence_assembly, "open_persistence_reader", open_reader)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv_tail])
    try:
        main_mod.main()
        code = 0
    except SystemExit as e:  # non-zero exits propagate through main()
        code = e.code or 0
    return code, capsys.readouterr().out

def test_pin_accepts_unique_prefix(monkeypatch, capsys):
    db = _FakeDB()
    code, out = _run(monkeypatch, capsys, ["pin", "20260315_092437"], db)
    assert db.pin_calls == [("20260315_092437_c9a6ff", True)]
    assert "Pinned session '20260315_092437_c9a6ff'." in out
    assert "(Alpha Work)" in out
    assert code == 0

def test_unpin_writes_false(monkeypatch, capsys):
    db = _FakeDB()
    _code, out = _run(monkeypatch, capsys, ["unpin", "20260315_092437_c9a6ff"], db)
    assert db.pin_calls == [("20260315_092437_c9a6ff", False)]
    assert "Unpinned session" in out

def test_pin_multiple_ids_one_missing(monkeypatch, capsys):
    db = _FakeDB(known=("aaa111", "bbb222"))
    code, out = _run(monkeypatch, capsys, ["pin", "aaa", "nope", "bbb"], db)
    assert ("aaa111", True) in db.pin_calls
    assert ("bbb222", True) in db.pin_calls
    assert "No session 'nope'" in out and "hermes sessions list" in out
    assert code == 1

def test_pinned_lists_only_pinned_rows(monkeypatch, capsys):
    rows = [
        {
            "id": "pinned_one",
            "title": "Keep Me",
            "source": "cli",
            "last_active": 1_700_000_000.0,
            "message_count": 3,
            "pinned": 1,
        },
        {
            "id": "recent_unpinned",
            "title": "Recent",
            "source": "cli",
            "last_active": 1_700_000_100.0,
            "message_count": 5,
            "pinned": 0,
        },
    ]
    db = _FakeDB(rows=rows)
    _code, out = _run(monkeypatch, capsys, ["pinned"], db)
    # include_pinned back-fill is what guarantees old pins surface
    assert db.sessions.list_query.include_pinned is True
    assert "pinned_one" in out
    assert "Keep Me" in out
    assert "recent_unpinned" not in out

def test_pinned_json_output(monkeypatch, capsys):
    rows = [
        {
            "id": "pinned_one",
            "title": "Keep Me",
            "source": "cli",
            "last_active": 1_700_000_000.0,
            "message_count": 3,
            "pinned": 1,
        },
    ]
    db = _FakeDB(rows=rows)
    _code, out = _run(monkeypatch, capsys, ["pinned", "--json"], db)
    payload = json.loads(out)
    assert payload == [
        {
            "id": "pinned_one",
            "title": "Keep Me",
            "source": "cli",
            "last_active": 1_700_000_000.0,
            "message_count": 3,
        }
    ]
