"""CLI pin / unpin / pinned subcommands (issue #52955).

Pin state is the durable "keep" flag in state.db that the Desktop sidebar
writes; these tests pin the CLI's read/write access to the SAME store
(SessionDB.set_session_pinned / list_sessions_rich(include_pinned=True)),
not a client-local list.
"""

import json
import sys


class _FakeDB:
    def __init__(self, rows=None, known=("20260315_092437_c9a6ff",)):
        self.rows = rows or []
        self.known = set(known)
        self.pin_calls = []
        self.list_kwargs = None
        self.closed = False

    def resolve_session_id(self, session_id):
        for k in self.known:
            if k.startswith(session_id):
                return k
        return None

    def set_session_pinned(self, session_id, pinned):
        self.pin_calls.append((session_id, pinned))
        return session_id in self.known

    def get_session_title(self, session_id):
        return "Alpha Work" if session_id in self.known else None

    def list_sessions_rich(self, **kwargs):
        self.list_kwargs = kwargs
        return self.rows

    def close(self):
        self.closed = True


def _run(monkeypatch, capsys, argv_tail, db):
    import hermes_cli.main as main_mod
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
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
    assert "Session 'nope' not found." in out
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
    assert db.list_kwargs["include_pinned"] is True
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


def test_pinned_empty_hint(monkeypatch, capsys):
    db = _FakeDB(rows=[])
    _code, out = _run(monkeypatch, capsys, ["pinned"], db)
    assert "No pinned sessions" in out
    assert "hermes sessions pin" in out
