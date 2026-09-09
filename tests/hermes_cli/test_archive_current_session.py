"""/archive from chat + `hermes sessions unarchive` (inspired by Factory Droid v0.209).

The archived flag and bulk `hermes sessions archive` predate this; these tests pin the
two new access paths: un-archiving by ID/prefix from the CLI, and archiving the session
you are currently IN (which must rotate to a fresh session BEFORE flagging the old row
so the archived transcript is complete).
"""

import sys
import types


class _FakeDB:
    def __init__(self, known=("20260909_101500_ab12cd",), message_count=3):
        self.known = set(known)
        self.message_count = message_count
        self.archive_calls = []

    def resolve_session_id(self, session_id):
        for k in self.known:
            if k.startswith(session_id):
                return k
        return None

    def get_session(self, session_id):
        if session_id in self.known:
            return {"id": session_id, "message_count": self.message_count}
        return None

    def set_session_archived(self, session_id, archived):
        self.archive_calls.append((session_id, archived))
        return session_id in self.known

    def get_session_title(self, session_id):
        return "Droid Port" if session_id in self.known else None

    def list_sessions_rich(self, **kwargs):
        return []

    def close(self):
        pass


def test_sessions_unarchive_accepts_unique_prefix(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    db = _FakeDB()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "unarchive", "20260909_101500"])
    try:
        main_mod.main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    out = capsys.readouterr().out
    assert code == 0
    assert db.archive_calls == [("20260909_101500_ab12cd", False)]
    assert "Un-archived session '20260909_101500_ab12cd'." in out


def test_cli_archive_rotates_before_flagging_old_session():
    """/archive must start the fresh session BEFORE archiving the old row (flush-then-archive),
    and must archive the OLD id, not the new one."""
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    db = _FakeDB()
    order = []

    stub = types.SimpleNamespace(
        _session_db=db,
        session_id="20260909_101500_ab12cd",
        _confirm_destructive_slash=lambda *a, **k: "once",
    )

    def _new_session(silent=False, title=None):
        order.append("rotate")
        stub.session_id = "20260909_110000_ffeedd"

    stub.new_session = _new_session
    _orig = db.set_session_archived

    def _tracked(session_id, archived):
        order.append("archive")
        return _orig(session_id, archived)

    db.set_session_archived = _tracked

    CLILoopsMixin._cmd_archive_session(stub, "/archive")
    assert order == ["rotate", "archive"]
    assert db.archive_calls == [("20260909_101500_ab12cd", True)]


def test_cli_archive_refuses_empty_session():
    """A session with no saved messages is not archived (nothing to keep) and is not rotated."""
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    db = _FakeDB(message_count=0)
    rotated = []
    stub = types.SimpleNamespace(
        _session_db=db,
        session_id="20260909_101500_ab12cd",
        _confirm_destructive_slash=lambda *a, **k: "once",
        new_session=lambda **k: rotated.append(True),
    )
    CLILoopsMixin._cmd_archive_session(stub, "/archive")
    assert db.archive_calls == []
    assert rotated == []
