"""CLI unhide subcommand — recovery affordance for the durable hidden flag.

Hiding is a legitimate write path (plugin-owned sessions, the REST PATCH on
api_server), but a stale or wrongly-adopted pointer can hide an ordinary
user conversation with no documented way back. These tests pin the CLI's
access to the SAME store the setters use (SessionDB.set_session_hidden(False)),
not a client-local list — the exact precedent of the pin/unpin tests.
"""


class _FakeDB:
    def __init__(self, known=("20260315_092437_c9a6ff",), hidden=()):
        self.known = set(known)
        self.hidden = set(hidden)
        self.hide_calls = []
        self.list_kwargs = None

    def resolve_session_id(self, session_id):
        for k in self.known:
            if k.startswith(session_id):
                return k
        return None

    def set_session_hidden(self, session_id, hidden):
        self.hide_calls.append((session_id, hidden))
        # SessionDB.set_session_hidden contract: True only when at least one
        # row actually changed. Unhiding an already-visible session (or its
        # already-visible lineage) is a no-op that returns False.
        was_hidden = session_id in self.hidden
        if hidden:
            self.hidden.add(session_id)
        else:
            self.hidden.discard(session_id)
        return was_hidden != hidden

    def get_session_title(self, session_id):
        return "Alpha Work" if session_id in self.known else None

    def list_sessions_rich(self, **kwargs):
        self.list_kwargs = kwargs
        return []

    def close(self):
        pass


def _run(monkeypatch, capsys, argv_tail, db):
    import sys

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


def test_unhide_accepts_unique_prefix(monkeypatch, capsys):
    db = _FakeDB(hidden=("20260315_092437_c9a6ff",))
    code, out = _run(monkeypatch, capsys, ["unhide", "20260315_092437"], db)
    assert db.hide_calls == [("20260315_092437_c9a6ff", False)]
    assert "Unhidden session '20260315_092437_c9a6ff'." in out
    assert "(Alpha Work)" in out
    assert code == 0


def test_unhide_multiple_ids_one_missing(monkeypatch, capsys):
    db = _FakeDB(known=("aaa111", "bbb222"), hidden=("aaa111", "bbb222"))
    code, out = _run(monkeypatch, capsys, ["unhide", "aaa", "nope", "bbb"], db)
    assert ("aaa111", False) in db.hide_calls
    assert ("bbb222", False) in db.hide_calls
    assert "Session 'nope' not found." in out
    assert code == 1


def test_unhide_already_visible_is_idempotent_success(monkeypatch, capsys):
    """The setter's False return means 'no row changed', not 'not found'.

    resolve_session_id() proved the row exists before the setter ran, so an
    idempotent second unhide of an already-visible session must report that
    state accurately and must NOT count as a failure (exit 1) — recovery
    scripts re-running unhide would otherwise read success as failure.
    """
    db = _FakeDB()  # session exists but is NOT hidden
    code, out = _run(monkeypatch, capsys, ["unhide", "20260315_092437"], db)
    assert db.hide_calls == [("20260315_092437_c9a6ff", False)]
    assert "not found" not in out.lower()
    assert "already visible" in out
    assert code == 0


def test_unhide_mixed_hidden_and_visible_counts_exit_zero(monkeypatch, capsys):
    db = _FakeDB(known=("aaa111", "bbb222"), hidden=("aaa111",))
    code, out = _run(monkeypatch, capsys, ["unhide", "aaa", "bbb"], db)
    assert "Unhidden session 'aaa111'." in out
    assert "already visible" in out
    assert code == 0


def test_list_include_hidden_flag_reaches_db(monkeypatch, capsys):
    db = _FakeDB()
    _code, _out = _run(monkeypatch, capsys, ["list", "--include-hidden"], db)
    assert db.list_kwargs is not None
    assert db.list_kwargs["include_hidden"] is True


def test_list_default_excludes_hidden(monkeypatch, capsys):
    db = _FakeDB()
    _code, _out = _run(monkeypatch, capsys, ["list"], db)
    assert db.list_kwargs is not None
    assert db.list_kwargs["include_hidden"] is False
