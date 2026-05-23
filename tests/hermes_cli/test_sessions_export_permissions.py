import json
import os
import stat
import sys


class _FakeSessionDB:
    def __init__(self):
        self.closed = False

    def export_all(self, source=None):
        assert source is None
        return [
            {
                "id": "sess-1",
                "source": "cli",
                "messages": [{"role": "user", "content": "private prompt"}],
            }
        ]

    def export_session(self, session_id):
        assert session_id == "resolved-sess-1"
        return {
            "id": "resolved-sess-1",
            "source": "cli",
            "messages": [{"role": "assistant", "content": "private reply"}],
        }

    def resolve_session_id(self, session_id):
        assert session_id == "sess-1"
        return "resolved-sess-1"

    def close(self):
        self.closed = True


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_sessions_export_creates_jsonl_0600(tmp_path, monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    db = _FakeSessionDB()
    output = tmp_path / "exports" / "sessions.jsonl"
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", str(output)])

    old_umask = os.umask(0)
    try:
        main_mod.main()
    finally:
        os.umask(old_umask)

    assert db.closed is True
    assert "Exported 1 sessions" in capsys.readouterr().out
    assert _mode(output.parent) == 0o700
    assert _mode(output) == 0o600
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["id"] == "sess-1"


def test_sessions_export_single_session_creates_jsonl_0600(tmp_path, monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    db = _FakeSessionDB()
    output = tmp_path / "session.jsonl"
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", str(output), "--session-id", "sess-1"],
    )

    old_umask = os.umask(0)
    try:
        main_mod.main()
    finally:
        os.umask(old_umask)

    assert db.closed is True
    assert "Exported 1 session" in capsys.readouterr().out
    assert _mode(output) == 0o600
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["id"] == "resolved-sess-1"


def test_sessions_export_stdout_does_not_create_file(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    db = _FakeSessionDB()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", "-"])

    main_mod.main()

    assert db.closed is True
    output = capsys.readouterr().out
    assert '"id": "sess-1"' in output
