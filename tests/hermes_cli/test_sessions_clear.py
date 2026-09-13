import sys
import pytest


def test_sessions_clear_accepts_unique_id_prefix(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "20260315_092437_c9a6ff"

        def get_session(self, session_id):
            return {"id": session_id, "title": "My Session", "pinned": 1}

        def clear_session_messages(self, session_id, **kwargs):
            captured["cleared"] = session_id
            captured["kwargs"] = kwargs
            return True

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "clear", "20260315_092437_c9a6", "--yes"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured["resolved_from"] == "20260315_092437_c9a6"
    assert captured["cleared"] == "20260315_092437_c9a6ff"
    assert captured["closed"] is True
    assert "Cleared transcript for session '20260315_092437_c9a6ff'." in output


def test_sessions_clear_with_last_n(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            return session_id

        def get_session(self, session_id):
            return {"id": session_id, "title": "My Session"}

        def clear_session_messages(self, session_id, keep_last_n=None, **kwargs):
            captured["session_id"] = session_id
            captured["keep_last_n"] = keep_last_n
            return True

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "clear", "sess123", "--last", "5", "--yes"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured["keep_last_n"] == 5
    assert "Cleared transcript for session 'sess123' (keeping last 5 messages)." in output


def test_sessions_clear_confirmation_abort(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            return session_id

        def get_session(self, session_id):
            return {"id": session_id, "title": "My Session"}

        def clear_session_messages(self, session_id, **kwargs):
            captured["cleared"] = True
            return True

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "clear", "sess123"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert "Cancelled." in output
    assert "cleared" not in captured
