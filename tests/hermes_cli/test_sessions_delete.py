import sys


def test_sessions_delete_accepts_unique_id_prefix(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "20260315_092437_c9a6ff"

        def delete_session(self, session_id):
            captured["deleted"] = session_id
            return True

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "delete", "20260315_092437_c9a6", "--yes"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured == {
        "resolved_from": "20260315_092437_c9a6",
        "deleted": "20260315_092437_c9a6ff",
        "closed": True,
    }
    assert "Deleted session '20260315_092437_c9a6ff'." in output


def test_sessions_delete_reports_not_found_when_prefix_is_unknown(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def resolve_session_id(self, session_id):
            return None

        def delete_session(self, session_id):
            raise AssertionError("delete_session should not be called when resolution fails")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "delete", "missing-prefix", "--yes"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert "Session 'missing-prefix' not found." in output
