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


def test_sessions_delete_handles_eoferror_on_confirm(monkeypatch, capsys):
    """sessions delete should not crash when stdin is closed (non-TTY)."""
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "20260315_092437_c9a6ff"

        def delete_session(self, session_id):
            raise AssertionError("delete_session should not be called when cancelled")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys, "argv",
        ["hermes", "sessions", "delete", "20260315_092437_c9a6"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))

    main_mod.main()

    output = capsys.readouterr().out
    assert "Cancelled" in output


def test_sessions_delete_accepts_multiple_session_ids(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {"resolved_from": [], "deleted": []}
    resolved = {
        "20260315_092437_c9a6": "20260315_092437_c9a6ff",
        "20260316_101500_abcd": "20260316_101500_abcd12",
    }

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"].append(session_id)
            return resolved.get(session_id)

        def delete_session(self, session_id):
            captured["deleted"].append(session_id)
            return True

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "delete",
            "20260315_092437_c9a6",
            "20260316_101500_abcd",
            "--yes",
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured == {
        "resolved_from": ["20260315_092437_c9a6", "20260316_101500_abcd"],
        "deleted": ["20260315_092437_c9a6ff", "20260316_101500_abcd12"],
        "closed": True,
    }
    assert "Deleted session '20260315_092437_c9a6ff'." in output
    assert "Deleted session '20260316_101500_abcd12'." in output
    assert "Deleted 2 session(s)." in output


def test_sessions_delete_continues_when_some_session_ids_are_not_found(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {"resolved_from": [], "deleted": []}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"].append(session_id)
            return {
                "existing-one": "20260315_092437_c9a6ff",
                "missing-one": None,
                "existing-two": "20260316_101500_abcd12",
            }[session_id]

        def delete_session(self, session_id):
            captured["deleted"].append(session_id)
            return True

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "delete",
            "existing-one",
            "missing-one",
            "existing-two",
            "--yes",
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured == {
        "resolved_from": ["existing-one", "missing-one", "existing-two"],
        "deleted": ["20260315_092437_c9a6ff", "20260316_101500_abcd12"],
        "closed": True,
    }
    assert "Deleted session '20260315_092437_c9a6ff'." in output
    assert "Session 'missing-one' not found." in output
    assert "Deleted session '20260316_101500_abcd12'." in output
    assert "Deleted 2 session(s)." in output
    assert "Failed to delete 1 session(s)." in output


def test_sessions_prune_handles_eoferror_on_confirm(monkeypatch, capsys):
    """sessions prune should not crash when stdin is closed (non-TTY)."""
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def prune_sessions(self, **kwargs):
            raise AssertionError("prune_sessions should not be called when cancelled")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys, "argv",
        ["hermes", "sessions", "prune"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))

    main_mod.main()

    output = capsys.readouterr().out
    assert "Cancelled" in output
