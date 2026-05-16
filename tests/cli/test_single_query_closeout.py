import types


def test_finalize_session_row_marks_agent_session_ended_and_syncs_id():
    from cli import HermesCLI

    calls = []

    class _DB:
        def end_session(self, session_id, end_reason):
            calls.append((session_id, end_reason))

    cli = object.__new__(HermesCLI)
    cli.session_id = "parent-session"
    cli.agent = types.SimpleNamespace(session_id="continuation-session")
    cli._session_db = _DB()

    cli._finalize_session_row("cli_close")

    assert calls == [("continuation-session", "cli_close")]
    assert cli.session_id == "continuation-session"


def test_finalize_session_row_noops_without_session_db():
    from cli import HermesCLI

    cli = object.__new__(HermesCLI)
    cli.session_id = "session-key"
    cli.agent = types.SimpleNamespace(session_id="session-key")
    cli._session_db = None

    cli._finalize_session_row("cli_close")
