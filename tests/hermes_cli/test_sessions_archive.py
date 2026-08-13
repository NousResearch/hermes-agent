from argparse import Namespace

import hermes_state
from hermes_cli.sessions_cmd import cmd_sessions


def test_sessions_archive_dry_run_matches_unended_title(monkeypatch, capsys, tmp_path):
    db_path = tmp_path / "state.db"
    session_db_cls = hermes_state.SessionDB
    db = session_db_cls(db_path)
    db.create_session("open", source="cli")
    db.set_session_title("open", "Purple Elephant Test")
    db.close()

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: session_db_cls(db_path))

    cmd_sessions(
        Namespace(
            sessions_action="archive",
            older_than=None,
            newer_than=None,
            before=None,
            after=None,
            source=None,
            title="Purple Elephant",
            end_reason=None,
            cwd=None,
            min_messages=None,
            max_messages=None,
            model=None,
            provider=None,
            user=None,
            chat_id=None,
            chat_type=None,
            branch=None,
            min_tokens=None,
            max_tokens=None,
            min_cost=None,
            max_cost=None,
            min_tool_calls=None,
            max_tool_calls=None,
            dry_run=True,
            yes=False,
        )
    )

    output = capsys.readouterr().out
    assert "1 session(s) match" in output
    assert "open" in output
    assert "Dry run" in output
