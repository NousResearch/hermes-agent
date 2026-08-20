"""Regression tests for classic CLI session browsing."""

from cli import HermesCLI
from hermes_state import SessionDB


def test_classic_cli_lists_human_sessions_across_sources(tmp_path):
    """CLI session browsing must match TUI's cross-source history policy."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("current-session", "cli")
        db.create_session("cli-session", "cli")
        db.set_session_title("cli-session", "CLI session")
        db.create_session("tui-session", "tui")
        db.set_session_title("tui-session", "TUI session")
        db.create_session("tool-session", "tool")
        db.set_session_title("tool-session", "Tool session")
        db.create_session("subagent-session", "subagent")
        db.set_session_title("subagent-session", "Subagent session")

        cli = HermesCLI.__new__(HermesCLI)
        cli._session_db = db
        cli.session_id = "current-session"

        rows = cli._list_recent_sessions(limit=10)

        assert {row["id"] for row in rows} == {"cli-session", "tui-session"}
    finally:
        db.close()
