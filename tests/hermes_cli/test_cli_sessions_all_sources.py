"""Regression for #44964 / #151584: the classic CLI can widen its session listing.

The shared listing policy stays source-scoped by design, so a bare ``/resume`` or ``/sessions``
must keep listing this CLI's own conversations only. What was missing is the discoverable opt-in:
a conversation started in the Desktop app (``gui``), the Ink TUI (``local`` + ``created_source``)
or a gateway lane was unreachable from the in-chat picker unless the user already knew its id or
title — while ``hermes sessions list --source gui`` and the Desktop sidebar listed it fine.

These tests pin the opt-in (``/sessions all`` and its ``/resume all`` spelling), the source labels
that make a widened row identifiable, and the invariant that a number shown next to a widened row
resumes that row rather than a CLI-only row that happens to sit at the same index.
"""

from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI
from hermes_cli.gateway_session_store import SurfaceLabelReader
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session


# (session id, stored source, created surface) — the newest row sorts first in the listing.
ROWS = [
    ("old-cli", "cli", "cli"),
    ("desk-gui", "gui", "gui"),
    ("tui-chat", "local", "tui"),
    ("tg-chat", "telegram", "telegram"),
    ("kanban-worker", "kanban", "kanban"),
    ("tool-run", "tool", "tool"),
    ("current-chat", "cli", "cli"),
]


@pytest.fixture()
def listing_cli(tmp_path, monkeypatch):
    """A CLI shell wired to a real store, so the listing SQL actually runs."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        for session_id, source, _created in ROWS:
            seed_session(store, session_id, source, title=f"title {session_id}")
        raw_holder(store)._execute_write(lambda conn: conn.executemany(
            "UPDATE sessions SET created_source=?, started_at=?, message_count=1 WHERE id=?",
            [(created, 1_000_000 + index, session_id)
             for index, (session_id, _source, created) in enumerate(ROWS)],
        ))
        cli = HermesCLI.__new__(HermesCLI)
        cli._session_db = SurfaceLabelReader(store)
        cli.session_id = "current-chat"
        cli._pending_resume_sessions = None
        yield cli


def _make_mock_cli():
    """The existing /resume test shape: a MagicMock store, minimal display state."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "current_session"
    cli._resumed = False
    cli._pending_title = None
    cli.conversation_history = []
    cli.agent = None
    cli._session_db = MagicMock()
    cli._pending_resume_sessions = None
    cli.resume_display = "minimal"
    return cli


def test_bare_sessions_stays_cli_scoped(listing_cli, capsys):
    """The default listing must not start showing other interfaces' conversations."""
    listing_cli._handle_sessions_command("/sessions")
    out = capsys.readouterr().out

    assert "old-cli" in out
    for hidden in ("desk-gui", "tui-chat", "tg-chat"):
        assert hidden not in out


@pytest.mark.parametrize("command", ["/sessions all", "/sessions --all"])
def test_sessions_all_lists_other_interfaces(listing_cli, capsys, command):
    listing_cli._handle_sessions_command(command)
    out = capsys.readouterr().out

    for session_id in ("old-cli", "desk-gui", "tui-chat", "tg-chat"):
        assert session_id in out
    # Internal runs stay hidden and the current chat is not offered for resume.
    for session_id in ("kanban-worker", "tool-run", "current-chat"):
        assert session_id not in out
    # A widened table can mix interfaces, so rows carry their origin.
    assert "[gui]" in out


def test_sessions_all_numbered_selection_resumes_the_widened_row(listing_cli):
    """A number printed beside a widened row must resolve to that row."""
    listing_cli._handle_sessions_command("/sessions all")
    ids = [row["id"] for row in listing_cli._last_session_listing]
    assert "desk-gui" in ids

    resolved = listing_cli._resolve_resume_target(str(ids.index("desk-gui") + 1))

    assert resolved is not None
    assert resolved[0] == "desk-gui"


def test_resume_all_is_a_listing_not_a_title(listing_cli, capsys):
    """`/resume all` must widen the list instead of looking for a session named "all"."""
    listing_cli._handle_resume_command("/resume all")
    out = capsys.readouterr().out

    assert "desk-gui" in out
    # The widened table is armed for a bare-number reply on the next line.
    assert any(row["id"] == "desk-gui" for row in listing_cli._pending_resume_sessions)


def test_sessions_search_is_scoped_unless_all_is_given(listing_cli, capsys):
    """`search` obeys the same scope rule as the plain list: widen explicitly."""
    listing_cli._handle_sessions_command("/sessions search old")
    scoped = capsys.readouterr().out

    assert "old-cli" in scoped
    assert "desk-gui" not in scoped

    listing_cli._handle_sessions_command("/sessions all search desk")
    widened = capsys.readouterr().out

    assert "desk-gui" in widened
    assert "old-cli" not in widened


def test_sessions_all_target_resumes_that_desktop_session():
    """`/sessions all <id>` keeps delegating to /resume, now for a non-CLI session."""
    cli = _make_mock_cli()
    desktop_id = "20261009_115229_148971"
    cli._session_db.get_session.return_value = {"id": desktop_id, "title": "Desk"}
    cli._session_db.get_resume_conversations.return_value = (
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "hello"}],
    )
    cli._session_db.resolve_resume_session_id.return_value = desktop_id

    with (
        patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=desktop_id),
        patch("cli._cprint"),
    ):
        cli._handle_sessions_command(f"/sessions all {desktop_id}")

    assert cli.session_id == desktop_id
