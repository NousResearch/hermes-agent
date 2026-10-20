"""Regression tests for /yield: an interactive CLI hands its cross-surface
active-session slot to another surface without ending the session (#124073).

The issue: a chat resumed into ``hermes chat --resume`` fenced every other
surface out of the session for the life of the CLI process; the only exit was
quitting the CLI (losing the chat for the surface that created it).

Regression tests for the PR: https://github.com/NousResearch/hermes-agent/pull/124112
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cli import HermesCLI
from hermes_cli.active_sessions import active_session_registry_snapshot


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A HermesCLI whose lease registry lives in a temp home and whose REPL app/chat are stubs."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cli = HermesCLI()
    cli._app = MagicMock()
    cli.chat = MagicMock()
    yield cli


def _live_entry(snapshot, session_id):
    return [e for e in snapshot if str(e.get("session_id") or "") == session_id]


def test_yield_drops_lease_without_ending_session(cli):
    assert cli._claim_active_session()
    lease = cli._active_session_lease
    assert lease is not None and not lease.released
    assert _live_entry(active_session_registry_snapshot(), cli.session_id)

    assert cli._handle_yield_command() is True
    assert cli._active_session_lease is None
    assert lease.released
    assert cli._yield_active_session_pending is True
    # The session itself survives: the same stored id, and no registry entry.
    assert cli.session_id
    assert active_session_registry_snapshot() == []


def test_foreign_surface_claims_after_yield_then_next_input_refuses(cli, monkeypatch):
    assert cli._claim_active_session()
    held_id = cli.session_id
    cli._handle_yield_command()

    # Another surface (gateway lease for the same stored session) takes the freed slot.
    lease, message = _acquire_for_foreign_surface(held_id)
    assert lease is not None and message is None

    # The CLI's next submitted input re-claims BEFORE the turn and is fenced out.
    cli._yield_active_session_pending = True
    cli._tui_process_one_input("hello")
    assert cli.chat.called is False
    assert cli._yield_active_session_pending is False
    assert cli._active_session_lease is None


def test_reclaim_succeeds_when_slot_still_free(cli):
    assert cli._claim_active_session()
    cli._handle_yield_command()
    assert cli._yield_active_session_pending is True

    cli._tui_process_one_input("still mine")
    assert cli.chat.called is True
    assert cli._active_session_lease is not None
    assert not cli._active_session_lease.released
    assert _live_entry(active_session_registry_snapshot(), cli.session_id)


def test_foreign_surface_takes_slot_while_cli_idle_between_turns(cli, monkeypatch):
    assert cli._claim_active_session()
    held_id = cli.session_id
    cli._handle_yield_command()

    lease, message = _acquire_for_foreign_surface(held_id)
    assert lease is not None and message is None


def _acquire_for_foreign_surface(session_id):
    from hermes_cli.active_sessions import try_acquire_active_session
    return try_acquire_active_session(
        session_id=session_id, surface="gateway:telegram", config=None,
        metadata={"live_session_id": "runtime-1"},
    )
