"""#124033: a queued /title or /new <title> must not be silently dropped
when the session DB row creation fails transiently.

Root cause: the agent-setup application point keeps `_pending_title` for
retry, but no code anywhere retries it — the status bar even displays the
unpersisted value, masking the failure until the auto-titler overwrites
the row with `title_source='llm'`.

Fix: idempotent `_apply_pending_title()` called from agent setup AND from
the top of every `chat()` turn (the retry the docstring already promised);
status bar prefers the persisted title and marks an unpersisted pending
title as `(pending)`.

RED/GREEN evidence:
- pre-fix: `_apply_pending_title` does not exist (collection error), and
  the status bar returns the raw pending value even when a DB exists (RED).
- post-fix: transient failures keep the pending title for the next turn;
  success clears it and writes the DB row; status bar shows the real
  persisted state with an explicit pending marker (GREEN).
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "session-1"
    cli_obj._pending_title = None
    cli_obj._session_db = None
    cli_obj.agent = None
    return cli_obj


def _attach_db(cli_obj, title=None):
    cli_obj._session_db = SimpleNamespace(  # type: ignore[assignment]
        get_session_title=lambda sid: title if sid == cli_obj.session_id else None,
        set_session_title=MagicMock(return_value=True),
    )
    return cli_obj._session_db


def _attach_agent(cli_obj, *, created=False, fail_then_succeed=False):
    state = {"calls": 0, "created": created}
    agent = SimpleNamespace(
        _session_db_created=created,
        _ensure_db_session=lambda: _bump(state),
    )

    def _bump(_state):
        _state["calls"] += 1
        if fail_then_succeed and _state["calls"] == 1:
            return  # first call leaves _session_db_created False (transient)
        _state["created"] = True
        agent._session_db_created = True

    # Rebind the closure after agent exists (bump references agent).
    agent._ensure_db_session = lambda: _bump(state)
    cli_obj.agent = agent
    return agent


class TestApplyPendingTitle:
    def test_transient_failure_keeps_pending_for_next_turn(self):
        cli_obj = _make_cli()
        db = _attach_db(cli_obj)
        cli_obj._pending_title = "weekly-digest"
        state = {"calls": 0, "created": False}
        agent = SimpleNamespace(_session_db_created=False, _ensure_db_session=lambda: state.update(calls=state["calls"] + 1))

        cli_obj.agent = agent
        cli_obj._apply_pending_title()

        # First attempt: row not created yet — pending must survive.
        assert cli_obj._pending_title == "weekly-digest"
        db.set_session_title.assert_not_called()

        # Next turn: row creation succeeds — pending is applied and cleared.
        agent._session_db_created = True
        cli_obj._apply_pending_title()

        assert cli_obj._pending_title is None
        db.set_session_title.assert_called_once_with("session-1", "weekly-digest")

    def test_exception_keeps_pending(self):
        cli_obj = _make_cli()
        db = _attach_db(cli_obj)
        cli_obj._pending_title = "keep-me"
        cli_obj.agent = SimpleNamespace(
            _session_db_created=False,
            _ensure_db_session=MagicMock(side_effect=RuntimeError("sqlite lock")),
        )

        cli_obj._apply_pending_title()

        assert cli_obj._pending_title == "keep-me"
        db.set_session_title.assert_not_called()

    def test_noop_without_pending_title(self):
        cli_obj = _make_cli()
        db = _attach_db(cli_obj)
        cli_obj.agent = SimpleNamespace(
            _session_db_created=False,
            _ensure_db_session=MagicMock(),
        )

        cli_obj._apply_pending_title()

        db.set_session_title.assert_not_called()
        cli_obj.agent._ensure_db_session.assert_not_called()

    def test_noop_without_session_db(self):
        cli_obj = _make_cli()
        cli_obj._pending_title = "no-db"
        cli_obj.agent = SimpleNamespace(_session_db_created=False, _ensure_db_session=MagicMock())

        cli_obj._apply_pending_title()

        assert cli_obj._pending_title == "no-db"
        cli_obj.agent._ensure_db_session.assert_not_called()


class TestStatusBarPendingMarker:
    def test_status_bar_marks_unpersisted_pending_title(self):
        cli_obj = _make_cli()
        _attach_db(cli_obj, title=None)  # DB exists but row has no title
        cli_obj._pending_title = "weekly-digest"

        title = cli_obj._get_status_bar_session_title()

        assert title == "weekly-digest (pending)"

    def test_status_bar_prefers_persisted_title_over_pending(self):
        cli_obj = _make_cli()
        _attach_db(cli_obj, title="user-profiles")
        cli_obj._pending_title = "weekly-digest"

        title = cli_obj._get_status_bar_session_title()

        assert title == "user-profiles"

    def test_status_bar_pending_without_db_shows_plain(self):
        cli_obj = _make_cli()
        cli_obj._pending_title = "weekly-digest"

        title = cli_obj._get_status_bar_session_title()

        assert title == "weekly-digest"


class TestRetryCallSites:
    def _read(self, path):
        from pathlib import Path
        return Path(__file__).parents[2].joinpath(path).read_text(encoding="utf-8")

    def test_chat_entry_invokes_apply_pending_title(self):
        src = self._read("hermes_cli/cli_chat_turn_mixin.py")
        assert "self._apply_pending_title()" in src, (
            "chat() must re-invoke _apply_pending_title() every turn so a "
            "transient DB-row failure retries instead of dropping the title"
        )

    def test_agent_setup_invokes_apply_pending_title(self):
        src = self._read("hermes_cli/cli_agent_setup_mixin.py")
        assert "self._apply_pending_title()" in src, (
            "agent setup must apply the queued title through the idempotent helper"
        )
        assert "def _apply_pending_title" in src, (
            "_apply_pending_title helper must live in cli_agent_setup_mixin"
        )
