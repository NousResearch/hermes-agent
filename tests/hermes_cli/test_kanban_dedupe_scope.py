"""``--dedupe-scope`` separates request idempotency from open-item dedup.

The two have always shared one key. Under the historical behaviour a
completed task keeps matching its key forever, so recurring automation goes
permanently silent after the first fix: it files once, a human closes the
task, and every later occurrence quietly returns the closed id — exiting 0
with a plausible task id, so the caller cannot detect the suppression.

`any` keeps that behaviour for webhook retries, where the second call really
is the same event. `open` is for alerts, where it is a new one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _set_status(conn, task_id: str, status: str) -> None:
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


# --- the shared behaviour both scopes keep ----------------------------------


@pytest.mark.parametrize("scope", ["any", "open"])
def test_open_task_always_dedupes(board, scope):
    """Neither scope should open a second ticket while one is still open."""
    first = kb.create_task(board, title="disk full", idempotency_key="k",
                           dedupe_scope=scope)
    second = kb.create_task(board, title="disk full", idempotency_key="k",
                            dedupe_scope=scope)
    assert second == first


@pytest.mark.parametrize("scope", ["any", "open"])
def test_archived_never_dedupes(board, scope):
    """Archiving has always meant "gone"; that is unchanged."""
    first = kb.create_task(board, title="disk full", idempotency_key="k",
                           dedupe_scope=scope)
    _set_status(board, first, "archived")
    second = kb.create_task(board, title="disk full", idempotency_key="k",
                            dedupe_scope=scope)
    assert second != first


# --- where they diverge: a completed task -----------------------------------


def test_any_scope_still_matches_a_done_task(board):
    """Backward compatibility. `any` is the default and must not change."""
    first = kb.create_task(board, title="import failed", idempotency_key="k",
                           dedupe_scope="any")
    _set_status(board, first, "done")
    second = kb.create_task(board, title="import failed", idempotency_key="k",
                            dedupe_scope="any")
    assert second == first


def test_default_scope_is_unchanged_behaviour(board):
    """Callers that pass no scope keep exactly what they had before."""
    first = kb.create_task(board, title="import failed", idempotency_key="k")
    _set_status(board, first, "done")
    second = kb.create_task(board, title="import failed", idempotency_key="k")
    assert second == first


def test_open_scope_files_again_after_a_fix(board):
    """The bug this flag exists for: a regression must reach the board."""
    first = kb.create_task(board, title="import failed", idempotency_key="k",
                           dedupe_scope="open")
    _set_status(board, first, "done")
    second = kb.create_task(board, title="import failed", idempotency_key="k",
                            dedupe_scope="open")
    assert second != first, "a completed task swallowed a fresh occurrence"


def test_open_scope_dedupes_against_the_newest_open_task(board):
    """After a regression files a second card, that card dedupes in turn.

    Otherwise the fix would trade silence for 144 cards a day.
    """
    first = kb.create_task(board, title="import failed", idempotency_key="k",
                           dedupe_scope="open")
    _set_status(board, first, "done")
    second = kb.create_task(board, title="import failed", idempotency_key="k",
                            dedupe_scope="open")
    third = kb.create_task(board, title="import failed", idempotency_key="k",
                           dedupe_scope="open")
    assert third == second != first


@pytest.mark.parametrize("status", ["blocked", "running", "review", "todo", "triage"])
def test_open_scope_treats_in_flight_states_as_open(board, status):
    """Anything a human or worker is still holding counts as open."""
    first = kb.create_task(board, title="import failed", idempotency_key="k",
                           dedupe_scope="open")
    _set_status(board, first, status)
    second = kb.create_task(board, title="import failed", idempotency_key="k",
                            dedupe_scope="open")
    assert second == first


# --- guardrails --------------------------------------------------------------


def test_unknown_scope_is_rejected(board):
    with pytest.raises(ValueError, match="dedupe_scope"):
        kb.create_task(board, title="x", idempotency_key="k", dedupe_scope="sometimes")


def test_scope_is_ignored_without_a_key(board):
    """No key means no dedup at all, whatever the scope says."""
    a = kb.create_task(board, title="x", dedupe_scope="open")
    b = kb.create_task(board, title="x", dedupe_scope="open")
    assert a != b
