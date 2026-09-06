"""Regression test for T0011: a duplicate, broken _tick_once_for_board
implementation used to live inline in gateway/kanban_watchers.py, referencing
never-assigned names (disabled_corrupt_boards, max_spawn, ...) and raising
NameError on every dispatcher tick. The fix deletes that duplicate and makes
the watcher call `_KanbanDispatcher.tick_once_for_board` directly.

`tests/gateway/test_kanban_watchers_mixin.py` only asserts the watcher
methods exist via hasattr() — it passed even while this bug was live, since
it never actually calls the tick logic. This test exercises the real
dispatcher against a real (empty) board DB so a future reintroduction of the
broken inline duplicate (e.g. via an upstream rebase, as happened here) would
be caught by an actual call failing, not just a missing attribute.
"""

from __future__ import annotations

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher


def _settings() -> _DispatcherSettings:
    return _DispatcherSettings(
        interval=60.0,
        max_spawn=1,
        max_in_progress=1,
        failure_limit=3,
        stale_timeout_seconds=300,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )


def test_tick_once_for_board_runs_without_nameerror():
    dispatcher = _KanbanDispatcher(kb, _settings())
    slug = kb.DEFAULT_BOARD
    kbc.connect(board=slug).close()  # ensure the board DB exists/migrated

    # Must not raise NameError (or anything else) against an empty board.
    dispatcher.tick_once_for_board(slug)


def test_tick_once_for_board_quarantine_dict_persists_across_calls():
    """The quarantine dict must be a real per-dispatcher instance attribute,
    not a bare name resolved from module/enclosing scope — calling the tick
    twice in a row must not raise even though the first call populates
    `dispatcher.disabled_corrupt_boards`."""
    dispatcher = _KanbanDispatcher(kb, _settings())
    slug = kb.DEFAULT_BOARD
    kbc.connect(board=slug).close()

    dispatcher.tick_once_for_board(slug)
    dispatcher.tick_once_for_board(slug)

    assert isinstance(dispatcher.disabled_corrupt_boards, dict)
