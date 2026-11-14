"""The embedded dispatcher's ready-probe must not call held-back work stuck.

``_KanbanDispatcher.ready_nonempty`` feeds the gateway's ``dispatcher stuck``
health counter. It receives the just-finished tick's per-board results and must
subtract the work that tick deliberately held back (per-profile cap, respawn
guard, rate-limit release) — a queue parked at
``kanban.max_in_progress_per_profile`` is the cap doing its configured job, not
a broken profile.
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace


def _isolated_home_with_profile(monkeypatch) -> str:
    test_home = tempfile.mkdtemp(prefix="kanban_stuck_hold_test_")
    for prof in ("alpha", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
        with open(os.path.join(test_home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")  # identity marker: a bare dir is not a profile
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    return test_home


def test_ready_nonempty_ignores_tasks_the_tick_held_back(monkeypatch):
    _isolated_home_with_profile(monkeypatch)
    from gateway import kanban_watchers_dispatcher as kwd
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)

    with kbc.connect_closing() as conn:
        kanban_db.create_board(slug="default", name="Test")
        ids = [kanban_db.create_task(conn, title=f"a{i}", assignee="alpha")
               for i in range(2)]

    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, 1)
    dispatcher = kwd._KanbanDispatcher(
        SimpleNamespace(DEFAULT_BOARD="default"), settings,
    )
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])

    # Unexplained ready work really is pending.
    assert dispatcher.ready_nonempty() is True
    # The same queue, with the ids the tick capped, is correctly idle.
    held = kbd.DispatchResult(
        skipped_per_profile_capped=[(tid, "alpha", 1) for tid in ids],
    )
    assert dispatcher.ready_nonempty([("default", held)]) is False
    # A board whose tick found nothing to hold keeps reporting the queue.
    assert dispatcher.ready_nonempty([("default", kbd.DispatchResult())]) is True
    # A whole-tick hold (board lock) skips the board entirely.
    locked = kbd.DispatchResult(skipped_locked=True)
    assert dispatcher.ready_nonempty([("default", locked)]) is False
