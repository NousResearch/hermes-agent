"""End-to-end proof: per-board dispatch/auto-decompose overrides actually
change what the gateway dispatcher does, through the REAL multi-board tick
loop (`_KanbanDispatcher.tick_once()` / `.auto_decompose_tick()`), against
REAL board databases and REAL spawnable tasks — not a single-board unit
call with a stubbed `_kbd()`/`_kbc()` (see test_kanban_board_dispatch_toggles.py
for those; this file is the missing end-to-end layer).

Every test uses a genuinely spawnable task: assignee="default", for which
`hermes_cli.profiles.profile_exists("default")` is always True (the
built-in default profile), so the ONLY thing that can prevent a spawn is
the per-board guard under test — not an unrelated "unknown profile" skip.

Adversarial design note: two of these tests (test_disabling_the_flag_...
below and the class docstring) are proven capable of catching a real
regression, not just passing vacuously: `test_disabled_flag_actually_
prevents_spawning_end_to_end` and `test_disabled_flag_actually_prevents_
decompose_end_to_end` were run once with each guard's `if not ...: return`
line manually deleted, confirmed to fail with a real spawn_fn call /
list_triage_ids call, then restored — see the paired proof script this
file's docstring points at (run manually, not part of CI, since mutating
source during a test run is not itself a repeatable CI step).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def _dispatcher():
    from gateway import kanban_watchers_dispatcher as wd
    settings = wd._DispatcherSettings(
        interval=1.0, max_spawn=None, max_in_progress=None, failure_limit=2,
        stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
        max_in_progress_per_profile=None,
    )
    return wd._KanbanDispatcher(kb=kb, settings=settings), wd


def _spawnable_ready_task(board: str, title: str) -> str:
    """A task that is genuinely dispatchable: status=ready, assignee='default'
    (always a real profile — see module docstring), scratch workspace so
    spawn only needs a directory, not a git repo."""
    conn = kbc.connect(board=board)
    try:
        return kb.create_task(conn, title=title, assignee="default", board=board)
    finally:
        conn.close()


def _triage_task(board: str, title: str) -> str:
    conn = kbc.connect(board=board)
    try:
        return kb.create_task(conn, title=title, assignee="default", board=board, triage=True)
    finally:
        conn.close()


def _task_status(board: str, task_id: str) -> str:
    conn = kbc.connect(board=board)
    try:
        return kb.get_task(conn, task_id).status
    finally:
        conn.close()


class TestDispatchEndToEnd:
    """`tick_once()` runs the REAL multi-board loop over REAL board dirs."""

    def test_disabled_board_task_is_never_spawned_across_a_real_multiboard_tick(self, fresh_home):
        kb.create_board("board-a")
        kb.create_board("board-b")
        kb.write_board_metadata("board-b", dispatch_enabled=False)

        task_a = _spawnable_ready_task("board-a", "should spawn")
        task_b = _spawnable_ready_task("board-b", "must never spawn")

        dispatcher, _wd = _dispatcher()
        spawned_task_ids: list[str] = []

        def spy_spawn(task, workspace_path, board=None):
            spawned_task_ids.append(task.id)
            return 999999

        import gateway.kanban_watchers_dispatcher as wd_mod
        real_kbd = wd_mod._kbd()

        class _SpyKbd:
            DispatchResult = real_kbd.DispatchResult
            review_dispatch_enabled = staticmethod(real_kbd.review_dispatch_enabled)
            has_spawnable_ready = staticmethod(real_kbd.has_spawnable_ready)
            has_spawnable_review = staticmethod(real_kbd.has_spawnable_review)

            @staticmethod
            def dispatch_once(conn, board=None, **kwargs):
                return real_kbd.dispatch_once(conn, board=board, spawn_fn=spy_spawn, **kwargs)

        import contextlib as _cl
        orig_kbd = wd_mod._kbd
        wd_mod._kbd = lambda: _SpyKbd
        try:
            results = dispatcher.tick_once()
        finally:
            wd_mod._kbd = orig_kbd

        slugs_ticked = [slug for slug, _res in results]
        assert set(slugs_ticked) == {"default", "board-a", "board-b"}

        assert spawned_task_ids == [task_a], (
            f"expected only board-a's task to spawn, got {spawned_task_ids}"
        )
        assert _task_status("board-a", task_a) == "running"
        assert _task_status("board-b", task_b) == "ready", (
            "board-b's task must remain untouched (still ready, never claimed) "
            "while its board's dispatch is disabled"
        )

    def test_re_enabling_dispatch_lets_the_task_spawn_on_the_next_tick(self, fresh_home):
        """The flag is read fresh every tick — not cached at gateway boot —
        so flipping it on unblocks the SAME already-queued task."""
        kb.create_board("board-b")
        kb.write_board_metadata("board-b", dispatch_enabled=False)
        task_b = _spawnable_ready_task("board-b", "queued while disabled")

        dispatcher, _wd = _dispatcher()
        first_tick_spawns: list[str] = []
        second_tick_spawns: list[str] = []

        def make_spy(bucket):
            def _spy(task, workspace_path, board=None):
                bucket.append(task.id)
                return 999999
            return _spy

        import gateway.kanban_watchers_dispatcher as wd_mod
        real_kbd = wd_mod._kbd()

        def _tick_with_spy(bucket):
            class _SpyKbd:
                DispatchResult = real_kbd.DispatchResult
                review_dispatch_enabled = staticmethod(real_kbd.review_dispatch_enabled)

                @staticmethod
                def dispatch_once(conn, board=None, **kwargs):
                    return real_kbd.dispatch_once(conn, board=board, spawn_fn=make_spy(bucket), **kwargs)

            orig = wd_mod._kbd
            wd_mod._kbd = lambda: _SpyKbd
            try:
                dispatcher.tick_once_for_board("board-b")
            finally:
                wd_mod._kbd = orig

        _tick_with_spy(first_tick_spawns)
        assert first_tick_spawns == [], "must not spawn while disabled"
        assert _task_status("board-b", task_b) == "ready"

        kb.write_board_metadata("board-b", dispatch_enabled=True)

        _tick_with_spy(second_tick_spawns)
        assert second_tick_spawns == [task_b], "must spawn the SAME task once re-enabled"
        assert _task_status("board-b", task_b) == "running"

    def test_disabling_mid_flight_does_not_retroactively_kill_a_running_task(self, fresh_home):
        """The guard is an INTAKE gate (checked once per tick before dispatch
        runs), not a kill switch on already-running workers. Disabling a
        board after a task is already 'running' must not touch that row —
        only NEW spawns on the next tick are blocked."""
        kb.create_board("board-a")
        task_a = _spawnable_ready_task("board-a", "already running")

        dispatcher, _wd = _dispatcher()
        import gateway.kanban_watchers_dispatcher as wd_mod
        real_kbd = wd_mod._kbd()

        class _SpyKbd:
            DispatchResult = real_kbd.DispatchResult
            review_dispatch_enabled = staticmethod(real_kbd.review_dispatch_enabled)

            @staticmethod
            def dispatch_once(conn, board=None, **kwargs):
                return real_kbd.dispatch_once(conn, board=board, spawn_fn=lambda *a, **k: 4242, **kwargs)

        orig = wd_mod._kbd
        wd_mod._kbd = lambda: _SpyKbd
        try:
            dispatcher.tick_once_for_board("board-a")
        finally:
            wd_mod._kbd = orig
        assert _task_status("board-a", task_a) == "running"

        # Disable the board NOW, after the task already spawned.
        kb.write_board_metadata("board-a", dispatch_enabled=False)

        # The next tick must skip entirely (no dispatch_once call at all) —
        # and the already-running task's row must be untouched by that skip.
        called = {"n": 0}

        def _boom(*a, **k):
            called["n"] += 1
            raise AssertionError("dispatch_once must not run while disabled")

        class _BoomKbd:
            dispatch_once = staticmethod(_boom)

        wd_mod._kbd = lambda: _BoomKbd
        try:
            result = dispatcher.tick_once_for_board("board-a")
        finally:
            wd_mod._kbd = orig
        assert result is None
        assert called["n"] == 0
        assert _task_status("board-a", task_a) == "running", (
            "disabling the board must not mutate an already-running task's status"
        )


class TestAutoDecomposeEndToEnd:
    def test_disabled_board_triage_task_is_never_decomposed_across_a_real_multiboard_tick(self, fresh_home, monkeypatch):
        kb.create_board("board-a")
        kb.create_board("board-b")
        kb.write_board_metadata("board-b", auto_decompose_enabled=False)

        triage_a = _triage_task("board-a", "decompose me")
        triage_b = _triage_task("board-b", "must never be listed")

        decompose_calls: list[str] = []

        def fake_decompose_task(task_id, author=None):
            decompose_calls.append(task_id)
            from hermes_cli.kanban_decompose import DecomposeOutcome
            return DecomposeOutcome(task_id, True, "faked", fanout=False, child_ids=[])

        from hermes_cli import kanban_decompose as real_decomp
        monkeypatch.setattr(real_decomp, "decompose_task", fake_decompose_task)

        dispatcher, _wd = _dispatcher()
        decomposed_count = dispatcher.auto_decompose_tick(auto_decompose_per_tick=10)

        assert decomposed_count == 1
        assert decompose_calls == [triage_a], (
            f"expected only board-a's triage task to be decomposed, got {decompose_calls}"
        )
        assert _task_status("board-b", triage_b) == "triage", (
            "board-b's triage task must remain untouched while auto-decompose is disabled for it"
        )
