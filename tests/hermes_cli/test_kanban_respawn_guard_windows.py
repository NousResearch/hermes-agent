"""Configurable ``check_respawn_guard`` windows.

The two guard windows — "a completed run this recent is proof the work is
done" and "a GitHub PR URL this recent means a PR is already open" — used to
be fixed at 1 h / 24 h. Operators whose cards legitimately need more work on a
still-open PR cannot wait the fix out; the windows are now read from
``kanban.respawn_guard_success_window_seconds`` /
``kanban.respawn_guard_pr_window_seconds``.

These are behavior contracts, not snapshots:

* with no config the guard must behave exactly as before (defaults preserved);
* a configured window must move the retain/release boundary by exactly that
  much, checked with timestamps relative to ``now``;
* an unusable value (text / negative / zero / null) must fall back to the
  default rather than disabling the guard or raising;
* the guard's two escape hatches — the re-queue exemption on
  ``recent_success`` and the cross-profile handoff exemption on
  ``active_pr`` — must survive an explicit window.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _config(monkeypatch: pytest.MonkeyPatch, kanban_cfg: dict) -> None:
    """Point the config loader at ``{"kanban": kanban_cfg}`` for this test."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config_readonly", lambda: {"kanban": kanban_cfg})


def _completed_run(conn, task_id: str, *, ended_at: int) -> None:
    """A completed run whose ``ended_at`` the test controls."""
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'dev', 'completed', 'completed', ?, ?)",
            (task_id, ended_at - 60, ended_at),
        )


def _pr_comment(conn, task_id: str, *, created_at: int) -> None:
    """A comment carrying a PR URL, aged by ``created_at``."""
    kb.add_comment(conn, task_id, author="dev",
                   body="Opened https://github.com/example/repo/pull/44 for review.")
    with kb.write_txn(conn):
        conn.execute("UPDATE task_comments SET created_at = ? WHERE task_id = ?",
                     (created_at, task_id))


# ---------------------------------------------------------------------------
# Default preserved when nothing is configured
# ---------------------------------------------------------------------------


def test_windows_default_to_the_shipped_values_when_config_is_silent(
    kanban_home, monkeypatch,
) -> None:
    """No keys in config → today's 1 h / 24 h windows, unchanged."""
    _config(monkeypatch, {})

    assert kbd.resolve_respawn_guard_windows() == (3600, 86400)


def test_recent_success_uses_the_default_window_when_config_is_silent(
    kanban_home, monkeypatch,
) -> None:
    """A completion 2 h old is outside the 1 h default → card releases."""
    _config(monkeypatch, {})
    now = int(time.time())

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="done 2h ago", assignee="dev")
        _completed_run(conn, tid, ended_at=now - 2 * 3600)

        assert kbd.check_respawn_guard(conn, tid) is None
        # And 30 minutes in it is still recent proof.
        _completed_run(conn, tid, ended_at=now - 30 * 60)
        conn.execute("DELETE FROM task_runs WHERE ended_at = ?", (now - 2 * 3600,))
        assert kbd.check_respawn_guard(conn, tid) == "recent_success"


def test_active_pr_uses_the_default_window_when_config_is_silent(
    kanban_home, monkeypatch,
) -> None:
    """A PR comment older than 24 h no longer holds the card."""
    _config(monkeypatch, {})
    now = int(time.time())

    with kbc.connect() as conn:
        old = kb.create_task(conn, title="pr 25h ago", assignee="dev")
        _pr_comment(conn, old, created_at=now - 25 * 3600)
        fresh = kb.create_task(conn, title="pr 1h ago", assignee="dev")
        _pr_comment(conn, fresh, created_at=now - 3600)

        assert kbd.check_respawn_guard(conn, old) is None
        assert kbd.check_respawn_guard(conn, fresh) == "active_pr"


# ---------------------------------------------------------------------------
# A configured window moves the boundary by exactly that much
# ---------------------------------------------------------------------------


def test_configured_success_window_moves_the_retain_boundary(
    kanban_home, monkeypatch,
) -> None:
    """The resolved window moves the retain boundary by exactly that much.

    A 15 min window releases a 20 min old completion and holds a 5 min old one;
    the default 1 h window would have held both.
    """
    _config(monkeypatch, {"respawn_guard_success_window_seconds": 900})
    success_window, _pr_window = kbd.resolve_respawn_guard_windows()
    assert success_window == 900
    now = int(time.time())

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="configurable", assignee="dev")
        _completed_run(conn, tid, ended_at=now - 20 * 60)
        assert kbd.check_respawn_guard(conn, tid, success_window=success_window) is None

        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET ended_at = ? WHERE task_id = ?",
                         (now - 300, tid))
        assert kbd.check_respawn_guard(
            conn, tid, success_window=success_window
        ) == "recent_success"

        # That same 20 min old completion is still "recent" to the default
        # window — proving the config value, not the clock, moved the boundary.
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET ended_at = ? WHERE task_id = ?",
                         (now - 20 * 60, tid))
        assert kbd.check_respawn_guard(
            conn, tid, success_window=kbd.RESPAWN_GUARD_SUCCESS_WINDOW_DEFAULT
        ) == "recent_success"


def test_configured_pr_window_moves_the_retain_boundary(
    kanban_home, monkeypatch,
) -> None:
    """With a 3 h PR window, a 4 h old PR comment releases and a 1 h holds."""
    _config(monkeypatch, {"respawn_guard_pr_window_seconds": 3 * 3600})
    _success_window, pr_window = kbd.resolve_respawn_guard_windows()
    assert pr_window == 3 * 3600
    now = int(time.time())

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="long open pr", assignee="dev")
        _pr_comment(conn, tid, created_at=now - 4 * 3600)
        assert kbd.check_respawn_guard(conn, tid, pr_window=pr_window) is None

        with kb.write_txn(conn):
            conn.execute("UPDATE task_comments SET created_at = ? WHERE task_id = ?",
                         (now - 3600, tid))
        assert kbd.check_respawn_guard(conn, tid, pr_window=pr_window) == "active_pr"

        # The 4 h comment again: inside the 24 h default, so only the
        # configured 3 h window releases it — a genuinely different verdict.
        with kb.write_txn(conn):
            conn.execute("UPDATE task_comments SET created_at = ? WHERE task_id = ?",
                         (now - 4 * 3600, tid))
        assert kbd.check_respawn_guard(
            conn, tid, pr_window=kbd.RESPAWN_GUARD_PR_WINDOW_DEFAULT
        ) == "active_pr"


def test_configured_windows_thread_through_dispatch_once(
    kanban_home, monkeypatch, all_assignees_spawnable,
) -> None:
    """The dispatch boundary resolves config and passes it to the guard.

    A ready card holding a 4 h old PR comment is guarded by default and
    spawns once ``respawn_guard_pr_window_seconds`` drops below 4 h.
    """
    _config(monkeypatch, {})
    now = int(time.time())

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="threaded", assignee="dev")
        _pr_comment(conn, tid, created_at=now - 4 * 3600)

        default = kbd.dispatch_once(conn, dry_run=True)
        assert dict(default.respawn_guarded).get(tid) == "active_pr"

        _config(monkeypatch, {"respawn_guard_pr_window_seconds": 3 * 3600})
        shortened = kbd.dispatch_once(conn, dry_run=True)
        assert dict(shortened.respawn_guarded).get(tid) is None
        assert tid in [s[0] for s in shortened.spawned]


# ---------------------------------------------------------------------------
# Unusable values fall back to the default, never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["soon", -1, 0, None, "", []])
def test_unusable_window_falls_back_to_the_default(kanban_home, monkeypatch, bad) -> None:
    """Text / negative / zero / null are not legal windows — use the default."""
    _config(
        monkeypatch,
        {
            "respawn_guard_success_window_seconds": bad,
            "respawn_guard_pr_window_seconds": bad,
        },
    )

    assert kbd.resolve_respawn_guard_windows() == (
        kbd.RESPAWN_GUARD_SUCCESS_WINDOW_DEFAULT,
        kbd.RESPAWN_GUARD_PR_WINDOW_DEFAULT,
    )


def test_unusable_window_never_releases_a_fresh_completion(
    kanban_home, monkeypatch,
) -> None:
    """A 0/negative success window must not turn into "guard everything off"."""
    _config(monkeypatch, {"respawn_guard_success_window_seconds": 0})

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="still recent", assignee="dev")
        _completed_run(conn, tid, ended_at=int(time.time()) - 60)
        assert kbd.check_respawn_guard(conn, tid) == "recent_success"


def test_explicit_parameter_wins_over_config(kanban_home, monkeypatch) -> None:
    """A caller-passed window is authoritative; config is only the fallback."""
    _config(monkeypatch, {"respawn_guard_pr_window_seconds": 86400})

    assert kbd.resolve_respawn_guard_windows(pr_window=120) == (3600, 120)


# ---------------------------------------------------------------------------
# The gateway boot boundary resolves and threads the same windows
# ---------------------------------------------------------------------------


def test_gateway_dispatcher_settings_resolve_the_windows(kanban_home) -> None:
    """The embedded dispatcher reads both windows once, at boot."""
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings

    configured = _resolve_dispatcher_settings(
        {"respawn_guard_success_window_seconds": 900,
         "respawn_guard_pr_window_seconds": 10800},
        kb,
    )
    assert (
        configured.respawn_guard_success_window_seconds,
        configured.respawn_guard_pr_window_seconds,
    ) == (900, 10800)

    # Unusable values fall back to the shipped defaults, never to "no guard".
    unset = _resolve_dispatcher_settings({}, kb)
    assert (
        unset.respawn_guard_success_window_seconds,
        unset.respawn_guard_pr_window_seconds,
    ) == (
        kbd.RESPAWN_GUARD_SUCCESS_WINDOW_DEFAULT,
        kbd.RESPAWN_GUARD_PR_WINDOW_DEFAULT,
    )
    garbage = _resolve_dispatcher_settings(
        {"respawn_guard_success_window_seconds": "later",
         "respawn_guard_pr_window_seconds": -1},
        kb,
    )
    assert (
        garbage.respawn_guard_success_window_seconds,
        garbage.respawn_guard_pr_window_seconds,
    ) == (
        kbd.RESPAWN_GUARD_SUCCESS_WINDOW_DEFAULT,
        kbd.RESPAWN_GUARD_PR_WINDOW_DEFAULT,
    )


def test_gateway_tick_passes_resolved_windows_to_dispatch_once(
    kanban_home, monkeypatch,
) -> None:
    """``tick_once_for_board`` forwards the boot-resolved windows verbatim."""
    from gateway.kanban_watchers_dispatcher import (
        _KanbanDispatcher,
        _resolve_dispatcher_settings,
    )

    captured: dict = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kbd.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)

    settings = _resolve_dispatcher_settings(
        {"respawn_guard_pr_window_seconds": 10800}, kb,
    )
    dispatcher = _KanbanDispatcher(kb, settings)
    dispatcher.tick_once_for_board(kb.DEFAULT_BOARD)

    assert captured.get("respawn_guard_pr_window_seconds") == 10800
    assert captured.get("respawn_guard_success_window_seconds") == 3600


# ---------------------------------------------------------------------------
# The two escape hatches survive an explicit window
# ---------------------------------------------------------------------------


def test_recent_success_requeue_exemption_survives_a_configured_window(
    kanban_home, monkeypatch,
) -> None:
    """An explicit re-queue after the completion still means "run it again"."""
    _config(monkeypatch, {"respawn_guard_success_window_seconds": 30 * 24 * 3600})

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="requeued", assignee="dev")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.complete_task(conn, tid, summary="done") is True
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))

        # Held by the (now month-long) window…
        assert kbd.check_respawn_guard(conn, tid) == "recent_success"
        # …until a deliberate block → unblock re-queues the card.
        assert kb.block_task(conn, tid, reason="operator wants another pass") is True
        assert kb.unblock_task(conn, tid) is True
        assert kbd.check_respawn_guard(conn, tid) is None


def test_active_pr_handoff_exemption_survives_a_configured_window(
    kanban_home, monkeypatch,
) -> None:
    """A handoff to a DIFFERENT profile lifts ``active_pr`` at any window size."""
    _config(monkeypatch, {"respawn_guard_pr_window_seconds": 30 * 24 * 3600})
    now = int(time.time())

    with kbc.connect() as conn:
        dev_id = kb.create_task(conn, title="dev own pr", assignee="dev")
        _pr_comment(conn, dev_id, created_at=now - 60)

        closer_id = kb.create_task(conn, title="closer recovery", assignee="dev")
        _pr_comment(conn, closer_id, created_at=now - 3600)
        assert kb.assign_task(conn, closer_id, "closer") is True

        assert kbd.check_respawn_guard(conn, dev_id) == "active_pr"
        assert kbd.check_respawn_guard(conn, closer_id) is None
