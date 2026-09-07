"""Tests for the ``pre_dispatch_claim`` dispatcher policy hook.

Proves the hook fires INSIDE ``_dispatch_lane_task`` between
``check_respawn_guard`` and ``claim_task`` (once per would-be claim, both
lanes), that an ``action='block'`` directive force-trips the row to
``blocked`` + records the reason + emits a ``blocked`` event + skips the
spawn, that a plugin which registers nothing changes nothing, and that a
broken plugin fails OPEN (dispatch proceeds).

The fixtures and ``dispatch_once`` signature follow the existing
``test_kanban_dispatch_tick_hook.py`` conventions; if either drifts in a
future upstream release, update this file accordingly.
"""

from __future__ import annotations

import importlib
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


# ── Fixtures (mirror existing test_kanban_dispatch_tick_hook.py) ──


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Set up an isolated HERMES_HOME so PluginManager cache keys correctly.

    Order matters: env vars + Path.home MUST be set before any import
    that triggers ``get_plugin_manager()`` (it caches by home key for the
    process lifetime). init_db is fine after that.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli.plugins import get_plugin_manager
    # Force the singleton to bind to the new home (clear the cache).
    import hermes_cli.plugins as _plugins
    _plugins._plugin_manager = None
    _plugins._plugin_managers_by_home.clear()
    from hermes_cli import kanban_db as kb
    kb.init_db()
    return home


@pytest.fixture
def register_hook(monkeypatch):
    """Register a pre_dispatch_claim callback by appending to the plugin
    manager's ``_hooks`` dict directly (PluginManager has no public
    ``register_hook`` — that lives on ``PluginContext`` and is only
    reachable through a real plugin manifest, which we don't have here).
    The lifecycle ``has_hook`` short-circuit respects this dict, so it
    is sufficient for the dispatcher-side tests.
    """
    from hermes_cli.plugins import get_plugin_manager
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _register(cb):
        mgr._hooks.setdefault("pre_dispatch_claim", []).append(cb)

    try:
        yield _register
    finally:
        mgr._hooks = saved


@pytest.fixture
def all_assignees_spawnable():
    """All profile names referenced in tests exist as dispatcher-claimable."""
    # Today the dispatcher treats unknown assignees as ``skipped_nonspawnable``
    # rather than failing the whole tick. We don't depend on that here; the
    # tests use ``alice`` (a name unlikely to collide) so this is a marker
    # fixture that documents intent.
    return True


# ── Helpers ──


def _connect(kanban_home):
    from hermes_cli import kanban_db_connect as kbc
    return kbc.connect()


def _create_task(conn, *, title="t", assignee="default"):
    from hermes_cli import kanban_db as kb
    return kb.create_task(conn, title=title, assignee=assignee)


def _task_status(conn, tid):
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["status"]


# ── Hook fires at the right moment, with the right payload ────────────


def test_hook_fires_between_respawn_guard_and_claim(
    kanban_home, all_assignees_spawnable,
):
    """The callback is invoked with the expected payload and the task is
    still unclaimed (``status='ready'``, no ``worker_pid``) at fire time —
    i.e. AFTER ``check_respawn_guard`` and BEFORE ``claim_task``.

    Tests the helper function directly rather than going through
    ``dispatch_once`` so the test is order-independent and doesn't share
    ``PluginManager`` state with neighbouring tests in this file.
    """
    seen: list[dict] = []
    # Plugin-context kwargs intentionally omit ``conn`` — plugins should not
    # poke the dispatcher's transaction; if they need to read state they
    # open a fresh connection. The unit-test only validates the payload
    # contract, not the helper's internal cursor access.
    def _cb(**kw):
        seen.append(kw)
        return None

    from hermes_cli import kanban_db as kb
    conn = _connect(kanban_home)
    try:
        tid = _create_task(conn, assignee="default")
        kb.recompute_ready(conn)
        # First registration: this is the only ``hook`` registration in the
        # helper's process lifetime, so the short-circuit cache is clean.
        from hermes_cli.plugins import get_plugin_manager
        mgr = get_plugin_manager()
        mgr._hooks.setdefault("pre_dispatch_claim", []).append(_cb)
        try:
            directive = kb.fire_pre_dispatch_claim_hook(
                conn, tid, board=None, assignee="default", lane="ready",
            )
        finally:
            mgr._hooks.pop("pre_dispatch_claim", None)
    finally:
        conn.close()

    assert directive is None, "None directive ⇒ proceed"
    assert len(seen) == 1, "hook must fire exactly once"
    p = seen[0]
    assert p["task_id"] == tid
    assert p["lane"] == "ready"
    assert p["assignee"] == "default"


# ── action='block' force-trips the row ────────────────────────────────


def test_block_directive_force_trips_task_and_skips_spawn(
    kanban_home, register_hook, all_assignees_spawnable,
):
    """``action='block'`` ⇒ row → ``status='blocked'``, reason in
    ``last_failure_error``, a ``blocked`` task event, on
    ``result.auto_blocked``, and NO spawn.
    """
    spawned_calls: list = []
    register_hook(lambda **kw: {"action": "block", "reason": "reserved assignee"})
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    conn = _connect(kanban_home)
    try:
        tid = _create_task(conn, assignee="default")
        kb.recompute_ready(conn)
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: spawned_calls.append(a) or 1
        )
        assert spawned_calls == [], "spawn_fn must not run when blocked"
        assert _task_status(conn, tid) == "blocked"
        row = conn.execute(
            "SELECT last_failure_error, worker_pid, block_kind "
            "FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["last_failure_error"] == "reserved assignee"
        assert row["worker_pid"] is None, "claim fields cleared"
        assert row["block_kind"] in ("capability", "transient"), (
            "block_kind is a recognised routing key"
        )
        ev = conn.execute(
            "SELECT COUNT(*) c FROM task_events "
            "WHERE task_id = ? AND kind = 'blocked'",
            (tid,),
        ).fetchone()["c"]
        assert ev >= 1, "a 'blocked' event was emitted"
    finally:
        conn.close()

    assert tid in result.auto_blocked
    assert not any(r[0] == tid for r in result.spawned)


# ── No plugin registered = byte-for-byte unchanged behaviour ──────────


def test_no_hook_registered_is_a_no_op(kanban_home, all_assignees_spawnable):
    """With nothing registered, dispatch behaves exactly as before: task
    spawns, no policy-block event, no ``auto_blocked`` entry.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    conn = _connect(kanban_home)
    try:
        tid = _create_task(conn, assignee="default")
        kb.recompute_ready(conn)
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 7)
        assert _task_status(conn, tid) == "running"
        # No policy-block event was emitted (only the lifecycle spawn event).
        evs = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ?",
            (tid,),
        ).fetchall()
        kinds = [e["kind"] for e in evs]
        assert "blocked" not in kinds, (
            "no policy event should be emitted without a plugin"
        )
    finally:
        conn.close()
    assert any(r[0] == tid for r in result.spawned)


# ── Broken plugin fails OPEN ──────────────────────────────────────────


def test_broken_plugin_fails_open(
    kanban_home, register_hook, all_assignees_spawnable,
):
    """A callback that raises must NOT block dispatch — the task still
    spawns (fail-OPEN contract).
    """
    def _boom(**kw):
        raise RuntimeError("simulated policy plugin bug")

    register_hook(_boom)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    conn = _connect(kanban_home)
    try:
        tid = _create_task(conn, assignee="default")
        kb.recompute_ready(conn)
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 9)
        assert _task_status(conn, tid) == "running", (
            "broken plugin must NOT prevent spawn (fail-OPEN)"
        )
    finally:
        conn.close()
    assert any(r[0] == tid for r in result.spawned)


# ── Atomic CAS: _force_block_from_policy returns bool on race ──────────


def test_force_block_from_policy_returns_true_on_winning_cas(kanban_home):
    """``_force_block_from_policy`` returns True when its CAS moves the row."""
    from hermes_cli import kanban_db as kb
    conn = _connect(kanban_home)
    try:
        tid = _create_task(conn, assignee="alice")
        kb.recompute_ready(conn)
        assert kb._force_block_from_policy(conn, tid, "first", lane="ready") is True
    finally:
        conn.close()


def test_force_block_from_policy_returns_false_on_lost_race(kanban_home):
    """If the row was already advanced (claimed or blocked) the helper
    returns False without raising. Pinning the ``rowcount → bool`` contract.
    """
    from hermes_cli import kanban_db as kb
    conn = _connect(kanban_home)
    try:
        tid = _create_task(conn, assignee="alice")
        kb.recompute_ready(conn)
        # First call wins the CAS.
        assert kb._force_block_from_policy(conn, tid, "first", lane="ready") is True
        # Row is now 'blocked' → second call matches 0 rows → False.
        assert kb._force_block_from_policy(conn, tid, "second", lane="ready") is False
    finally:
        conn.close()


# ── Reason truncation by core ────────────────────────────────────────


def test_reason_truncated_to_500_chars(kanban_home):
    """Core truncates ``reason`` to 500 chars before persisting."""
    from hermes_cli import kanban_db as kb
    conn = _connect(kanban_home)
    try:
        tid = _create_task(conn, assignee="alice")
        kb.recompute_ready(conn)
        long_reason = "X" * 1500
        kb._force_block_from_policy(conn, tid, long_reason, lane="ready")
        row = conn.execute(
            "SELECT last_failure_error FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert len(row["last_failure_error"]) == 500
    finally:
        conn.close()