"""Regression tests for the rate-limit respawn budget.

A ``rate_limited`` requeue records no failure, so ``check_respawn_guard`` used
to retry forever — once per cooldown tick — for as long as the provider quota
wall lasted. On a paid lane every one of those spawns bills a full worker
context before the 429 comes back; one stalled card sat behind a quota wall for
244 runs, all of them billed. The budget below bounds that path:

* the trailing run of ``rate_limited`` outcomes is counted from durable
  ``task_runs`` rows, so a gateway restart cannot reset it mid-loop,
* it trips at ``HERMES_KANBAN_RATE_LIMIT_MAX_STREAK`` (default 3),
* any other closed outcome breaks the streak, so the budget never accumulates
  across unrelated attempts,
* ``0`` restores the old unbounded behaviour,
* the dispatch site parks the tripped card ``blocked`` (sticky) so it stops
  respawning *and* stops looking merely stuck in ``ready``.
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
    # Pin the cooldown: the ambient environment carries a 24h override, and the
    # backdated runs below only have to clear the *budget*, not that window.
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _runs(conn, task_id: str, n: int, *, outcome: str = "rate_limited") -> None:
    """Insert ``n`` closed runs, backdated well past any cooldown."""
    base = int(time.time()) - 7200
    with kb.write_txn(conn):
        for i in range(n):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'default', ?, ?, ?, ?)",
                (task_id, outcome, outcome, base + i, base + i + 1),
            )


# ---------------------------------------------------------------------------
# The budget itself
# ---------------------------------------------------------------------------


def test_below_budget_stays_retryable(kanban_home: Path) -> None:
    """Under the budget the card still retries — the cooldown, not the budget,
    is what spaces those attempts."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wall, short")
        _runs(conn, tid, 2)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_budget_trips_at_the_limit(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wall, long")
        _runs(conn, tid, 3)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_streak"


def test_streak_is_durable_across_a_restart(kanban_home: Path) -> None:
    """The count lives in ``task_runs``, so a gateway restart mid-loop does not
    hand the card a fresh budget."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wall across a restart")
        _runs(conn, tid, 3)
    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_streak"


def test_non_rate_limited_outcome_breaks_the_streak(kanban_home: Path) -> None:
    """A real run supersedes the quota wall: only the *trailing* run counts."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="recovered, then limited again")
        _runs(conn, tid, 3)
        _runs(conn, tid, 1, outcome="success")
        _runs(conn, tid, 2)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_env_override_sets_the_budget(kanban_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_STREAK", "5")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="operator raised the budget")
        _runs(conn, tid, 4)
        assert kbd.check_respawn_guard(conn, tid) is None
        _runs(conn, tid, 1)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_streak"


def test_zero_disables_the_budget(kanban_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_STREAK", "0")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unbounded by request")
        _runs(conn, tid, 12)
        assert kbd.check_respawn_guard(conn, tid) is None


# ---------------------------------------------------------------------------
# The dispatch site parks the card
# ---------------------------------------------------------------------------


def test_dispatch_reports_the_guard_and_spawns_nothing(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.profiles as profmod

    # ``worker`` is a routing label here, not an installed profile: without
    # this the row is bucketed ``skipped_nonspawnable`` and never guarded.
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked, not spawned", assignee="worker")
        _runs(conn, tid, 3)
        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid not in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(tid) == "rate_limit_streak"


def test_parking_blocks_sticky_and_survives_recompute(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked for an operator")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (tid,)
            )
            kbd._park_rate_limit_streak(conn, tid, 3, 3)

        assert kb.get_task(conn, tid).status == "blocked"
        # A quota wall is the provider's fault, not the card's: the park must not
        # spend the crash budget, or an unblocked card re-blocks on one failure.
        assert kb.get_task(conn, tid).consecutive_failures == 1
        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert gave_up, "the park must be visible in ``hermes kanban tail``"
        payload = gave_up[-1].payload
        assert payload.get("sticky") is True
        assert payload.get("rate_limited_streak") == 3

        # Sticky: the next tick must not promote it back to ``ready``.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"
