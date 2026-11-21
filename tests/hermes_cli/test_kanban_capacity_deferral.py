"""A board sitting at its concurrency cap is deferring work, not stuck (#46800).

``_tick_spawn_budget`` returns before any spawn attempt when ``max_spawn`` or
``max_in_progress`` is already full, and used to record nothing on the
``DispatchResult`` — so the dispatcher health check could not tell "every slot
is busy" from "the profile's venv is gone". Each test here pins one contract:

1. both cap gates record ``capacity_deferred`` without attempting a spawn;
2. a REAL spawn failure with a free slot stays unexplained, so the "stuck"
   warning still fires for it;
3. ``idle_reason`` never explains away a fault (``auto_blocked`` outranks a
   benign sibling; ``blocker_auth`` and unknown guard reasons are faults;
   ``elevated`` memory pressure still permits a spawn);
4. the streak HOLDS across a deferred tick instead of resetting — otherwise a
   board that interleaves a deferral with a genuine failure never reaches the
   health window at all;
5. the CLI names the cap (``--json`` and text) and the legacy daemon callback
   applies the same three-state streak as the gateway.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _start_worker(conn: sqlite3.Connection, title: str, assignee: str = "alice") -> str:
    """Create a task and claim it, so the board has one ``running`` worker."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    return tid


# ---------------------------------------------------------------------------
# 1. Both cap gates record the deferral, and attempt no spawn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "caps, expected",
    [
        # The operator's serial 1/1 board: the BOARD gate fires first and the
        # host gate is never evaluated, so recording only the host gate
        # misses the configuration that hits this every tick.
        ({"max_spawn": 1, "max_in_progress": 1}, "max_spawn"),
        ({"max_in_progress": 1}, "max_in_progress"),
    ],
)
def test_full_cap_records_the_gate_that_fired_without_a_spawn_attempt(
    kanban_home, all_assignees_spawnable, caps, expected,
):
    attempts: list = []

    def exploding_spawn(task, workspace, board=None):
        attempts.append(task.id)
        raise RuntimeError("venv missing")

    with kbc.connect() as conn:
        _start_worker(conn, "already-running")
        kb.create_task(conn, title="waiting-for-slot", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=exploding_spawn, **caps)

    # A capped tick cannot hide a spawn failure: it never attempts one.
    assert attempts == []
    assert not res.spawned
    assert res.capacity_deferred == expected
    assert kbd.idle_reason(res) == f"capacity:{expected}"


# ---------------------------------------------------------------------------
# 2. A real spawn failure must stay unexplained
# ---------------------------------------------------------------------------


def test_spawn_failure_with_free_slot_is_not_explained(kanban_home, all_assignees_spawnable):
    """The signal the warning exists for must survive this change.

    A first-attempt spawn failure leaves NO field on the result (the breaker
    only appends to ``auto_blocked`` on the trip), so ``idle_reason`` must
    return None: spawnable work, a free slot, zero spawns, no reason.
    """
    def exploding_spawn(task, workspace, board=None):
        raise RuntimeError("venv missing")

    with kbc.connect() as conn:
        kb.create_task(conn, title="cannot-launch", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=exploding_spawn, max_spawn=2, max_in_progress=2,
        )

    assert not res.spawned
    assert not res.auto_blocked  # first failure of two — breaker has not tripped
    assert res.capacity_deferred is None
    assert kbd.idle_reason(res) is None


# ---------------------------------------------------------------------------
# 3. idle_reason never explains away a fault
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        # "elevated" still allows one spawn, so a zero-spawn elevated tick is
        # NOT explained by memory pressure (only "critical" spawns nothing).
        pytest.param({"memory_pressure": "elevated"}, id="elevated"),
        # blocker_auth is a quota/auth FAULT dressed as a respawn guard.
        pytest.param(
            {"respawn_guarded": [("t1", "recent_success"), ("t2", "blocker_auth")]},
            id="blocker_auth",
        ),
        # An unknown future guard reason must not be assumed benign.
        pytest.param({"respawn_guarded": [("t1", "something_new")]}, id="unknown_guard"),
        # A fault on the same tick always wins over a benign sibling.
        pytest.param(
            {"auto_blocked": ["t1"], "capacity_deferred": "max_spawn"},
            id="auto_blocked_outranks_cap",
        ),
    ],
)
def test_idle_reason_never_explains_a_fault(kwargs):
    assert kbd.idle_reason(kbd.DispatchResult(**kwargs)) is None


# ---------------------------------------------------------------------------
# 4. The health streak must HOLD across a deferred tick
# ---------------------------------------------------------------------------


def test_interleaved_failure_and_deferral_still_reaches_health_window():
    """The regression a reset-on-benign policy would introduce.

    One board, two profiles: ``bob`` is at his per-profile cap (benign, marked
    on every tick) while ``alice`` cannot launch at all. A FIRST spawn failure
    records nothing, so that tick reads as explained-by-bob's-cap; the breaker
    only marks ``auto_blocked`` when it trips on the second attempt. Ticks
    therefore alternate explained / unexplained, and a counter that RESET on
    the explained tick would oscillate 1, 0, 1, 0 and never reach the six-tick
    window — silently losing a warning that fires on main today. Holding the
    streak lets it climb.
    """
    first_attempt = kbd.DispatchResult(
        skipped_per_profile_capped=[("b1", "bob", 1)],
    )  # alice's first failure leaves no field at all
    breaker_trip = kbd.DispatchResult(
        skipped_per_profile_capped=[("b1", "bob", 1)], auto_blocked=["a1"],
    )  # second failure trips the breaker -> a fault outranks bob's cap

    held = reset_on_benign = 0
    for tick in range(12):
        res = first_attempt if tick % 2 == 0 else breaker_trip
        stalled = kbd.idle_reason(res) is None
        held = kbd.next_bad_tick_count(held, stalled=stalled, deferred=not stalled)
        # The two-state policy this change deliberately does NOT use.
        reset_on_benign = reset_on_benign + 1 if stalled else 0

    assert held >= 6, f"warning lost: streak stalled at {held}"
    assert reset_on_benign < 6, "the rejected policy would also have warned"


# ---------------------------------------------------------------------------
# 5. CLI surface — the operator's proof that the cap is holding
# ---------------------------------------------------------------------------


def test_cli_dispatch_reports_the_cap_in_json_and_text(monkeypatch, capsys, kanban_home):
    from hermes_cli import kanban as kb_cli

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    monkeypatch.setattr(
        kbd, "dispatch_once",
        lambda conn, **kw: kbd.DispatchResult(capacity_deferred="max_in_progress"),
    )

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=True)
    assert kb_cli._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capacity_deferred"] == "max_in_progress"

    args.json = False
    assert kb_cli._cmd_dispatch(args) == 0
    out = capsys.readouterr().out
    # max_in_progress is a HOST-level cap (other boards' workers count), so
    # the operator must not be told the board is full.
    assert "Deferred (host already at the kanban.max_in_progress cap" in out


# ---------------------------------------------------------------------------
# 6. The deprecated `hermes kanban daemon --force` path stays in lockstep
# ---------------------------------------------------------------------------


def test_daemon_tick_holds_through_deferrals_and_never_warns_on_a_capped_board(
    kanban_home, monkeypatch, capsys,
):
    """Drive the real ``_cmd_daemon`` health callback with the gateway's policy.

    #69010's review: the legacy callback must not count a capped tick, and it
    must hold (not reset) the streak across a benign deferral.
    """
    from hermes_cli import kanban as kb_cli

    captured: dict = {}
    monkeypatch.setattr(kbd, "run_daemon", lambda **kw: captured.update(kw))
    monkeypatch.setattr(kbd, "has_spawnable_ready", lambda conn: True)
    args = argparse.Namespace(
        force=True, interval=1, max=None, failure_limit=2, pidfile=None, verbose=False,
    )
    assert kb_cli._cmd_daemon(args) == 0
    on_tick = captured["on_tick"]
    capsys.readouterr()

    for _ in range(40):
        on_tick(kbd.DispatchResult(capacity_deferred="max_spawn"))
    assert "dispatcher stuck" not in capsys.readouterr().err

    for _ in range(3):
        on_tick(kbd.DispatchResult())            # unexplained: 1, 2, 3
    on_tick(kbd.DispatchResult(respawn_guarded=[("t1", "active_pr")]))  # held at 3
    assert "dispatcher stuck" not in capsys.readouterr().err
    for _ in range(3):
        on_tick(kbd.DispatchResult())            # 4, 5, 6 -> warn

    err = capsys.readouterr().err
    assert "dispatcher stuck" in err
    assert "6 unexplained ticks" in err
