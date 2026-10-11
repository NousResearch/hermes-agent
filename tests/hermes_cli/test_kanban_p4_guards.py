"""Phase2 P4 kanban guards (t_3f196ad5): cooldown, sweep, do-not-dispatch.

Covers, against an isolated board (``HERMES_HOME`` in tmp):

* P4-1 ``gave_up`` cooldown re-dispatch with backoff + guard event reason —
  fires inside the window, backs off (300/600/1200... capped 3600), elapses
  into permission (never sticky), exempts ``rate_limited`` runs, clears on a
  later ``completed`` run, and falls through to ``blocker_auth`` (no trap).
* P4-2 ledger-integrity sweep — warn-only: lists done-pointer / orphan-run /
  leaked-open-run findings and performs zero writes.
* P4-3 ``do_not_dispatch`` hold — set/clear roundtrip with audit events,
  guard priority-0, TTL lapse without a write, refusal on terminal/held
  tasks, migration defaults.
* P4-4 narrow-UPDATE lint rule — clean on the transition engine, flags
  synthetic violations.
* P4-5 scoped unblock confirmation signal — fires only on loop history.
* P4-6 diagnostics surfacing — ``dispatch_held`` visible, ``stranded``
  suppressed for held tasks, lapsed holds quiet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kd

REPO_ROOT = Path(__file__).resolve().parents[2]
# Module-level import without permanently mutating sys.path: a leaked
# scripts/ entry changes the spawn boundary's probe verdict for every test
# collected afterwards in the same process (only per-file sharded CI hid it).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
try:
    import check_kanban_narrow_updates as narrow_lint
finally:
    sys.path.remove(str(REPO_ROOT / "scripts"))


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB (post-migration schema)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_GAVE_UP_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_LEDGER_SWEEP", raising=False)
    kb.init_db()
    return home


NOW = 5_000_000


def _freeze_time(monkeypatch: pytest.MonkeyPatch, now: int) -> None:
    monkeypatch.setattr(kb.time, "time", lambda: now)


def _gave_up(conn, task_id: str, at: int, n: int = 1) -> None:
    for _ in range(n):
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'gave_up', '{}', ?)",
            (task_id, at),
        )
    conn.commit()


def _event_kinds(conn, task_id: str) -> list[str]:
    return [e.kind for e in kb.list_events(conn, task_id)]


# ---------------------------------------------------------------------------
# P4-1 gave_up cooldown
# ---------------------------------------------------------------------------


def test_gave_up_cooldown_fires_inside_window(kanban_home, monkeypatch):
    _freeze_time(monkeypatch, NOW)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="crasher", assignee="a")
        _gave_up(conn, tid, NOW - 100)
        assert kbd.check_respawn_guard(conn, tid) == "gave_up_cooldown"


def test_gave_up_cooldown_backoff_grows(kanban_home, monkeypatch):
    # n=1 -> 300s; n=2 -> 600s; n=3 -> 1200s.
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="repeat crasher", assignee="a")
        _gave_up(conn, tid, NOW - 400, n=1)
        _freeze_time(monkeypatch, NOW)
        assert kbd.check_respawn_guard(conn, tid) is None  # 400 > 300 elapsed
        _gave_up(conn, tid, NOW - 400, n=1)  # now n=2 -> 600s window
        assert kbd.check_respawn_guard(conn, tid) == "gave_up_cooldown"
        _gave_up(conn, tid, NOW - 400, n=1)  # now n=3 -> 1200s window
        _freeze_time(monkeypatch, NOW + 700)  # elapsed 1100 < 1200
        assert kbd.check_respawn_guard(conn, tid) == "gave_up_cooldown"
        _freeze_time(monkeypatch, NOW + 900)  # elapsed 1300 > 1200
        assert kbd.check_respawn_guard(conn, tid) is None


def test_gave_up_cooldown_caps_at_one_hour(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="chronic crasher", assignee="a")
        _gave_up(conn, tid, NOW - 3500, n=12)  # 300*2**11 >> 3600 cap
        _freeze_time(monkeypatch, NOW)
        assert kbd.check_respawn_guard(conn, tid) == "gave_up_cooldown"
        _freeze_time(monkeypatch, NOW + 200)  # elapsed 3700 > 3600 cap
        assert kbd.check_respawn_guard(conn, tid) is None


def test_gave_up_cooldown_exempts_rate_limited(kanban_home, monkeypatch):
    """A quota wall stays on the rate-limit path, never the gave_up path —
    even with older gave_up debt on the same card."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota + crashes", assignee="a")
        _gave_up(conn, tid, NOW - 50, n=3)
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=? WHERE id=?",
            (NOW - 10, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        _freeze_time(monkeypatch, NOW)
        # Inside the rate-limit cooldown: the quota reason, NOT gave_up.
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"


def test_gave_up_debt_cleared_by_later_completion(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="healed", assignee="a")
        _gave_up(conn, tid, NOW - 100, n=5)
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='completed', status='completed', "
            "ended_at=? WHERE id=?",
            (NOW - 50, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL WHERE id=?", (tid,),
        )
        conn.commit()
        _freeze_time(monkeypatch, NOW)
        # The gave_up debt is cleared: the guard answers with the ordinary
        # recent-success signal, never the crash-loop cooldown.
        assert kbd.check_respawn_guard(conn, tid) == "recent_success"


def test_gave_up_cooldown_disabled_by_env(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_GAVE_UP_COOLDOWN_SECONDS", "0")
    _freeze_time(monkeypatch, NOW)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="opt-out", assignee="a")
        _gave_up(conn, tid, NOW - 10, n=4)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_gave_up_elapsed_falls_through_to_blocker_auth(kanban_home, monkeypatch):
    """Unlike the rate-limit path, an elapsed gave_up cooldown does NOT skip
    blocker_auth — a quota-flavored crash signature must still park."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota crash", assignee="a")
        _gave_up(conn, tid, NOW - 10_000, n=1)  # long elapsed
        conn.execute(
            "UPDATE tasks SET last_failure_error=? WHERE id=?",
            ("quota exceeded 429 — try later", tid),
        )
        conn.commit()
        _freeze_time(monkeypatch, NOW)
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"


# ---------------------------------------------------------------------------
# P4-2 ledger sweep (warn-only)
# ---------------------------------------------------------------------------


def _sweep_state(conn, tid: str):
    statuses = {
        r["id"]: r["status"]
        for r in conn.execute("SELECT id, status FROM tasks").fetchall()
    }
    n_events = conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()["n"]
    return statuses, n_events


def test_sweep_clean_board_is_quiet(kanban_home):
    with kbc.connect() as conn:
        kb.create_task(conn, title="fine", assignee="a")
        assert kbd.ledger_integrity_sweep(conn) == []


def test_sweep_done_pointer_variants(kanban_home):
    with kbc.connect() as conn:
        open_tid = kb.create_task(conn, title="done-open", assignee="a")
        kb.claim_task(conn, open_tid)
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (open_tid,))

        stale_tid = kb.create_task(conn, title="done-stale", assignee="a")
        kb.claim_task(conn, stale_tid)
        run_id = kb.get_task(conn, stale_tid).current_run_id
        conn.execute("UPDATE task_runs SET ended_at=? WHERE id=?", (NOW, run_id))
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (stale_tid,))

        dangling_tid = kb.create_task(conn, title="done-dangling", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='done', current_run_id=999999 WHERE id=?",
            (dangling_tid,),
        )
        conn.commit()

        before = _sweep_state(conn, open_tid)
        findings = kbd.ledger_integrity_sweep(conn)
        assert before == _sweep_state(conn, open_tid)  # zero writes
        by_kind = {}
        for f in findings:
            kind, rest = f.split(" ", 1)
            by_kind.setdefault(kind, []).append(rest)
        assert any(open_tid in r and "open" in r for r in by_kind.get("done_pointer", []))
        assert any(stale_tid in r and "stale" in r for r in by_kind.get("done_pointer", []))
        assert any(dangling_tid in r and "dangling" in r for r in by_kind.get("done_pointer", []))


def test_sweep_orphan_and_leaked_runs(kanban_home):
    with kbc.connect() as conn:
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at) "
            "VALUES ('t_missing_nope', 'crashed', ?)",
            (NOW,),
        )
        leaked_tid = kb.create_task(conn, title="leaked", assignee="a")
        kb.claim_task(conn, leaked_tid)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (leaked_tid,))
        conn.commit()

        before = _sweep_state(conn, leaked_tid)
        findings = " ".join(kbd.ledger_integrity_sweep(conn))
        assert before == _sweep_state(conn, leaked_tid)  # zero writes
        assert "orphan_run t_missing_nope" in findings
        assert f"leaked_open_run {leaked_tid}" in findings


def test_sweep_kill_switch(kanban_home, monkeypatch):
    """The tick's reclaim phase carries sweep findings onto the result —
    unless the kill-switch is set."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='done', current_run_id=424242 WHERE id=?", (tid,),
        )
        conn.commit()
        result = kbd.DispatchResult()
        kbd._run_reclaim_phase(
            conn, result, stale_timeout_seconds=0,
            failure_limit=kbd.DEFAULT_FAILURE_LIMIT, reconcile_orphans=False,
        )
        assert any("done_pointer" in w and tid in w for w in result.ledger_warnings)

    monkeypatch.setenv("HERMES_KANBAN_LEDGER_SWEEP", "0")
    assert not kbd._ledger_sweep_enabled()
    with kbc.connect() as conn:
        result = kbd.DispatchResult()
        kbd._run_reclaim_phase(
            conn, result, stale_timeout_seconds=0,
            failure_limit=kbd.DEFAULT_FAILURE_LIMIT, reconcile_orphans=False,
        )
        assert result.ledger_warnings == []


# ---------------------------------------------------------------------------
# P4-3 do_not_dispatch hold
# ---------------------------------------------------------------------------


def test_hold_release_roundtrip_with_audit(kanban_home, monkeypatch):
    _freeze_time(monkeypatch, NOW)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held work", assignee="a")
        assert kb.set_do_not_dispatch(conn, tid, reason="waiting on vendor", by="op") is True
        assert kb.set_do_not_dispatch(conn, tid, reason="again") is False  # already held
        task = kb.get_task(conn, tid)
        assert task.do_not_dispatch is True
        assert task.do_not_dispatch_reason == "waiting on vendor"
        assert kbd.check_respawn_guard(conn, tid) == "do_not_dispatch"
        assert "dispatch_hold" in _event_kinds(conn, tid)

        assert kb.clear_do_not_dispatch(conn, tid, by="op") is True
        assert kb.clear_do_not_dispatch(conn, tid) is False  # not held
        assert kbd.check_respawn_guard(conn, tid) is None
        assert "dispatch_release" in _event_kinds(conn, tid)


def test_hold_beats_cooldown_and_survives_unblock(kanban_home, monkeypatch):
    """Priority-0: an operator hold wins even over an elapsed cooldown, and
    an unblock does not clear dispatch-level intent."""
    _freeze_time(monkeypatch, NOW)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held crasher", assignee="a")
        _gave_up(conn, tid, NOW - 10_000, n=1)
        assert kb.set_do_not_dispatch(conn, tid, reason="hands off") is True
        assert kbd.check_respawn_guard(conn, tid) == "do_not_dispatch"
        kb.block_task(conn, tid, reason="pause", kind="capability")
        assert kb.get_task(conn, tid).do_not_dispatch is True
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).do_not_dispatch is True


def test_hold_ttl_lapses_without_a_write(kanban_home, monkeypatch):
    _freeze_time(monkeypatch, NOW)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ttl hold", assignee="a")
        assert kb.set_do_not_dispatch(conn, tid, ttl_seconds=100) is True
        assert kbd.check_respawn_guard(conn, tid) == "do_not_dispatch"
        _freeze_time(monkeypatch, NOW + 200)
        # Expired: dispatches normally, flag row still present (auditable).
        assert kbd.check_respawn_guard(conn, tid) is None
        assert kb.get_task(conn, tid).do_not_dispatch is True
        assert kb.clear_do_not_dispatch(conn, tid) is True


def test_hold_refused_on_terminal_or_unknown(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="done soon", assignee="a")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
        conn.commit()
        assert kb.set_do_not_dispatch(conn, tid) is False
        assert kb.set_do_not_dispatch(conn, "t_nope_missing", reason="x") is False
        assert kb.set_do_not_dispatch(conn, tid, ttl_seconds=0) is False
        assert kb.set_do_not_dispatch(conn, tid, ttl_seconds=-5) is False


def test_hold_migration_defaults(kanban_home):
    """Fresh + legacy-shaped rows read as unheld (fail open, never silent)."""
    with kbc.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        assert {"do_not_dispatch", "do_not_dispatch_reason", "do_not_dispatch_until"} <= cols
        tid = kb.create_task(conn, title="legacy read", assignee="a")
        row = conn.execute(
            "SELECT id, title, body, assignee, status, priority, created_by, created_at, "
            "started_at, completed_at, workspace_kind, workspace_path, claim_lock, claim_expires "
            "FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()
        task = kb.Task.from_row(row)
        assert task.do_not_dispatch is False
        assert task.do_not_dispatch_reason is None
        assert task.do_not_dispatch_until is None
        assert kb.dispatch_hold_active({}, int(time.time())) is False


# ---------------------------------------------------------------------------
# P4-4 narrow-UPDATE lint rule
# ---------------------------------------------------------------------------


def test_narrow_updates_clean_on_transition_engine(tmp_path):
    findings = []
    for name in (
        "kanban_db.py", "kanban_db_dispatch.py", "kanban.py",
        "kanban_ops.py", "kanban_output.py", "kanban_diagnostics.py",
        "kanban_db_workspace.py", "kanban_db_notify.py", "kanban_db_graph.py",
    ):
        p = REPO_ROOT / "hermes_cli" / name
        if p.is_file():
            findings.extend(narrow_lint._scan_file(p))
    assert findings == []


def test_narrow_updates_flags_synthetic_violations(tmp_path):
    bad = tmp_path / "bad_kanban.py"
    bad.write_text(
        "x = 1\n"
        "cur = conn.execute(\"UPDATE tasks SET status = 'ready' WHERE id = ?\", (tid,))\n"
        "cur2 = conn.execute(\"UPDATE tasks SET title = ?\", (title,))\n"
        "ok = conn.execute(\"UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'\", (s, tid))\n",
        encoding="utf-8",
    )
    rules = {f.rule for f in narrow_lint._scan_file(bad)}
    assert rules == {"status-without-predicate", "missing-WHERE-id"}


# ---------------------------------------------------------------------------
# P4-5 scoped unblock confirmation signal
# ---------------------------------------------------------------------------


def test_unblock_loop_signal_only_on_history(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="loop risk", assignee="a")
        assert kb.unblock_loop_signal(conn, tid) is None  # ready: nothing to confirm
        kb.block_task(conn, tid, reason="need input", kind="needs_input")
        # First block, never unblocked: routine unblock stays silent.
        assert kb.unblock_loop_signal(conn, tid) is None
        assert kb.unblock_task(conn, tid) is True
        assert kb.unblock_loop_signal(conn, tid) is None  # ready again: silent
        # Re-blocked after an unblock (different kind): proven cycle -> signal.
        kb.block_task(conn, tid, reason="flaky?", kind="transient")
        sig = kb.unblock_loop_signal(conn, tid)
        assert sig is not None
        assert sig["block_recurrences"] == 1
        assert sig["block_kind"] == "transient"
        assert sig["trips_triage_next"] is True  # LIMIT=2: one more routes to triage

        # Same-kind re-block escalates to triage, where there is no unblock
        # transition left to confirm.
        tid2 = kb.create_task(conn, title="same-kind loop", assignee="a")
        kb.block_task(conn, tid2, reason="need input", kind="needs_input")
        assert kb.unblock_task(conn, tid2) is True
        kb.block_task(conn, tid2, reason="need input again", kind="needs_input")
        assert kb.get_task(conn, tid2).status == "triage"
        assert kb.unblock_loop_signal(conn, tid2) is None
        assert kb.unblock_loop_signal(conn, "t_nope_missing") is None


def test_unblock_parser_has_force_and_hold_release_parse():
    import argparse

    from hermes_cli import kanban_parser as kp

    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="cmd")
    kp.build_parser(sub)
    ns, _ = top.parse_known_args(["kanban", "unblock", "t_x"])
    assert ns.kanban_action == "unblock" and ns.force is False
    ns, _ = top.parse_known_args(["kanban", "unblock", "t_x", "--force"])
    assert ns.force is True
    ns, _ = top.parse_known_args(["kanban", "hold", "t_x", "--reason", "wait", "--ttl", "60"])
    assert (ns.kanban_action, ns.ttl) == ("hold", 60)
    ns, _ = top.parse_known_args(["kanban", "release", "t_x"])
    assert ns.kanban_action == "release"


def test_hold_release_in_delegated_deny_list():
    """New mutating verbs must join the delegated-child CLI deny list."""
    from hermes_cli.kanban import _DELEGATED_CHILD_DENIED_ACTIONS

    assert {"hold", "release"} <= set(_DELEGATED_CHILD_DENIED_ACTIONS)


def _run_hermes_cli(home: Path, *args: str, marker: bool = False):
    """Real CLI subprocess (pattern from test_kanban_cli_exit_status.py)."""
    import os as _os
    import subprocess as _sp

    env = _os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    for name in (
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(REPO_ROOT) + _os.pathsep + env.get("PYTHONPATH", "")
    if marker:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    return _sp.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_hold_release_cli_smoke_and_child_refusal(tmp_path):
    """End-to-end on a scratch board: hold -> show holds -> release, and a
    delegated child context is refused both verbs."""
    import json as _json

    home = tmp_path / "hermes"
    home.mkdir()
    created = _run_hermes_cli(home, "kanban", "create", "p4 cli smoke", "--json")
    assert created.returncode == 0, created.stderr
    tid = _json.loads(created.stdout)["id"]

    held = _run_hermes_cli(home, "kanban", "hold", tid, "--reason", "vendor freeze")
    assert held.returncode == 0, held.stderr
    assert "do-not-dispatch" in held.stdout

    shown = _run_hermes_cli(home, "kanban", "show", tid)
    assert shown.returncode == 0, shown.stderr
    assert "dispatch-hold" in shown.stdout and "HELD" in shown.stdout

    released = _run_hermes_cli(home, "kanban", "release", tid)
    assert released.returncode == 0, released.stderr

    for verb in ("hold", "release"):
        refused = _run_hermes_cli(home, "kanban", verb, tid, marker=True)
        assert refused.returncode == 1
        assert "cannot mutate Kanban tasks via the CLI" in refused.stderr

    help_out = _run_hermes_cli(home, "kanban", "--help")
    assert help_out.returncode == 0
    assert "hold" in help_out.stdout and "release" in help_out.stdout


# ---------------------------------------------------------------------------
# P4-6 diagnostics surfacing
# ---------------------------------------------------------------------------


def test_dispatch_held_visible_and_stranded_suppressed(kanban_home, monkeypatch):
    _freeze_time(monkeypatch, NOW)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held ready", assignee="backend")
        # Age it past the stranded threshold with no events after creation.
        conn.execute(
            "UPDATE tasks SET status='ready', created_at=? WHERE id=?", (NOW - 7200, tid),
        )
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (NOW - 7200, tid))
        conn.commit()
        assert kb.set_do_not_dispatch(conn, tid, reason="vendor freeze") is True

        task = kb.get_task(conn, tid)
        diags = kd.compute_task_diagnostics(
            task, kb.list_events(conn, tid), kb.list_runs(conn, tid),
            graph=kb.task_graph_context(conn, tid),
        )
        kinds = {d.kind for d in diags}
        assert "dispatch_held" in kinds
        assert "stranded_in_ready" not in kinds

        # Control: same age without the hold IS stranded, with no held rule.
        tid2 = kb.create_task(conn, title="plain ready", assignee="backend")
        conn.execute(
            "UPDATE tasks SET status='ready', created_at=? WHERE id=?", (NOW - 7200, tid2),
        )
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (NOW - 7200, tid2))
        conn.commit()
        task2 = kb.get_task(conn, tid2)
        diags2 = kd.compute_task_diagnostics(
            task2, kb.list_events(conn, tid2), kb.list_runs(conn, tid2),
            graph=kb.task_graph_context(conn, tid2),
        )
        kinds2 = {d.kind for d in diags2}
        assert "stranded_in_ready" in kinds2
        assert "dispatch_held" not in kinds2


def test_lapsed_hold_is_quiet(kanban_home, monkeypatch):
    _freeze_time(monkeypatch, NOW)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="lapsed", assignee="backend")
        assert kb.set_do_not_dispatch(conn, tid, ttl_seconds=100) is True
        _freeze_time(monkeypatch, NOW + 500)
        task = kb.get_task(conn, tid)
        diags = kd.compute_task_diagnostics(
            task, kb.list_events(conn, tid), kb.list_runs(conn, tid),
            graph=kb.task_graph_context(conn, tid),
        )
        assert "dispatch_held" not in {d.kind for d in diags}
