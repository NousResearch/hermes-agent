"""Delivery-orchestration repair: explicit continuation + visible dispatch holds.

Two reproduced defects, both of which make the board look idle while work is
silently parked:

1. ``check_respawn_guard`` returns ``"active_pr"`` for an implementation card
   that already has a Draft PR, and NOTHING overrides it — not an operator
   ``done -> ready`` drag, not ``unblock``, not a review handing findings back
   to the same card. The ``recent_success`` rule already honours an explicit
   re-queue; ``active_pr`` did not, so a deliberate "continue this PR" request
   sat behind the 24h PR window with no way out short of deleting the comment.

2. ``hermes kanban dispatch`` never surfaced the holds. The JSON payload
   omitted ``respawn_guarded`` / ``rate_limited`` / ``skipped_locked`` /
   ``memory_pressure`` entirely, and the text output omitted the last three —
   so a fully-held tick printed a bare ``Spawned: 0``, indistinguishable from
   an idle board.

The continuation is *bounded*: one spawn per explicit authorization. Once a
run has started after the re-queue event, the guard re-arms. That keeps the
duplicate-worker protection the guard exists for while letting an operator
(or a review returning findings) say "yes, continue the SAME card/PR".
"""

from __future__ import annotations

from hermes_cli import kanban_db_connect, kanban_db_dispatch

import argparse
import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


PR_URL = "https://github.com/millermindsolutions-com/reefmind-project/pull/4242"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kanban_db_connect.init_db()
    return home


@pytest.fixture
def conn(kanban_home, monkeypatch):
    # The PR-liveness probe shells out to `gh`. Pin it to "still open" so
    # these tests exercise the guard logic, never the network.
    # dispatch_once skips any assignee that is not a real Hermes profile
    # BEFORE the respawn guard runs. The tmp HERMES_HOME has no profiles, so
    # without this every fixture card would land in skipped_nonspawnable and
    # never reach the code under test.
    from hermes_cli import profiles as _profiles
    monkeypatch.setattr(_profiles, "profile_exists", lambda name: True)
    with kanban_db_connect.connect() as c:
        yield c


def _run_row(conn, task_id, *, started_at, ended_at, outcome="completed"):
    with kanban_db_connect.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, "done", started_at, ended_at, outcome),
        )


def _event(conn, task_id, kind, *, created_at):
    payload = {
        "status": {"status": "ready"},
        "promoted_manual": {"actor": "operator"},
        "reclaimed": {"manual": True},
    }.get(kind)
    with kanban_db_connect.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, kind, json.dumps(payload) if payload is not None else None, created_at),
        )


def _comment(conn, task_id, body, *, created_at):
    with kanban_db_connect.write_txn(conn):
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, "worker", body, created_at),
        )


def _card_with_draft_pr(conn, *, pr_age=600, run_age=7200):
    """Implementation card whose worker already opened a Draft PR.

    ``run_age`` sits outside ``_RESPAWN_GUARD_SUCCESS_WINDOW`` on purpose so
    these tests isolate the ``active_pr`` rule from ``recent_success``.
    """
    now = int(time.time())
    tid = kb.create_task(conn, title="implement thing", assignee="integrator")
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    _run_row(conn, tid, started_at=now - run_age - 60, ended_at=now - run_age)
    _comment(conn, tid, f"Opened Draft PR {PR_URL}", created_at=now - pr_age)
    return tid, now


# ---------------------------------------------------------------------------
# 1. Bounded explicit-authorized continuation of the SAME card / PR
# ---------------------------------------------------------------------------

def test_active_pr_still_holds_without_an_explicit_requeue(conn):
    """Baseline protection retained: a fresh PR with no operator signal is
    still a duplicate-work hold."""
    tid, _now = _card_with_draft_pr(conn)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"


@pytest.mark.parametrize("kind", ["status", "promoted_manual", "unblocked", "reclaimed"])
def test_explicit_requeue_after_the_pr_releases_the_active_pr_hold(conn, kind):
    """An operator drag / manual promotion / unblock / manual reclaim that lands
    AFTER the PR comment is a deliberate "continue this card" request."""
    tid, now = _card_with_draft_pr(conn)
    _event(conn, tid, kind, created_at=now - 60)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None


def test_requeue_older_than_the_pr_does_not_release_the_hold(conn):
    """Only a re-queue NEWER than the PR comment authorizes continuation.
    A stale re-queue from before the PR was opened proves nothing."""
    tid, now = _card_with_draft_pr(conn, pr_age=600)
    _event(conn, tid, "status", created_at=now - 3600)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"


def test_authorization_is_consumed_by_the_spawn_it_authorized(conn):
    """Bounded, not unrestricted: once a run has STARTED after the re-queue
    event, that authorization is spent and the guard re-arms. Without this the
    re-queue event lives in task_events forever and every subsequent tick
    would respawn the same card — an unbounded retry loop."""
    tid, now = _card_with_draft_pr(conn)
    _event(conn, tid, "unblocked", created_at=now - 300)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None, "authorized continuation"
    # The dispatcher handed the card to a worker; that run then crashed.
    _event(conn, tid, "claimed", created_at=now - 200)
    _run_row(conn, tid, started_at=now - 200, ended_at=now - 100, outcome="crashed")
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr", "re-armed"


def test_bookkeeping_runs_do_not_spend_the_authorization(conn):
    """`block_task` writes a synthetic ``blocked`` run row whose ``started_at``
    lands in the same second as the ``unblocked`` event. Keying consumption on
    run start timestamps let that pair cancel its own authorization, so the
    canonical block -> unblock re-queue was a no-op against ``active_pr``."""
    tid, now = _card_with_draft_pr(conn)
    _event(conn, tid, "blocked", created_at=now - 120)
    _run_row(conn, tid, started_at=now - 120, ended_at=now - 120, outcome="blocked")
    _event(conn, tid, "unblocked", created_at=now - 120)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None


def test_explicit_requeue_does_not_override_the_auth_blocker(conn):
    """Auth/quota protection is NOT relaxed by the continuation path — a
    re-queue must not let the dispatcher hammer a dead credential."""
    tid, now = _card_with_draft_pr(conn)
    with kanban_db_connect.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("401 Unauthorized: OAuth token expired", tid),
        )
    _event(conn, tid, "unblocked", created_at=now - 60)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "blocker_auth"


def test_review_lane_is_unaffected_by_the_continuation_path(conn):
    """The review lane already skips active_pr — a PR URL is the review
    handoff's precondition. Continuation must not change that."""
    tid, _now = _card_with_draft_pr(conn)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid, lane="review") is None


# ---------------------------------------------------------------------------
# 2. Hold events are recorded once, not once per dispatcher tick
# ---------------------------------------------------------------------------

def test_respawn_guarded_event_is_deduped_across_ticks(conn):
    """The hold must be visible in `hermes kanban tail` — but the dispatcher
    ticks every 60s, and re-stamping an identical respawn_guarded event every
    tick buries the real history under thousands of duplicates."""
    tid, _now = _card_with_draft_pr(conn)

    def never_spawn(task, workspace_path, board=None):  # pragma: no cover
        raise AssertionError("guarded task must not spawn")

    for _ in range(4):
        res = kanban_db_dispatch.dispatch_once(conn, spawn_fn=never_spawn)
        assert (tid, "active_pr") in res.respawn_guarded

    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'respawn_guarded'",
        (tid,),
    ).fetchall()
    assert len(rows) == 1, f"expected one hold event, got {len(rows)}"


def test_a_changed_hold_reason_is_recorded_again(conn):
    """Dedupe is per-reason: a genuinely different hold is still news."""
    tid, _now = _card_with_draft_pr(conn)

    def never_spawn(task, workspace_path, board=None):  # pragma: no cover
        raise AssertionError("guarded task must not spawn")

    kanban_db_dispatch.dispatch_once(conn, spawn_fn=never_spawn)
    with kanban_db_connect.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("429 rate limit exceeded for this API key", tid),
        )
    kanban_db_dispatch.dispatch_once(conn, spawn_fn=never_spawn)

    reasons = [
        json.loads(r["payload"])["reason"]
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'respawn_guarded' ORDER BY id",
            (tid,),
        ).fetchall()
    ]
    assert reasons == ["active_pr", "blocker_auth"]


# ---------------------------------------------------------------------------
# 3. `hermes kanban dispatch` must name every hold, in JSON and in text
# ---------------------------------------------------------------------------

def _held_result():
    res = kanban_db_dispatch.DispatchResult()
    res.respawn_guarded = [("t_aaa", "active_pr"), ("t_bbb", "rate_limit_cooldown")]
    res.rate_limited = ["t_ccc"]
    res.skipped_locked = True
    res.memory_pressure = "critical"
    return res


def _dispatch_args(**kw):
    ns = argparse.Namespace(
        dry_run=False, max=None, json=False, failure_limit=2,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_dispatch_json_names_every_hold_bucket(conn, monkeypatch, capsys):
    monkeypatch.setattr(kanban_db_dispatch, "dispatch_once", lambda *a, **k: _held_result())
    assert kcli._cmd_dispatch(_dispatch_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["respawn_guarded"] == [
        {"task_id": "t_aaa", "reason": "active_pr"},
        {"task_id": "t_bbb", "reason": "rate_limit_cooldown"},
    ]
    assert payload["rate_limited"] == ["t_ccc"]
    assert payload["skipped_locked"] is True
    assert payload["memory_pressure"] == "critical"


def test_dispatch_text_names_lock_memory_and_rate_limit_holds(conn, monkeypatch, capsys):
    monkeypatch.setattr(kanban_db_dispatch, "dispatch_once", lambda *a, **k: _held_result())
    assert kcli._cmd_dispatch(_dispatch_args()) == 0
    out = capsys.readouterr().out

    assert "Spawned:      0" in out
    assert "respawn guard" in out.lower() and "t_aaa" in out and "active_pr" in out
    assert "rate limit" in out.lower() and "t_ccc" in out
    assert "dispatch lock" in out.lower()
    assert "memory pressure" in out.lower() and "critical" in out


def test_quiet_tick_stays_quiet(conn, monkeypatch, capsys):
    """No holds -> no hold lines. The new output must not add noise to the
    ordinary idle tick."""
    monkeypatch.setattr(kanban_db_dispatch, "dispatch_once", lambda *a, **k: kanban_db_dispatch.DispatchResult())
    assert kcli._cmd_dispatch(_dispatch_args()) == 0
    out = capsys.readouterr().out.lower()
    for phrase in ("respawn guard", "rate limit", "dispatch lock", "memory pressure"):
        assert phrase not in out


# ---------------------------------------------------------------------------
# 4. The hold is visible in diagnostics, not only in a live dispatch tick
# ---------------------------------------------------------------------------

def _diag_kinds(task, events, runs=()):
    return {
        d.kind: d
        for d in kd.compute_task_diagnostics(
            task, list(events), list(runs), config=dict(kd.DEFAULT_CONFIG)
        )
    }


def test_diagnostics_surface_a_held_ready_task(conn):
    """`hermes kanban diagnostics` reads events, so the respawn_guarded event
    is enough to explain WHY a ready card never gets a worker."""
    tid, _now = _card_with_draft_pr(conn)

    def never_spawn(task, workspace_path, board=None):  # pragma: no cover
        raise AssertionError("guarded task must not spawn")

    kanban_db_dispatch.dispatch_once(conn, spawn_fn=never_spawn)

    diags = _diag_kinds(
        kb.get_task(conn, tid), kb.list_events(conn, tid), kb.list_runs(conn, tid)
    )
    assert "respawn_guard_hold" in diags
    d = diags["respawn_guard_hold"]
    assert "active_pr" in d.detail or "active_pr" in json.dumps(d.data)
    assert d.severity in {"warning", "error"}
    assert "respawn_guard_hold" in kd.DIAGNOSTIC_KINDS


def test_no_hold_diagnostic_once_the_task_is_running(conn):
    """A card that got its worker is not held. Stale hold events must not
    keep flagging a task that has since been dispatched."""
    tid, now = _card_with_draft_pr(conn)
    _event(conn, tid, "respawn_guarded", created_at=now - 300)
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))

    diags = _diag_kinds(kb.get_task(conn, tid), kb.list_events(conn, tid))
    assert "respawn_guard_hold" not in diags
