"""A dead Kanban worker is booked the same way whichever process notices it.

``_recent_worker_exits`` is filled by ``os.waitpid`` and so only knows children of
the process running the sweep; a per-tick ``hermes kanban dispatch`` process finds it
empty. The worker's own exit trailer in its log is the durable witness the sweep reads
instead, and a tripped protocol-violation budget must hold the card until an operator
unblocks it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER, exit_single_query


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _dead_worker_with_log(
    conn, tid: str, pid: int, rc: int, message: str = "the model said something", cause: str = "",
) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its log — never reaped here."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        suffix = f" cause={cause}" if cause else ""
        f.write(f"{message}\n\nResume this session with:\n  hermes --resume x\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}{suffix}\n")


@pytest.mark.parametrize(
    "rc, event, failure_counted",
    [(0, "protocol_violation", False), (kb.KANBAN_RATE_LIMIT_EXIT_CODE, "rate_limited", False)],
)
def test_fresh_process_sweep_books_the_logged_exit_code(kanban_home, rc, event, failure_counted):
    """Empty reap registry + exit trailer in the log: a clean exit is the protocol violation
    (marker, streak, no unified-budget hit) and a 75 is a rate-limit requeue — not a bare
    ``pid N not alive`` crash that counts a failure."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 70001, rc)

        kbd.detect_crashed_workers(conn)

        ev = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        run = conn.execute(
            "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,)).fetchone()
        task = kb.get_task(conn, tid)
        assert ev["kind"] == event
        assert "not alive" not in (run["error"] or "")
        assert task.status == "ready"
        assert task.consecutive_failures == (1 if failure_counted else 0)
        # The decoded rc lands in the run row so quota (75) vs crash stays tellable after
        # the fact even though the worker_output tail is trimmed (#113611).
        assert kb._json_dict(run["metadata"]).get("exit_code") == rc
        if rc == 0:
            assert kb._json_dict(run["metadata"]).get("protocol_violation") is True
            assert kbd._protocol_violation_streak(conn, tid) == 1
            assert KANBAN_WORKER_EXIT_TRAILER not in (run["error"] or "")
        else:
            assert run["outcome"] == "rate_limited"


def test_violation_budget_trip_holds_until_operator_unblock(kanban_home):
    """The third consecutive clean exit trips the violation budget and ``recompute_ready``
    must not promote the card back the same tick (``consecutive_failures`` is still below
    ``failure_limit``); ``unblock_task`` lifts the hold."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="loop", assignee="a")
        for i in range(kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT):
            _dead_worker_with_log(conn, tid, 71000 + i, 0)
            kbd.detect_crashed_workers(conn)
            kb.recompute_ready(conn, failure_limit=10)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures < 10

        kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        kb.recompute_ready(conn, failure_limit=10)
        assert kb.get_task(conn, tid).status == "ready"


def test_plain_budget_trip_requires_recovery_not_reassignment(kanban_home):
    """A unified-budget trip is released when its configured failure limit rises,
    but reassignment is not evidence that a provider/security failure cleared.

    ``assign_task`` must preserve the failure state so only the audited narrow
    recovery lifecycle can erase it; otherwise reassignment bypasses the hold.
    """
    with kbc.connect() as conn:
        tids = [kb.create_task(conn, title=t, assignee="a") for t in ("raise-limit", "reassign")]
        for tid in tids:
            for i in range(2):
                kbd._record_task_failure(
                    conn, tid, error=f"boom{i}", outcome="crashed", failure_limit=2,
                    release_claim=False, end_run=False,
                )
            assert kb.get_task(conn, tid).status == "blocked"
        assert kb.recompute_ready(conn, failure_limit=2) == 0

        assert kb.recompute_ready(conn, failure_limit=5) == 2
        assert kb.get_task(conn, tids[0]).status == "ready"

        for i in range(2):
            kbd._record_task_failure(
                conn, tids[1], error=f"again{i}", outcome="crashed", failure_limit=2,
                release_claim=False, end_run=False,
            )
        assert kb.get_task(conn, tids[1]).status == "blocked"
        kb.assign_task(conn, tids[1], "other-profile")
        assert kb.recompute_ready(conn, failure_limit=2) == 0
        task = kb.get_task(conn, tids[1])
        assert task is not None
        assert (task.status, task.assignee) == ("blocked", "other-profile")
        assert task.consecutive_failures >= 2


def test_unstructured_worker_output_cannot_block_peer_cards(kanban_home):
    """Provider-looking prose in one worker log is not authority to mutate peer cards."""
    with kbc.connect() as conn:
        failed = kb.create_task(conn, title="failed", assignee="broken-profile")
        queued = kb.create_task(conn, title="queued", assignee="broken-profile")
        other = kb.create_task(conn, title="other", assignee="healthy-profile")
        _dead_worker_with_log(
            conn, failed, 72001, 1,
            "No API key found for provider 'openrouter'",
        )

        crashed = kbd.detect_crashed_workers(conn)

        assert crashed == [failed]
        assert kb.get_task(conn, failed).status == "ready"
        assert kb.get_task(conn, queued).status == "ready"
        assert kb.get_task(conn, other).status == "ready"
        assert "config_fatal" not in [e.kind for e in kb.list_events(conn, queued)]


def test_structured_provider_config_cause_blocks_the_affected_profile(kanban_home):
    with kbc.connect() as conn:
        failed = kb.create_task(conn, title="failed", assignee="broken-profile")
        queued = kb.create_task(conn, title="queued", assignee="broken-profile")
        _dead_worker_with_log(conn, failed, 72001, 1, cause="provider_config")

        blocked = kbd.detect_crashed_workers(conn)

        assert blocked == [failed]
        assert kbd.detect_crashed_workers._last_auto_blocked == [failed, queued]
        assert kb.get_task(conn, failed).status == "blocked"
        assert kb.get_task(conn, queued).status == "blocked"
        assert "config_fatal" in [event.kind for event in kb.list_events(conn, queued)]
        comment = conn.execute(
            "SELECT body FROM task_comments WHERE task_id=? ORDER BY id DESC LIMIT 1", (queued,),
        ).fetchone()
        assert "config_fatal" in comment["body"]


def test_config_fatal_trip_records_provider_for_recovery(kanban_home, monkeypatch):
    """The trip names the failing provider on the card so a later recovery probe (and a
    human) knows what was actually broken, not just that *something* was (#t_f9a0fdf7)."""
    monkeypatch.setattr(
        kbd, "_run_credential_probe",
        lambda profile, timeout=20.0: {"ok": False, "provider": "openrouter"},
    )
    with kbc.connect() as conn:
        failed = kb.create_task(conn, title="failed", assignee="broken-profile")
        _dead_worker_with_log(conn, failed, 72001, 1, cause="provider_config")

        kbd.detect_crashed_workers(conn)

        trip = next(e for e in kb.list_events(conn, failed) if e.kind == "config_fatal")
        assert trip.payload["provider"] == "openrouter"
        assert trip.payload["prev_status"] == "ready"


def test_recheck_requeues_once_after_credential_recovers(kanban_home, monkeypatch):
    """A confirmed-passing probe requeues every config_fatal-parked task for that profile,
    resets the failure counter, and a later tick does not touch them again — one-shot."""
    monkeypatch.setattr(kbd, "_config_fatal_recheck_interval_seconds", lambda: 0)
    monkeypatch.setattr(
        kbd, "_run_credential_probe",
        lambda profile, timeout=20.0: {"ok": True, "provider": "openrouter"},
    )
    with kbc.connect() as conn:
        failed = kb.create_task(conn, title="failed", assignee="broken-profile")
        queued = kb.create_task(conn, title="queued", assignee="broken-profile")
        _dead_worker_with_log(conn, failed, 72001, 1, cause="provider_config")
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, failed).status == "blocked"
        assert kb.get_task(conn, queued).status == "blocked"

        recovered = kbd._recheck_config_fatal_credentials(conn)

        assert set(recovered) == {failed, queued}
        for tid in (failed, queued):
            task = kb.get_task(conn, tid)
            assert task.status == "ready"
            assert task.consecutive_failures == 0
            assert [e.kind for e in kb.list_events(conn, tid)][-1] == "config_fatal_recovered"
        comment = conn.execute(
            "SELECT body FROM task_comments WHERE task_id=? ORDER BY id DESC LIMIT 1", (queued,),
        ).fetchone()
        assert "recovered" in comment["body"]

        # The profile no longer has anything config_fatal-parked: a later tick is a no-op,
        # proving the requeue is one-shot per trip rather than a repeating unblock.
        assert kbd._recheck_config_fatal_credentials(conn) == []


def test_recheck_recovers_after_failed_probe_once_interval_elapses(kanban_home, monkeypatch):
    """A failed recheck stays parked and throttled, then a later healthy probe requeues once."""
    clock = [1_000]
    calls: list[str] = []
    responses = iter((
        {"ok": False, "provider": "openrouter", "reason": "credential_rejected"},
        {"ok": False, "provider": "openrouter", "reason": "credential_rejected"},
        {"ok": True, "provider": "openrouter"},
    ))
    monkeypatch.setattr(kbd.time, "time", lambda: clock[0])
    monkeypatch.setattr(kbd, "_config_fatal_recheck_interval_seconds", lambda: 5)
    monkeypatch.setattr(
        kbd, "_run_credential_probe",
        lambda profile, timeout=20.0: (calls.append(profile), next(responses))[1],
    )
    with kbc.connect() as conn:
        failed = kb.create_task(conn, title="failed", assignee="broken-profile")
        _dead_worker_with_log(conn, failed, 72001, 1, cause="provider_config")
        kbd.detect_crashed_workers(conn)

        # The initial config-fatal trip itself is the first probe. At the interval boundary,
        # the next failed probe records a recheck without unblocking the task.
        clock[0] += 5
        assert kbd._recheck_config_fatal_credentials(conn) == []

        task = kb.get_task(conn, failed)
        assert task is not None
        assert task.status == "blocked"
        last_event = kb.list_events(conn, failed)[-1]
        assert last_event.kind == "config_fatal_recheck"
        assert last_event.payload["ok"] is False

        # A dispatcher tick inside the interval must not probe again.
        clock[0] += 1
        assert kbd._recheck_config_fatal_credentials(conn) == []
        assert calls == ["broken-profile", "broken-profile"]

        # Once the recheck interval elapses and the credential heals, restore precisely once.
        clock[0] += 4
        assert kbd._recheck_config_fatal_credentials(conn) == [failed]
        assert kb.get_task(conn, failed).status == "ready"
        assert [event.kind for event in kb.list_events(conn, failed)].count(
            "config_fatal_recovered",
        ) == 1

        clock[0] += 5
        assert kbd._recheck_config_fatal_credentials(conn) == []
        assert calls == ["broken-profile", "broken-profile", "broken-profile"]


def test_recheck_throttles_repeat_probes_within_interval(kanban_home, monkeypatch):
    """Never hammer the provider every dispatcher tick: inside the recheck interval a
    second tick must not re-invoke the credential probe for the same parked profile."""
    monkeypatch.setattr(kbd, "_config_fatal_recheck_interval_seconds", lambda: 5)
    calls: list[str] = []
    monkeypatch.setattr(
        kbd, "_run_credential_probe",
        lambda profile, timeout=20.0: (calls.append(profile), {"ok": False, "provider": "openrouter"})[1],
    )
    with kbc.connect() as conn:
        failed = kb.create_task(conn, title="failed", assignee="broken-profile")
        _dead_worker_with_log(conn, failed, 72001, 1, cause="provider_config")
        kbd.detect_crashed_workers(conn)
        calls.clear()
        # Backdate the trip event past the interval floor so the first recheck below is
        # eligible to probe; without this the trip's own naming probe would already have
        # consumed the throttle window and the test couldn't tell "ran once" from "never ran".
        conn.execute("UPDATE task_events SET created_at = created_at - 10 WHERE kind = 'config_fatal'")
        conn.commit()

        kbd._recheck_config_fatal_credentials(conn)
        assert calls == ["broken-profile"]

        kbd._recheck_config_fatal_credentials(conn)
        assert calls == ["broken-profile"]  # second tick inside the floor: no extra call


def test_non_credential_failures_are_untouched_by_recheck(kanban_home, monkeypatch):
    """A card blocked for an ordinary (non-config_fatal) reason is never selected by the
    credential-recovery scan and never triggers a credential probe for its profile."""
    def _must_not_run(profile, timeout=20.0):
        raise AssertionError("credential probe must not run for a non-config_fatal block")
    monkeypatch.setattr(kbd, "_run_credential_probe", _must_not_run)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ordinary", assignee="healthy-profile")
        for i in range(3):
            kbd._record_task_failure(
                conn, tid, error=f"boom{i}", outcome="crashed", failure_limit=2,
                release_claim=False, end_run=False,
            )
        assert kb.get_task(conn, tid).status == "blocked"

        recovered = kbd._recheck_config_fatal_credentials(conn)

        assert recovered == []
        assert kb.get_task(conn, tid).status == "blocked"


def test_exit_single_query_writes_trailer_only_for_kanban_workers(monkeypatch, capsys):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with pytest.raises(SystemExit) as exc:
        exit_single_query(1)
    assert exc.value.code == 1
    assert KANBAN_WORKER_EXIT_TRAILER not in capsys.readouterr().err

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    with pytest.raises(SystemExit) as exc:
        exit_single_query(kb.KANBAN_RATE_LIMIT_EXIT_CODE, kanban_cause="provider_config")
    assert exc.value.code == kb.KANBAN_RATE_LIMIT_EXIT_CODE
    assert (
        f"{KANBAN_WORKER_EXIT_TRAILER}{kb.KANBAN_RATE_LIMIT_EXIT_CODE} cause=provider_config"
        in capsys.readouterr().err
    )
