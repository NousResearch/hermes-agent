"""Single automatic re-run for cron fires aborted by the stale-call circuit breaker.

Contract (cron/stale_abort_retry.py): a recurring job whose run FAILED on the exact
``_check_stale_giveup`` signature ("… consecutive stale attempts … aborting this call")
with zero completed model calls gets exactly ONE requeue at a jittered +30–60 min.
The retry run that re-aborts is not requeued again (no cascade) and the next natural
occurrence gets its own single requeue; any other RuntimeError, a paused job, and a
natural occurrence landing within the requeue window all keep the schedule; a run that
reaches the model (either outcome) clears the marker.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from cron import stale_abort_retry as sar
from cron.jobs import create_job, get_due_jobs, get_job, mark_job_run, update_job

# The exact _check_stale_giveup abort text (agent/chat_completion_helpers.py).
STALE_ABORT_MSG = (
    "Provider has been unresponsive (no response received) for 5 consecutive stale "
    "attempts — aborting this call to avoid an indefinite stall. Switch models or "
    "start a new session, then retry."
)


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _abort() -> RuntimeError:
    return RuntimeError(STALE_ABORT_MSG)


def test_stale_abort_requeues_once_in_window_then_no_cascade(tmp_cron_home):
    """(a) exact signature → exactly 1 requeue in [30, 60] min; (b) the retry run
    re-aborting the same way → 0 requeues, the natural schedule stands; (c) the next
    natural occurrence aborting gets its own single requeue."""
    # Interval, not a cron expression: the natural next fire is always a full day out,
    # so the requeue instant can never collide with it (mirrors test_unreachable_retry).
    job = create_job("weekly digest", "every 24h")
    job_id = job["id"]
    now = datetime.now(UTC)

    # (a) First stale abort: exactly one requeue, inside the approved window.
    assert mark_job_run(job_id, False, STALE_ABORT_MSG, stale_abort=True)
    j = get_job(job_id)
    nxt = datetime.fromisoformat(j["next_run_at"])
    assert timedelta(seconds=sar.RETRY_MIN_SECONDS) <= nxt - now <= timedelta(
        seconds=sar.RETRY_MAX_SECONDS + 120), f"requeue {nxt - now} outside [30,60]min"
    assert j[sar.STATE_KEY]["attempt"] == 1
    assert j["last_status"] == "error"
    # The requeued job is due again at the requeue instant (it IS its next occurrence).
    assert all(d["id"] != job_id for d in get_due_jobs())

    # (b) The retry run re-aborts on the same signature: NO second requeue (no cascade)
    # — the job falls back to its natural schedule.
    assert mark_job_run(job_id, False, STALE_ABORT_MSG, stale_abort=True)
    j = get_job(job_id)
    assert sar.STATE_KEY not in j
    assert datetime.fromisoformat(j["next_run_at"]) - now > timedelta(hours=23)

    # (c) The next natural occurrence aborting again is a new occurrence: one fresh requeue.
    assert mark_job_run(job_id, False, STALE_ABORT_MSG, stale_abort=True)
    j = get_job(job_id)
    assert j[sar.STATE_KEY]["attempt"] == 1
    assert datetime.fromisoformat(j["next_run_at"]) - now <= timedelta(
        seconds=sar.RETRY_MAX_SECONDS + 120)


def test_other_runtime_error_never_requeues(tmp_cron_home):
    """(c) any other RuntimeError — even one echoing a fragment of the signature —
    never requeues and clears any stale marker."""
    job = create_job("hourly job", "every 24h")
    job_id = job["id"]

    assert mark_job_run(job_id, False, "RuntimeError: provider exploded (5 consecutive stale attempts)")
    j = get_job(job_id)
    assert sar.STATE_KEY not in j
    assert datetime.fromisoformat(j["next_run_at"]) - datetime.now(UTC) > timedelta(hours=23)

    # A normal failure (model reached) also spends a pending requeue.
    assert mark_job_run(job_id, False, STALE_ABORT_MSG, stale_abort=True)
    assert sar.STATE_KEY in get_job(job_id)
    assert mark_job_run(job_id, False, "agent tool failed")
    assert sar.STATE_KEY not in get_job(job_id)


def test_paused_or_imminent_next_run_never_requeues(tmp_cron_home):
    """(d) paused jobs and jobs whose natural occurrence comes within the requeue
    window are left exactly where the schedule put them."""
    paused = create_job("paused digest", "every 24h", paused=True, paused_reason="ops")
    assert mark_job_run(paused["id"], False, STALE_ABORT_MSG, stale_abort=True)
    j = get_job(paused["id"])
    assert sar.STATE_KEY not in j
    assert j["state"] == "paused"

    # A sub-hourly cadence: the natural occurrence (+30m) lands inside the requeue
    # window, so plan_retry refuses — the schedule itself is the retry (no double
    # digest on frequent jobs).
    soon = create_job("frequent job", "every 30m")
    assert mark_job_run(soon["id"], False, STALE_ABORT_MSG, stale_abort=True)
    j = get_job(soon["id"])
    gap = datetime.fromisoformat(j["next_run_at"]) - datetime.now(UTC)
    assert timedelta(0) < gap <= timedelta(minutes=45), f"natural slot moved: {gap}"
    assert sar.STATE_KEY not in j
    assert j["state"] == "scheduled"


def test_reaching_the_model_clears_marker_and_onehots_never_requeue(tmp_cron_home):
    """A successful (or any model-reaching) run clears the marker; one-shots are
    excluded (at-most-times dispatch accounting, #38758)."""
    job = create_job("digest", "every 24h")
    assert mark_job_run(job["id"], False, STALE_ABORT_MSG, stale_abort=True)
    assert sar.STATE_KEY in get_job(job["id"])
    assert mark_job_run(job["id"], True, None)
    assert sar.STATE_KEY not in get_job(job["id"])
    assert get_job(job["id"])["last_status"] == "ok"

    once = create_job("one shot", datetime.now(UTC).isoformat())
    assert mark_job_run(once["id"], False, STALE_ABORT_MSG, stale_abort=True)
    remaining = get_job(once["id"])
    assert remaining is None or sar.STATE_KEY not in remaining


def test_signature_is_strict_type_and_both_fragments():
    """Only a bare RuntimeError carrying BOTH fragments matches; subclasses, other
    types, and payload echoes do not. Zero completed model calls required."""
    assert sar.is_stale_abort_failure(_abort()) is True

    class _StaleSub(RuntimeError):
        pass

    assert sar.is_stale_abort_failure(_StaleSub(STALE_ABORT_MSG)) is False
    assert sar.is_stale_abort_failure(Exception(STALE_ABORT_MSG)) is False
    assert sar.is_stale_abort_failure(RuntimeError("consecutive stale attempts only")) is False
    assert sar.is_stale_abort_failure(RuntimeError("aborting this call only")) is False
    assert sar.is_stale_abort_failure(RuntimeError("")) is False

    class _Agent:
        session_api_calls = 3

    # The final call stalled after a full conversation ran: normal failure, alerted.
    assert sar.is_stale_abort_failure(_abort(), agent=_Agent()) is False


def test_disabled_config_stands_down(tmp_cron_home):
    """``cron.retry_stale_abort: false`` stands the whole mechanism down."""
    with patch("hermes_cli.config.load_config", return_value={"cron": {"retry_stale_abort": False}}):
        assert sar.retry_enabled() is False
        job = create_job("digest", "every 24h")
        assert sar.will_retry(job) is False
        assert mark_job_run(job["id"], False, STALE_ABORT_MSG, stale_abort=True)
        j = get_job(job["id"])
        assert sar.STATE_KEY not in j
        assert datetime.fromisoformat(j["next_run_at"]) - datetime.now(UTC) > timedelta(hours=23)


def test_will_retry_gates_notice_suppression(tmp_cron_home):
    """will_retry (the scheduler's notice-suppression gate) is True exactly when plan_retry
    will park a re-run: False for a re-abort, a paused job, a disabled config, and a cadence
    whose natural occurrence lands inside the re-run window (that failure must alert)."""
    job = create_job("digest", "every 24h")
    assert sar.will_retry(job) is True

    assert mark_job_run(job["id"], False, STALE_ABORT_MSG, stale_abort=True)
    pending = get_job(job["id"])
    assert sar.will_retry(pending) is False  # marker set → this IS the retry run

    fresh = get_job(create_job("other digest", "every 24h")["id"])
    fresh["state"] = "paused"
    assert sar.will_retry(fresh) is False
    fresh["state"] = "scheduled"
    with patch("hermes_cli.config.load_config", return_value={"cron": {"retry_stale_abort": False}}):
        assert sar.will_retry(fresh) is False

    # Sub-hourly cadence: plan_retry refuses, so the notice must not be held either —
    # otherwise every failure of a frequent job would be swallowed while the provider is down.
    frequent = create_job("frequent job", "every 30m")
    assert sar.will_retry(frequent) is False


def test_update_job_edit_clears_marker(tmp_cron_home):
    """Rescheduling a job recomputes next_run_at; the +30-60min pull must not survive
    as a stale artifact on a record whose schedule changed under it."""
    job = create_job("digest", "every 24h")
    assert mark_job_run(job["id"], False, STALE_ABORT_MSG, stale_abort=True)
    assert sar.STATE_KEY in get_job(job["id"])
    j = update_job(job["id"], {"schedule": "every 12h"})
    assert sar.STATE_KEY not in j
    assert datetime.fromisoformat(j["next_run_at"]) - datetime.now(UTC) <= timedelta(hours=12)


def test_cron_expression_requeue_fires_off_lattice_instead_of_reanchoring(tmp_cron_home, monkeypatch):
    """The requeued instant is OFF the cron lattice. It must fire once at the
    requeue instant (recognised by _reanchor_stale_cron via is_retry_fire), not be re-anchored
    to the next natural occurrence without firing (which silently lost the retry for every
    cron-expression job). A direct jobs.json expr edit while parked still re-anchors."""
    from cron.jobs import load_jobs, save_jobs

    pinned = datetime(2026, 9, 18, 12, 17, tzinfo=UTC)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: pinned)
    monkeypatch.setattr(sar, "_hermes_now", lambda: pinned)
    weekly = create_job("weekly digest", "0 12 * * 5")
    assert mark_job_run(weekly["id"], False, STALE_ABORT_MSG, stale_abort=True)
    j = get_job(weekly["id"])
    retry_at = datetime.fromisoformat(j["next_run_at"])
    assert timedelta(seconds=sar.RETRY_MIN_SECONDS) <= retry_at - pinned <= timedelta(seconds=sar.RETRY_MAX_SECONDS)
    assert j[sar.STATE_KEY]["expr"] == "0 12 * * 5"
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: retry_at + timedelta(seconds=1))
    assert weekly["id"] in {due["id"] for due in get_due_jobs()}

    # Direct expression edit while parked: no inherited exception, re-anchor without firing.
    jobs = load_jobs()
    next(x for x in jobs if x["id"] == weekly["id"])["schedule"]["expr"] = "0 9 * * 1"
    save_jobs(jobs)
    assert weekly["id"] not in {due["id"] for due in get_due_jobs()}


def test_rerun_through_tick_and_bookkeeping_does_not_spend_repeat_budget(tmp_cron_home, monkeypatch):
    """End to end through the tick's claim hand-off and the bookkeeping tail: the flagged
    occurrence parks one re-run, the re-run fires at that instant, does not count toward
    ``repeat`` again, and its second abort falls back to the natural schedule."""
    import cron.scheduler as sched
    from cron.scheduler_bookkeeping import finish_completed_run

    clock = [datetime.now(UTC)]
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: clock[0])
    monkeypatch.setattr(sar, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(sched, "finish_execution", lambda *_a, **_kw: None)
    runs = []
    monkeypatch.setattr(sched, "run_one_job", lambda job, **_kw: runs.append(job) or True)
    job_id = create_job("digest", "every 24h", repeat=3)["id"]

    held = []
    for attempt in range(2):  # the occurrence, then its single re-run
        clock[0] = datetime.fromisoformat(get_job(job_id)["next_run_at"]) + timedelta(seconds=1)
        due = next(d for d in get_due_jobs() if d["id"] == job_id)
        assert sched._process_due_job(dict(due, execution_id=f"exec-{attempt}"), None, None, False)
        run = runs[-1]
        run["_stale_abort"] = True
        held.append(sar.will_retry(run))
        assert finish_completed_run(
            sched._RunDelivery(job=run, success=False, error=STALE_ABORT_MSG),
            run["fire_claim"]["by"], f"exec-{attempt}")
        j = get_job(job_id)
        assert j["repeat"]["completed"] == 1, "only the occurrence itself counts"
        if attempt == 0:
            gap = datetime.fromisoformat(j["next_run_at"]) - clock[0]
            assert timedelta(seconds=sar.RETRY_MIN_SECONDS) <= gap <= timedelta(
                seconds=sar.RETRY_MAX_SECONDS)

    j = get_job(job_id)
    assert sar.STATE_KEY not in j and j["state"] == "scheduled"
    assert datetime.fromisoformat(j["next_run_at"]) - clock[0] > timedelta(hours=23)
    # The first abort's notice is held for the re-run; the re-run's own abort alerts.
    assert held == [True, False]
