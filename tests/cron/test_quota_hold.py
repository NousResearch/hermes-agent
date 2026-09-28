"""Provider quota windows park a cron job instead of re-firing into them (#89376;
ai-velho-oy/chairman#3).

Contract (cron/quota_hold.py): a failed run whose cause is a rate-limited ``AuthError`` with a
``retry after <N>s`` hint parks a recurring job's ``next_run_at`` past the window and stamps
``quota_hold_until``; the stale-error re-arm leaves a held job alone; a run that reaches the
model clears the marker. A mid-run 429 reaches the same place through the verdict the scheduler
stamps on the error it raises for a failed turn result. The hint is read only from the
AuthError / the stamped 429 in the cause chain, never from arbitrary failure text, and the
parsed wait is clamped to the longest window a provider really names.
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import cron.scheduler as sched
from cron import quota_hold as qh
from cron.jobs import (
    _job_is_stale_error_recurring, create_job, get_due_jobs, get_job, mark_job_run, update_job,
)
from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, AuthError

QUOTA_MSG = "Codex provider quota exhausted (429); retry after 123518s. Credentials are still valid."

# A provider resolution the scheduler accepts, so the tick reaches the agent (the tests that
# need a run rather than a pre-flight failure).
_RUNTIME_OK = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


def _resolve_ok(**_kw):
    return dict(_RUNTIME_OK)


def _stamped_429(text: str, *, retry_after=None) -> RuntimeError:
    """A failure the scheduler raises for a rate-limited run: bare RuntimeError + stamped context."""
    exc = RuntimeError(text)
    exc.status_code = 429
    if retry_after is not None:
        exc.retry_after = retry_after
    return exc


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _quota_error() -> AuthError:
    return AuthError(QUOTA_MSG, provider="openai-codex", code=CODEX_RATE_LIMITED_CODE)


def test_hold_seconds_only_from_rate_limited_auth_error_in_cause_chain():
    """The scheduler wraps the resolve failure in a RuntimeError ``from`` the AuthError; the
    hint survives through the cause chain, and text alone (or a re-login AuthError) never
    parks a job."""
    try:
        raise RuntimeError(QUOTA_MSG) from _quota_error()
    except RuntimeError as wrapped:
        assert qh.hold_seconds_from_failure(wrapped) == 123518.0

    assert qh.hold_seconds_from_failure(RuntimeError(QUOTA_MSG)) is None
    assert qh.hold_seconds_from_failure(RuntimeError("HTTP 429: retry after 60s")) is None
    relogin = AuthError(QUOTA_MSG, provider="openai-codex", code="expired", relogin_required=True)
    assert qh.hold_seconds_from_failure(relogin) is None
    structured = AuthError("quota", code=CODEX_RATE_LIMITED_CODE, retry_after=900)
    assert qh.hold_seconds_from_failure(structured) == 900.0


def test_weekly_cron_retries_when_quota_recovers_before_next_occurrence(
    tmp_cron_home, monkeypatch,
):
    """A weekly fire blocked by a shorter quota window retries when the provider reopens;
    it is not silently deferred until the following week's natural occurrence."""
    now = datetime(2026, 9, 18, 12, 1, tzinfo=timezone.utc)
    natural_next = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    job = create_job("weekly digest", "0 12 * * 5")
    assert datetime.fromisoformat(job["next_run_at"]) == natural_next

    assert mark_job_run(
        job["id"], False, QUOTA_MSG, quota_hold_seconds=20 * 60 * 60,
        recover_consumed_fire=True,
    )

    held = get_job(job["id"])
    assert held is not None
    retry_at = datetime.fromisoformat(held["next_run_at"])
    assert retry_at == now + timedelta(hours=20, seconds=qh.HOLD_SLACK_SECONDS) < natural_next
    assert qh.is_recovery_fire(held, held["next_run_at"])
    edited = {**held, "schedule": {"kind": "cron", "expr": "0 9 * * *"}}
    assert not qh.is_recovery_fire(edited, edited["next_run_at"])

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: retry_at + timedelta(seconds=1))
    assert job["id"] in {due["id"] for due in get_due_jobs()}


def test_recovery_fire_skips_dense_schedules_and_never_re_parks(monkeypatch):
    """Recovery is one attempt for sparse schedules only: an hourly job whose hold ends two
    minutes before :00 keeps its natural :00 (no off-lattice near-duplicate), and a job that is
    already the recovery fire (carries quota_hold_until) failing again is not re-parked."""
    now = datetime(2026, 9, 18, 12, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)

    dense = {
        "schedule": {"kind": "cron", "expr": "0 * * * *"},
        "next_run_at": datetime(2026, 9, 18, 13, 0, tzinfo=timezone.utc).isoformat(),
    }
    natural = dense["next_run_at"]
    assert not qh.plan_hold(dense, hold_seconds=57 * 60 - qh.HOLD_SLACK_SECONDS,
                            recover_consumed_fire=True)
    assert dense["next_run_at"] == natural
    assert qh.STATE_KEY not in dense

    held_again = {
        "schedule": {"kind": "cron", "expr": "0 12 * * 5"},
        "next_run_at": (now + timedelta(days=7)).isoformat(),
        qh.STATE_KEY: now.isoformat(),
        qh.SCHEDULE_EXPR_KEY: "0 12 * * 5",
    }
    natural = held_again["next_run_at"]
    assert not qh.plan_hold(held_again, hold_seconds=20 * 60 * 60, recover_consumed_fire=True)
    assert held_again["next_run_at"] == natural
    assert qh.STATE_KEY not in held_again


def _raise_quota(**_kw):
    raise _quota_error()


def _tick(job, home, deliveries, resolve, *, result=None):
    """One real scheduler tick (preflight ON) with the provider resolver replaced by *resolve*.

    *result* is handed back by ``run_conversation`` (a failed turn result, say); without it the
    agent call raises, which is the pre-flight-failure shape.
    """
    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
         patch.object(sched, "_deliver_result",
                      side_effect=lambda jb, content, **kw: deliveries.append(content)), \
         patch("run_agent.AIAgent") as agent_cls:
        if result is None:
            agent_cls.return_value.run_conversation.side_effect = RuntimeError("model said no")
        else:
            agent_cls.return_value.run_conversation.return_value = dict(result)
        sched.run_one_job(dict(job))


def test_quota_hold_parks_past_window_survives_stale_rearm_and_clears_on_model_reach(tmp_cron_home):
    """A 30-minute job whose provider resolve raises the Codex quota AuthError ('retry after
    123518s') is parked by the real scheduler tick: preflight lets the rate-limited AuthError
    through (it is not a missing credential), the one delivered alert carries the hold notice,
    and the job does not fire again inside the window (not even after the stale-error re-arm's
    cadence+grace). The marker clears once a run reaches the model."""
    job = create_job("portfolio triage", "every 30m", deliver="local")
    job_id = job["id"]
    now = datetime.now(timezone.utc)
    deliveries: list = []

    _tick(get_job(job_id), tmp_cron_home, deliveries, _raise_quota)
    j = get_job(job_id)
    assert j["last_status"] == "error"
    assert len(deliveries) == 1, deliveries
    parked = datetime.fromisoformat(j["next_run_at"])
    assert parked - now >= timedelta(seconds=123518), "next_run_at must land past the window"
    assert j[qh.STATE_KEY] == j["next_run_at"]
    assert "_quota_hold_seconds" not in j

    # Two hours later the job looks like a wedged stale-error record (#62002) — the hold says
    # it is parked on purpose, so it is neither re-armed nor due.
    update_job(job_id, {"last_run_at": (now - timedelta(hours=2)).isoformat()})
    j = get_job(job_id)
    assert not _job_is_stale_error_recurring(j, j["schedule"], now)
    assert all(d["id"] != job_id for d in get_due_jobs())
    assert datetime.fromisoformat(get_job(job_id)["next_run_at"]) == parked

    # A run that reached the model (either outcome) clears the marker.
    assert mark_job_run(job_id, False, "RuntimeError: model said no")
    j = get_job(job_id)
    assert qh.STATE_KEY not in j
    assert datetime.fromisoformat(j["next_run_at"]) - now < timedelta(hours=1)

    # Editing the schedule recomputes next_run_at from the new cadence; the stale marker must
    # not linger on a record that is no longer parked where it says.
    assert mark_job_run(job_id, False, QUOTA_MSG, quota_hold_seconds=123518)
    assert qh.STATE_KEY in get_job(job_id)
    j = update_job(job_id, {"schedule": "every 15m"})
    assert qh.STATE_KEY not in j
    assert datetime.fromisoformat(j["next_run_at"]) - now < timedelta(hours=1)


def test_failed_result_stamps_its_rate_limit_window_on_the_error_the_scheduler_raises():
    """A mid-run 429 is reported through the turn result, not as an exception: the scheduler
    raises a bare ``RuntimeError`` for a failed run, so the classifier's verdict and the window
    the provider named are carried onto that error. Without them ``hold_seconds_from_failure``
    never saw a 429 at all and a window-naming 429 was re-fired on every cadence tick."""
    resets_at = time.time() + 46 * 60
    with pytest.raises(RuntimeError) as caught:
        sched._final_response_from_result(
            {"failed": True, "completed": False,
             "error": "HTTP 429: Your usage window refills in 46 minutes.",
             "failure_reason": "rate_limit", "failure_retryable": True,
             "failure_resets_at": resets_at},
            "job-1", "job one", None)

    assert caught.value.status_code == 429
    assert caught.value.retry_after == pytest.approx(46 * 60, abs=5)
    assert qh.hold_seconds_from_failure(caught.value) == pytest.approx(46 * 60, abs=5)

    # A failure with no rate-limit verdict carries no status: the error TEXT alone never parks a
    # job, so an unrelated "retry after" in an agent's output cannot hold one.
    with pytest.raises(RuntimeError) as other:
        sched._final_response_from_result(
            {"failed": True, "completed": False, "error": "HTTP 429: retry after 600s",
             "failure_reason": "unknown"},
            "job-1", "job one", None)
    assert not hasattr(other.value, "status_code")
    assert qh.hold_seconds_from_failure(other.value) is None


def test_mid_run_429_result_parks_the_job_through_the_real_scheduler_tick(tmp_cron_home):
    """End to end: a failed turn result whose verdict is a rate limit parks a sub-hourly job past
    the window the provider named, instead of re-firing it into the wall every 30 minutes."""
    job = create_job("portfolio triage", "every 30m", deliver="local")
    job_id = job["id"]
    now = datetime.now(timezone.utc)
    deliveries: list = []

    _tick(get_job(job_id), tmp_cron_home, deliveries, _resolve_ok, result={
        "failed": True, "completed": False, "final_response": "",
        "error": "HTTP 429: Your usage window refills in 46 minutes.",
        "failure_reason": "rate_limit", "failure_retryable": True,
        "failure_resets_at": (now + timedelta(minutes=46)).timestamp(),
    })

    j = get_job(job_id)
    assert j["last_status"] == "error"
    parked = datetime.fromisoformat(j["next_run_at"])
    assert parked - now >= timedelta(minutes=46), "next_run_at must land past the window the 429 named"
    assert parked - now <= timedelta(minutes=48)
    assert j[qh.STATE_KEY] == j["next_run_at"]
    assert "model said no" not in (j.get("last_error") or "")


def test_pathological_window_is_clamped_to_the_longest_real_window(caplog):
    """A sentinel in a stringified ``resets_in_seconds`` field must not park a job for years: the
    parsed wait is clamped to the longest window a provider really names (a week), and a
    structured hint is bounded by the same ceiling."""
    from agent.retry_utils import min_reset_delay_from_message

    body = "HTTP 429: usage window closed: {'resets_in_seconds': 999999999999}"
    assert min_reset_delay_from_message(body) == pytest.approx(999999999999)

    with caplog.at_level("WARNING"):
        assert qh.hold_seconds_from_failure(_stamped_429(body)) == qh.MAX_HOLD_SECONDS
    assert "clamping the hold" in caplog.text

    assert qh.hold_seconds_from_failure(
        _stamped_429("HTTP 429: quota exhausted", retry_after=10 ** 12)) == qh.MAX_HOLD_SECONDS
    assert qh.hold_seconds_from_failure(
        AuthError("quota", code=CODEX_RATE_LIMITED_CODE, retry_after=10 ** 12)) == qh.MAX_HOLD_SECONDS
    # a real long window is untouched — the Codex weekly quota hint is ~34h
    assert qh.hold_seconds_from_failure(
        _stamped_429("HTTP 429: quota exhausted", retry_after=123518)) == 123518.0
