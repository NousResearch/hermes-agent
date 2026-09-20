"""Provider quota windows park a cron job instead of re-firing into them (#89376;
ai-velho-oy/chairman#3).

Contract (cron/quota_hold.py): a failed run whose cause is a rate-limited ``AuthError`` with a
``retry after <N>s`` hint parks a recurring job's ``next_run_at`` past the window and stamps
``quota_hold_until``; the stale-error re-arm leaves a held job alone; a run that reaches the
model clears the marker. The hint is read only from the AuthError in the cause chain, never from
arbitrary failure text.
"""

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
        quota_recover_occurrence=True,
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


def test_hold_policy_coalesces_dense_cron_without_accelerating_sparse_interval(monkeypatch):
    now = datetime(2026, 9, 18, 12, 1, tzinfo=timezone.utc)
    window_end = now + timedelta(hours=20, seconds=qh.HOLD_SLACK_SECONDS)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)

    dense_cron = {
        "schedule": {"kind": "cron", "expr": "0 * * * *"},
        "next_run_at": (now + timedelta(hours=1)).isoformat(),
    }
    assert qh.plan_hold(dense_cron, hold_seconds=20 * 60 * 60)
    assert datetime.fromisoformat(dense_cron["next_run_at"]) > window_end

    sparse_interval = {
        "schedule": {"kind": "interval", "minutes": 7 * 24 * 60},
        "next_run_at": (now + timedelta(days=7)).isoformat(),
    }
    original_next = sparse_interval["next_run_at"]
    assert not qh.plan_hold(sparse_interval, hold_seconds=20 * 60 * 60)
    assert sparse_interval["next_run_at"] == original_next
    assert qh.STATE_KEY not in sparse_interval

    manual_sparse_cron = {
        "schedule": {"kind": "cron", "expr": "0 12 * * 5"},
        "next_run_at": (now + timedelta(days=7)).isoformat(),
    }
    assert not qh.plan_hold(manual_sparse_cron, hold_seconds=20 * 60 * 60)
    assert qh.STATE_KEY not in manual_sparse_cron


def _raise_quota(**_kw):
    raise _quota_error()


def _tick(job, home, deliveries, resolve):
    """One real scheduler tick (preflight ON) with the provider resolver replaced by *resolve*."""
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
        agent_cls.return_value.run_conversation.side_effect = RuntimeError("model said no")
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
    assert len(deliveries) == 1 and "This job is held" in deliveries[0], deliveries
    assert "provider credential missing" not in deliveries[0]
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
