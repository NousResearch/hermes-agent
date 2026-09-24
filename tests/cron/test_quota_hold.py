"""Provider quota windows park a cron job instead of re-firing into them (#89376).

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
