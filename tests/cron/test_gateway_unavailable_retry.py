"""One automatic re-run for a cron fire the gateway refused before the agent ran.

Contract (cron/unreachable_retry.py): a recurring job whose run fails with
``GatewayUnavailableError`` (owner draining for an update/restart, still starting, or
absent) before ANY model call gets exactly ONE automatic re-run,
``GATEWAY_RETRY_DELAYS_SECONDS`` out, then the schedule's own next occurrence. A run that
reaches the model resets it, and the network ladder keeps its own three rungs.
"""

from datetime import UTC, datetime, timedelta

import pytest

from cron import unreachable_retry as ur
from cron.jobs import create_job, get_job, mark_job_run

REFUSAL = "GatewayUnavailableError: Gateway draining: update_paused"


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class _Agent:
    """Stand-in for the cron agent: only the API-call counter matters to the classifier."""

    def __init__(self, calls: int = 0):
        self.session_api_calls = calls


def test_a_refusal_with_zero_api_calls_classifies_and_nothing_else_does(tmp_cron_home):
    from hermes_cli.gateway_client import GatewayUnavailableError

    refused = GatewayUnavailableError(
        "Gateway draining: update_paused", "draining", "update_paused")
    assert ur.is_gateway_unavailable_failure(refused, _Agent(0))
    assert ur.is_gateway_unavailable_failure(refused, None), "no agent yet: still free to re-run"
    assert not ur.is_gateway_unavailable_failure(refused, _Agent(2)), (
        "the agent already ran: a re-run could double a side effect")
    assert not ur.is_gateway_unavailable_failure(RuntimeError("boom"), _Agent(0))
    # The scheduler wraps transport errors; a refusal anywhere in the chain still counts.
    wrapped = RuntimeError("fire refused")
    wrapped.__cause__ = refused
    assert ur.is_gateway_unavailable_failure(wrapped, _Agent(0))


def test_canonical_dial_refusal_marks_retry_without_an_admission(tmp_cron_home, monkeypatch):
    """A gateway drain on the canonical dial must reach the retry bookkeeping."""
    from contextlib import asynccontextmanager

    from cron.scheduler import run_job
    from cron.scheduler_authority import journal_path
    from gateway import session_cron
    from hermes_cli import gateway_client

    monkeypatch.setattr(session_cron, "owner_for_home", lambda _home: None)

    @asynccontextmanager
    async def refused_gateway():
        raise gateway_client.GatewayUnavailableError(
            "Gateway draining: update_paused", "draining", "update_paused")
        yield  # unreachable: the owner refused before admission

    monkeypatch.setattr(gateway_client, "connect_gateway", refused_gateway)
    job = get_job(create_job("calendar mirror", "every 30m")["id"])
    execution_id = "draining-before-admission"
    success, _output, _answer, error = run_job(job, execution_id=execution_id)

    assert success is False and error.startswith("GatewayUnavailableError:")
    assert not journal_path(job["id"], execution_id).exists(), "no submit was attempted"
    assert job.get("_gateway_unavailable") is True, (
        "canonical run_job returns a failure tuple before its inline exception classifier")


def test_a_refusal_re_arms_once_then_yields_to_the_schedule(tmp_cron_home):
    job = create_job("calendar mirror", "every 30m")
    job_id = job["id"]
    now = datetime.now(UTC)

    assert mark_job_run(job_id, False, REFUSAL, gateway_unavailable=True)
    j = get_job(job_id)
    delay = ur.GATEWAY_RETRY_DELAYS_SECONDS[0]
    parked = datetime.fromisoformat(j["next_run_at"])
    assert timedelta(0) < parked - now <= timedelta(seconds=delay + 120), (
        "the single rung is a couple of minutes out, not a full period")
    assert j[ur.STATE_KEY]["attempt"] == 1
    assert j[ur.STATE_KEY]["kind"] == ur.KIND_GATEWAY
    assert ur.is_retry_fire(j, j["next_run_at"]), "the parked instant must be recognized as a re-run"
    from cron.occurrences import scheduled_instant
    assert ur.is_retry_run(dict(j, _scheduled_instant=scheduled_instant(j["next_run_at"]))), (
        "a re-run must not spend another repeat slot")

    # One shot: a refusal on the re-run leaves the natural 30-minute schedule.
    assert mark_job_run(job_id, False, REFUSAL, gateway_unavailable=True)
    j = get_job(job_id)
    assert j.get(ur.STATE_KEY) is None
    assert datetime.fromisoformat(j["next_run_at"]) - now > timedelta(minutes=25)


def test_switching_failure_class_starts_that_ladder_at_its_first_rung(tmp_cron_home):
    job = create_job("nightly digest", "every 24h")
    job_id = job["id"]
    assert mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
    assert get_job(job_id)[ur.STATE_KEY]["kind"] == ur.KIND_MODEL

    now = datetime.now(UTC)
    assert mark_job_run(job_id, False, REFUSAL, gateway_unavailable=True)
    j = get_job(job_id)
    assert j[ur.STATE_KEY]["kind"] == ur.KIND_GATEWAY
    assert j[ur.STATE_KEY]["attempt"] == 1, "a different failure class must not inherit the count"
    assert datetime.fromisoformat(j["next_run_at"]) - now <= timedelta(
        seconds=ur.GATEWAY_RETRY_DELAYS_SECONDS[0] + 120)


def test_notice_hold_and_config_switch_follow_the_gateway_ladder(tmp_cron_home, monkeypatch):
    fast = create_job("minute poll", "every 1m")
    snapshot = get_job(fast["id"])  # what the scheduler holds while composing the notice
    assert mark_job_run(fast["id"], False, REFUSAL, gateway_unavailable=True)
    jf = get_job(fast["id"])
    assert jf.get(ur.STATE_KEY) is None, "the 1-minute cadence beats the 2-minute rung"
    assert ur.will_retry(snapshot, gateway=True) is False, (
        "no re-run scheduled: the notice must go out")

    slow = create_job("calendar mirror", "every 30m")
    snapshot = get_job(slow["id"])
    assert mark_job_run(slow["id"], False, REFUSAL, gateway_unavailable=True)
    assert ur.will_retry(snapshot, gateway=True) is True, (
        "the 2-minute rung beats the 30-minute cadence: hold the notice")

    off = create_job("calendar mirror paused knob", "every 30m")
    monkeypatch.setattr(ur, "gateway_retry_enabled", lambda cfg=None: False)
    assert mark_job_run(off["id"], False, REFUSAL, gateway_unavailable=True)
    assert get_job(off["id"]).get(ur.STATE_KEY) is None


def test_network_ladder_still_has_all_three_rungs(tmp_cron_home):
    job = create_job("nightly report", "every 24h")
    job_id = job["id"]
    for attempt, delay in enumerate(ur.RETRY_DELAYS_SECONDS, start=1):
        assert mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
        state = get_job(job_id)[ur.STATE_KEY]
        assert state["attempt"] == attempt and state["kind"] == ur.KIND_MODEL
    assert len(ur.RETRY_DELAYS_SECONDS) == 3
    assert ur.RETRY_DELAYS_SECONDS[0] > ur.GATEWAY_RETRY_DELAYS_SECONDS[0]


def test_bookkeeping_turns_the_run_flag_into_the_one_shot(tmp_cron_home, monkeypatch):
    from cron import scheduler as sched
    from cron.scheduler_bookkeeping import finish_completed_run

    job = get_job(create_job("calendar mirror", "every 30m")["id"])
    captured: dict = {}
    monkeypatch.setattr(
        sched, "mark_job_run",
        lambda job_id, success, error=None, **kw: captured.update(kw) or True)
    monkeypatch.setattr(sched, "finish_execution", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "self_removal_delivery_allowed", lambda _id: False)

    d = sched._RunDelivery(
        job=dict(job, _gateway_unavailable=True), success=False, error=REFUSAL)
    assert finish_completed_run(d, None, "exec-refused")
    assert captured.get("gateway_unavailable") is True, (
        "run_job's verdict must reach mark_job_run, where plan_retry parks the rung")
    assert "_gateway_unavailable" not in d.job, "the transient flag is consumed, not persisted"
