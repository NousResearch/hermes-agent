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
    _job_is_stale_error_recurring, compute_next_run, create_job, get_due_jobs, get_job, mark_job_run,
    update_job,
)
from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, AuthError

QUOTA_MSG = "Codex provider quota exhausted (429); retry after 123518s. Credentials are still valid."


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(qh, "_release_probe_at", {}, raising=False)
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


def _held_job(name, schedule, now):
    """A recurring job parked by a failed run against the Codex quota window."""
    job = create_job(name, schedule, deliver="local")
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=123518)
    held = get_job(job["id"])
    assert qh.hold_active(held, now) and held[qh.STATE_KEY] == held["next_run_at"]
    return held


def _run_tick(monkeypatch, resolve, usage_restored=True):
    """One real ``tick()`` with job execution stubbed; returns the dispatched job ids and the
    number of provider probes. *usage_restored* is what the Codex usage endpoint reports."""
    import hermes_cli.auth as auth

    dispatched, probes = [], []

    def _probe(**kwargs):
        probes.append(kwargs)
        return resolve(**kwargs)

    monkeypatch.setattr(sched, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(sched, "_process_due_job",
                        lambda job, *_a, **_k: dispatched.append(job["id"]) or True)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _probe)
    monkeypatch.setattr(auth, "_probe_codex_quota_restored", lambda *_a, **_k: usage_restored)
    sched.tick(verbose=False, sync=True)
    return dispatched, probes


def test_hold_is_released_when_the_provider_reopens_before_the_window_ends(
    tmp_cron_home, monkeypatch,
):
    """Codex can reopen days before the reset it announced (a banked reset, a plan change, a
    rotated account). A held job must not sit out the rest of the stale window once the
    provider resolves and its usage window is open again: the interval job fires on the next
    tick, the cron job moves to its next legal occurrence (never off-lattice), both markers
    clear, and one probe serves every held job on the same route."""
    now = datetime.now(timezone.utc)
    interval = _held_job("crossfeed", "every 10m", now)
    daily = _held_job("nightly synthesis", "0 22 * * *", now)
    daily_parked = daily["next_run_at"]
    from agent import secret_scope
    scoped = []

    def _resolve(**_kw):
        # The tick holds no secret scope; the probe must run inside the profile's own.
        scoped.append(secret_scope.current_secret_scope() is not None)
        return {"provider": "openai-codex"}

    dispatched, probes = _run_tick(monkeypatch, _resolve)

    assert len(probes) == 1 and scoped == [True]
    assert secret_scope.current_secret_scope() is None
    assert interval["id"] in dispatched
    assert daily["id"] not in dispatched
    released = get_job(daily["id"])
    assert qh.STATE_KEY not in released
    assert released["next_run_at"] == compute_next_run(daily["schedule"], now.isoformat())
    assert released["next_run_at"] < daily_parked
    assert qh.STATE_KEY not in get_job(interval["id"])


def test_hold_stays_while_the_provider_is_still_closed_or_unresolvable(tmp_cron_home, monkeypatch):
    """Only a resolve that succeeds with its usage window open releases a hold. The same quota
    error, any other resolve failure (the real run reports those), or a successful resolve whose
    usage window is still closed or unknown leaves the job parked exactly where it was."""
    now = datetime.now(timezone.utc)
    held = _held_job("crossfeed", "every 10m", now)

    def _resolves(**_kw):
        return {"provider": "openai-codex", "api_key": "token", "base_url": None}

    cases = [(f, True) for f in (_quota_error(), RuntimeError("network down"),
                                 AuthError("expired", provider="openai-codex", code="expired",
                                           relogin_required=True))]
    cases += [(None, False), (None, None)]
    for failure, usage in cases:
        def _resolve(**_kw):
            if failure is not None:
                raise failure
            return _resolves()

        monkeypatch.setattr(qh, "_release_probe_at", {}, raising=False)
        dispatched, probes = _run_tick(monkeypatch, _resolve, usage_restored=usage)
        assert probes and dispatched == []
        still = get_job(held["id"])
        assert still[qh.STATE_KEY] == held[qh.STATE_KEY]
        assert still["next_run_at"] == held["next_run_at"]


def test_jobs_without_an_active_hold_are_never_probed(tmp_cron_home, monkeypatch):
    """Nothing to release, nothing to ask: plain jobs and paused held jobs cost no probe."""
    create_job("plain", "every 10m", deliver="local")
    paused = _held_job("paused", "every 10m", datetime.now(timezone.utc))
    update_job(paused["id"], {"state": "paused"})

    _dispatched, probes = _run_tick(monkeypatch, lambda **_kw: {"provider": "openai-codex"})

    assert probes == []


def test_a_still_closed_provider_is_probed_at_most_once_per_interval(tmp_cron_home, monkeypatch):
    """The tick runs every minute for the whole window; the release check must not turn that into
    a probe per minute against a provider that is still closed."""
    import time

    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    held = _held_job("crossfeed", "every 10m", datetime.now(timezone.utc))

    def _closed(**_kw):
        raise _quota_error()

    counts = []
    for step in (0, 60, 120, qh.RELEASE_PROBE_INTERVAL_SECONDS - 1,
                 qh.RELEASE_PROBE_INTERVAL_SECONDS, qh.RELEASE_PROBE_INTERVAL_SECONDS + 60):
        clock[0] = 1000.0 + step
        _dispatched, probes = _run_tick(monkeypatch, _closed)
        counts.append(len(probes))

    assert counts == [1, 0, 0, 0, 1, 0]
    assert get_job(held["id"])[qh.STATE_KEY] == held[qh.STATE_KEY]


def test_early_release_keeps_a_sparse_jobs_one_recovery_fire(tmp_cron_home, monkeypatch):
    """A weekly job parked on its off-lattice recovery fire (#121451) takes that one retry as soon
    as the provider reopens, instead of losing the blocked week to its next natural occurrence."""
    now = datetime(2026, 9, 18, 12, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    job = create_job("weekly digest", "0 12 * * 5", deliver="local")
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=20 * 60 * 60,
                        recover_consumed_fire=True)
    held = get_job(job["id"])
    assert qh.is_recovery_fire(held, held["next_run_at"])

    later = now + timedelta(hours=2)  # still well inside the announced 20h window
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: later)
    monkeypatch.setattr(qh, "_hermes_now", lambda: later)
    dispatched, probes = _run_tick(monkeypatch, lambda **_kw: {"provider": "openai-codex"})

    assert len(probes) == 1
    assert job["id"] in dispatched


def test_release_skips_a_job_whose_run_finished_during_the_probe(tmp_cron_home, monkeypatch):
    """The probe runs outside the jobs lock. A job whose record changed meanwhile (its own run
    cleared or re-parked the hold) is left as that run wrote it."""
    now = datetime.now(timezone.utc)
    held = _held_job("crossfeed", "every 10m", now)

    def _resolve_while_a_run_lands(**_kw):
        assert mark_job_run(held["id"], True)  # the run reached the model: hold cleared
        return {"provider": "openai-codex"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        _resolve_while_a_run_lands)

    assert qh.release_reopened_holds() == 0
    after_run = get_job(held["id"])
    assert qh.STATE_KEY not in after_run
    assert after_run["last_status"] == "ok"


def _jwt(exp: int) -> str:
    import base64
    import json

    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    claims = {"exp": exp, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}
    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


def _singleton_codex_login(home) -> str:
    """A valid, unexpired singleton Codex login (no pool): it resolves whether or not the usage
    window is open, so the resolve alone says nothing about the quota."""
    import json
    import time

    token = _jwt(int(time.time()) + 30 * 86400)
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"openai-codex": {
        "tokens": {"access_token": token, "refresh_token": "refresh", "id_token": token},
        "last_refresh": "2026-10-01T00:00:00Z", "auth_mode": "chatgpt"}}}))
    return token


@pytest.mark.parametrize("usage_says", [False, None])
def test_a_valid_singleton_login_does_not_release_a_hold_while_the_window_is_closed(
    tmp_cron_home, monkeypatch, usage_says,
):
    """The credentials stay valid while the window is shut, so a successful resolve is not a
    reopened window. With a singleton login the real resolve succeeds; only the usage endpoint
    can say the quota is back. Closed (False) or unknown (None) keeps the hold."""
    import hermes_cli.auth as auth

    token = _singleton_codex_login(tmp_cron_home)
    asked = []

    def _usage(access_token, **kwargs):
        asked.append(access_token)
        return usage_says

    monkeypatch.setattr(auth, "_probe_codex_quota_restored", _usage)
    monkeypatch.setattr(sched, "_should_yield_tick_to_fresh_gateway", lambda: None)
    dispatched = []
    monkeypatch.setattr(sched, "_process_due_job",
                        lambda job, *_a, **_k: dispatched.append(job["id"]) or True)
    job = create_job("crossfeed", "every 10m", deliver="local",
                     provider="openai-codex", model="gpt-5.5")
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=123518)
    held = get_job(job["id"])

    sched.tick(verbose=False, sync=True)

    assert asked == [token]
    assert dispatched == []
    still = get_job(job["id"])
    assert still[qh.STATE_KEY] == held[qh.STATE_KEY]
    assert still["next_run_at"] == held["next_run_at"]


def test_a_singleton_login_releases_the_hold_once_the_usage_window_reopens(
    tmp_cron_home, monkeypatch,
):
    """Same singleton login, usage endpoint reports headroom again: the hold is released and the
    interval job fires on this tick."""
    import hermes_cli.auth as auth

    _singleton_codex_login(tmp_cron_home)
    monkeypatch.setattr(auth, "_probe_codex_quota_restored", lambda *_a, **_k: True)
    monkeypatch.setattr(sched, "_should_yield_tick_to_fresh_gateway", lambda: None)
    dispatched = []
    monkeypatch.setattr(sched, "_process_due_job",
                        lambda job, *_a, **_k: dispatched.append(job["id"]) or True)
    job = create_job("crossfeed", "every 10m", deliver="local",
                     provider="openai-codex", model="gpt-5.5")
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=123518)

    sched.tick(verbose=False, sync=True)

    assert job["id"] in dispatched
    assert qh.STATE_KEY not in get_job(job["id"])
