"""Provider quota windows park a cron job instead of re-firing into them (#89376;
ai-velho-oy/chairman#3).

Contract (cron/quota_hold.py): a failed run whose cause is a rate-limited ``AuthError`` with a
``retry after <N>s`` hint parks a recurring job's ``next_run_at`` past the window and stamps
``quota_hold_until``; the stale-error re-arm leaves a held job alone; a run that reaches the
model clears the marker. The hint is read only from the AuthError in the cause chain, never from
arbitrary failure text.
"""

import base64
import json
import time
from datetime import datetime, timedelta, timezone, UTC
from threading import Event
from unittest.mock import MagicMock, patch

import httpx
import pytest

import cron.scheduler as sched
from cron import quota_hold as qh
from cron.jobs import (
    _job_is_stale_error_recurring, create_job, get_due_jobs, get_job, mark_job_run, pause_job,
    trigger_job, update_job,
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


def _codex_login(home, subject="fixture"):
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    token = f"{part({'alg': 'none'})}.{part({'sub': subject, 'exp': int(time.time()) + 3600})}.fake"
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {
        "openai-codex": {"tokens": {"access_token": token, "refresh_token": "fixture"}},
    }}), encoding="utf-8")
    return token


@pytest.fixture
def usage_transport(monkeypatch):
    """Use the real auth/usage parser. Wait for its background result without a wall-clock guess."""
    from concurrent.futures import Future
    from hermes_cli import auth_codex
    from cron import quota_recovery

    auth_codex._codex_quota_probe_cache.clear()
    submitted = []

    def track():
        future = Future()
        submitted.append(future)
        return future

    def configure(handler):
        monkeypatch.setattr(quota_recovery, "Future", track)
        monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kw: httpx.Client(
            transport=httpx.MockTransport(handler), **kw))
        return submitted

    yield configure
    for future in submitted:
        future.result(timeout=4)
    auth_codex._codex_quota_probe_cache.clear()


def _usage_response(used_percent):
    return httpx.Response(200, json={"rate_limit": {"primary_window": {"used_percent": used_percent}}})


def _held_job(now, name, schedule, **kwargs):
    job = create_job(name, schedule, provider="openai-codex", deliver="local", **kwargs)
    assert mark_job_run(job["id"], False, QUOTA_MSG, quota_hold_seconds=24 * 3600)
    held = get_job(job["id"])
    assert held is not None and qh.hold_active(held, now)
    return held


def test_reset_releases_stale_interval_hold_at_the_next_slot(tmp_cron_home, monkeypatch, usage_transport):
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    token = _codex_login(tmp_cron_home)

    def usage(request):
        assert request.headers["Authorization"] == f"Bearer {token}"
        assert request.url.path == "/backend-api/wham/usage"
        return _usage_response(5)

    probes = usage_transport(usage)
    held = _held_job(now, "interval digest", "every 30m")
    assert held["quota_hold_until"] == "2026-09-19T12:01:00+00:00"

    # The saved error is old enough for stale-error recovery. It must not cause a run-now
    # when the quota hold lifts, and the last failure must still be available for inspection.
    now = datetime(2026, 9, 18, 15, 5, tzinfo=UTC)
    with patch("cron.scheduler_authority.run_canonical_job", return_value=(True, "result", "", None)) as run:
        assert sched.tick(verbose=False) == 0
        probes[0].result(timeout=4)
        assert sched.tick(verbose=False) == 0
        recovered = get_job(held["id"])
        assert recovered is not None
        assert recovered["next_run_at"] == "2026-09-18T15:30:00+00:00"
        assert "quota_hold_until" not in recovered
        assert recovered["last_status"] == "error"
        run.assert_not_called()

        now = datetime(2026, 9, 18, 15, 30, tzinfo=UTC)
        assert sched.tick(verbose=False) == 1
        assert sched.tick(verbose=False) == 0
        run.assert_called_once()


def test_pool_only_quota_hold_recovers_without_background_auth_writes(tmp_cron_home, monkeypatch, usage_transport):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    token = _codex_login(tmp_cron_home)
    auth_file = tmp_cron_home / "auth.json"
    auth_file.write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {
        "openai-codex": [{"id": "fixture", "label": "fixture", "auth_type": "oauth", "priority": 0,
                          "source": "device_code", "access_token": token, "refresh_token": "fixture",
                          "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 429,
                          "last_error_reason": "usage_limit_reached", "last_error_reset_at": time.time() + 86400}],
    }}), encoding="utf-8")
    used = 100

    def usage(request):
        assert request.headers["Authorization"] == f"Bearer {token}"
        return _usage_response(used)

    probes = usage_transport(usage)
    job = create_job("pool digest", "every 30m", provider="openai-codex", deliver="local")
    with pytest.raises(AuthError) as exc:
        resolve_runtime_provider(requested="openai-codex")
    seconds = qh.hold_seconds_from_failure(exc.value)
    assert seconds is not None
    assert mark_job_run(job["id"], False, str(exc.value), quota_hold_seconds=seconds)
    auth_before = auth_file.read_bytes()

    used = 3
    assert sched.tick(verbose=False) == 0
    probes[0].result(timeout=4)
    assert sched.tick(verbose=False) == 0
    restored = get_job(job["id"])
    assert restored is not None and restored["next_run_at"] == "2026-09-18T12:30:00+00:00"
    assert auth_file.read_bytes() == auth_before

    selected = []

    def run(jb, **_kw):
        selected.append(resolve_runtime_provider(requested=jb["provider"])["api_key"])
        return True, "result", "", None

    now = datetime(2026, 9, 18, 12, 30, tzinfo=UTC)
    monkeypatch.setattr("cron.scheduler_authority.run_canonical_job", run)
    assert sched.tick(verbose=False) == 1
    assert selected == [token]


def test_usage_from_an_ambiguous_codex_login_cannot_reopen_a_hold(tmp_cron_home, monkeypatch, usage_transport):
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    pool_token = _codex_login(tmp_cron_home, "pool")
    fallback_token = _codex_login(tmp_cron_home, "singleton")
    auth_file = tmp_cron_home / "auth.json"
    auth = json.loads(auth_file.read_text(encoding="utf-8"))
    auth["credential_pool"] = {"openai-codex": [{
        "id": "pool", "auth_type": "oauth", "source": "device_code", "access_token": pool_token,
        "last_status": "exhausted", "last_error_code": 429, "last_error_reset_at": time.time() + 86400,
    }]}
    auth_file.write_text(json.dumps(auth), encoding="utf-8")

    def usage(request):
        return _usage_response(5 if request.headers["Authorization"] == f"Bearer {fallback_token}" else 100)

    probes = usage_transport(usage)
    held = _held_job(now, "digest", "every 10m")
    assert sched.tick(verbose=False) == 0
    probes[0].result(timeout=4)
    assert sched.tick(verbose=False) == 0
    assert get_job(held["id"]) == held


@pytest.mark.parametrize("response", ["closed", "refused", "unknown", "timeout"])
def test_active_hold_requires_a_new_positive_usage_answer(tmp_cron_home, monkeypatch, usage_transport, response):
    from hermes_cli.auth_codex import _probe_codex_quota_restored

    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    token = _codex_login(tmp_cron_home)
    answers = iter(["old-open", response])

    def usage(request):
        match next(answers):
            case "old-open":
                return _usage_response(5)
            case "closed":
                return _usage_response(100)
            case "refused":
                return httpx.Response(200, json={"rate_limit": {
                    "allowed": False, "primary_window": {"used_percent": 5},
                }})
            case "timeout":
                raise httpx.ReadTimeout("synthetic timeout", request=request)
            case _:
                return httpx.Response(200, json={})

    probes = usage_transport(usage)
    assert _probe_codex_quota_restored(token) is True
    held = _held_job(now, "digest", "every 10m")
    assert sched.tick(verbose=False) == 0
    probes[0].result(timeout=4)
    assert sched.tick(verbose=False) == 0
    assert next(answers, None) is None  # The cached pre-429 open was not reused.
    assert get_job(held["id"]) == held


def test_probe_uses_the_served_profiles_and_does_not_block_the_next_one(tmp_path, monkeypatch, usage_transport):
    from cron.scheduler_provider import _profile_cron_scope

    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    root = tmp_path / "isolated"
    monkeypatch.setenv("HERMES_HOME", str(root))
    alpha, beta = root / "profiles" / "alpha", root / "profiles" / "beta"
    for home in (alpha, beta):
        home.mkdir(parents=True)
        (home / ".env").write_text(
            f"HERMES_CODEX_BASE_URL=https://{home.name}.invalid/backend-api/codex\n", encoding="utf-8")
    alpha_token, beta_token = _codex_login(alpha, "alpha"), _codex_login(beta, "beta")
    started, release = Event(), Event()

    def usage(request):
        host = request.url.host
        assert request.headers["Authorization"] == "Bearer " + {
            "alpha.invalid": alpha_token, "beta.invalid": beta_token,
        }[host]
        if host == "alpha.invalid":
            started.set()
            assert release.wait(4)
            return _usage_response(5)
        return _usage_response(100)

    probes = usage_transport(usage)
    with _profile_cron_scope(alpha):
        a = _held_job(now, "pinned digest", "0 9 * * *", timezone="Asia/Tokyo")
        assert sched.tick(verbose=False) == 0
    try:
        assert started.wait(4)
        with _profile_cron_scope(beta):
            b = _held_job(now, "pinned digest", "0 9 * * *", timezone="Asia/Tokyo")
            assert sched.tick(verbose=False) == 0
            probes[1].result(timeout=4)
            assert sched.tick(verbose=False) == 0
            assert get_job(b["id"]) == b
        assert not probes[0].done()
    finally:
        release.set()

    probes[0].result(timeout=4)
    with _profile_cron_scope(alpha):
        assert sched.tick(verbose=False) == 0
        recovered = get_job(a["id"])
        assert recovered is not None
        assert recovered["next_run_at"] == "2026-09-19T09:00:00+09:00"
        assert "quota_hold_until" not in recovered


@pytest.mark.parametrize("edit", ["pause", "run-now", "schedule", "completion", "auth"])
def test_a_late_usage_answer_cannot_override_new_state(tmp_cron_home, monkeypatch, usage_transport, edit):
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)
    _codex_login(tmp_cron_home)
    entered, release = Event(), Event()

    def usage(_request):
        entered.set()
        assert release.wait(4)
        return _usage_response(5)

    probes = usage_transport(usage)
    held = _held_job(now, "digest", "every 30m")
    job_id = held["id"]
    assert sched.tick(verbose=False) == 0
    try:
        assert entered.wait(4)
        actions = {
            "pause": lambda: pause_job(job_id),
            "run-now": lambda: trigger_job(job_id),
            "schedule": lambda: update_job(job_id, {"schedule": "every 4h"}),
            "completion": lambda: mark_job_run(job_id, True),
            "auth": lambda: _codex_login(tmp_cron_home, "new-login"),
        }
        assert actions[edit]()
        edited = get_job(job_id)
        assert edited is not None
    finally:
        release.set()

    probes[0].result(timeout=4)
    with patch("cron.scheduler_authority.run_canonical_job", return_value=(True, "result", "", None)) as run:
        if edit == "run-now":
            assert sched.tick(verbose=False) == 1
            assert sched.tick(verbose=False) == 0
            run.assert_called_once()
        else:
            assert sched.tick(verbose=False) == 0
            assert get_job(job_id) == edited
            run.assert_not_called()


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
    now = datetime(2026, 9, 18, 12, 1, tzinfo=UTC)
    natural_next = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
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
    now = datetime(2026, 9, 18, 12, 1, tzinfo=UTC)
    monkeypatch.setattr(qh, "_hermes_now", lambda: now)

    dense = {
        "schedule": {"kind": "cron", "expr": "0 * * * *"},
        "next_run_at": datetime(2026, 9, 18, 13, 0, tzinfo=UTC).isoformat(),
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
    """One real scheduler tick (preflight ON) with the provider resolver replaced by *resolve*; returns the
    stand-in for the run's store opener."""
    with patch("cron.scheduler._hermes_home", home), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("persistence_assembly.open_persistence", return_value=MagicMock()) as open_store, \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
         patch("cron.delivery_queue.enqueue",
               side_effect=lambda execution_id, jb, content, **kw: (
                   deliveries.append(content), {"status": "pending"})[1]), \
         patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.run_conversation.side_effect = RuntimeError("model said no")
        sched.run_one_job(dict(job))
    return open_store


def test_quota_hold_parks_past_window_survives_stale_rearm_and_clears_on_model_reach(tmp_cron_home, cron_owner):
    """A 30-minute job whose provider resolve raises the Codex quota AuthError ('retry after
    123518s') is parked by the real scheduler tick: preflight lets the rate-limited AuthError
    through (it is not a missing credential), the one delivered alert carries the hold notice,
    and the job does not fire again inside the window (not even after the stale-error re-arm's
    cadence+grace). The marker clears once a run reaches the model."""
    # A real (non-local) lane: the alert rides the durable delivery queue, whose enqueue is the seam.
    (tmp_cron_home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n", encoding="utf-8")
    job = create_job("portfolio triage", "every 30m", deliver="slack:alerts")
    job_id = job["id"]
    now = datetime.now(UTC)
    deliveries: list = []

    open_store = _tick(get_job(job_id), tmp_cron_home, deliveries, _raise_quota)
    # The provider resolve fails before the run opens its store.
    open_store.assert_not_called()
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
