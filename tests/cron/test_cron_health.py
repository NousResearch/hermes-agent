from __future__ import annotations

import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import cron.jobs as jobs
from cron.job_health import (
    SCHEMA_VERSION,
    classify_failure,
    coerce_health,
    default_health,
    record_result,
    suppress_or_claim,
)
from cron.jobs import (
    advance_next_run,
    claim_job_for_fire,
    create_job,
    get_due_jobs,
    get_job,
    mark_job_run,
    rearm_oneshot,
    trigger_job,
)
from cron.scheduler import should_escalate_cron_failure


FIXED = datetime.datetime(2026, 9, 7, 10, 0, tzinfo=datetime.timezone.utc)
QUOTA_ERROR = "RuntimeError: Codex provider quota exhausted (429); retry after 133241s. Credentials are still valid."


def test_explicit_oneshot_rearm_clears_inherited_circuit(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    first_at = (FIXED + datetime.timedelta(minutes=1)).isoformat()
    job = create_job(prompt="once", schedule=first_at)
    assert mark_job_run(job["id"], False, "HTTP 401 Unauthorized")
    opened = get_job(job["id"])
    assert opened is not None
    assert opened["health"]["state"] == "circuit_open"

    replacement = rearm_oneshot(
        job["id"],
        (FIXED + datetime.timedelta(minutes=2)).isoformat(),
    )

    assert replacement is not None
    assert replacement["health"]["state"] == "unknown"
    assert replacement["health"]["retry_not_before"] is None


def test_missing_provider_key_is_classified_as_auth_error():
    assert (
        classify_failure("No API key configured for provider 'openrouter'")[0] == "auth"
    )


def test_numeric_path_components_are_not_http_statuses():
    assert classify_failure("failed to read /tmp/reports/401/input.csv")[0] == "unknown"
    assert classify_failure("HTTP status 401 Unauthorized")[0] == "auth"
    sdk_quota = (
        "RateLimitError: Error code: 429 - You exceeded your current quota "
        "(code=insufficient_quota)"
    )
    assert classify_failure(sdk_quota)[0] == "provider_quota_exhausted"


def test_create_job_persists_nested_cron_health_contract(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="health", schedule="every 5m")

    persisted = get_job(job["id"])
    assert persisted is not None
    health = persisted["health"]
    assert health["schema_version"] == SCHEMA_VERSION
    assert health["job_id"] == job["id"]
    assert health["profile"] == "default"
    assert health["state"] == "unknown"
    assert health["consecutive_failures"] == 0
    assert health["probe_claim"] is None


def test_long_provider_retry_after_opens_immediately_and_persists_absolute_deadline(
    monkeypatch,
):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="failing", schedule="every 1m")

    assert mark_job_run(job["id"], False, QUOTA_ERROR)
    health = get_job(job["id"])["health"]

    assert health["state"] == "circuit_open"
    assert health["reason_code"] == "provider_quota_exhausted"
    assert health["consecutive_failures"] == 1
    assert health["retry_after_seconds"] == 133241
    assert health["retry_not_before"] == "2026-09-08T23:00:41Z"
    assert "credentials" not in health["reason_summary"]
    assert "Codex" not in json.dumps(health)


def test_corrupt_or_type_invalid_health_fails_safe_to_unknown():
    valid = default_health("job", "default", FIXED.isoformat())
    malformed = [
        {**valid, "consecutive_failures": "3"},
        {**valid, "retry_after_seconds": 604801},
        {**valid, "retry_after_clamped": 1},
        {**valid, "failure_fingerprint": "not-a-sha256"},
        {**valid, "state": ["healthy"]},
        {**valid, "reason_code": {"code": "auth"}},
        {**valid, "unexpected": "field"},
    ]

    for raw in malformed:
        coerced = coerce_health(raw, "job", "default")
        assert coerced["state"] == "unknown"
        assert coerced["consecutive_failures"] == 0

    malformed_job = {"id": "job", "health": {**valid, "state": ["healthy"]}}
    assert not should_escalate_cron_failure(
        malformed_job,
        "transport connection reset",
        now=FIXED,
    )


def test_health_never_persists_raw_provider_or_prompt_content():
    secret = "SECRET-PROMPT TOKEN-SHOULD-NOT-RENDER"
    health = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error=f"unknown provider response: {secret}",
        now=FIXED,
    )

    serialized = json.dumps(health)
    assert secret not in serialized
    assert "SECRET-PROMPT" not in serialized
    assert health["reason_summary"] == "terminal cron failure"


def test_preserved_usage_limit_fixture_defaults_retry_to_one_hour(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="kanban sync", schedule="every 1m")

    assert mark_job_run(
        job["id"], False, "RuntimeError: HTTP 429: The usage limit has been reached"
    )
    health = get_job(job["id"])["health"]

    assert health["state"] == "circuit_open"
    assert health["reason_code"] == "provider_quota_exhausted"
    assert health["retry_after_seconds"] == 3600
    assert health["retry_not_before"] == "2026-09-07T11:00:00Z"


def test_rate_limit_retry_after_overrides_local_backoff_after_threshold():
    health = None
    for minute in range(3):
        health = record_result(
            health,
            job_id="job",
            profile="default",
            success=False,
            error="HTTP 429 rate limit; retry-after: 7200",
            now=FIXED + datetime.timedelta(minutes=minute),
        )

    assert health["state"] == "circuit_open"
    assert health["reason_code"] == "provider_rate_limited"
    assert health["retry_after_seconds"] == 7200
    assert health["retry_not_before"] == "2026-09-07T12:02:00Z"


def test_retry_deadline_is_clamped_to_seven_days():
    health = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error="HTTP 429: provider quota exhausted; retry after 999999999s",
        now=FIXED,
    )

    assert health["retry_after_seconds"] == 604800
    assert health["retry_after_clamped"] is True
    assert health["retry_not_before"] == "2026-09-14T10:00:00Z"


def test_retry_and_probe_deadlines_use_elapsed_utc_across_dst_fold():
    zone = ZoneInfo("America/Denver")
    local_now = datetime.datetime(2026, 11, 1, 1, 58, tzinfo=zone, fold=0)
    opened = record_result(
        None,
        job_id="dst",
        profile="default",
        success=False,
        error="HTTP 429 provider quota exhausted",
        now=local_now,
    )
    retry_at = datetime.datetime.fromisoformat(
        opened["retry_not_before"].replace("Z", "+00:00")
    )
    assert (retry_at - local_now.astimezone(datetime.timezone.utc)).total_seconds() == 3600

    action, claimed = suppress_or_claim(
        opened,
        job_id="dst",
        profile="default",
        now=retry_at.astimezone(zone),
    )
    assert action == "probe"
    lease_at = datetime.datetime.fromisoformat(
        claimed["probe_claim"]["lease_expires_at"].replace("Z", "+00:00")
    )
    assert (lease_at - retry_at).total_seconds() == 300


def test_retry_deadlines_follow_classifier_open_policy():
    quota_without_deadline = record_result(
        None,
        job_id="quota-no-deadline",
        profile="default",
        success=False,
        error="HTTP 429 provider quota exhausted",
        now=FIXED,
    )
    assert quota_without_deadline["state"] == "circuit_open"
    assert quota_without_deadline["retry_after_seconds"] == 3600

    timeout_with_deadline = None
    for minute in range(3):
        timeout_with_deadline = record_result(
            timeout_with_deadline,
            job_id="timeout-with-deadline",
            profile="default",
            success=False,
            error="provider request timed out; retry after 600s",
            now=FIXED + datetime.timedelta(minutes=minute),
        )
    assert timeout_with_deadline is not None
    assert timeout_with_deadline["state"] == "circuit_open"
    assert timeout_with_deadline["retry_after_seconds"] == 600


def test_structured_retry_hint_precedes_provider_error_text():
    health = record_result(
        None,
        job_id="structured-hint",
        profile="default",
        success=False,
        error="HTTP 429 provider quota exhausted; retry after 123518s",
        now=FIXED,
        retry_after_seconds=72000,
    )
    assert health["retry_after_seconds"] == 72000

    dense_job = create_job(prompt="dense structured", schedule="every 1m")
    dense_job["_quota_hold_seconds"] = 72000
    assert should_escalate_cron_failure(
        dense_job,
        "HTTP 429 provider quota exhausted",
        now=FIXED,
    )


def test_due_ticks_are_suppressed_per_job_without_replacing_attempt_status(monkeypatch):
    now = [FIXED]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    limited = create_job(prompt="limited", schedule="every 1m")
    healthy = create_job(prompt="healthy", schedule="every 1m")
    assert mark_job_run(limited["id"], False, QUOTA_ERROR)
    prior_status = get_job(limited["id"])["last_status"]

    for minute in range(1, 25):
        now[0] = FIXED + datetime.timedelta(minutes=minute)
        due_ids = {job["id"] for job in get_due_jobs()}
        assert limited["id"] not in due_ids
        assert healthy["id"] in due_ids

    limited_job = get_job(limited["id"])
    assert limited_job["health"]["suppressed_runs"] == 24
    assert limited_job["health"]["consecutive_failures"] == 1
    assert limited_job["last_status"] == prior_status
    evidence = sorted(
        (jobs._job_output_dir(limited["id"]) / "suppressed").glob("*-skipped.md")
    )
    assert len(evidence) == 24
    for path in evidence:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("## Skipped")
        assert "provider_quota_exhausted" in content
        assert "Credentials are still valid" not in content


def test_circuit_hold_is_not_rearmed_as_a_stale_error(monkeypatch):
    now = [FIXED]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    job = create_job(prompt="held interval", schedule="every 10m")
    assert mark_job_run(job["id"], False, "HTTP 401 Unauthorized")

    now[0] = FIXED + datetime.timedelta(minutes=10)
    assert job["id"] not in {item["id"] for item in get_due_jobs()}
    first = get_job(job["id"])
    assert first is not None
    assert first["health"]["suppressed_runs"] == 1
    next_run_at = first["next_run_at"]

    now[0] = FIXED + datetime.timedelta(minutes=12, seconds=1)
    assert job["id"] not in {item["id"] for item in get_due_jobs()}
    second = get_job(job["id"])
    assert second is not None
    assert second["health"]["suppressed_runs"] == 1
    assert second["next_run_at"] == next_run_at


def test_three_identical_failures_open_and_different_fingerprint_resets_series():
    health = None
    for minute in range(3):
        health = record_result(
            health,
            job_id="job",
            profile="default",
            success=False,
            error="tool invocation failed in a stable way",
            now=FIXED + datetime.timedelta(minutes=minute),
        )
    assert health["state"] == "circuit_open"
    assert health["consecutive_failures"] == 3

    changed = record_result(
        health,
        job_id="job",
        profile="default",
        success=False,
        error="transport connection failed",
        now=FIXED + datetime.timedelta(minutes=3),
    )
    assert changed["state"] == "degraded"
    assert changed["consecutive_failures"] == 1
    assert changed["reason_code"] == "transport"


def test_same_fingerprint_after_hour_starts_a_new_failure_series():
    first = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error="tool invocation failed",
        now=FIXED,
    )
    second = record_result(
        first,
        job_id="job",
        profile="default",
        success=False,
        error="tool invocation failed",
        now=FIXED + datetime.timedelta(hours=2),
    )

    assert second["state"] == "degraded"
    assert second["consecutive_failures"] == 1
    assert second["series_started_at"] == "2026-09-07T12:00:00Z"


def test_half_open_claim_is_single_and_success_resets_health():
    health = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error=QUOTA_ERROR,
        now=FIXED,
    )
    retry_at = datetime.datetime.fromisoformat(
        health["retry_not_before"].replace("Z", "+00:00")
    )

    action, claimed = suppress_or_claim(
        health,
        job_id="job",
        profile="default",
        now=retry_at,
    )
    second_action, suppressed = suppress_or_claim(
        claimed,
        job_id="job",
        profile="default",
        now=retry_at,
    )
    assert action == "probe"
    assert second_action == "suppress"
    assert suppressed["state"] == "half_open"
    assert claimed["probe_claim"]["token"]

    recovered = record_result(
        claimed,
        job_id="job",
        profile="default",
        success=True,
        error=None,
        now=retry_at + datetime.timedelta(seconds=1),
    )
    assert recovered["state"] == "healthy"
    assert recovered["consecutive_failures"] == 0
    assert recovered["retry_not_before"] is None
    assert recovered["suppressed_runs"] == 0


def test_manual_and_no_agent_runs_do_not_consume_provider_retry_budget(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    manual = create_job(prompt="manual", schedule="every 1h")
    trigger_job(manual["id"])
    claimed_manual = claim_job_for_fire(manual["id"], return_job=True)
    assert isinstance(claimed_manual, dict)
    advance_next_run(manual["id"])
    assert mark_job_run(
        manual["id"],
        False,
        QUOTA_ERROR,
        expected_fire_owner=claimed_manual["fire_claim"]["by"],
    )
    assert get_job(manual["id"])["health"]["state"] == "unknown"

    script_job = create_job(
        prompt=None,
        schedule="every 1h",
        script="printf ok",
        no_agent=True,
    )
    assert mark_job_run(script_job["id"], False, QUOTA_ERROR)
    assert get_job(script_job["id"])["health"]["state"] == "unknown"


def test_inference_change_clears_provider_specific_cooldown(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="provider", schedule="every 1m", provider="openrouter")
    assert mark_job_run(job["id"], False, QUOTA_ERROR)
    changed = jobs.update_job(job["id"], {"provider": "anthropic"})
    assert changed is not None
    assert changed["health"]["state"] == "unknown"
    assert changed["health"]["retry_not_before"] is None


def test_effective_default_model_change_clears_provider_cooldown(monkeypatch):
    now = [FIXED]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    config_path = jobs.get_hermes_home() / "config.yaml"
    config_path.write_text("model:\n  default: old-model\n", encoding="utf-8")
    job = create_job(prompt="default model", schedule="every 1m")
    assert mark_job_run(
        job["id"], False, QUOTA_ERROR, quota_hold_seconds=72000
    )
    opened = get_job(job["id"])
    assert opened is not None
    assert opened["health"]["state"] == "circuit_open"

    config_path.write_text("model:\n  default: new-model\n", encoding="utf-8")
    now[0] = FIXED + datetime.timedelta(minutes=1)
    assert job["id"] not in {item["id"] for item in get_due_jobs()}
    changed = get_job(job["id"])
    assert changed is not None
    assert changed["health"]["state"] == "unknown"
    assert changed["health_route"]["model"] == "new-model"
    assert changed.get("quota_hold_until") is None
    now[0] = FIXED + datetime.timedelta(minutes=2)
    assert job["id"] in {item["id"] for item in get_due_jobs()}


def test_effective_base_url_change_clears_provider_cooldown(monkeypatch):
    now = [FIXED]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    config_path = jobs.get_hermes_home() / "config.yaml"
    config_path.write_text(
        "model:\n  default: same-model\n  base_url: https://old.invalid\n",
        encoding="utf-8",
    )
    job = create_job(prompt="endpoint", schedule="every 1m")
    assert mark_job_run(job["id"], False, "HTTP 401 Unauthorized")

    config_path.write_text(
        "model:\n  default: same-model\n  base_url: https://new.invalid\n",
        encoding="utf-8",
    )
    now[0] = FIXED + datetime.timedelta(minutes=1)
    assert job["id"] in {item["id"] for item in get_due_jobs()}
    changed = get_job(job["id"])
    assert changed is not None
    assert changed["health_route"]["base_url"] == "https://new.invalid"


def test_stale_attempt_cannot_poison_new_provider_route(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(
        prompt="route switch",
        schedule="every 1m",
        provider="openrouter",
    )
    claimed = claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]
    changed = jobs.update_job(job["id"], {"provider": "anthropic"})
    assert changed is not None

    assert mark_job_run(
        job["id"],
        False,
        QUOTA_ERROR,
        expected_fire_owner=owner,
        quota_hold_seconds=72000,
    )
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["health"]["state"] == "unknown"
    assert stored["health_route"]["provider"] == "anthropic"
    assert stored.get("quota_hold_until") is None


def test_queued_manual_intent_does_not_reclassify_scheduled_attempt(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="scheduled", schedule="every 1m")
    claimed = claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    trigger_job(job["id"])
    assert mark_job_run(
        job["id"],
        False,
        QUOTA_ERROR,
        expected_fire_owner=claimed["fire_claim"]["by"],
    )
    assert get_job(job["id"])["health"]["state"] == "circuit_open"


def test_escalation_occurs_only_when_failure_first_opens_circuit():
    job = {"id": "job", "health": default_health("job", "default", FIXED.isoformat())}
    assert not should_escalate_cron_failure(
        job,
        "tool invocation failed",
        now=FIXED,
    )

    first = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error="tool invocation failed",
        now=FIXED,
    )
    second = record_result(
        first,
        job_id="job",
        profile="default",
        success=False,
        error="tool invocation failed",
        now=FIXED + datetime.timedelta(minutes=1),
    )
    job["health"] = second
    assert should_escalate_cron_failure(
        job,
        "tool invocation failed",
        now=FIXED + datetime.timedelta(minutes=2),
    )

    opened = record_result(
        second,
        job_id="job",
        profile="default",
        success=False,
        error="tool invocation failed",
        now=FIXED + datetime.timedelta(minutes=2),
    )
    job["health"] = opened
    assert not should_escalate_cron_failure(
        job,
        "tool invocation failed",
        now=FIXED + datetime.timedelta(minutes=3),
    )


def test_quota_retry_after_escalates_immediately_but_probe_failure_does_not_realert():
    job = {"id": "job", "health": default_health("job", "default", FIXED.isoformat())}
    assert should_escalate_cron_failure(job, QUOTA_ERROR, now=FIXED)

    opened = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error=QUOTA_ERROR,
        now=FIXED,
    )
    retry_at = datetime.datetime.fromisoformat(
        opened["retry_not_before"].replace("Z", "+00:00")
    )
    action, half_open = suppress_or_claim(
        opened,
        job_id="job",
        profile="default",
        now=retry_at,
    )
    assert action == "probe"
    job["health"] = half_open
    assert not should_escalate_cron_failure(job, QUOTA_ERROR, now=retry_at)
    assert should_escalate_cron_failure(job, "HTTP 401 Unauthorized", now=retry_at)


def test_exhausted_unreachable_retry_still_alerts(monkeypatch):
    opened = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error=QUOTA_ERROR,
        now=FIXED,
    )
    retry_at = datetime.datetime.fromisoformat(
        opened["retry_not_before"].replace("Z", "+00:00")
    )
    action, half_open = suppress_or_claim(
        opened,
        job_id="job",
        profile="default",
        now=retry_at,
    )
    assert action == "probe"
    monkeypatch.setattr("cron.unreachable_retry.will_retry", lambda job: False)
    job = {"id": "job", "health": half_open, "_model_unreachable": True}
    assert should_escalate_cron_failure(
        job,
        "transport connection reset",
        now=retry_at,
    )


def test_suppressed_probe_refreshes_incident_without_realert(monkeypatch):
    import cron.scheduler as scheduler

    opened = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error=QUOTA_ERROR,
        now=FIXED,
    )
    retry_at = datetime.datetime.fromisoformat(
        opened["retry_not_before"].replace("Z", "+00:00")
    )
    action, half_open = suppress_or_claim(
        opened,
        job_id="job",
        profile="default",
        now=retry_at,
    )
    assert action == "probe"
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_upsert_incident_for_failure",
        lambda job, error, output_file=None: calls.append((error, output_file))
        or (False, "incident"),
    )
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda job_id: False)

    content, *_ = scheduler._compose_run_delivery(
        {"id": "job", "name": "job", "health": half_open},
        success=False,
        error=QUOTA_ERROR,
        final_response="",
        output_file="probe.md",
    )

    assert content == ""
    assert calls == [(QUOTA_ERROR, "probe.md")]


def test_self_removed_normal_failure_still_delivers(monkeypatch):
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda job_id: True)
    monkeypatch.setattr(
        scheduler,
        "_upsert_incident_for_failure",
        lambda job, error, output_file=None: (False, "incident"),
    )
    content, *_ = scheduler._compose_run_delivery(
        {
            "id": "removed",
            "name": "removed",
            "health": default_health("removed", "default"),
        },
        success=False,
        error="tool execution failed",
        final_response="",
        output_file=None,
        agent_declared=True,
    )
    assert content


def test_different_probe_failure_starts_a_new_degraded_series():
    opened = record_result(
        None,
        job_id="job",
        profile="default",
        success=False,
        error=QUOTA_ERROR,
        now=FIXED,
    )
    retry_at = datetime.datetime.fromisoformat(
        opened["retry_not_before"].replace("Z", "+00:00")
    )
    action, half_open = suppress_or_claim(
        opened,
        job_id="job",
        profile="default",
        now=retry_at,
    )
    assert action == "probe"

    failed_probe = record_result(
        half_open,
        job_id="job",
        profile="default",
        success=False,
        error="transport connection reset",
        now=retry_at,
    )

    assert failed_probe["state"] == "degraded"
    assert failed_probe["reason_code"] == "transport"
    assert failed_probe["consecutive_failures"] == 1
    assert failed_probe["probe_failures"] == 0
    assert failed_probe["retry_after_seconds"] is None


def test_low_frequency_and_one_shot_failures_still_notify():
    interval_job = create_job(prompt="slow", schedule="every 2h")
    boundary_job = create_job(prompt="boundary", schedule="every 30m")
    finite_job = create_job(prompt="finite", schedule="every 1m", repeat=2)
    finite_job["repeat"]["completed"] = 1
    hourly_job = create_job(prompt="hourly", schedule="0 * * * *")
    assert should_escalate_cron_failure(
        interval_job,
        "transport connection reset",
        now=FIXED,
    )
    assert should_escalate_cron_failure(
        boundary_job,
        "transport connection reset",
        now=FIXED,
    )
    assert should_escalate_cron_failure(
        finite_job,
        "transport connection reset",
        now=FIXED,
    )
    assert should_escalate_cron_failure(
        hourly_job,
        "transport connection reset",
        now=FIXED,
    )
    assert should_escalate_cron_failure(
        {
            "id": "once",
            "health": default_health("once", "default", FIXED.isoformat()),
            "schedule": {"kind": "once"},
        },
        "transport connection reset",
        now=FIXED,
    )


def test_delivery_failure_is_not_recorded_as_model_success(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="deliver", schedule="every 1h")

    assert mark_job_run(
        job["id"],
        True,
        delivery_error="transport connection failed while delivering",
    )

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["health"]["state"] == "degraded"
    assert stored["health"]["reason_code"] == "transport"
    assert stored["health"]["last_success_at"] is None


def test_external_fire_respects_open_circuit_and_claims_one_elapsed_probe(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="external", schedule="every 1m")
    assert mark_job_run(job["id"], False, QUOTA_ERROR)
    actual_output = jobs.save_job_output(job["id"], "actual failure output")

    assert claim_job_for_fire(job["id"], return_job=True) is False
    assert actual_output.exists()
    evidence = list(actual_output.parent.glob("*.md"))
    assert actual_output in evidence
    skipped = list((actual_output.parent / "suppressed").glob("*-skipped.md"))
    assert len(skipped) == 1
    assert skipped[0].read_text(encoding="utf-8").startswith("## Skipped")
    suppressed = get_job(job["id"])
    assert suppressed is not None
    assert suppressed["health"]["state"] == "suppressed"
    assert list(actual_output.parent.glob("*.md")) == [actual_output]
    for _ in range(55):
        jobs.save_job_suppression_event(job["id"], suppressed["health"], FIXED)
    assert actual_output.exists()
    assert len(list((actual_output.parent / "suppressed").glob("*.md"))) == 50

    retry_at = datetime.datetime.fromisoformat(
        suppressed["health"]["retry_not_before"].replace("Z", "+00:00")
    )
    monkeypatch.setattr(jobs, "_hermes_now", lambda: retry_at)
    claimed = claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    assert claimed["health"]["state"] == "half_open"
    assert claim_job_for_fire(job["id"], return_job=True) is False


def test_concurrent_elapsed_ticks_atomically_claim_exactly_one_probe(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="probe race", schedule="every 1m")
    assert mark_job_run(job["id"], False, QUOTA_ERROR)
    opened = get_job(job["id"])
    assert opened is not None
    retry_at = datetime.datetime.fromisoformat(
        opened["health"]["retry_not_before"].replace("Z", "+00:00")
    )
    monkeypatch.setattr(jobs, "_hermes_now", lambda: retry_at)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: claim_job_for_fire(job["id"], return_job=True),
                range(8),
            )
        )

    claimed = [result for result in results if isinstance(result, dict)]
    assert len(claimed) == 1
    assert claimed[0]["health"]["state"] == "half_open"


def test_scheduled_completion_preserves_concurrently_queued_manual_context(monkeypatch):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="manual queued", schedule="every 1h")
    claimed = claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    owner = claimed["fire_claim"]["by"]

    triggered = trigger_job(job["id"], extra_prompt="one-time instruction")
    assert triggered is not None
    assert mark_job_run(job["id"], True, expected_fire_owner=owner)

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["next_run_at"] == stored["manual_run_at"]
    assert stored["manual_run_prompt"] == "one-time instruction"


def test_ticker_preserves_manual_identity_after_pre_advance(monkeypatch):
    import cron.scheduler as scheduler

    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED)
    job = create_job(prompt="manual ticker", schedule="every 1h")
    assert mark_job_run(job["id"], False, QUOTA_ERROR)
    opened = get_job(job["id"])
    assert opened is not None
    before = opened["health"]
    triggered = trigger_job(job["id"], extra_prompt="manual context")
    assert triggered is not None
    due = next(item for item in get_due_jobs() if item["id"] == job["id"])
    assert due["manual_run_at"] == due["next_run_at"]
    assert advance_next_run(job["id"])
    due["execution_id"] = "test-manual-execution"

    def _finish(claimed_job, *, adapters=None, loop=None, verbose=False):
        assert claimed_job["fire_claim"]["manual"] is True
        assert mark_job_run(claimed_job["id"], False, QUOTA_ERROR)
        return True

    monkeypatch.setattr(scheduler, "run_one_job", _finish)
    assert scheduler._process_due_job(due, adapters=None, loop=None, verbose=False)

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["health"] == before
    assert "manual_run_at" not in stored
    assert "manual_run_prompt" not in stored


def test_direct_tool_run_marks_attempt_manual_before_execution(monkeypatch):
    from tools import cronjob_tools

    job = create_job(prompt="manual direct", schedule="every 1h")

    def _finish(claimed_job, *, extra_prompt=None):
        assert claimed_job["fire_claim"]["manual"] is True
        assert mark_job_run(claimed_job["id"], False, QUOTA_ERROR)
        return {"claimed": True, "success": False, "error": QUOTA_ERROR}

    monkeypatch.setattr(cronjob_tools, "_run_claimed_job", _finish)
    result = cronjob_tools._execute_job_now(job)

    assert result["claimed"] is True
    assert get_job(job["id"])["health"]["state"] == "unknown"
