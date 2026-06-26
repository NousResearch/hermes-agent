from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli.signal_coo.live_profile_verify import (
    clear_live_profile_investigation_request,
    consume_live_profile_investigation_request,
    live_profile_failure_fingerprint,
    render_verification_failure,
    stage_live_profile_investigation_request,
    update_live_profile_alert_state,
    verify_torben_live_profile,
)


def _write_jobs(profile: Path, jobs: list[dict]) -> None:
    jobs_path = profile / "cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text(json.dumps({"jobs": jobs}, indent=2) + "\n", encoding="utf-8")


def _write_script(root: Path, name: str, body: str = "print('ok')\n") -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / name).write_text(body, encoding="utf-8")


def _write_gmail_health_state(profile: Path, *, expiration_at: str, pull_generated_at: str) -> None:
    state = profile / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "torben-gmail-watch-state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "last_watch_registration_status": "pass",
                "accounts": {
                    "personal": {
                        "alias": "personal",
                        "email": "eric@example.com",
                        "history_id": "123",
                        "watch_expiration_at": expiration_at,
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (state / "torben-gmail-pubsub-pull-latest.json").write_text(
        json.dumps(
            {
                "task": "torben_gmail_pubsub_pull",
                "wakeAgent": False,
                "generated_at": pull_generated_at,
                "reason": "no pubsub notifications",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_backend_artifact(profile: Path, name: str, payload: dict) -> None:
    state = profile / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_live_profile_verify_passes_enabled_scripts_and_snapshot_sync(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    snapshot = tmp_path / "snapshot"
    _write_jobs(
        profile,
        [
            {"name": "active", "enabled": True, "script": "active.py", "last_status": "ok"},
            {"name": "disabled", "enabled": False, "script": "missing-disabled.py", "last_status": "error"},
        ],
    )
    _write_script(profile, "active.py")
    _write_script(snapshot, "active.py")

    payload = verify_torben_live_profile(profile_home=profile, repo_snapshot_home=snapshot)

    assert payload["status"] == "pass"
    assert payload["wakeAgent"] is False
    assert payload["submanager_contract_health"]["status"] == "pass"
    assert payload["submanager_contract_health"]["contracts"]["finance"]["provider"] == "ratatosk_robinhood_v01"
    active = [item for item in payload["script_checks"] if item["name"] == "active"][0]
    assert active["exists"] is True
    assert active["compiles"] is True
    assert active["snapshot_in_sync"] is True


def test_live_profile_verify_fails_missing_enabled_script(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    _write_jobs(profile, [{"name": "missing", "enabled": True, "script": "missing.py"}])

    payload = verify_torben_live_profile(profile_home=profile)

    assert payload["status"] == "fail"
    assert payload["wakeAgent"] is True
    assert "missing: enabled cron script missing" in payload["errors"][0]


def test_live_profile_verify_fails_compile_error(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    _write_jobs(profile, [{"name": "bad", "enabled": True, "script": "bad.py"}])
    _write_script(profile, "bad.py", "def nope(:\n")

    payload = verify_torben_live_profile(profile_home=profile)

    assert payload["status"] == "fail"
    assert any("script does not compile" in error for error in payload["errors"])


def test_live_profile_verify_fails_stale_cron_errors(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    _write_jobs(
        profile,
        [
            {
                "name": "errored",
                "enabled": True,
                "script": "ok.py",
                "last_status": "ok",
                "last_error": "script missing",
                "last_delivery_error": "delivery failed",
            }
        ],
    )
    _write_script(profile, "ok.py")

    payload = verify_torben_live_profile(profile_home=profile)

    assert payload["status"] == "fail"
    assert any("last_error is set" in error for error in payload["errors"])
    assert any("last_delivery_error is set" in error for error in payload["errors"])


def test_live_profile_verify_fails_backend_error_artifact(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    _write_jobs(profile, [{"name": "torben-gtm-radar", "enabled": True, "script": "torben_gtm_radar.py"}])
    _write_script(profile, "torben_gtm_radar.py")
    _write_backend_artifact(
        profile,
        "torben-gtm-radar-latest.json",
        {
            "task": "torben_gtm_radar_adapter",
            "wakeAgent": True,
            "error": {
                "type": "RuntimeError",
                "message": "Magnus GTM radar refresh failed",
            },
            "public_actions_taken": 0,
            "external_mutations": 0,
        },
    )

    payload = verify_torben_live_profile(profile_home=profile)

    assert payload["status"] == "fail"
    assert any("backend_artifacts: torben_gtm_radar.py: latest artifact has error" in error for error in payload["errors"])
    assert payload["backend_artifact_health"]["artifacts"][0]["status"] == "fail"


def test_live_profile_verify_fails_backend_forbidden_mutation_count(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    _write_jobs(profile, [{"name": "torben-finance-radar", "enabled": True, "script": "torben_finance_radar.py"}])
    _write_script(profile, "torben_finance_radar.py")
    _write_backend_artifact(
        profile,
        "torben-finance-radar-latest.json",
        {
            "task": "torben_finance_radar",
            "wakeAgent": True,
            "generated_at": "2026-06-26T12:00:00Z",
            "public_actions_taken": 0,
            "external_mutations": 0,
            "broker_orders_submitted": 1,
        },
    )

    payload = verify_torben_live_profile(profile_home=profile)

    assert payload["status"] == "fail"
    assert any("backend_artifacts: torben_finance_radar.py: broker_orders_submitted is nonzero: 1" in error for error in payload["errors"])


def test_live_profile_verify_ignores_stale_self_status_after_repair(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    snapshot = tmp_path / "snapshot"
    _write_jobs(
        profile,
        [
            {
                "name": "torben-live-profile-verify",
                "enabled": True,
                "script": "torben_live_profile_verify.py",
                "last_status": "error",
                "last_error": "previous drift before repair",
                "last_delivery_error": "previous delivery failure",
            }
        ],
    )
    _write_script(profile, "torben_live_profile_verify.py")
    _write_script(snapshot, "torben_live_profile_verify.py")

    payload = verify_torben_live_profile(profile_home=profile, repo_snapshot_home=snapshot)

    assert payload["status"] == "pass"
    assert payload["wakeAgent"] is False
    assert payload["errors"] == []
    assert any("ignoring stale verifier self last_error" in warning for warning in payload["warnings"])
    assert any("ignoring stale verifier self last_status" in warning for warning in payload["warnings"])


def test_live_profile_verify_fails_snapshot_drift(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    snapshot = tmp_path / "snapshot"
    _write_jobs(profile, [{"name": "drifted", "enabled": True, "script": "job.py"}])
    _write_script(profile, "job.py", "print('live')\n")
    _write_script(snapshot, "job.py", "print('snapshot')\n")

    payload = verify_torben_live_profile(profile_home=profile, repo_snapshot_home=snapshot)

    assert payload["status"] == "fail"
    assert any("live script differs from repo snapshot" in error for error in payload["errors"])


def test_live_profile_verify_passes_healthy_gmail_realtime_state(tmp_path: Path) -> None:
    now = datetime(2026, 6, 25, 20, 0, tzinfo=timezone.utc)
    profile = tmp_path / "profile"
    _write_jobs(
        profile,
        [
            {"name": "pull", "enabled": True, "script": "torben_gmail_pubsub_pull.py", "last_status": "ok"},
            {"name": "renew", "enabled": True, "script": "torben_gmail_watch_register.py", "last_status": "ok"},
        ],
    )
    _write_script(profile, "torben_gmail_pubsub_pull.py")
    _write_script(profile, "torben_gmail_watch_register.py")
    _write_gmail_health_state(
        profile,
        expiration_at="2026-07-02T20:00:00Z",
        pull_generated_at="2026-06-25T19:59:30Z",
    )

    payload = verify_torben_live_profile(profile_home=profile, now=now)

    assert payload["status"] == "pass"
    assert payload["gmail_realtime_health"]["status"] == "pass"
    assert payload["gmail_realtime_health"]["accounts_checked"] == 1


def test_live_profile_verify_fails_gmail_watch_inside_renewal_floor(tmp_path: Path) -> None:
    now = datetime(2026, 6, 25, 20, 0, tzinfo=timezone.utc)
    profile = tmp_path / "profile"
    _write_jobs(
        profile,
        [
            {"name": "pull", "enabled": True, "script": "torben_gmail_pubsub_pull.py", "last_status": "ok"},
            {"name": "renew", "enabled": True, "script": "torben_gmail_watch_register.py", "last_status": "ok"},
        ],
    )
    _write_script(profile, "torben_gmail_pubsub_pull.py")
    _write_script(profile, "torben_gmail_watch_register.py")
    _write_gmail_health_state(
        profile,
        expiration_at="2026-06-27T19:59:00Z",
        pull_generated_at="2026-06-25T19:59:30Z",
    )

    payload = verify_torben_live_profile(profile_home=profile, now=now)

    assert payload["status"] == "fail"
    assert payload["gmail_realtime_health"]["status"] == "fail"
    assert any("gmail watch expires within 48h" in error for error in payload["errors"])


def test_render_verification_failure_is_actionable(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    _write_jobs(profile, [{"name": "missing", "enabled": True, "script": "missing.py"}])
    payload = verify_torben_live_profile(profile_home=profile)

    text = render_verification_failure(payload)

    assert "Torben live-profile verification failed." in text
    assert "enabled cron jobs may be blind or noisy" in text
    assert "missing: enabled cron script missing" in text


def test_live_profile_alert_state_suppresses_duplicate_failure(tmp_path: Path) -> None:
    state_path = tmp_path / "alert-state.json"
    now = datetime(2026, 6, 25, 20, 0, tzinfo=timezone.utc)
    payload = {
        "task": "torben_live_profile_verify",
        "status": "fail",
        "wakeAgent": True,
        "errors": ["torben-gmail-pubsub-pull: live script differs from repo snapshot"],
    }

    first_suppressed = update_live_profile_alert_state(payload=payload, state_path=state_path, now=now)
    repeated = dict(payload)
    repeated.pop("alert_dedupe", None)
    second_suppressed = update_live_profile_alert_state(
        payload=repeated,
        state_path=state_path,
        now=now.replace(minute=15),
    )

    assert first_suppressed is False
    assert payload["alert_dedupe"]["status"] == "new_failure"
    assert second_suppressed is True
    assert repeated["alert_dedupe"]["status"] == "duplicate_failure_suppressed"
    assert repeated["alert_dedupe"]["duplicate_count"] == 1
    assert repeated["alert_dedupe"]["fingerprint"] == live_profile_failure_fingerprint(payload)


def test_live_profile_alert_state_alerts_changed_failure_and_clears_on_pass(tmp_path: Path) -> None:
    state_path = tmp_path / "alert-state.json"
    now = datetime(2026, 6, 25, 20, 0, tzinfo=timezone.utc)
    first = {"status": "fail", "errors": ["old drift"], "wakeAgent": True}
    changed = {"status": "fail", "errors": ["new drift"], "wakeAgent": True}
    passed = {"status": "pass", "errors": [], "wakeAgent": False}

    assert update_live_profile_alert_state(payload=first, state_path=state_path, now=now) is False
    assert update_live_profile_alert_state(payload=changed, state_path=state_path, now=now.replace(minute=15)) is False
    assert changed["alert_dedupe"]["status"] == "new_failure"
    assert update_live_profile_alert_state(payload=passed, state_path=state_path, now=now.replace(minute=30)) is False

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "pass"
    assert state["active_fingerprint"] is None
    assert passed["alert_dedupe"]["status"] == "cleared"


def test_live_profile_investigation_request_stages_failure_evidence(tmp_path: Path) -> None:
    request_path = tmp_path / "investigation-request.json"
    payload = {
        "task": "torben_live_profile_verify",
        "status": "fail",
        "wakeAgent": True,
        "generated_at": "2026-06-25T20:00:00Z",
        "profile_home": "/tmp/profile",
        "jobs_path": "/tmp/profile/cron/jobs.json",
        "repo_snapshot_home": "/tmp/repo/profiles/torben",
        "errors": ["torben-gmail-pubsub-pull: live script differs from repo snapshot"],
        "warnings": ["gmail_realtime: pull latest wake reason: test"],
        "script_checks": [
            {
                "name": "torben-gmail-pubsub-pull",
                "script": "torben_gmail_pubsub_pull.py",
                "errors": ["live script differs from repo snapshot"],
            },
            {"name": "healthy", "script": "healthy.py", "errors": []},
        ],
        "gmail_realtime_health": {"status": "pass"},
    }

    request = stage_live_profile_investigation_request(payload=payload, request_path=request_path)

    assert request["wakeAgent"] is True
    assert request["status"] == "pending"
    assert request["failure_fingerprint"] == live_profile_failure_fingerprint(payload)
    assert request["failing_script_checks"][0]["name"] == "torben-gmail-pubsub-pull"
    assert request["investigation_contract"]["approval_required_before_fix"] is True
    assert payload["investigation_request"]["status"] == "staged"


def test_live_profile_investigation_consumer_hands_failure_to_llm_once(tmp_path: Path) -> None:
    request_path = tmp_path / "investigation-request.json"
    state_path = tmp_path / "investigation-state.json"
    payload = {
        "status": "fail",
        "wakeAgent": True,
        "errors": ["missing script"],
        "warnings": [],
        "script_checks": [],
    }
    stage_live_profile_investigation_request(payload=payload, request_path=request_path)

    first = consume_live_profile_investigation_request(request_path=request_path, state_path=state_path)
    second = consume_live_profile_investigation_request(request_path=request_path, state_path=state_path)

    assert first["wakeAgent"] is True
    assert first["status"] == "investigation_requested"
    assert first["request"]["errors"] == ["missing script"]
    assert first["operator_boundary"]["llm_must_not"][0] == "edit files"
    assert second["wakeAgent"] is False
    assert second["status"] == "duplicate_investigation_suppressed"


def test_live_profile_investigation_request_clears_on_pass(tmp_path: Path) -> None:
    request_path = tmp_path / "investigation-request.json"
    state_path = tmp_path / "investigation-state.json"
    passed = {"status": "pass", "wakeAgent": False, "errors": []}

    clear_live_profile_investigation_request(payload=passed, request_path=request_path)
    result = consume_live_profile_investigation_request(request_path=request_path, state_path=state_path)

    assert result["wakeAgent"] is False
    assert result["status"] == "idle"
    assert passed["investigation_request"]["status"] == "cleared"
