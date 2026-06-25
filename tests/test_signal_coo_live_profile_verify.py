from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli.signal_coo.live_profile_verify import render_verification_failure, verify_torben_live_profile


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
