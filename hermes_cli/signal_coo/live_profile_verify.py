"""Live Torben profile verification helpers."""

from __future__ import annotations

import json
import py_compile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

GMAIL_WATCH_RENEWAL_FLOOR = timedelta(hours=48)
GMAIL_PUBSUB_PULL_FRESHNESS = timedelta(minutes=10)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


@dataclass
class ScriptCheck:
    name: str
    enabled: bool
    script: str | None
    live_path: str | None
    exists: bool = False
    compiles: bool | None = None
    snapshot_in_sync: bool | None = None
    last_status: str | None = None
    last_error: str | None = None
    last_delivery_error: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "script": self.script,
            "live_path": self.live_path,
            "exists": self.exists,
            "compiles": self.compiles,
            "snapshot_in_sync": self.snapshot_in_sync,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "last_delivery_error": self.last_delivery_error,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list):
        raise ValueError(f"Cron jobs file must contain a jobs list: {path}")
    return [job for job in jobs if isinstance(job, dict)]


def _enabled_scripts(jobs: list[dict[str, Any]]) -> set[str]:
    scripts: set[str] = set()
    for job in jobs:
        if not bool(job.get("enabled", True)):
            continue
        script = str(job.get("script") or "").strip()
        if script:
            scripts.add(Path(script).name)
    return scripts


def _verify_gmail_realtime_health(profile: Path, jobs: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    """Verify Gmail realtime health without reading Gmail or Pub/Sub.

    The check uses only local artifacts from the watch renewer and Pub/Sub
    puller so the live-profile verifier stays cheap and non-mutating.
    """

    enabled = _enabled_scripts(jobs)
    if not {"torben_gmail_pubsub_pull.py", "torben_gmail_watch_register.py"} & enabled:
        return {"enabled": False, "status": "skipped", "errors": [], "warnings": []}

    state_dir = profile / "state"
    watch_state_path = state_dir / "torben-gmail-watch-state.json"
    watch_state = _load_json(watch_state_path)
    errors: list[str] = []
    warnings: list[str] = []
    accounts_checked = 0
    soonest_expiration_at: str | None = None

    if not watch_state:
        errors.append(f"gmail realtime watch state missing or unreadable: {watch_state_path}")
    else:
        if watch_state.get("last_watch_registration_status") != "pass":
            errors.append(
                "gmail watch registration status is not pass: "
                f"{watch_state.get('last_watch_registration_status') or 'missing'}"
            )
        accounts = watch_state.get("accounts") if isinstance(watch_state.get("accounts"), dict) else {}
        if not accounts:
            errors.append("gmail realtime watch state has no registered accounts")
        for alias, account_state in sorted(accounts.items()):
            if not isinstance(account_state, dict):
                errors.append(f"{alias}: gmail watch state is malformed")
                continue
            accounts_checked += 1
            expires_at = _parse_timestamp(account_state.get("watch_expiration_at"))
            if expires_at is None:
                errors.append(f"{alias}: gmail watch expiration missing or invalid")
                continue
            if soonest_expiration_at is None or expires_at.isoformat() < soonest_expiration_at:
                soonest_expiration_at = expires_at.isoformat().replace("+00:00", "Z")
            remaining = expires_at - now
            if remaining <= timedelta(0):
                errors.append(f"{alias}: gmail watch is expired")
            elif remaining <= GMAIL_WATCH_RENEWAL_FLOOR:
                errors.append(
                    f"{alias}: gmail watch expires within {int(GMAIL_WATCH_RENEWAL_FLOOR.total_seconds() // 3600)}h"
                )

    pull_latest_path = state_dir / "torben-gmail-pubsub-pull-latest.json"
    pull_latest = _load_json(pull_latest_path)
    pubsub_latest_at: str | None = None
    if "torben_gmail_pubsub_pull.py" in enabled:
        if not pull_latest:
            errors.append(f"gmail Pub/Sub pull latest artifact missing or unreadable: {pull_latest_path}")
        else:
            generated_at = _parse_timestamp(pull_latest.get("generated_at"))
            if generated_at is None:
                errors.append("gmail Pub/Sub pull latest artifact has missing or invalid generated_at")
            else:
                pubsub_latest_at = generated_at.isoformat().replace("+00:00", "Z")
                if now - generated_at > GMAIL_PUBSUB_PULL_FRESHNESS:
                    errors.append(
                        "gmail Pub/Sub pull latest artifact is stale: "
                        f"{pubsub_latest_at}"
                    )
            if pull_latest.get("wakeAgent") and pull_latest.get("reason"):
                warnings.append(f"gmail Pub/Sub pull latest wake reason: {pull_latest.get('reason')}")

    status = "pass" if not errors else "fail"
    return {
        "enabled": True,
        "status": status,
        "accounts_checked": accounts_checked,
        "soonest_watch_expiration_at": soonest_expiration_at,
        "watch_renewal_floor_seconds": int(GMAIL_WATCH_RENEWAL_FLOOR.total_seconds()),
        "pubsub_latest_at": pubsub_latest_at,
        "pubsub_freshness_seconds": int(GMAIL_PUBSUB_PULL_FRESHNESS.total_seconds()),
        "errors": errors,
        "warnings": warnings,
    }


def _script_path(profile_home: Path, script: str | None) -> Path | None:
    if not script:
        return None
    path = Path(script)
    if path.is_absolute():
        return path
    return profile_home / "scripts" / script


def _compile_script(path: Path) -> tuple[bool, str | None]:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return False, str(exc)
    return True, None


def verify_torben_live_profile(
    *,
    profile_home: str | Path,
    repo_snapshot_home: str | Path | None = None,
    check_snapshot_sync: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    profile = Path(profile_home)
    jobs_path = profile / "cron" / "jobs.json"
    snapshot = Path(repo_snapshot_home) if repo_snapshot_home else None
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    errors: list[str] = []
    warnings: list[str] = []

    if not profile.exists():
        return {
            "task": "torben_live_profile_verify",
            "wakeAgent": True,
            "generated_at": _utc_now(),
            "status": "fail",
            "profile_home": str(profile),
            "errors": [f"profile_home missing: {profile}"],
            "warnings": [],
            "script_checks": [],
        }
    if not jobs_path.exists():
        return {
            "task": "torben_live_profile_verify",
            "wakeAgent": True,
            "generated_at": _utc_now(),
            "status": "fail",
            "profile_home": str(profile),
            "errors": [f"cron jobs file missing: {jobs_path}"],
            "warnings": [],
            "script_checks": [],
        }

    jobs = _load_jobs(jobs_path)
    checks: list[ScriptCheck] = []
    for job in jobs:
        enabled = bool(job.get("enabled", True))
        script = str(job.get("script") or "").strip() or None
        name = str(job.get("name") or job.get("id") or "unnamed")
        live_path = _script_path(profile, script)
        check = ScriptCheck(
            name=name,
            enabled=enabled,
            script=script,
            live_path=str(live_path) if live_path else None,
            last_status=job.get("last_status"),
            last_error=job.get("last_error"),
            last_delivery_error=job.get("last_delivery_error"),
        )
        checks.append(check)

        if not enabled:
            continue
        if not script:
            continue
        if live_path is None or not live_path.exists():
            check.errors.append(f"enabled cron script missing: {live_path}")
            continue
        check.exists = True
        compiles, compile_error = _compile_script(live_path)
        check.compiles = compiles
        if not compiles:
            check.errors.append(f"script does not compile: {compile_error}")

        ignore_stale_self_status = script == "torben_live_profile_verify.py"
        if job.get("last_error") and not ignore_stale_self_status:
            check.errors.append(f"last_error is set: {job.get('last_error')}")
        elif job.get("last_error"):
            check.warnings.append(f"ignoring stale verifier self last_error: {job.get('last_error')}")
        if job.get("last_delivery_error") and not ignore_stale_self_status:
            check.errors.append(f"last_delivery_error is set: {job.get('last_delivery_error')}")
        elif job.get("last_delivery_error"):
            check.warnings.append(
                f"ignoring stale verifier self last_delivery_error: {job.get('last_delivery_error')}"
            )
        if (
            str(job.get("last_status") or "").lower() in {"error", "failed", "fail"}
            and not ignore_stale_self_status
        ):
            check.errors.append(f"last_status is failing: {job.get('last_status')}")
        elif str(job.get("last_status") or "").lower() in {"error", "failed", "fail"}:
            check.warnings.append(f"ignoring stale verifier self last_status: {job.get('last_status')}")

        if check_snapshot_sync and snapshot is not None and script and not Path(script).is_absolute():
            snapshot_script = snapshot / "scripts" / script
            if not snapshot_script.exists():
                check.warnings.append(f"repo snapshot script missing: {snapshot_script}")
                check.snapshot_in_sync = None
            else:
                check.snapshot_in_sync = snapshot_script.read_bytes() == live_path.read_bytes()
                if not check.snapshot_in_sync:
                    check.errors.append(f"live script differs from repo snapshot: {snapshot_script}")

    for check in checks:
        errors.extend(f"{check.name}: {error}" for error in check.errors)
        warnings.extend(f"{check.name}: {warning}" for warning in check.warnings)

    gmail_realtime_health = _verify_gmail_realtime_health(profile, jobs, now_utc)
    errors.extend(f"gmail_realtime: {error}" for error in gmail_realtime_health.get("errors") or [])
    warnings.extend(f"gmail_realtime: {warning}" for warning in gmail_realtime_health.get("warnings") or [])

    status = "pass" if not errors else "fail"
    return {
        "task": "torben_live_profile_verify",
        "wakeAgent": status != "pass",
        "generated_at": now_utc.isoformat().replace("+00:00", "Z"),
        "status": status,
        "profile_home": str(profile),
        "jobs_path": str(jobs_path),
        "repo_snapshot_home": str(snapshot) if snapshot else None,
        "enabled_jobs_checked": sum(1 for job in jobs if bool(job.get("enabled", True))),
        "script_checks": [check.to_dict() for check in checks],
        "gmail_realtime_health": gmail_realtime_health,
        "errors": errors,
        "warnings": warnings,
    }


def render_verification_failure(payload: dict[str, Any]) -> str:
    errors = list(payload.get("errors") or [])
    warnings = list(payload.get("warnings") or [])
    lines = [
        "Torben live-profile verification failed.",
        "",
        "Why it matters: enabled cron jobs may be blind or noisy until live profile drift is fixed.",
    ]
    if errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in errors[:20])
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings[:10])
    return "\n".join(lines).strip() + "\n"
