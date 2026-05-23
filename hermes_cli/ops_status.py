"""Read-only, redacted operator status for Hermes.

This command is intentionally metadata-first. It may inspect file metadata and
bounded log tails for counts, but it does not print raw log lines, cron prompts,
private memory, env values, credentials, or provider facts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.redact import redact_sensitive_text
from hermes_cli.config import get_hermes_home
from hermes_cli.gateway_validation import (
    CANONICAL_LAUNCHD_LABEL,
    LEGACY_LAUNCHD_LABEL,
    build_gateway_validation_report,
)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_TAIL_BYTES = 64 * 1024
DISK_WARN_FREE_BYTES = 2 * 1024 * 1024 * 1024
DISK_WARN_USED_FRACTION = 0.90
SAFE_CRON_STATUS_BUCKETS = {
    "unknown",
    "ok",
    "error",
    "success",
    "failed",
    "running",
    "pending",
    "paused",
    "blocked",
    "cancelled",
    "skipped",
}


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {str(k): _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


def _iso_from_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    try:
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except (OSError, OverflowError, ValueError):
        return None


def _age_seconds(timestamp: float | None) -> int | None:
    if timestamp is None:
        return None
    try:
        return max(0, int(datetime.now(tz=timezone.utc).timestamp() - timestamp))
    except (OSError, OverflowError, ValueError):
        return None


def _file_metadata(path: Path, *, include_mode: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "path": _redact(str(path)),
        "exists": False,
        "kind": "missing",
    }
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return data
    except OSError as exc:
        data.update(
            {
                "kind": "unknown",
                "error": _redact(type(exc).__name__),
            }
        )
        return data

    if path.is_dir():
        kind = "directory"
    elif path.is_file():
        kind = "file"
    else:
        kind = "other"
    data.update(
        {
            "exists": True,
            "kind": kind,
            "bytes": stat_result.st_size,
            "mtime": _iso_from_timestamp(stat_result.st_mtime),
            "age_seconds": _age_seconds(stat_result.st_mtime),
        }
    )
    if include_mode:
        data["mode"] = oct(stat_result.st_mode & 0o777)
    return data


def _read_tail(path: Path, max_bytes: int = LOG_TAIL_BYTES) -> str:
    try:
        with path.open("rb") as fh:
            try:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes))
            except OSError:
                fh.seek(0)
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return ""


def _log_metadata(path: Path) -> dict[str, Any]:
    data = _file_metadata(path)
    data["content_included"] = False
    data["recent_warning_count"] = 0
    data["recent_error_count"] = 0
    if data.get("exists") and data.get("kind") == "file":
        tail = _read_tail(path)
        upper = tail.upper()
        data["recent_warning_count"] = upper.count("WARNING") + upper.count("WARN ")
        data["recent_error_count"] = upper.count("ERROR") + upper.count("CRITICAL")
    return data


def _operator_root() -> Path:
    return Path(os.environ.get("HERMES_OPERATOR_ROOT") or (Path.home() / "Operator"))


def _health_loop_root() -> Path:
    return Path(
        os.environ.get("HERMES_HEALTH_LOOP_ROOT")
        or (_operator_root() / "health-loop")
    )


def _cron_jobs_path(hermes_home: Path) -> Path:
    return hermes_home / "cron" / "jobs.json"


def _cron_summary(hermes_home: Path) -> dict[str, Any]:
    jobs_path = _cron_jobs_path(hermes_home)
    summary: dict[str, Any] = {
        "path": _redact(str(jobs_path)),
        "exists": jobs_path.exists(),
        "content_included": False,
        "total_jobs": 0,
        "enabled_jobs": 0,
        "paused_jobs": 0,
        "last_status_counts": {},
        "status": "missing",
    }
    if not jobs_path.exists():
        return summary
    try:
        raw = jobs_path.read_text(encoding="utf-8", errors="replace")
        payload = json.loads(raw or "[]")
    except json.JSONDecodeError:
        summary["status"] = "parse_error"
        return summary
    except OSError as exc:
        summary["status"] = "read_error"
        summary["error"] = _redact(type(exc).__name__)
        return summary

    if isinstance(payload, dict):
        candidate = payload.get("jobs")
        jobs = candidate if isinstance(candidate, list) else []
    elif isinstance(payload, list):
        jobs = payload
    else:
        jobs = []

    status_counts: dict[str, int] = {}
    enabled = 0
    paused = 0
    for item in jobs:
        if not isinstance(item, dict):
            continue
        if item.get("enabled", True) is False or item.get("paused") is True:
            paused += 1
        else:
            enabled += 1
        raw_status = str(item.get("last_status") or "unknown").strip().lower()
        status = raw_status if raw_status in SAFE_CRON_STATUS_BUCKETS else "other"
        status_counts[status] = status_counts.get(status, 0) + 1

    summary.update(
        {
            "status": "ok",
            "total_jobs": len([item for item in jobs if isinstance(item, dict)]),
            "enabled_jobs": enabled,
            "paused_jobs": paused,
            "last_status_counts": status_counts,
        }
    )
    return summary


def _disk_summary(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    data: dict[str, Any] = {
        "path": _redact(str(path)),
        "probe_path": _redact(str(probe)),
        "exists": path.exists(),
        "status": "unknown",
    }
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        data["error"] = _redact(type(exc).__name__)
        return data
    used_fraction = usage.used / usage.total if usage.total else math.nan
    free_warn = usage.free < DISK_WARN_FREE_BYTES
    used_warn = bool(math.isfinite(used_fraction) and used_fraction >= DISK_WARN_USED_FRACTION)
    data.update(
        {
            "status": "warn" if free_warn or used_warn else "ok",
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": round(used_fraction * 100, 1) if math.isfinite(used_fraction) else None,
        }
    )
    return data


def _health_guardian_summary() -> dict[str, Any]:
    root = _health_loop_root()
    status_md = root / "status.md"
    status_json = root / "status.json"
    return {
        "root": _redact(str(root)),
        "content_included": False,
        "status_md": _file_metadata(status_md),
        "status_json": _file_metadata(status_json),
    }


def _logs_summary(hermes_home: Path) -> dict[str, Any]:
    logs_dir = hermes_home / "logs"
    files = {
        "gateway": logs_dir / "gateway.log",
        "gateway_error": logs_dir / "gateway.error.log",
        "agent": logs_dir / "agent.log",
        "errors": logs_dir / "errors.log",
    }
    return {
        "directory": _file_metadata(logs_dir, include_mode=True),
        "content_included": False,
        "tail_bytes_scanned_per_file": LOG_TAIL_BYTES,
        "files": {name: _log_metadata(path) for name, path in files.items()},
    }


def _receipt_paths(hermes_home: Path) -> dict[str, Any]:
    root = _health_loop_root()
    return {
        "health_loop_status_md": _redact(str(root / "status.md")),
        "health_loop_status_json": _redact(str(root / "status.json")),
        "gateway_incident_bundle_command": (
            "hermes gateway incident-bundle --output /tmp/hermes-gateway-incident --force"
        ),
        "operator_quickstart": _redact(str(PROJECT_ROOT / "docs" / "HERMES_OPERATOR_QUICKSTART.md")),
        "build_log": _redact(str(PROJECT_ROOT / "docs" / "HERMES_BUILD_LOG.md")),
        "logs_directory": _redact(str(hermes_home / "logs")),
    }


def _positive_timeout(value: Any, name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number of seconds") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return timeout


def build_ops_status_report(
    *,
    check_health: bool = True,
    launchctl_timeout: float = 5.0,
    health_timeout: float = 2.0,
) -> dict[str, Any]:
    """Build a redacted, read-only status report for local operators."""
    hermes_home = get_hermes_home()
    gateway_report = build_gateway_validation_report(
        check_health=check_health,
        launchctl_timeout=launchctl_timeout,
        health_timeout=health_timeout,
    )
    gateway_report = _redact(gateway_report)

    health_guardian = _health_guardian_summary()
    cron = _cron_summary(hermes_home)
    logs = _logs_summary(hermes_home)
    disk = {
        "hermes_home": _disk_summary(hermes_home),
        "repo": _disk_summary(PROJECT_ROOT),
        "operator": _disk_summary(_operator_root()),
    }

    checks: list[dict[str, Any]] = []
    if gateway_report.get("overall_status") == "fail":
        checks.append(
            {
                "id": "gateway.validation",
                "status": "fail",
                "severity": "error",
                "message": "Gateway validation reported a startup failure.",
            }
        )
    else:
        checks.append(
            {
                "id": "gateway.validation",
                "status": "pass",
                "severity": "info",
                "message": "Gateway validation did not report startup failure.",
            }
        )

    if cron.get("status") == "parse_error":
        checks.append(
            {
                "id": "cron.jobs_metadata",
                "status": "warn",
                "severity": "warning",
                "message": "Cron jobs metadata exists but could not be parsed.",
            }
        )
    elif cron.get("status") == "read_error":
        checks.append(
            {
                "id": "cron.jobs_metadata",
                "status": "warn",
                "severity": "warning",
                "message": "Cron jobs metadata exists but could not be read.",
            }
        )
    else:
        checks.append(
            {
                "id": "cron.jobs_metadata",
                "status": "pass",
                "severity": "info",
                "message": "Cron summary includes counts only.",
            }
        )

    recent_log_errors = 0
    recent_log_warnings = 0
    for item in logs["files"].values():
        if isinstance(item, dict):
            recent_log_errors += int(item.get("recent_error_count") or 0)
            recent_log_warnings += int(item.get("recent_warning_count") or 0)
    if recent_log_errors:
        checks.append(
            {
                "id": "logs.recent_errors",
                "status": "warn",
                "severity": "warning",
                "message": "Recent log tails contain error markers; raw lines are not included.",
                "evidence": {"recent_error_count": recent_log_errors},
            }
        )
    if recent_log_warnings:
        checks.append(
            {
                "id": "logs.recent_warnings",
                "status": "warn",
                "severity": "warning",
                "message": "Recent log tails contain warning markers; raw lines are not included.",
                "evidence": {"recent_warning_count": recent_log_warnings},
            }
        )

    for disk_id, disk_item in disk.items():
        if disk_item.get("status") == "warn":
            checks.append(
                {
                    "id": f"disk.{disk_id}",
                    "status": "warn",
                    "severity": "warning",
                    "message": "Disk usage is near the local warning threshold.",
                    "evidence": {
                        "free_bytes": disk_item.get("free_bytes"),
                        "used_percent": disk_item.get("used_percent"),
                    },
                }
            )

    errors = sum(1 for item in checks if item.get("status") == "fail")
    warnings = sum(1 for item in checks if item.get("status") == "warn")
    overall_status = "pass" if errors == 0 else "fail"
    return {
        "schema_version": 1,
        "owner": "hermes-ops-plane",
        "risk_tier": "R0",
        "read_only": True,
        "redacted": True,
        "overall_status": overall_status,
        "summary": {
            "checks": len(checks),
            "errors": errors,
            "warnings": warnings,
        },
        "runtime": {
            "project_root": _redact(str(PROJECT_ROOT)),
            "hermes_home": _redact(str(hermes_home)),
            "operator_root": _redact(str(_operator_root())),
            "canonical_gateway_label": CANONICAL_LAUNCHD_LABEL,
            "legacy_gateway_label": LEGACY_LAUNCHD_LABEL,
        },
        "gateway": gateway_report,
        "health_guardian": health_guardian,
        "cron": cron,
        "api": gateway_report.get("health", {}),
        "disk": disk,
        "logs": logs,
        "receipts": _receipt_paths(hermes_home),
        "checks": checks,
        "next_actions": [
            "hermes gateway validate",
            "hermes doctor",
            "hermes logs gateway --since 30m --level WARNING",
        ],
    }


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_bytes(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while amount >= 1024 and idx < len(units) - 1:
        amount /= 1024
        idx += 1
    if idx == 0:
        return f"{int(amount)} {units[idx]}"
    return f"{amount:.1f} {units[idx]}"


def format_ops_status_text(report: dict[str, Any]) -> str:
    gateway = report.get("gateway", {})
    launchd = gateway.get("launchd", {}) if isinstance(gateway, dict) else {}
    program = launchd.get("program_summary") or {}
    legacy = launchd.get("legacy_label_state") or {}
    health = report.get("api", {}) if isinstance(report.get("api"), dict) else {}
    cron = report.get("cron", {}) if isinstance(report.get("cron"), dict) else {}
    logs = report.get("logs", {}).get("files", {}) if isinstance(report.get("logs"), dict) else {}
    receipts = report.get("receipts", {}) if isinstance(report.get("receipts"), dict) else {}

    lines = [
        f"Hermes ops status: {str(report['overall_status']).upper()}",
        (
            f"Read only: {_format_bool(report.get('read_only'))}  "
            f"Redacted: {_format_bool(report.get('redacted'))}  "
            f"Checks: {report['summary']['checks']}  "
            f"Errors: {report['summary']['errors']}  "
            f"Warnings: {report['summary']['warnings']}"
        ),
        "",
        "Gateway",
        f"  Validation: {str(gateway.get('overall_status', 'unknown')).upper()}",
        f"  Active label: {launchd.get('active_label') or 'unknown'}",
        f"  Canonical label: {CANONICAL_LAUNCHD_LABEL}",
        f"  Legacy loaded: {_format_bool(legacy.get('loaded'))}",
        f"  Wrapper backed: {_format_bool(program.get('uses_expected_wrapper'))}",
        f"  Expected wrapper: {launchd.get('expected_wrapper') or 'unknown'}",
        "",
        "API",
        f"  Enabled: {_format_bool(health.get('enabled'))}",
        f"  Auth configured: {_format_bool(health.get('auth_configured'))}",
        f"  /health: {health.get('health_status', 'skipped')}",
        f"  /health/detailed: {health.get('detailed_status', 'skipped')}",
        "",
        "Cron",
        f"  Status: {cron.get('status', 'unknown')}",
        f"  Jobs: {cron.get('total_jobs', 0)} total, {cron.get('enabled_jobs', 0)} enabled, {cron.get('paused_jobs', 0)} paused",
        "",
        "Logs",
    ]
    for name, item in logs.items():
        if not isinstance(item, dict):
            continue
        lines.append(
            "  "
            f"{name}: exists={_format_bool(item.get('exists'))} "
            f"size={_format_bytes(item.get('bytes'))} "
            f"warnings={item.get('recent_warning_count', 0)} "
            f"errors={item.get('recent_error_count', 0)}"
        )
    lines.extend(["", "Disk"])
    for name, item in report.get("disk", {}).items():
        if not isinstance(item, dict):
            continue
        lines.append(
            "  "
            f"{name}: {item.get('status', 'unknown')} "
            f"free={_format_bytes(item.get('free_bytes'))} "
            f"used={item.get('used_percent', 'unknown')}%"
        )
    lines.extend(
        [
            "",
            "Receipts",
            f"  Health guardian: {receipts.get('health_loop_status_md', 'unknown')}",
            f"  Incident bundle: {receipts.get('gateway_incident_bundle_command', 'unknown')}",
            "",
            "Checks",
        ]
    )
    for item in report.get("checks", []):
        lines.append(f"  [{str(item.get('status', 'unknown')).upper()}] {item.get('id')}: {item.get('message')}")
    lines.extend(["", "Next actions"])
    for action in report.get("next_actions", []):
        lines.append(f"  {action}")
    return "\n".join(lines) + "\n"


def format_ops_status_markdown(report: dict[str, Any]) -> str:
    gateway = report.get("gateway", {})
    launchd = gateway.get("launchd", {}) if isinstance(gateway, dict) else {}
    program = launchd.get("program_summary") or {}
    legacy = launchd.get("legacy_label_state") or {}
    health = report.get("api", {}) if isinstance(report.get("api"), dict) else {}
    cron = report.get("cron", {}) if isinstance(report.get("cron"), dict) else {}
    logs = report.get("logs", {}).get("files", {}) if isinstance(report.get("logs"), dict) else {}
    receipts = report.get("receipts", {}) if isinstance(report.get("receipts"), dict) else {}

    lines = [
        "# Hermes Ops Status",
        "",
        f"- Status: `{str(report['overall_status']).upper()}`",
        f"- Read only: `{str(bool(report.get('read_only'))).lower()}`",
        f"- Redacted: `{str(bool(report.get('redacted'))).lower()}`",
        f"- Risk tier: `{report.get('risk_tier', 'unknown')}`",
        (
            f"- Checks: `{report['summary']['checks']}` "
            f"(errors `{report['summary']['errors']}`, warnings `{report['summary']['warnings']}`)"
        ),
        "",
        "## Gateway",
        "",
        f"- Validation: `{str(gateway.get('overall_status', 'unknown')).upper()}`",
        f"- Active label: `{launchd.get('active_label') or 'unknown'}`",
        f"- Canonical label: `{CANONICAL_LAUNCHD_LABEL}`",
        f"- Legacy loaded: `{str(bool(legacy.get('loaded'))).lower()}`",
        f"- Wrapper backed: `{str(bool(program.get('uses_expected_wrapper'))).lower()}`",
        f"- Expected wrapper: `{launchd.get('expected_wrapper') or 'unknown'}`",
        "",
        "## API",
        "",
        f"- Enabled: `{str(bool(health.get('enabled'))).lower()}`",
        f"- Auth configured: `{str(bool(health.get('auth_configured'))).lower()}`",
        f"- `/health`: `{health.get('health_status', 'skipped')}`",
        f"- `/health/detailed`: `{health.get('detailed_status', 'skipped')}`",
        "",
        "## Cron",
        "",
        f"- Status: `{cron.get('status', 'unknown')}`",
        f"- Jobs: `{cron.get('total_jobs', 0)}` total, `{cron.get('enabled_jobs', 0)}` enabled, `{cron.get('paused_jobs', 0)}` paused",
        "- Content included: `false`",
        "",
        "## Logs",
        "",
        "| Log | Exists | Size | Warnings | Errors |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, item in logs.items():
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            f"`{name}` | "
            f"`{str(bool(item.get('exists'))).lower()}` | "
            f"`{_format_bytes(item.get('bytes'))}` | "
            f"`{item.get('recent_warning_count', 0)}` | "
            f"`{item.get('recent_error_count', 0)}` |"
        )
    lines.extend(
        [
            "",
            "Raw log lines are not included.",
            "",
            "## Disk",
            "",
            "| Path | Status | Free | Used |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name, item in report.get("disk", {}).items():
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            f"`{name}` | "
            f"`{item.get('status', 'unknown')}` | "
            f"`{_format_bytes(item.get('free_bytes'))}` | "
            f"`{item.get('used_percent', 'unknown')}%` |"
        )
    lines.extend(
        [
            "",
            "## Receipts",
            "",
            f"- Health guardian: `{receipts.get('health_loop_status_md', 'unknown')}`",
            f"- Incident bundle: `{receipts.get('gateway_incident_bundle_command', 'unknown')}`",
            f"- Operator quickstart: `{receipts.get('operator_quickstart', 'unknown')}`",
            "",
            "## Checks",
            "",
        ]
    )
    for item in report.get("checks", []):
        lines.append(
            f"- `{str(item.get('status', 'unknown')).upper()}` "
            f"`{item.get('id')}`: {item.get('message')}"
        )
    lines.extend(["", "## Next Actions", ""])
    for action in report.get("next_actions", []):
        lines.append(f"- `{action}`")
    return "\n".join(lines) + "\n"


def run_ops_status(args: Any) -> bool:
    raw_launchctl_timeout = getattr(args, "launchctl_timeout", None)
    if raw_launchctl_timeout is None:
        raw_launchctl_timeout = 5.0
    raw_health_timeout = getattr(args, "health_timeout", None)
    if raw_health_timeout is None:
        raw_health_timeout = 2.0
    try:
        launchctl_timeout = _positive_timeout(raw_launchctl_timeout, "--launchctl-timeout")
        health_timeout = _positive_timeout(raw_health_timeout, "--health-timeout")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False

    report = build_ops_status_report(
        check_health=not bool(getattr(args, "no_health", False)),
        launchctl_timeout=launchctl_timeout,
        health_timeout=health_timeout,
    )
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    elif getattr(args, "markdown", False):
        print(format_ops_status_markdown(report), end="")
    else:
        print(format_ops_status_text(report), end="")
    return report["overall_status"] == "pass"


def ops_command(args: Any) -> bool:
    subcommand = getattr(args, "ops_command", None)
    if subcommand in {None, "", "status"}:
        return run_ops_status(args)
    print(f"Unknown ops command: {subcommand}", file=sys.stderr)
    print("Usage: hermes ops status [--json|--markdown] [--no-health]", file=sys.stderr)
    return False
