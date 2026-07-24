"""Automation-friendly Hermes health checks.

``hermes doctor`` is the interactive diagnostic command for humans. It may run
provider/network checks and prints remediation detail. ``hermes health`` is the
small, offline-by-default contract for automation: cron jobs, service managers,
containers, dashboards, and monitoring scripts can depend on stable exit codes
and structured output without spending model/provider quota.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_cli import __version__ as HERMES_VERSION
from hermes_cli.config import get_hermes_home
from hermes_cli.profiles import get_active_profile_name

_STATUS_RANK = {"healthy": 0, "warning": 1, "critical": 2}
_EXIT_CODE = {"healthy": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class HealthRow:
    """One health check result.

    ``status`` is intentionally constrained to the public exit-code contract:
    healthy -> 0, warning -> 1, critical -> 2. ``id`` is the stable machine key;
    ``subsystem`` is human-facing display text.
    """

    id: str
    subsystem: str
    status: str
    detail: str
    action: str = ""


def _aggregate_status(rows: list[HealthRow]) -> str:
    if not rows:
        return "healthy"
    return max(rows, key=lambda row: _STATUS_RANK.get(row.status, 2)).status


def _exit_code(status: str) -> int:
    return _EXIT_CODE.get(status, 2)


def _configured_route(config: dict[str, Any] | None) -> tuple[str, str]:
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider") or "auto")
        model = str(model_cfg.get("default") or model_cfg.get("name") or "(not set)")
    elif isinstance(model_cfg, str):
        provider = "auto"
        model = model_cfg or "(not set)"
    else:
        provider = "auto"
        model = "(not set)"
    return provider, model


def _read_raw_config(config_path: Path) -> tuple[dict[str, Any] | None, HealthRow | None]:
    """Read config.yaml strictly enough for health exit-code semantics.

    ``load_config()`` intentionally falls back to defaults for normal CLI
    resilience. Health is an automation contract, so invalid syntax or a
    non-mapping config file must be surfaced as critical before that fallback.
    This helper is read-only and deliberately does not run config repair/backup
    paths; ``hermes config check`` remains the repair-oriented command.
    """
    if not config_path.exists():
        return None, None
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, HealthRow(
            "profile_config",
            "profile/config",
            "critical",
            f"config.yaml is not valid YAML for profile {get_active_profile_name()}: {exc}",
            "run: hermes config check",
        )
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return None, HealthRow(
            "profile_config",
            "profile/config",
            "critical",
            f"config.yaml must be a mapping/object, got {type(raw).__name__}",
            "run: hermes config check",
        )
    return raw, None


def _check_profile_config(home: Path) -> tuple[HealthRow, dict[str, Any] | None]:
    config_path = home / "config.yaml"
    raw_config, raw_error = _read_raw_config(config_path)
    if raw_error is not None:
        return raw_error, None

    if not config_path.exists():
        return (
            HealthRow(
                "profile_config",
                "profile/config",
                "warning",
                f"profile={get_active_profile_name()} home={home}; config.yaml missing, using defaults",
                "run: hermes setup or hermes config edit",
            ),
            None,
        )
    config = raw_config if isinstance(raw_config, dict) else None
    provider, model = _configured_route(config)
    return (
        HealthRow(
            "profile_config",
            "profile/config",
            "healthy",
            f"profile={get_active_profile_name()} provider={provider} model={model}",
        ),
        config,
    )


def _check_state_db(home: Path) -> HealthRow:
    db_path = home / "state.db"
    if not db_path.exists():
        return HealthRow(
            "state_db",
            "state DB availability",
            "warning",
            f"{db_path} missing; a new DB may be created on first session",
            "run a session or verify HERMES_HOME",
        )
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return HealthRow(
            "state_db",
            "state DB availability",
            "critical",
            f"{db_path} is not readable as SQLite: {exc}",
            "check permissions or restore state.db from backup",
        )
    return HealthRow("state_db", "state DB availability", "healthy", f"readable: {db_path}")


def _last_cron_run_timestamp(jobs: list[dict[str, Any]]) -> str | None:
    """Return the latest valid timestamp from cron's persisted job records."""
    parsed: list[tuple[datetime, str]] = []
    for job in jobs:
        value = job.get("last_run_at")
        if not isinstance(value, str) or not value:
            continue
        try:
            instant = datetime.fromisoformat(value)
        except ValueError:
            continue
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        parsed.append((instant.astimezone(timezone.utc), value))
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def _read_cron_jobs_read_only(home: Path) -> tuple[list[dict[str, Any]], str, str | None]:
    """Read cron/jobs.json without invoking cron repair or migration code."""
    jobs_path = home / "cron" / "jobs.json"
    if not jobs_path.exists():
        return [], "missing", None
    try:
        payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], "invalid", str(exc)
    if isinstance(payload, dict):
        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list):
            return [], "invalid", "jobs field is not a list"
        return [job for job in jobs if isinstance(job, dict)], "current", None
    if isinstance(payload, list):
        return [job for job in payload if isinstance(job, dict)], "legacy_list", None
    return [], "invalid", f"expected object or list, got {type(payload).__name__}"


def _check_cron(home: Path) -> HealthRow:
    jobs, storage_status, error = _read_cron_jobs_read_only(home)
    if storage_status == "invalid":
        return HealthRow(
            "cron_storage",
            "cron scheduler status",
            "critical",
            f"unable to read cron jobs without repair: {error}",
            "run: hermes cron list",
        )
    active = sum(1 for job in jobs if job.get("enabled", True))
    last_tick = _last_cron_run_timestamp(jobs)
    suffix = f"; last run {last_tick}" if last_tick else "; no run history yet"
    if storage_status == "missing":
        detail = f"0 active / 0 total jobs; jobs.json missing{suffix}"
    else:
        detail = f"{active} active / {len(jobs)} total jobs{suffix}"
    if storage_status == "legacy_list":
        return HealthRow(
            "cron_storage",
            "cron scheduler status",
            "warning",
            detail + "; legacy jobs.json list format detected",
            "run: hermes cron list to migrate storage format",
        )
    return HealthRow("cron_storage", "cron scheduler status", "healthy", detail)


def _check_provider_routing(config: dict[str, Any] | None) -> HealthRow:
    provider, model = _configured_route(config)
    if model == "(not set)":
        return HealthRow(
            "provider_routing",
            "provider routing config",
            "warning",
            "no default model configured",
            "run: hermes model",
        )
    return HealthRow(
        "provider_routing",
        "provider routing config",
        "healthy",
        f"configured route {provider}/{model}; no provider/network probe run",
        "run: hermes doctor for live provider diagnostics",
    )


def _check_disk(home: Path) -> HealthRow:
    try:
        usage = shutil.disk_usage(home)
    except Exception as exc:
        return HealthRow("disk", "disk/free space", "critical", f"cannot read disk usage for {home}: {exc}", "check filesystem")
    free_gb = usage.free / (1024 ** 3)
    free_ratio = usage.free / usage.total if usage.total else 0
    detail = f"{free_gb:.1f} GiB free ({free_ratio:.0%}) at {home}"
    if usage.free < 100 * 1024 ** 2 or free_ratio < 0.02:
        return HealthRow("disk", "disk/free space", "critical", detail, "free disk space immediately")
    if usage.free < 1024 ** 3 or free_ratio < 0.10:
        return HealthRow("disk", "disk/free space", "warning", detail, "clean logs/caches or expand disk")
    return HealthRow("disk", "disk/free space", "healthy", detail)


def _check_runtime_modules() -> HealthRow:
    missing = [name for name in ("toolsets", "tools.registry") if importlib.util.find_spec(name) is None]
    if missing:
        return HealthRow(
            "runtime_modules",
            "runtime modules",
            "critical",
            f"core module(s) unavailable: {', '.join(missing)}",
            "run: hermes doctor",
        )
    return HealthRow(
        "runtime_modules",
        "runtime modules",
        "healthy",
        "core tool registry modules are import-resolvable; full imports skipped",
        "run: hermes doctor for dependency-level checks",
    )


def collect_health() -> dict[str, Any]:
    """Collect offline, low-cost health rows and aggregate status."""
    home = get_hermes_home()
    config_row, config = _check_profile_config(home)
    rows: list[HealthRow] = [
        config_row,
        _check_state_db(home),
        _check_cron(home),
        _check_provider_routing(config),
        _check_disk(home),
        _check_runtime_modules(),
    ]
    status = _aggregate_status(rows)
    return {
        "schema_version": 1,
        "status": status,
        "exit_code": _exit_code(status),
        "profile": get_active_profile_name(),
        "hermes_home": str(home),
        "hermes_version": HERMES_VERSION,
        "checks": [asdict(row) for row in rows],
    }


def _print_table(result: dict[str, Any]) -> None:
    print()
    print("Hermes Health")
    print(f"Status: {result['status']} (exit {result['exit_code']})")
    print(f"Profile: {result['profile']}")
    print(f"Hermes home: {result['hermes_home']}")
    print()
    print(f"{'Subsystem':<27} {'Status':<9} {'Detail':<58} Action")
    print("-" * 120)
    for row in result["checks"]:
        detail = str(row.get("detail", ""))
        if len(detail) > 58:
            detail = detail[:55] + "..."
        print(f"{row.get('subsystem',''):<27} {row.get('status',''):<9} {detail:<58} {row.get('action','')}")
    print()


def run_health(args) -> int:
    result = collect_health()
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif getattr(args, "quiet", False) and result["status"] == "healthy":
        pass
    else:
        _print_table(result)
    return int(result["exit_code"])
