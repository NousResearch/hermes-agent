"""Read-only Automation Health summary for Cockpit.

BIF-522 keeps this projection intentionally small: it reads existing local safe
sources, folds them into Marco-facing buckets, and exposes no repair/retry/run
controls or mutation paths.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from gateway.status import get_running_pid, read_runtime_status
from hermes_cli.config import get_hermes_home
from hermes_cli.cockpit import COCKPIT_SCHEMA_VERSION
from hermes_cli.cockpit_daily_ops import get_daily_ops_radar_payload
from hermes_cli.cockpit_n8n import get_n8n_daily_checks_local_payload

_MAX_DETAIL_CHARS = 160
_MAX_SUMMARY_CHARS = 220
_N8N_STALE_SECONDS = 36 * 60 * 60
_DAILY_OPS_STALE_SECONDS = 36 * 60 * 60

_SECRET_RE = re.compile(
    r"\b(?:api[_-]?key|token|secret|password|authorization|credential)\b\s*[:=]?\s*[^\s,;]+",
    re.IGNORECASE,
)
_KEY_RE = re.compile(r"\b(?:sk|pk|ghp|gho|github_pat|xox[abprs])-[-_A-Za-z0-9]{8,}\b", re.IGNORECASE)
_PLATFORM_ID_RE = re.compile(r"(?<![A-Z0-9])[CGDUT](?=[A-Z0-9]*\d)[A-Z0-9]{7,}(?![A-Z0-9])", re.IGNORECASE)
_LONG_ID_RE = re.compile(r"\b[A-Za-z0-9_-]{18,}\b")
_LOCAL_PATH_RE = re.compile(r"(?<!\w)(?:~|/(?:Users|opt|tmp|var|private|Volumes))(?:(?:/|\s+)[^\s`'\")]+)+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)

_MUTATION_SENTINEL: Any = None


def _clean(value: Any, *, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    text = "" if value is None else str(value)
    text = _URL_RE.sub("[link hidden]", text)
    text = _LOCAL_PATH_RE.sub("[local path hidden]", text)
    text = _SECRET_RE.sub(lambda match: f"{match.group(0).split()[0].split(':')[0].split('=')[0]} [redacted]", text)
    text = _KEY_RE.sub("[key redacted]", text)
    text = _PLATFORM_ID_RE.sub("[id hidden]", text)
    text = _LONG_ID_RE.sub("[id hidden]", text)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _epoch_from_any(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        from datetime import datetime

        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return None


def _is_stale(value: Any, max_age_seconds: int, *, now: float) -> bool:
    epoch = _epoch_from_any(value)
    if epoch is None:
        return True
    return now - epoch > max_age_seconds


def _display_last_checked(value: Any) -> str:
    epoch = _epoch_from_any(value)
    if epoch is not None:
        return datetime.fromtimestamp(epoch).isoformat(timespec="seconds")
    return _clean(value, max_chars=80)


def _safe_last_checked(*values: Any, fallback: str = "available now") -> str:
    for value in values:
        text = _display_last_checked(value)
        if text:
            return text
    return fallback


def _card(title: str, bucket: str, summary: str, *, source: str, last_checked: str, details: list[str] | None = None) -> dict[str, Any]:
    safe_details = [_clean(item, max_chars=_MAX_DETAIL_CHARS) for item in (details or [])]
    safe_details = [item for item in safe_details if item][:3]
    payload: dict[str, Any] = {
        "title": _clean(title, max_chars=80),
        "bucket": bucket if bucket in {"attention", "healthy", "stale_or_failing"} else "stale_or_failing",
        "summary": _clean(summary),
        "last_checked": _clean(last_checked, max_chars=100),
        "source": _clean(source, max_chars=80),
    }
    if safe_details:
        payload["details"] = safe_details
    return payload


def _n8n_card(now: float) -> dict[str, Any]:
    try:
        payload = get_n8n_daily_checks_local_payload()
    except Exception:
        return _card(
            "Daily automation checks",
            "stale_or_failing",
            "Daily n8n checks could not be read, so Marco should treat automation freshness as unknown.",
            source="n8n daily checks",
            last_checked="read failed",
        )
    checks = payload.get("checks") if isinstance(payload, Mapping) else []
    checks = checks if isinstance(checks, list) else []
    bad = []
    healthy = 0
    unknown = 0
    latest: Any = payload.get("inventory_checked_at") or payload.get("generated_at") if isinstance(payload, Mapping) else None
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        status = str(check.get("execution_status") or check.get("status") or "unknown").lower()
        if any(term in status for term in ("fail", "error", "crashed")):
            bad.append(_clean(check.get("name") or "Daily check", max_chars=80))
        elif any(term in status for term in ("success", "ok", "observed", "complete")):
            healthy += 1
        else:
            unknown += 1
        candidate_latest = check.get("last_completed") or check.get("last_started")
        if candidate_latest and (_epoch_from_any(candidate_latest) or 0) >= (_epoch_from_any(latest) or 0):
            latest = candidate_latest
    stale = bool(payload.get("stale") or payload.get("fallback")) if isinstance(payload, Mapping) else True
    if _is_stale(latest, _N8N_STALE_SECONDS, now=now):
        stale = True
    if stale:
        bucket = "stale_or_failing"
        if bad:
            summary = (
                "Daily automation checks are stale or incomplete; the local snapshot still contains "
                f"{len(bad)} stale error signal(s), so it should not be treated as current live failure state."
            )
        elif healthy:
            summary = f"Daily automation checks are stale or incomplete; {healthy} look healthy and {unknown} are unknown."
        else:
            summary = "Daily automation checks are stale or fallback-only; no live repair action is wired."
    elif bad:
        bucket = "attention"
        summary = f"{len(bad)} daily automation check needs review; {healthy} look healthy."
    elif unknown:
        bucket = "stale_or_failing"
        if healthy:
            summary = f"Daily automation checks are stale or incomplete; {healthy} look healthy and {unknown} are unknown."
        else:
            summary = "Daily automation checks are stale or fallback-only; no live repair action is wired."
    else:
        bucket = "healthy"
        summary = f"Daily automation checks look healthy across {healthy or len(checks)} observed row(s)."
    return _card(
        "Daily automation checks",
        bucket,
        summary,
        source="n8n daily checks",
        last_checked=_safe_last_checked(latest, payload.get("generated_at") if isinstance(payload, Mapping) else None),
        details=(bad if bad and not stale else ["No repair, retry, or workflow trigger controls are exposed here."]),
    )


def _daily_ops_card(now: float) -> dict[str, Any]:
    try:
        payload = get_daily_ops_radar_payload()
    except Exception:
        return _card(
            "Daily Ops Radar",
            "stale_or_failing",
            "Daily Ops Radar metadata could not be read from existing cron output.",
            source="Daily Ops Radar cron output",
            last_checked="read failed",
        )
    job = payload.get("job") if isinstance(payload, Mapping) else {}
    summary = payload.get("summary") if isinstance(payload, Mapping) else {}
    job = job if isinstance(job, Mapping) else {}
    summary = summary if isinstance(summary, Mapping) else {}
    last_run = summary.get("last_run") or job.get("last_run_at") or payload.get("generated_at")
    status = str(job.get("status") or "unknown").lower()
    relevant = summary.get("relevant_change_count")
    behind = summary.get("behind_count")
    stale = _is_stale(last_run, _DAILY_OPS_STALE_SECONDS, now=now)
    if any(term in status for term in ("fail", "error")):
        bucket = "attention"
        text = "Daily Ops Radar last run reported a failure and needs Marco review."
    elif stale or payload.get("source") == "cron_output_missing":
        bucket = "stale_or_failing"
        text = "Daily Ops Radar has no fresh output yet, so upgrade awareness may be stale."
    elif isinstance(relevant, int) and relevant > 0:
        bucket = "attention"
        text = f"Daily Ops Radar found {relevant} ops-adjacent upstream change(s) across {behind or 0} behind commit(s); this is an upgrade-review signal, not an automation failure."
    else:
        bucket = "healthy"
        text = "Daily Ops Radar ran recently and has no urgent upstream-change review showing."
    return _card(
        "Daily Ops Radar",
        bucket,
        text,
        source="Daily Ops Radar cron output",
        last_checked=_safe_last_checked(last_run),
        details=["Read-only review metadata only; no git, service, external-send, or automation repair action is wired."],
    )


def _cron_card(now: float) -> dict[str, Any]:
    path = Path(get_hermes_home()) / "cron" / "jobs.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _card(
            "Scheduled jobs",
            "stale_or_failing",
            "Cron job metadata is not readable from the local jobs store.",
            source="local cron jobs metadata",
            last_checked="read failed",
        )
    jobs = payload.get("jobs", []) if isinstance(payload, Mapping) else []
    jobs = jobs if isinstance(jobs, list) else []
    disabled = 0
    failing = 0
    stale = 0
    newest: Any = None
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        if job.get("enabled") is False:
            disabled += 1
        status = str(job.get("last_status") or job.get("status") or "").lower()
        if any(term in status for term in ("fail", "error")):
            failing += 1
        last = job.get("last_run_at") or job.get("updated_at")
        if _is_stale(last, 7 * 24 * 60 * 60, now=now):
            stale += 1
        newest = last or newest
    if failing:
        bucket = "attention"
        text = f"{failing} scheduled job(s) last reported failure; review before relying on automations."
    elif stale and jobs:
        bucket = "stale_or_failing"
        text = f"{stale} scheduled job(s) have no recent run metadata."
    else:
        bucket = "healthy"
        text = f"{len(jobs)} scheduled job(s) are readable; {disabled} intentionally disabled."
    return _card(
        "Scheduled jobs",
        bucket,
        text,
        source="local cron jobs metadata",
        last_checked=_safe_last_checked(newest, fallback="metadata read now"),
        details=[f"{disabled} disabled job(s) hidden from action controls."],
    )


def _gateway_card() -> dict[str, Any]:
    try:
        pid = get_running_pid(cleanup_stale=False)
        runtime = read_runtime_status() or {}
    except Exception:
        pid = None
        runtime = {}
    updated = runtime.get("updated_at") or runtime.get("last_heartbeat") if isinstance(runtime, Mapping) else None
    platforms = runtime.get("platforms") if isinstance(runtime, Mapping) else None
    platform_count = len(platforms) if isinstance(platforms, Mapping) else 0
    if pid:
        return _card(
            "Gateway/local status",
            "healthy",
            f"Gateway appears to be running locally with {platform_count} platform status group(s) visible.",
            source="gateway runtime status",
            last_checked=_safe_last_checked(updated, fallback="runtime read now"),
            details=["Process identifiers and raw platform internals are hidden."],
        )
    return _card(
        "Gateway/local status",
        "stale_or_failing",
        "Gateway is not currently visible as running from local status metadata.",
        source="gateway runtime status",
        last_checked=_safe_last_checked(updated, fallback="runtime read now"),
    )


def _dashboard_card() -> dict[str, Any]:
    return _card(
        "Dashboard access",
        "healthy",
        "Cockpit is serving authenticated read-only automation health. This section is observer-only and exposes no action controls.",
        source="dashboard local status",
        last_checked="request time",
    )


def get_automation_health_payload() -> dict[str, Any]:
    """Return a simplified, authenticated, read-only automation health summary."""
    now = time.time()
    cards = [_n8n_card(now), _daily_ops_card(now), _cron_card(now), _gateway_card(), _dashboard_card()]
    order = {"attention": 0, "stale_or_failing": 1, "healthy": 2}
    cards = sorted(cards, key=lambda card: (order.get(str(card.get("bucket")), 9), str(card.get("title"))))[:6]
    counts = {
        "attention": sum(1 for card in cards if card.get("bucket") == "attention"),
        "healthy": sum(1 for card in cards if card.get("bucket") == "healthy"),
        "stale_or_failing": sum(1 for card in cards if card.get("bucket") == "stale_or_failing"),
    }
    if counts["attention"]:
        headline = f"{counts['attention']} automation area(s) need Marco review."
    elif counts["stale_or_failing"]:
        headline = f"{counts['stale_or_failing']} automation area(s) look stale or incomplete."
    else:
        headline = "Automation checks look healthy from the safe local sources."
    return {
        "schema_version": COCKPIT_SCHEMA_VERSION,
        "read_only": True,
        "actions_enabled": False,
        "mutation_enabled": False,
        "external_delivery_enabled": False,
        "generated_at": now,
        "reading_model": {
            "attention": "Needs Marco review now.",
            "healthy": "Looks fine from local read-only metadata.",
            "stale_or_failing": "Missing, old, or failing metadata.",
            "last_checked_source": "When and where Cockpit read the summary from.",
        },
        "summary": {
            **counts,
            "headline": headline,
            "last_checked": "request time",
            "source": "n8n daily checks, Daily Ops Radar, cron metadata, gateway runtime, dashboard local status",
        },
        "cards": cards,
    }
