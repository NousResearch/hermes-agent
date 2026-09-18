"""Honcho production smoke monitor.

This module powers the standalone cron/watchdog script in ``scripts/honcho_monitor.py``.
It keeps the output compact and model-aware while checking for the silent-failure
cases that matter in production:

- loaded embedding config in the Honcho API container
- DB document/message embedding dimensions
- queue/doc growth deltas between runs
- recent representation-save / 401 errors from the deriver
- spark-goat / spark-polarbear chat latency, not just model liveness
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# Use Teleport SSH by default. The Honcho host is enrolled and available via
# the cron tbot config, which keeps unattended runs off the interactive LAN path.
# Keep HONCHO_MONITOR_HOST as an explicit escape hatch.
HONCHO_TARGET = os.environ.get("HONCHO_MONITOR_HOST", "ubuntu@honcho.teleport.batumi.works")
STATE_PATH = Path(
    os.environ.get(
        "HONCHO_MONITOR_STATE",
        str(Path.home() / ".hermes" / "cache" / "honcho_monitor.json"),
    )
)
HERMES_STATE_DB = Path(os.environ.get("HONCHO_MONITOR_HERMES_STATE_DB", str(Path.home() / ".hermes" / "state.db")))
INGESTION_STALE_SECONDS = int(os.environ.get("HONCHO_MONITOR_INGESTION_STALE_SECONDS", "3600"))
DERIVER_STALE_ACTIVE_SECONDS = int(os.environ.get("HONCHO_MONITOR_DERIVER_STALE_ACTIVE_SECONDS", "600"))
DREAM_STALE_ACTIVE_SECONDS = int(os.environ.get("HONCHO_MONITOR_DREAM_STALE_ACTIVE_SECONDS", "1800"))
SSH_FAILURE_ALERT_THRESHOLD = int(os.environ.get("HONCHO_MONITOR_SSH_FAILURE_ALERT_THRESHOLD", "2"))
SPARK_CHAT_DEGRADED_SECONDS = 5.0
SPARK_CHAT_CRITICAL_SECONDS = 30.0
LOCAL_DISK_MIN_FREE_BYTES = 10 * 1024**3
LOCAL_DISK_MAX_USED_PERCENT = 95


HOST_MAP = {
    "192.168.10.211:8001": "spark-goat",
    "192.168.10.211:11435": "spark-goat",
    "100.69.54.37:8001": "spark-goat",
    "100.69.54.37:11435": "spark-goat",
    "100.71.155.95:8001": "spark-polarbear",
    "100.71.155.95:11435": "spark-polarbear",
    "100.71.155.95:18081": "spark-polarbear",
    "100.110.104.77:8087": "mac-studio",
    "192.168.100.14:8088": "mac-horse",
    "192.168.10.154:8081": "proxmox-geekom",
    "192.168.10.104:8000": "mac-zebra",
    "api.openai.com": "openai",
    "openrouter.ai/api/v1": "openrouter",
}

# Parse the log in Python rather than sending a Python regex to ``grep -E``.
# Only error-level records and unquoted OpenAI exception lines are eligible;
# derived/boxed message content, access-log ports, and timestamp milliseconds
# must not be mistaken for current authentication or representation failures.
_DERIVER_ERROR_LOG_LINE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})? - .+? - )?"
    r"(?:ERROR|CRITICAL) - (?P<message>.*)$"
)
_AUTH_EXCEPTION_LINE = re.compile(
    r"^(?:openai\.)?(?:AuthenticationError|APIStatusError):",
    re.IGNORECASE,
)
_AUTH_ERROR_SIGNAL = re.compile(
    r"(?:\bError code:\s*401\b|\b401 Unauthorized\b|\binvalid_api_key\b|"
    r"\bUnauthorized\b|\bAuthenticationError\b|api\.openai\.com.*\b401\b)",
    re.IGNORECASE,
)
_SAVE_REPRESENTATION_SIGNAL = re.compile(
    r"\bFailed to save representation\b",
    re.IGNORECASE,
)

# Prefer the LAN endpoint for spark-goat. The historical Tailscale address
# (100.69.54.37) can refuse :8001 while Honcho and Hermes both reach the live
# vLLM service through the current LAN address advertised in Honcho's loaded
# config.
SPARK_CHAT_BASE = os.environ.get("HONCHO_MONITOR_SPARK_CHAT_BASE", "http://192.168.10.211:8001")


@dataclass(frozen=True)
class HonchoSnapshot:
    services: dict[str, bool]
    pipeline: dict[str, dict[str, str]]
    db: dict[str, Any]
    queue: dict[str, int]
    queue_by_type: dict[str, dict[str, int]]
    errors: dict[str, int]
    spark_goat: dict[str, Any]
    deriver: dict[str, Any]
    ingestion: dict[str, Any] = field(default_factory=dict)
    observer: dict[str, Any] = field(default_factory=dict)


def sanitize_base_url(url: str | None) -> str:
    """Remove URL credentials, query parameters, and fragments."""
    if not url:
        return ""
    try:
        parts = urlsplit(str(url))
        host = parts.hostname
        if not parts.scheme or not host:
            return "[redacted-url]"
        rendered_host = f"[{host}]" if ":" in host else host
        if parts.port is not None:
            rendered_host = f"{rendered_host}:{parts.port}"
        return urlunsplit((parts.scheme, rendered_host, parts.path, "", ""))
    except (TypeError, ValueError):
        return "[redacted-url]"


def short_host(url: str | None) -> str:
    if not url:
        return "?"
    url = sanitize_base_url(url)
    for needle, short in HOST_MAP.items():
        if needle in url:
            return short
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if not host:
            return "?"
        rendered_host = f"[{host}]" if ":" in host else host
        return f"{rendered_host}:{parts.port}" if parts.port is not None else rendered_host
    except (TypeError, ValueError):
        return "?"


def mask(value: str | None) -> str:
    if not value:
        return "****"
    value = str(value)
    if len(value) < 8:
        return "****"
    return f"{value[:4]}{'*' * max(4, len(value) - 4)}"


def parse_pipeline_env(raw: str) -> dict[str, dict[str, str]]:
    """Extract the pipeline model config from ``docker exec ... env`` output."""
    fields = {
        "embedding": {
            "model": "EMBEDDING_MODEL_CONFIG__MODEL",
            "base_url": "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL",
            "dimensions_mode": "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE",
            "vector_dimensions": "EMBEDDING_VECTOR_DIMENSIONS",
        },
        "deriver": {
            "model": "DERIVER_MODEL_CONFIG__MODEL",
            "base_url": "DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL",
            "workers": "DERIVER_WORKERS",
        },
        "summary": {
            "model": "SUMMARY_MODEL_CONFIG__MODEL",
            "base_url": "SUMMARY_MODEL_CONFIG__OVERRIDES__BASE_URL",
        },
        "dream": {
            "model": "DREAM_DEDUCTION_MODEL_CONFIG__MODEL",
            "base_url": "DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__BASE_URL",
        },
        "dialectic": {
            "model": "DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL",
            "base_url": "DIALECTIC_LEVELS__low__MODEL_CONFIG__OVERRIDES__BASE_URL",
        },
    }

    env: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

    parsed: dict[str, dict[str, str]] = {}
    for stage, mapping in fields.items():
        parsed[stage] = {
            out_key: (
                sanitize_base_url(env.get(in_key, ""))
                if out_key == "base_url"
                else env.get(in_key, "")
            )
            for out_key, in_key in mapping.items()
        }
    return parsed


def load_state(path: Path | str = STATE_PATH) -> dict[str, Any]:
    state_path = Path(path)
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_state(path: Path | str, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{state_path.name}.",
        suffix=".tmp",
        dir=str(state_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, state_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def preserve_last_valid_probe_state(
    snapshot: HonchoSnapshot,
    current_state: dict[str, Any],
    previous_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep counter baselines when a probe failed instead of recording zeroes."""
    merged = dict(current_state)
    if snapshot.db.get("probe_ok") is False and previous_state:
        for key in ("documents_total", "messages_total"):
            if key in previous_state:
                merged[key] = previous_state[key]
    return merged


def _now_utc(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _fmt_s(value: Any) -> str:
    try:
        return f"{float(value):.1f}s"
    except Exception:
        return "?"


def _service_row(services: dict[str, bool]) -> str:
    labels = (("api_ok", "API"), ("deriver_up", "Deriver"), ("db_ok", "DB"), ("redis_ok", "Redis"))
    if services.get("ssh_ok") is False:
        return " ".join(f"⚪ {label}" for _, label in labels)
    return " ".join(
        f"{'🟢' if services.get(key, False) else '🔴'} {label}"
        for key, label in labels
    )


def _ssh_error_reason(services: dict[str, Any]) -> str:
    raw = str(services.get("ssh_error") or "unknown transport error")
    raw = raw.removeprefix("__SSH_ERROR__").strip()
    raw = re.sub(r"https://login\.tailscale\.com/a/\S+", "<redacted-auth-url>", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:180] or "unknown transport error"


def restore_last_valid_remote_sample(
    snapshot: HonchoSnapshot,
    previous_state: dict[str, Any] | None,
) -> HonchoSnapshot:
    """Show the last valid remote sample when only the SSH transport failed."""
    if snapshot.services.get("ssh_ok") is not False or not previous_state:
        return snapshot

    replacements: dict[str, Any] = {}
    for key in ("pipeline", "db", "queue", "queue_by_type", "errors", "deriver"):
        value = previous_state.get(key)
        if isinstance(value, dict) and value:
            replacements[key] = value
    return replace(snapshot, **replacements) if replacements else snapshot


def _queue_type_delta(snapshot: HonchoSnapshot, previous_state: dict[str, Any], task_type: str) -> int:
    """Return the non-negative done-count delta for one queue task type."""
    current_done = int(snapshot.queue_by_type.get(task_type, {}).get("done", 0))
    previous_by_type = previous_state.get("queue_by_type", {})
    if not isinstance(previous_by_type, dict) or not previous_by_type:
        return 0
    prev_done = int(previous_by_type.get(task_type, {}).get("done", 0))
    return max(0, current_done - prev_done)


def _trend_line(snapshot: HonchoSnapshot, previous_state: dict[str, Any] | None) -> str | None:
    if not previous_state:
        return None

    prev_docs = int(previous_state.get("documents_total", snapshot.db.get("documents_total", 0)))
    docs_delta = max(0, int(snapshot.db.get("documents_total", 0)) - prev_docs)
    rep_delta = _queue_type_delta(snapshot, previous_state, "representation")
    reconciler_delta = _queue_type_delta(snapshot, previous_state, "reconciler")
    webhook_delta = _queue_type_delta(snapshot, previous_state, "webhook")
    dream_delta = _queue_type_delta(snapshot, previous_state, "dream")
    return (
        "Δ since previous sample: "
        f"representation +{rep_delta} · "
        f"reconciler +{reconciler_delta} · "
        f"webhook +{webhook_delta} · "
        f"dream +{dream_delta} · "
        f"docs +{docs_delta}"
    )


def _embedding_vector_dimensions_display(embed: dict[str, str], db: dict[str, Any]) -> str:
    configured = (embed.get("vector_dimensions") or "").strip()
    if configured:
        return configured
    doc_dims = db.get("documents_dims")
    msg_dims = db.get("messages_dims")
    if (
        doc_dims not in (None, "", 0, "0")
        and msg_dims not in (None, "", 0, "0")
        and str(doc_dims) == str(msg_dims)
    ):
        return f"{doc_dims} (inferred from DB)"
    return ""


def _has_fresh_dream_contention(snapshot: HonchoSnapshot) -> bool:
    spark = snapshot.spark_goat or {}
    deriver = snapshot.deriver or {}
    return (
        bool(spark.get("ok"))
        and not bool(spark.get("thinking"))
        and float(spark.get("latency_s", 0.0) or 0.0) > SPARK_CHAT_DEGRADED_SECONDS
        and int(deriver.get("active_dream_count") or 0) > 0
        and int(float(deriver.get("active_dream_oldest_age_s") or 0)) < DREAM_STALE_ACTIVE_SECONDS
    )


def next_dream_contention_streak(
    snapshot: HonchoSnapshot,
    previous_state: dict[str, Any] | None = None,
) -> int:
    if not _has_fresh_dream_contention(snapshot):
        return 0
    return int((previous_state or {}).get("dream_contention_streak", 0)) + 1


def _has_representation_stall_candidate(
    snapshot: HonchoSnapshot,
    previous_state: dict[str, Any] | None,
) -> bool:
    if not previous_state:
        return False
    prev_rep_state = previous_state.get("queue_by_type", {}).get("representation")
    if prev_rep_state is None:
        return False

    deriver = snapshot.deriver or {}
    if int(deriver.get("runs_15m") or 0) > 0:
        return False

    active_count = int(deriver.get("active_count") or 0)
    has_per_task_active_stats = "active_representation_count" in deriver
    active_representation_count = (
        int(deriver.get("active_representation_count") or 0)
        if has_per_task_active_stats
        else active_count
    )
    active_dream_count = int(deriver.get("active_dream_count") or 0)
    active_dream_oldest_age_s = int(float(deriver.get("active_dream_oldest_age_s") or 0))
    active_other_count = int(deriver.get("active_other_count") or 0)
    active_other_oldest_age_s = int(float(deriver.get("active_other_oldest_age_s") or 0))
    try:
        deriver_workers = int(snapshot.pipeline.get("deriver", {}).get("workers") or 0)
    except (TypeError, ValueError):
        deriver_workers = 0
    sole_worker_has_fresh_non_representation = deriver_workers == 1 and (
        (active_dream_count > 0 and active_dream_oldest_age_s < DREAM_STALE_ACTIVE_SECONDS)
        or (active_other_count > 0 and active_other_oldest_age_s < DERIVER_STALE_ACTIVE_SECONDS)
    )

    rep_done = int(snapshot.queue_by_type.get("representation", {}).get("done", 0))
    prev_rep_done = int(prev_rep_state.get("done", 0))
    rep_delta = max(0, rep_done - prev_rep_done)
    prev_docs = int(previous_state.get("documents_total", snapshot.db.get("documents_total", 0)))
    docs_delta = max(0, int(snapshot.db.get("documents_total", 0)) - prev_docs)
    rep_pending = int(snapshot.queue_by_type.get("representation", {}).get("pending", 0))
    prev_rep_pending = int(prev_rep_state.get("pending", 0))
    return (
        active_representation_count == 0
        and not sole_worker_has_fresh_non_representation
        and rep_pending > 0
        and prev_rep_pending > 0
        and rep_delta == 0
        and docs_delta == 0
    )


def next_representation_stall_streak(
    snapshot: HonchoSnapshot,
    previous_state: dict[str, Any] | None = None,
) -> int:
    if not _has_representation_stall_candidate(snapshot, previous_state):
        return 0
    return int((previous_state or {}).get("representation_stall_streak", 0)) + 1


def build_alerts(snapshot: HonchoSnapshot, previous_state: dict[str, Any] | None = None) -> list[str]:
    alerts: list[str] = []

    observer = snapshot.observer or {}
    disk_free_bytes = int(observer.get("disk_free_bytes") or 0)
    disk_used_percent = int(observer.get("disk_used_percent") or 0)
    if observer and (
        disk_free_bytes < LOCAL_DISK_MIN_FREE_BYTES
        or disk_used_percent >= LOCAL_DISK_MAX_USED_PERCENT
    ):
        alerts.append(
            "Local observer disk critically low "
            f"({_format_bytes(disk_free_bytes)} free, {disk_used_percent}% used)"
        )

    svc = snapshot.services
    if svc.get("ssh_ok") is False:
        previous_streak = int((previous_state or {}).get("ssh_failure_streak", 0))
        streak = previous_streak + 1
        if streak >= SSH_FAILURE_ALERT_THRESHOLD:
            alerts.append(
                f"Honcho SSH probe failed ({streak} consecutive): {_ssh_error_reason(svc)}"
            )
        return alerts
    for key, label in (("api_ok", "API"), ("deriver_up", "Deriver"), ("db_ok", "DB"), ("redis_ok", "Redis")):
        if not svc.get(key, False):
            alerts.append(f"{label} down")

    if svc.get("db_ok") and snapshot.db.get("probe_ok") is False:
        alerts.append("DB stats probe failed")
        return alerts

    embed = snapshot.pipeline.get("embedding", {})
    embed_model = embed.get("model", "")
    embed_url = embed.get("base_url", "")
    vector_dims = (embed.get("vector_dimensions") or "").strip()
    doc_dims = snapshot.db.get("documents_dims")
    msg_dims = snapshot.db.get("messages_dims")
    db_dims_consistent = (
        doc_dims not in (None, "", 0, "0")
        and msg_dims not in (None, "", 0, "0")
        and str(doc_dims) == str(msg_dims)
    )
    if not embed_model or not embed_url:
        alerts.append("Embedding config missing")
    elif "api.openai.com" in embed_url or embed_model == "text-embedding-3-small":
        alerts.append("Embedding config looks like OpenAI fallback")
    elif not vector_dims and not db_dims_consistent:
        alerts.append("Embedding vector dimensions missing from env")

    if vector_dims and doc_dims not in (None, "", 0, "0") and str(doc_dims) != vector_dims:
        alerts.append("Document embedding dims mismatch")
    if vector_dims and msg_dims not in (None, "", 0, "0") and str(msg_dims) != vector_dims:
        alerts.append("Message embedding dims mismatch")

    if int(snapshot.errors.get("save_representation", 0)) > 0 or int(snapshot.errors.get("four_oh_one", 0)) > 0:
        alerts.append("Recent representation save / 401 errors")

    spark = snapshot.spark_goat
    if not spark.get("skipped"):
        if not spark.get("ok"):
            alerts.append("spark-goat chat failed")
        elif spark.get("thinking"):
            alerts.append("spark-goat thinking still enabled")
        elif float(spark.get("latency_s", 0.0) or 0.0) > SPARK_CHAT_DEGRADED_SECONDS:
            deriver = snapshot.deriver or {}
            active_dream_count = int(deriver.get("active_dream_count") or 0)
            active_dream_age_s = int(float(deriver.get("active_dream_oldest_age_s") or 0))
            if active_dream_count > 0 and active_dream_age_s < DREAM_STALE_ACTIVE_SECONDS:
                previous_streak = int((previous_state or {}).get("dream_contention_streak", 0))
                if (
                    float(spark.get("latency_s") or 0.0) >= SPARK_CHAT_CRITICAL_SECONDS
                    or previous_streak >= 1
                ):
                    alerts.append(
                        "spark-goat dream contention "
                        f"(chat latency {float(spark.get('latency_s') or 0.0):.1f}s)"
                    )
            else:
                alerts.append("spark-goat chat latency degraded")

    ingestion = snapshot.ingestion or {}
    if ingestion.get("source_fresh") and not ingestion.get("downstream_fresh"):
        try:
            drift_s = int(float(ingestion.get("drift_s") or 0))
        except Exception:
            drift_s = 0
        if drift_s > INGESTION_STALE_SECONDS:
            alerts.append(f"Hermes→Honcho ingestion stale ({_format_age(drift_s)} drift)")

    deriver = snapshot.deriver or {}
    active_count = int(deriver.get("active_count") or 0)
    active_oldest_age_s = int(float(deriver.get("active_oldest_age_s") or 0))
    has_per_task_active_stats = "active_representation_count" in deriver
    active_representation_count = (
        int(deriver.get("active_representation_count") or 0)
        if has_per_task_active_stats
        else active_count
    )
    active_dream_count = int(deriver.get("active_dream_count") or 0)
    active_dream_oldest_age_s = int(float(deriver.get("active_dream_oldest_age_s") or 0))
    active_other_count = int(deriver.get("active_other_count") or 0)
    active_other_oldest_age_s = int(float(deriver.get("active_other_oldest_age_s") or 0))

    if previous_state:
        prev_rep_state = previous_state.get("queue_by_type", {}).get("representation")
        if prev_rep_state is not None:
            prev_rep_done = int(prev_rep_state.get("done", 0))
            rep_done = int(snapshot.queue_by_type.get("representation", {}).get("done", 0))
            prev_docs = int(previous_state.get("documents_total", snapshot.db.get("documents_total", 0)))
            docs_delta = max(0, snapshot.db.get("documents_total", 0) - prev_docs)
            rep_delta = max(0, rep_done - prev_rep_done)
            # A one-item gap is normal: a representation unit can complete before
            # the corresponding document counter is visible in the next snapshot.
            # Alert only on a larger drift, not routine tick jitter.
            if rep_delta >= 5 and rep_delta - docs_delta > 1:
                alerts.append("Representation queue advancing faster than documents")

            if (
                _has_representation_stall_candidate(snapshot, previous_state)
                and int(previous_state.get("representation_stall_streak", 0)) >= 1
            ):
                alerts.append("Deriver stalled: representation backlog with no progress")

    active_work_unit_key = str(deriver.get("active_oldest_work_unit_key") or "")
    active_task_type = active_work_unit_key.partition(":")[0]
    if has_per_task_active_stats:
        active_groups = (
            (
                "Representation",
                active_representation_count,
                int(float(deriver.get("active_representation_oldest_age_s") or 0)),
                DERIVER_STALE_ACTIVE_SECONDS,
            ),
            (
                "Dream",
                active_dream_count,
                active_dream_oldest_age_s,
                DREAM_STALE_ACTIVE_SECONDS,
            ),
            (
                "Other",
                active_other_count,
                active_other_oldest_age_s,
                DERIVER_STALE_ACTIVE_SECONDS,
            ),
        )
        for label, count, oldest_age_s, threshold in active_groups:
            if count > 0 and oldest_age_s >= threshold:
                alerts.append(
                    f"{label} active work stale ({count} active, oldest {_format_age(oldest_age_s)})"
                )
    else:
        # Backward compatibility for cached state and callers that predate the
        # per-task query. New snapshots always report all three active groups.
        stale_threshold = (
            DREAM_STALE_ACTIVE_SECONDS
            if active_task_type == "dream"
            else DERIVER_STALE_ACTIVE_SECONDS
        )
        if active_count > 0 and active_oldest_age_s >= stale_threshold:
            label = "Dream" if active_task_type == "dream" else "Deriver"
            alerts.append(
                f"{label} active work stale ({active_count} active, oldest {_format_age(active_oldest_age_s)})"
            )

    return alerts


def _format_age(seconds: int | float | None) -> str:
    try:
        total = max(0, int(float(seconds or 0)))
    except Exception:
        return "?"
    if total >= 86400:
        return f"{total / 86400:.1f}d"
    if total >= 3600:
        return f"{total / 3600:.1f}h"
    if total >= 60:
        return f"{total // 60}m"
    return f"{total}s"


def _format_bytes(value: int | float | None) -> str:
    size = max(0.0, float(value or 0))
    for unit in ("bytes", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _parse_pg_timestamptz(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("+00"):
        text = text[:-3] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    return _now_utc(dt)


def latest_local_message_timestamp(path: Path | str = HERMES_STATE_DB) -> float | None:
    try:
        con = sqlite3.connect(Path(path))
        try:
            row = con.execute(
                """
                SELECT max(m.timestamp)
                FROM messages AS m
                LEFT JOIN sessions AS s ON s.id = m.session_id
                WHERE m.role IN ('user','assistant')
                  AND m.content IS NOT NULL
                  AND length(m.content) > 0
                  AND (s.source IS NULL OR s.source != 'cron')
                """
            ).fetchone()
        finally:
            con.close()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except Exception:
        return None


def build_ingestion_status(local_ts: float | None, honcho_latest_raw: str | None, now: datetime | None = None) -> dict[str, Any]:
    now_dt = _now_utc(now)
    honcho_dt = _parse_pg_timestamptz(honcho_latest_raw)
    local_dt = datetime.fromtimestamp(local_ts, tz=timezone.utc) if local_ts else None
    source_age_s = int((now_dt - local_dt).total_seconds()) if local_dt else None
    downstream_age_s = int((now_dt - honcho_dt).total_seconds()) if honcho_dt else None
    drift_s = int((local_dt - honcho_dt).total_seconds()) if local_dt and honcho_dt else None
    return {
        "local_latest_ts": local_ts,
        "local_latest_iso": local_dt.isoformat() if local_dt else "",
        "honcho_latest_raw": (honcho_latest_raw or "").strip(),
        "honcho_latest_iso": honcho_dt.isoformat() if honcho_dt else "",
        "source_age_s": source_age_s,
        "downstream_age_s": downstream_age_s,
        "drift_s": drift_s,
        "source_fresh": source_age_s is not None and source_age_s <= INGESTION_STALE_SECONDS,
        "downstream_fresh": downstream_age_s is not None and downstream_age_s <= INGESTION_STALE_SECONDS,
    }


def _normalize_queue_by_type(raw: Any) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    if not isinstance(raw, dict):
        return result
    for task_type, counts in raw.items():
        if isinstance(counts, dict):
            result[task_type] = {
                "pending": int(counts.get("pending", 0)),
                "done": int(counts.get("done", 0)),
            }
    return result


def format_report(snapshot: dict[str, Any] | HonchoSnapshot, previous_state: dict[str, Any] | None = None, now: datetime | None = None) -> str:
    if isinstance(snapshot, dict):
        snapshot = HonchoSnapshot(
            services=dict(snapshot.get("services", {})),
            pipeline={k: dict(v) for k, v in snapshot.get("pipeline", {}).items()},
            db=dict(snapshot.get("db", {})),
            queue={k: int(v) for k, v in snapshot.get("queue", {}).items()},
            queue_by_type=_normalize_queue_by_type(snapshot.get("queue_by_type")),
            errors={k: int(v) for k, v in snapshot.get("errors", {}).items()},
            spark_goat=dict(snapshot.get("spark_goat", {})),
            deriver=dict(snapshot.get("deriver", {})),
            ingestion=dict(snapshot.get("ingestion", {})),
            observer=dict(snapshot.get("observer", {})),
        )

    lines = [f"🩺 Honcho — {_now_utc(now).strftime('%H:%M UTC')}"]
    lines.append(_service_row(snapshot.services))
    if snapshot.services.get("ssh_ok") is False:
        lines.append("  Remote sample: stale (SSH unavailable; last valid counters/config shown)")

    for label, key in (
        ("Embedding", "embedding"),
        ("Deriver", "deriver"),
        ("Summary", "summary"),
        ("Dream", "dream"),
        ("Dialectic", "dialectic"),
    ):
        stage = snapshot.pipeline.get(key, {})
        lines.append(f"  {label}: {stage.get('model', '?')} @ {short_host(stage.get('base_url'))}")

    embed = snapshot.pipeline.get("embedding", {})
    vector_dims_display = _embedding_vector_dimensions_display(embed, snapshot.db)
    lines.append(
        "  Embedding env: "
        f"model={embed.get('model', '?')} "
        f"base_url={sanitize_base_url(embed.get('base_url')) or '?'} "
        f"dims_mode={embed.get('dimensions_mode', '?')} "
        f"vector_dims={vector_dims_display}"
    )

    lines.append(
        "  Embedding DB: "
        f"docs {snapshot.db.get('documents_with_embeddings', 0)}/{snapshot.db.get('documents_total', 0)} "
        f"dims={snapshot.db.get('documents_dims', '?')} · "
        f"messages {snapshot.db.get('messages_with_embeddings', 0)}/{snapshot.db.get('messages_total', 0)} "
        f"dims={snapshot.db.get('messages_dims', '?')}"
    )

    trend = _trend_line(snapshot, previous_state)
    if trend:
        lines.append(f"  {trend}")

    lines.append(f"📋 Queue: {snapshot.queue.get('pending', 0)} pending · {snapshot.queue.get('done', 0)} done")
    save_representation_errors = int(snapshot.errors.get("save_representation", 0))
    auth_errors = int(snapshot.errors.get("four_oh_one", 0))
    if save_representation_errors or auth_errors:
        lines.append(f"⚠️ Recent errors: save-repr={save_representation_errors} · 401={auth_errors}")

    spark = snapshot.spark_goat
    spark_tag = " ⚠️thinking" if spark.get("thinking") else ""
    spark_detail = f" (model={spark.get('model', '?')})" if spark.get("model") else ""
    spark_latency = float(spark.get("latency_s", 0.0) or 0.0)
    if spark.get("skipped"):
        lines.append("⚪ spark-goat chat: not routed (skipped)")
    else:
        if not spark.get("ok"):
            spark_icon = "🔴"
        elif spark.get("thinking") or spark_latency > SPARK_CHAT_DEGRADED_SECONDS:
            spark_icon = "🟠"
        else:
            spark_icon = "🟢"
        lines.append(f"{spark_icon} spark-goat chat: {_fmt_s(spark_latency)}{spark_tag}{spark_detail}")

    deriver = snapshot.deriver
    if deriver:
        active = ""
        if int(deriver.get("active_count") or 0) > 0:
            work_unit_key = str(deriver.get("active_oldest_work_unit_key") or "")
            task_type = work_unit_key.partition(":")[0]
            task_label = f" {task_type}" if task_type else ""
            active = f" · active={deriver.get('active_count')}{task_label} oldest={_format_age(deriver.get('active_oldest_age_s'))}"
        lines.append(
            f"⚡ {deriver.get('runs_15m', '?')} runs/15m · last={_fmt_s(deriver.get('last_duration_s'))} · {deriver.get('conclusions', '?')} total conclusions{active}"
        )

    ingestion = snapshot.ingestion or {}
    if ingestion:
        lines.append(
            "🔁 Ingestion: "
            f"local age={_format_age(ingestion.get('source_age_s'))} · "
            f"Honcho age={_format_age(ingestion.get('downstream_age_s'))} · "
            f"drift={_format_age(ingestion.get('drift_s'))}"
        )

    observer = snapshot.observer or {}
    if observer:
        lines.append(
            "💾 Observer disk: "
            f"{_format_bytes(observer.get('disk_free_bytes'))} free · "
            f"{int(observer.get('disk_used_percent') or 0)}% used"
        )

    alerts = build_alerts(snapshot, previous_state=previous_state)
    if alerts:
        lines.append(f"⚠️ {len(alerts)}: {'; '.join(alerts)}")
    else:
        lines.append("✅ All nominal")

    return "\n".join(lines)


def ssh(command: str, timeout: int = 20) -> str:
    ssh_cmd = ["ssh"]
    tbot_config = Path("/var/lib/tbot/hermes-cron-ssh/ssh_config")
    if tbot_config.exists():
        ssh_cmd.extend(["-F", str(tbot_config)])
    ssh_cmd.extend(
        [
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            HONCHO_TARGET,
            command,
        ]
    )
    last_error = ""
    for attempt in range(2):
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"__SSH_ERROR__ timeout={timeout}s stderr={str(exc)[:200]}"
        else:
            if result.returncode == 0:
                return result.stdout.strip()
            detail = (result.stderr or result.stdout or "").strip().replace("\n", " ")[:300]
            last_error = f"__SSH_ERROR__ rc={result.returncode} stderr={detail}"
        if attempt == 0:
            time.sleep(1)
    return last_error


def curl_json(url: str, timeout: int = 10) -> tuple[bool, dict[str, Any] | None]:
    try:
        result = subprocess.run(
            ["curl", "-s", "-m", str(timeout), url],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode != 0 or not result.stdout:
            return False, None
        return True, json.loads(result.stdout)
    except Exception:
        return False, None


def curl_text(url: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            ["curl", "-s", "-m", str(timeout), url],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def curl_post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> tuple[bool, dict[str, Any] | None, float]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-m",
                str(timeout),
                "-X",
                "POST",
                url,
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        dt = time.perf_counter() - start
        if result.returncode != 0 or not result.stdout:
            return False, None, dt
        return True, json.loads(result.stdout), dt
    except Exception:
        return False, None, time.perf_counter() - start


def _parse_service_status(raw: str) -> dict[str, Any]:
    services: dict[str, Any] = {"api_ok": False, "deriver_up": False, "db_ok": False, "redis_ok": False}
    if raw.startswith("__SSH_ERROR__"):
        services["ssh_ok"] = False
        services["ssh_error"] = raw
        return services
    services["ssh_ok"] = True
    if not raw.strip():
        services["ssh_ok"] = False
        services["ssh_error"] = "empty SSH probe output"
        return services
    for line in raw.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        name, status = parts
        key = name.replace("honcho-", "").replace("-1", "")
        up = status.startswith("Up")
        if key == "api":
            services["api_ok"] = up
        elif key == "deriver":
            services["deriver_up"] = up
        elif key == "database":
            services["db_ok"] = up
        elif key == "redis":
            services["redis_ok"] = up
    return services


def _parse_kv_lines(raw: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) != 2:
            continue
        data[parts[0]] = parts[1]
    return data


def parse_deriver_error_counts(raw: str) -> dict[str, int]:
    """Count real recent deriver failures without matching quoted content."""
    counts = {"save_representation": 0, "four_oh_one": 0}
    for line in raw.splitlines():
        log_match = _DERIVER_ERROR_LOG_LINE.fullmatch(line)
        if log_match:
            message = log_match.group("message")
            if _SAVE_REPRESENTATION_SIGNAL.search(message):
                counts["save_representation"] += 1
            if _AUTH_ERROR_SIGNAL.search(message):
                counts["four_oh_one"] += 1
            continue

        if _AUTH_EXCEPTION_LINE.match(line) and _AUTH_ERROR_SIGNAL.search(line):
            counts["four_oh_one"] += 1
    return counts


def collect_recent_deriver_errors(ssh_fn=None) -> dict[str, int]:
    """Fetch the bounded deriver log window and parse it with production code."""
    if ssh_fn is None:
        ssh_fn = ssh
    raw = ssh_fn("docker logs honcho-deriver-1 --since 15m 2>&1", timeout=30)
    if raw.startswith("__SSH_ERROR__"):
        return {"save_representation": 0, "four_oh_one": 0}
    return parse_deriver_error_counts(raw)


def _int_or_zero(value: str | None) -> int:
    try:
        return int(value) if value is not None else 0
    except Exception:
        return 0


def parse_queue_raw(queue_raw: str) -> dict[str, int]:
    """Parse PostgreSQL boolean text output for queue processed counts.

    Handles both legacy 2-column rows ``processed|count`` and the current
    3-column per-task-type rows ``task_type|processed|count``.
    """
    pending = done = 0
    for row in queue_raw.splitlines():
        parts = row.split("|")
        if len(parts) < 2:
            continue
        if len(parts) >= 3:
            # task_type|processed|count
            state, count = parts[1], parts[2]
        else:
            state, count = parts[0], parts[1]
        state = state.strip().lower()
        if state in ("f", "false"):
            pending += _int_or_zero(count)
        elif state in ("t", "true"):
            done += _int_or_zero(count)
    return {"pending": pending, "done": done}


def parse_queue_by_type_raw(queue_raw: str) -> dict[str, dict[str, int]]:
    """Parse per-task-type queue counts from PostgreSQL boolean text output.

    Expected rows: ``task_type|processed|count``. Returns a dict keyed by
    task_type with ``pending`` and ``done`` totals. Unknown task types are
    still preserved so the report stays truthful about queue composition.
    """
    by_type: dict[str, dict[str, int]] = {}
    for row in queue_raw.splitlines():
        parts = row.split("|")
        if len(parts) < 3:
            continue
        task_type, processed, count = parts[0], parts[1].strip().lower(), parts[2]
        bucket = by_type.setdefault(task_type, {"pending": 0, "done": 0})
        if processed in ("f", "false"):
            bucket["pending"] += _int_or_zero(count)
        elif processed in ("t", "true"):
            bucket["done"] += _int_or_zero(count)
    return by_type


def is_spark_goat_chat_url(url: str | None) -> bool:
    """Return True when ``url`` points at the spark-goat chat service."""
    if not url:
        return False
    return short_host(url) == "spark-goat" and ":8001" in url


def select_spark_model(pipeline: dict[str, dict[str, str]]) -> str:
    """Return the model name that should be used for the spark-goat chat smoke.

    If the pipeline's deriver base_url is on the spark-goat chat host, use its
    model; otherwise fall back to aeon-ultimate, which is known to be hosted on
    spark-goat. This prevents using a model routed elsewhere (e.g. mac-studio)
    against the spark-goat endpoint. Accept both current LAN and historical
    Tailscale addresses as spark-goat identity, even when the actual smoke
    probe uses the configured current endpoint.
    """
    deriver = pipeline.get("deriver", {})
    deriver_base = deriver.get("base_url", "")
    if is_spark_goat_chat_url(deriver_base):
        return deriver.get("model") or "aeon-ultimate"
    return "aeon-ultimate"


def build_db_stats_query() -> str:
    """Return the embedding inventory query used by the health monitor.

    Dimension checks intentionally sample one populated vector. Aggregating
    ``array_length(embedding::real[], 1)`` across the full documents table
    repeatedly materializes every large vector and can exceed the monitor's
    timeout as the database grows.
    """
    return (
        "SELECT "
        "count(*) FILTER (WHERE deleted_at IS NULL)::text, "
        "count(*) FILTER (WHERE deleted_at IS NULL AND embedding IS NOT NULL)::text, "
        "COALESCE((SELECT array_length(embedding::real[], 1)::text FROM documents "
        "WHERE deleted_at IS NULL AND embedding IS NOT NULL LIMIT 1), '')::text, "
        "(SELECT count(*)::text FROM message_embeddings), "
        "(SELECT count(*)::text FROM message_embeddings WHERE embedding IS NOT NULL), "
        "COALESCE((SELECT array_length(embedding::real[], 1)::text FROM message_embeddings "
        "WHERE embedding IS NOT NULL LIMIT 1), '')::text "
        "FROM documents;"
    )


def collect_deriver_conclusions(ssh_fn=ssh) -> int:
    """Count deriver conclusions from the database.

    The old source — grepping deriver logs for ``N total conclusions`` — went
    stale: the deployed Honcho version never emits that line (and has no
    ``conclusions`` table); conclusions live as ``documents`` rows with level
    ``inductive`` or ``deductive``. Fall back to 0 only when the query fails.
    """
    raw = ssh_fn(
        "docker exec honcho-database-1 psql -U postgres -t -A -c \""
        "SELECT count(*) FROM documents WHERE deleted_at IS NULL "
        "AND level IN ('inductive','deductive');\"",
        timeout=20,
    )
    return _int_or_zero(raw.strip().splitlines()[-1]) if raw and not raw.startswith("__SSH_ERROR__") else 0


def collect_snapshot() -> tuple[HonchoSnapshot, dict[str, Any]]:
    disk_usage = shutil.disk_usage(STATE_PATH.parent)
    observer = {
        "disk_total_bytes": disk_usage.total,
        "disk_free_bytes": disk_usage.free,
        "disk_used_percent": round((disk_usage.used / disk_usage.total) * 100) if disk_usage.total else 0,
    }

    services_raw = ssh("cd /opt/honcho/honcho && sudo docker compose ps --format '{{.Name}} {{.Status}}'", timeout=30)
    services = _parse_service_status(services_raw)

    env_raw = ssh("docker exec honcho-api-1 env", timeout=20) if services.get("api_ok") else ""
    pipeline = parse_pipeline_env(env_raw)

    # Count rows, but sample one populated vector per table for dimensions.
    # Casting every stored vector to real[] made this probe increasingly costly
    # and eventually caused 30-second timeouts on the production database.
    db_stats_query = build_db_stats_query().replace('"', '\\"')
    db_stats_raw = ssh(
        "docker exec honcho-database-1 psql -U postgres -t -A -F '|' -c \""
        + db_stats_query
        + "\"",
        timeout=30,
    )
    db_parts = [part.strip() for part in db_stats_raw.split("|")] if db_stats_raw else []
    db_probe_ok = bool(db_parts) and len(db_parts) >= 6 and not db_stats_raw.startswith("__SSH_ERROR__")
    db = {
        "probe_ok": db_probe_ok,
        "probe_error": "" if db_probe_ok else (db_stats_raw[:300] if db_stats_raw else "empty DB stats probe output"),
        "documents_total": _int_or_zero(db_parts[0]) if db_probe_ok else 0,
        "documents_with_embeddings": _int_or_zero(db_parts[1]) if db_probe_ok else 0,
        "documents_dims": _int_or_zero(db_parts[2]) if db_probe_ok and db_parts[2] else 0,
        "messages_total": _int_or_zero(db_parts[3]) if db_probe_ok else 0,
        "messages_with_embeddings": _int_or_zero(db_parts[4]) if db_probe_ok else 0,
        "messages_dims": _int_or_zero(db_parts[5]) if db_probe_ok and db_parts[5] else 0,
    }

    honcho_latest_raw = ssh(
        "docker exec honcho-database-1 psql -U postgres -t -A -c \"SELECT COALESCE(max(created_at)::text, '') FROM messages;\"",
        timeout=20,
    )
    ingestion = build_ingestion_status(latest_local_message_timestamp(), honcho_latest_raw)

    queue_raw = ssh(
        "docker exec honcho-database-1 psql -U postgres -t -A -F '|' -c \""
        "SELECT task_type, processed::text, count(*)::text FROM queue "
        "GROUP BY task_type, processed ORDER BY task_type, processed;\"",
        timeout=20,
    )
    queue_counts = parse_queue_raw(queue_raw)
    queue_by_type = parse_queue_by_type_raw(queue_raw)
    pending = queue_counts["pending"]
    done = queue_counts["done"]

    error_counts = collect_recent_deriver_errors()

    # Pick a model actually served by the spark-goat chat stage (port 8001).
    # If the loaded pipeline routes the deriver elsewhere, fall back to the
    # aeon-ultimate model known to be hosted on spark-goat.
    spark_model = select_spark_model(pipeline)

    # Only probe spark-goat when the live pipeline actually routes a stage to
    # its chat endpoint. After the cloud cutover no stage routes there, and
    # probing a decommissioned host produced a permanent false "chat failed"
    # alert even while every pipeline stage was healthy elsewhere.
    spark_routed = any(is_spark_goat_chat_url(stage.get("base_url")) for stage in pipeline.values())
    if spark_routed:
        spark_ok, spark_data, spark_dt = curl_post_json(
            f"{SPARK_CHAT_BASE}/v1/chat/completions",
            {
                "model": spark_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
        )
    else:
        spark_ok, spark_data, spark_dt = True, {"_skipped": True}, 0.0
    thinking = False
    spark_http_ok = spark_ok
    spark_model_ok = False
    spark_skipped = bool((spark_data or {}).get("_skipped"))
    if spark_data and not spark_skipped:
        msg = (spark_data.get("choices") or [{}])[0].get("message", {})
        thinking = bool(msg.get("reasoning_content"))
        spark_model_ok = bool(msg.get("content"))
        spark_ok = spark_ok and spark_model_ok
    elif spark_ok and not spark_skipped:
        # HTTP succeeded but body wasn't JSON / no choices -> treat as failure
        spark_ok = False

    logs_raw = ssh(
        "docker logs honcho-deriver-1 --since 15m > /tmp/honcho-deriver-monitor.log 2>&1; "
        "printf 'count|%s\n' \"$(grep -c 'Observation Count' /tmp/honcho-deriver-monitor.log || true)\"; "
        "printf 'last|%s\\n' \\\"$(grep 'Llm Call Duration' /tmp/honcho-deriver-monitor.log | tail -1 || true)\\\"",
        timeout=30,
    )
    log_parts = _parse_kv_lines(logs_raw)
    runs_15m = _int_or_zero(log_parts.get("count", "0"))
    # Conclusions come from the DB now (collect_deriver_conclusions); the old
    # log-grep source went stale — the "N total conclusions" line no longer
    # exists in the deployed Honcho version.
    conclusions = collect_deriver_conclusions(ssh_fn=ssh)
    last_duration_s = 0
    last_match = re.search(r"([\d,]+)\s+ms", log_parts.get("last", ""))
    if last_match:
        try:
            last_duration_s = int(int(last_match.group(1).replace(",", "")) / 1000)
        except Exception:
            last_duration_s = 0

    active_raw = ssh(
        "docker exec honcho-database-1 psql -U postgres -t -A -F '|' -c \""
        "SELECT count(*)::text, COALESCE(EXTRACT(EPOCH FROM (now() - min(last_updated)))::int::text, '0'), "
        "COALESCE((SELECT work_unit_key FROM active_queue_sessions ORDER BY last_updated LIMIT 1), ''), "
        "count(*) FILTER (WHERE work_unit_key LIKE 'representation:%')::text, "
        "COALESCE(EXTRACT(EPOCH FROM (now() - min(last_updated) FILTER (WHERE work_unit_key LIKE 'representation:%')))::int::text, '0'), "
        "count(*) FILTER (WHERE work_unit_key LIKE 'dream:%')::text, "
        "COALESCE(EXTRACT(EPOCH FROM (now() - min(last_updated) FILTER (WHERE work_unit_key LIKE 'dream:%')))::int::text, '0'), "
        "count(*) FILTER (WHERE work_unit_key NOT LIKE 'representation:%' AND work_unit_key NOT LIKE 'dream:%')::text, "
        "COALESCE(EXTRACT(EPOCH FROM (now() - min(last_updated) FILTER (WHERE work_unit_key NOT LIKE 'representation:%' AND work_unit_key NOT LIKE 'dream:%')))::int::text, '0') "
        "FROM active_queue_sessions;\"",
        timeout=20,
    )
    active_parts = [part.strip() for part in active_raw.split("|")] if active_raw else []
    active_count = _int_or_zero(active_parts[0]) if len(active_parts) > 0 else 0
    active_oldest_age_s = _int_or_zero(active_parts[1]) if len(active_parts) > 1 else 0
    active_oldest_work_unit_key = active_parts[2] if len(active_parts) > 2 else ""
    active_representation_count = _int_or_zero(active_parts[3]) if len(active_parts) > 3 else 0
    active_representation_oldest_age_s = _int_or_zero(active_parts[4]) if len(active_parts) > 4 else 0
    active_dream_count = _int_or_zero(active_parts[5]) if len(active_parts) > 5 else 0
    active_dream_oldest_age_s = _int_or_zero(active_parts[6]) if len(active_parts) > 6 else 0
    active_other_count = _int_or_zero(active_parts[7]) if len(active_parts) > 7 else 0
    active_other_oldest_age_s = _int_or_zero(active_parts[8]) if len(active_parts) > 8 else 0

    snapshot = HonchoSnapshot(
        services=services,
        pipeline=pipeline,
        db=db,
        queue={"pending": pending, "done": done},
        queue_by_type=queue_by_type,
        errors={"save_representation": error_counts.get("save_representation", 0), "four_oh_one": error_counts.get("four_oh_one", 0)},
        spark_goat={
            "ok": spark_ok,
            "latency_s": spark_dt,
            "thinking": thinking,
            "model": spark_model,
            "http_ok": spark_http_ok,
            "model_ok": spark_model_ok,
            "skipped": spark_skipped,
        },
        deriver={
            "runs_15m": runs_15m,
            "last_duration_s": last_duration_s,
            "conclusions": conclusions,
            "active_count": active_count,
            "active_oldest_age_s": active_oldest_age_s,
            "active_oldest_work_unit_key": active_oldest_work_unit_key,
            "active_representation_count": active_representation_count,
            "active_representation_oldest_age_s": active_representation_oldest_age_s,
            "active_dream_count": active_dream_count,
            "active_dream_oldest_age_s": active_dream_oldest_age_s,
            "active_other_count": active_other_count,
            "active_other_oldest_age_s": active_other_oldest_age_s,
        },
        ingestion=ingestion,
        observer=observer,
    )

    current_state = {
        "queue_done": done,
        "queue_by_type": queue_by_type,
        "documents_total": db.get("documents_total", 0),
        "messages_total": db.get("messages_total", 0),
        "pipeline": pipeline,
        "db": db,
        "queue": {"pending": pending, "done": done},
        "errors": {"save_representation": error_counts.get("save_representation", 0), "four_oh_one": error_counts.get("four_oh_one", 0)},
        "deriver": {
            "runs_15m": runs_15m,
            "last_duration_s": last_duration_s,
            "conclusions": conclusions,
            "active_count": active_count,
            "active_oldest_age_s": active_oldest_age_s,
            "active_oldest_work_unit_key": active_oldest_work_unit_key,
            "active_representation_count": active_representation_count,
            "active_representation_oldest_age_s": active_representation_oldest_age_s,
            "active_dream_count": active_dream_count,
            "active_dream_oldest_age_s": active_dream_oldest_age_s,
            "active_other_count": active_other_count,
            "active_other_oldest_age_s": active_other_oldest_age_s,
        },
        "ingestion": ingestion,
        "observer": observer,
        "deriver_active_count": active_count,
        "deriver_active_oldest_age_s": active_oldest_age_s,
    }

    return snapshot, current_state


def should_emit_report(snapshot: HonchoSnapshot, previous_state: dict[str, Any] | None = None) -> bool:
    """Return True when the cron should send a Telegram message.

    The Honcho monitor is a script-only cron job: any non-empty stdout is
    delivered verbatim. Keep routine healthy runs silent, but leave an explicit
    operator override for manual debugging.
    """
    if os.environ.get("HONCHO_MONITOR_ALWAYS_PRINT", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return bool(build_alerts(snapshot, previous_state=previous_state))


def main() -> int:
    snapshot, current_state = collect_snapshot()
    previous_state = load_state(STATE_PATH)
    current_state["dream_contention_streak"] = next_dream_contention_streak(snapshot, previous_state)
    current_state["representation_stall_streak"] = next_representation_stall_streak(
        snapshot, previous_state
    )
    report_snapshot = restore_last_valid_remote_sample(snapshot, previous_state)
    if should_emit_report(report_snapshot, previous_state=previous_state):
        report = format_report(report_snapshot, previous_state=previous_state)
        print(report)
    if snapshot.services.get("ssh_ok") is False:
        current_state["ssh_failure_streak"] = int(previous_state.get("ssh_failure_streak", 0)) + 1
        for key in ("pipeline", "db", "queue", "queue_by_type", "errors", "deriver"):
            if key in previous_state:
                current_state[key] = previous_state[key]
    else:
        current_state["ssh_failure_streak"] = 0
    current_state = preserve_last_valid_probe_state(snapshot, current_state, previous_state)
    try:
        save_state(STATE_PATH, current_state)
    except OSError as exc:
        print(f"⚠️ Local observer state save failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
