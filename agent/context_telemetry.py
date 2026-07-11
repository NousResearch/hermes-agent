"""
Context telemetry recorder — measures context window composition per model call.

Records per-API-call breakdown of system prompt components so we can
identify which layers consume the most tokens and where bloat lives.

Stores daily JSONL files in ~/.hermes/data/context_telemetry/ with
30-day auto-prune.

Usage:
    from agent.context_telemetry import record_context_call, ContextBreakdown

    breakdown = ContextBreakdown(
        soul_md_chars=1234,
        tool_guidance_chars=567,
        system_message_chars=89,
        memory_chars=9407,
        user_profile_chars=4735,
        external_memory_chars=0,
        skills_chars=8200,
        context_files_chars=0,
        timestamp_model_chars=100,
        platform_hints_chars=200,
        environment_hints_chars=50,
        total_system_prompt_chars=24582,
    )
    record_context_call(
        session_id="cron_abc123_20260603_041200",
        cron_job_name="AMC Calibration",
        provider="deepseek",
        model="deepseek-v4-pro",
        total_chars=45000,
        total_approx_tokens=11250,
        message_count=15,
        system_breakdown=breakdown,
    )
"""

import fcntl
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTEXT_TELEMETRY_DIR = get_hermes_home() / "data" / "context_telemetry"
_MAX_DAILY_FILES = 30  # auto-prune older files
_MAX_QUEUE_SIZE = 5000
_FLUSH_INTERVAL_S = 5.0

# ---------------------------------------------------------------------------
# Breakdown dataclass
# ---------------------------------------------------------------------------


@dataclass
class ContextBreakdown:
    """Char-count breakdown of system prompt components."""

    soul_md_chars: int = 0
    tool_guidance_chars: int = 0
    system_message_chars: int = 0
    memory_chars: int = 0
    user_profile_chars: int = 0
    external_memory_chars: int = 0
    skills_chars: int = 0
    context_files_chars: int = 0
    timestamp_model_chars: int = 0
    platform_hints_chars: int = 0
    environment_hints_chars: int = 0
    other_chars: int = 0  # catch-all for future layers
    total_system_prompt_chars: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Background flush worker
# ---------------------------------------------------------------------------

_queue: list = []
_queue_lock = threading.Lock()
_dropped_count = 0
_flush_thread: Optional[threading.Thread] = None
_flush_running = False


def _daily_file() -> Path:
    """Return today's JSONL file path (ET date)."""
    from hermes_time import now as _hermes_now

    today_str = _hermes_now().strftime("%Y%m%d")
    _CONTEXT_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_files()
    return _CONTEXT_TELEMETRY_DIR / f"context_calls_{today_str}.jsonl"


def _prune_old_files() -> None:
    """Remove files older than _MAX_DAILY_FILES days."""
    try:
        files = sorted(_CONTEXT_TELEMETRY_DIR.glob("context_calls_*.jsonl"))
        if len(files) > _MAX_DAILY_FILES:
            for old_file in files[: len(files) - _MAX_DAILY_FILES]:
                try:
                    old_file.unlink()
                except OSError:
                    pass
    except Exception:
        pass


def _flush_worker() -> None:
    """Background thread: flush queue to disk every _FLUSH_INTERVAL_S seconds."""
    global _flush_running, _queue, _dropped_count
    while _flush_running:
        time.sleep(_FLUSH_INTERVAL_S)
        drained = []
        dropped = 0
        with _queue_lock:
            if _queue:
                drained = _queue[:]
                _queue.clear()
                dropped = _dropped_count
                _dropped_count = 0
        if drained:
            _write_batch(drained, dropped)


def _write_batch(records: list, dropped_before: int = 0) -> None:
    """Write a batch of records to today's JSONL file (thread-safe)."""
    filepath = _daily_file()
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if dropped_before > 0:
                    f.write(
                        json.dumps(
                            {
                                "_meta": "dropped",
                                "count": dropped_before,
                                "timestamp": time.time(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        logger.warning("Context telemetry: failed to write %d records: %s", len(records), e)
        # Re-queue them if possible; preserve dropped count for retry
        with _queue_lock:
            _could_not_queue = 0
            for record in records:
                if len(_queue) < _MAX_QUEUE_SIZE:
                    _queue.append(record)
                else:
                    _could_not_queue += 1
            global _dropped_count
            _dropped_count += dropped_before + _could_not_queue


def _start_flush_thread() -> None:
    """Ensure the background flush thread is running."""
    global _flush_thread, _flush_running
    if _flush_thread is not None and _flush_thread.is_alive():
        return
    _flush_running = True
    _flush_thread = threading.Thread(target=_flush_worker, daemon=True, name="ctx-telemetry-flush")
    _flush_thread.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_context_call(
    session_id: str,
    cron_job_name: str,
    provider: str,
    model: str,
    total_chars: int,
    total_approx_tokens: int,
    message_count: int,
    system_breakdown: ContextBreakdown,
    *,
    user_session: bool = False,
    platform: str = "",
) -> None:
    """
    Record a context telemetry entry for a single model API call.

    Args:
        session_id: Unique session identifier (cron_* or user session ID)
        cron_job_name: Cron job name (empty string for user sessions)
        provider: LLM provider (e.g., 'deepseek', 'openai')
        model: Model name (e.g., 'deepseek-v4-pro')
        total_chars: Total character count of all API messages
        total_approx_tokens: Approximate token count (chars/4)
        message_count: Number of messages in the API call
        system_breakdown: Breakdown of system prompt component sizes
        user_session: True if this is a user-initiated session (not cron)
        platform: Platform name (e.g., 'slack', 'telegram', 'cli')
    """
    _start_flush_thread()

    # Compute conversation chars: total - system prompt
    conv_chars = max(0, total_chars - system_breakdown.total_system_prompt_chars)

    record = {
        "session_id": session_id,
        "cron_job_name": cron_job_name,
        "user_session": user_session,
        "platform": platform,
        "provider": provider,
        "model": model,
        "timestamp": time.time(),
        "total_chars": total_chars,
        "total_approx_tokens": total_approx_tokens,
        "message_count": message_count,
        "system_prompt_chars": system_breakdown.total_system_prompt_chars,
        "conversation_chars": conv_chars,
        "breakdown": system_breakdown.to_dict(),
    }

    with _queue_lock:
        if len(_queue) < _MAX_QUEUE_SIZE:
            _queue.append(record)
        else:
            global _dropped_count
            _dropped_count += 1


def get_stats(
    cron_job_name: Optional[str] = None,
    days: int = 7,
    include_calls: bool = False,
) -> dict:
    """
    Aggregate context telemetry stats.

    Args:
        cron_job_name: Filter to a specific cron job (None = all)
        days: Number of days to analyze
        include_calls: Include individual call records in output

    Returns:
        Dict with aggregate stats and optional call records.
    """
    records = []
    for fpath in sorted(_CONTEXT_TELEMETRY_DIR.glob("context_calls_*.jsonl")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "_meta" in rec:
                            continue  # skip dropped-count markers
                        # Filter by age
                        ts = rec.get("timestamp", 0)
                        if time.time() - ts > days * 86400:
                            continue
                        # Filter by cron job
                        if cron_job_name and rec.get("cron_job_name") != cron_job_name:
                            continue
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    if not records:
        return {"total_calls": 0, "cron_jobs": {}, "summary": "No data for the selected period."}

    # Aggregate by cron job
    by_job = defaultdict(lambda: {"count": 0, "total_tokens": 0, "total_chars": 0, "breakdown_totals": defaultdict(int)})
    for rec in records:
        job = rec.get("cron_job_name") or "(user session)"
        by_job[job]["count"] += 1
        by_job[job]["total_tokens"] += rec.get("total_approx_tokens", 0)
        by_job[job]["total_chars"] += rec.get("total_chars", 0)
        bd = rec.get("breakdown", {})
        for key, val in bd.items():
            by_job[job]["breakdown_totals"][key] += val

    # Compute averages
    result = {
        "total_calls": len(records),
        "total_tokens": sum(r.get("total_approx_tokens", 0) for r in records),
        "total_chars": sum(r.get("total_chars", 0) for r in records),
        "days": days,
        "cron_jobs": {},
    }

    for job, stats in sorted(by_job.items(), key=lambda x: -x[1]["total_tokens"]):
        n = stats["count"]
        result["cron_jobs"][job] = {
            "call_count": n,
            "avg_tokens_per_call": stats["total_tokens"] // n if n else 0,
            "avg_chars_per_call": stats["total_chars"] // n if n else 0,
            "total_tokens": stats["total_tokens"],
            "total_chars": stats["total_chars"],
            "avg_system_prompt_pct": (
                (stats["breakdown_totals"].get("total_system_prompt_chars", 0) / stats["total_chars"] * 100)
                if stats["total_chars"]
                else 0
            ),
            "avg_breakdown": {
                k: v // n if n else 0 for k, v in stats["breakdown_totals"].items()
            },
        }

    if include_calls:
        result["calls"] = records

    return result


def get_top_consumers(component: str = "memory_chars", days: int = 7, limit: int = 10) -> list:
    """
    Rank cron jobs by a specific context component's consumption.

    Args:
        component: Breakdown field name (e.g., 'memory_chars', 'skills_chars')
        days: Number of days to analyze
        limit: Max results

    Returns:
        List of (cron_job_name, avg_chars_per_call) sorted descending.
    """
    stats = get_stats(days=days)
    rankings = []
    for job, job_stats in stats.get("cron_jobs", {}).items():
        avg_breakdown = job_stats.get("avg_breakdown", {})
        avg_val = avg_breakdown.get(component, 0)
        if avg_val > 0:
            rankings.append((job, avg_val, job_stats["call_count"]))
    rankings.sort(key=lambda x: -x[1])
    return rankings[:limit]


# ---------------------------------------------------------------------------
# Atexit flush
# ---------------------------------------------------------------------------


def _flush_on_exit() -> None:
    """Drain remaining queue on process exit."""
    global _flush_running, _queue, _dropped_count
    _flush_running = False
    drained = []
    with _queue_lock:
        drained = _queue[:]
        _queue.clear()
        dropped = _dropped_count
        _dropped_count = 0
    if drained:
        _write_batch(drained, dropped)


import atexit

atexit.register(_flush_on_exit)
