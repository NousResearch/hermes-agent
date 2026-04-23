"""
Cron job audit logging for Hermes Agent.

Records state changes to cron jobs in an append-only JSONL log file.
Opt-in via config.yaml (cron.audit_log: true) or env var (HERMES_CRON_AUDIT_LOG=1).

Log format: one JSON object per line:
    {"ts": "ISO", "job_id": "...", "job_name": "...", "action": "...", "actor": "...", "details": {...}}
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_config_cache: Optional[Dict[str, Any]] = None
_config_lock = threading.Lock()


def _load_audit_config() -> Dict[str, Any]:
    """Load audit config from config.yaml with caching."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    hermes_home = get_hermes_home()
    defaults = {
        "enabled": False,
        "log_path": str(hermes_home / "cron" / "audit.log"),
        "max_mb": 10,
        "log_ticks": False,
    }

    # Env var takes priority
    if os.getenv("HERMES_CRON_AUDIT_LOG", "").strip() in ("1", "true", "yes"):
        defaults["enabled"] = True

    # Load from config.yaml (uses hermes_cli.config which resolves the real path)
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}

        if cron_cfg.get("audit_log", False):
            defaults["enabled"] = True

        if cron_cfg.get("audit_log_path"):
            defaults["log_path"] = str(cron_cfg["audit_log_path"]).strip()

        if cron_cfg.get("audit_log_max_mb"):
            try:
                defaults["max_mb"] = int(cron_cfg["audit_log_max_mb"])
            except (ValueError, TypeError):
                pass

        if cron_cfg.get("audit_log_ticks", False):
            defaults["log_ticks"] = True
    except Exception as e:
        logger.debug("Failed to load cron audit config: %s", e)

    with _config_lock:
        _config_cache = defaults
    return defaults


def reload_audit_config() -> None:
    """Force reload audit config (useful after config changes)."""
    global _config_cache
    with _config_lock:
        _config_cache = None


def is_audit_enabled() -> bool:
    """Check if audit logging is enabled."""
    return _load_audit_config().get("enabled", False)


# ---------------------------------------------------------------------------
# Actor detection
# ---------------------------------------------------------------------------

def _detect_actor() -> str:
    """Detect who initiated the action."""
    # If running inside a cron tick, actor is scheduler
    if os.getenv("HERMES_CRON_SESSION"):
        return "scheduler"
    # If there's an active gateway session (user interacting via chat), actor is user
    if os.getenv("HERMES_GATEWAY_SESSION") or os.getenv("HERMES_INTERACTIVE"):
        return "user"
    return "system"


# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------

def _rotate_if_needed(log_path: str, max_mb: int) -> None:
    """Rotate the audit log if it exceeds max_mb."""
    path = Path(log_path)
    if not path.exists():
        return
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb >= max_mb:
            rotated = path.with_suffix(f".log.{_hermes_now().strftime('%Y%m%d_%H%M%S')}")
            path.rename(rotated)
            logger.info("Rotated audit log to %s", rotated)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def audit_event(
    job_id: str,
    job_name: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    actor: Optional[str] = None,
) -> None:
    """Log an audit event for a cron job.

    Args:
        job_id: The job ID.
        job_name: Human-readable job name.
        action: One of created, updated, paused, resumed, removed, completed, disabled, enabled, scheduler_tick.
        details: Optional dict with action-specific context.
        actor: Who initiated the action. Auto-detected if None.
    """
    cfg = _load_audit_config()
    if not cfg.get("enabled"):
        return

    # Suppress tick events unless explicitly enabled
    if action == "scheduler_tick" and not cfg.get("log_ticks"):
        return

    entry = {
        "ts": _hermes_now().isoformat(),
        "job_id": job_id,
        "job_name": job_name,
        "action": action,
        "actor": actor or _detect_actor(),
        "details": details or {},
    }

    log_path = cfg["log_path"]
    max_mb = cfg.get("max_mb", 10)

    try:
        # Ensure directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        with _write_lock:
            _rotate_if_needed(log_path, max_mb)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.warning("Failed to write cron audit log: %s", e)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def audit_created(job: Dict[str, Any]) -> None:
    audit_event(
        job_id=job["id"],
        job_name=job.get("name", ""),
        action="created",
        details={
            "schedule": job.get("schedule_display", ""),
            "deliver": job.get("deliver", ""),
        },
    )


def audit_updated(job_id: str, job_name: str, updates: Dict[str, Any]) -> None:
    # Sanitize updates — don't log full prompts
    safe = {}
    for k, v in updates.items():
        if k == "prompt":
            safe[k] = f"<{len(str(v))} chars>"
        else:
            safe[k] = v
    audit_event(job_id=job_id, job_name=job_name, action="updated", details={"changes": safe})


def audit_paused(job_id: str, job_name: str, reason: Optional[str] = None) -> None:
    details = {}
    if reason:
        details["reason"] = reason
    audit_event(job_id=job_id, job_name=job_name, action="paused", details=details)


def audit_resumed(job_id: str, job_name: str) -> None:
    audit_event(job_id=job_id, job_name=job_name, action="resumed")


def audit_removed(job_id: str, job_name: str) -> None:
    audit_event(job_id=job_id, job_name=job_name, action="removed")


def audit_run_completed(job_id: str, job_name: str, success: bool, error: Optional[str] = None) -> None:
    action = "completed" if success else "failed"
    details = {}
    if error:
        details["error"] = error
    audit_event(job_id=job_id, job_name=job_name, action=action, details=details)


def audit_disabled(job_id: str, job_name: str, reason: str) -> None:
    """Log when a job is automatically disabled (e.g. one-shot completion, repeat limit)."""
    audit_event(job_id=job_id, job_name=job_name, action="disabled", details={"reason": reason})


def audit_enabled(job_id: str, job_name: str, reason: str = "manual") -> None:
    """Log when a job is re-enabled."""
    audit_event(job_id=job_id, job_name=job_name, action="enabled", details={"reason": reason})


def audit_tick(due_count: int, total_active: int) -> None:
    """Log a scheduler tick (only if audit_log_ticks is enabled)."""
    audit_event(
        job_id="",
        job_name="",
        action="scheduler_tick",
        details={"due_count": due_count, "total_active": total_active},
    )
