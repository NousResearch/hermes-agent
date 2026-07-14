"""Opt-in append-only audit log for cron job state changes."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MB = 10
# Keyed by the active HERMES_HOME so a multi-profile gateway (dashboard cron
# requests scope HERMES_HOME per profile) never serves one profile's enabled
# flag / log path to another. See reload_audit_config() to invalidate.
_config_cache: Dict[str, Dict[str, Any]] = {}
_config_lock = threading.Lock()
_write_lock = threading.Lock()


def _load_audit_config() -> Dict[str, Any]:
    """Load audit config from ``config.yaml``, cached per active HERMES_HOME.

    Behavioral configuration is exposed through ``config.yaml`` only
    (``cron.audit_log`` / ``cron.audit_log_path`` / ``cron.audit_log_max_mb``);
    per AGENTS.md, non-secret settings must not add user-facing ``HERMES_*``
    env vars.
    """
    hermes_home = get_hermes_home()
    cache_key = str(hermes_home)

    with _config_lock:
        cached = _config_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

    cfg: Dict[str, Any] = {
        "enabled": False,
        "log_path": str(hermes_home / "cron" / "audit.log"),
        "max_mb": _DEFAULT_MAX_MB,
    }

    try:
        from hermes_cli.config import load_config

        root_cfg = load_config() or {}
        cron_cfg = root_cfg.get("cron", {}) if isinstance(root_cfg, dict) else {}
        if cron_cfg.get("audit_log") is True:
            cfg["enabled"] = True
        if cron_cfg.get("audit_log_path"):
            cfg["log_path"] = str(cron_cfg["audit_log_path"])
        if cron_cfg.get("audit_log_max_mb"):
            try:
                cfg["max_mb"] = max(1, int(cron_cfg["audit_log_max_mb"]))
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid cron.audit_log_max_mb=%r", cron_cfg.get("audit_log_max_mb"))
    except Exception as exc:  # pragma: no cover - config load failures must not break cron
        logger.debug("Failed to load cron audit config: %s", exc)

    with _config_lock:
        _config_cache[cache_key] = dict(cfg)
    return cfg


def reload_audit_config() -> None:
    """Clear the cached audit config for all profiles."""
    with _config_lock:
        _config_cache.clear()


def is_audit_enabled() -> bool:
    return bool(_load_audit_config().get("enabled"))


def audit_log_path() -> Path:
    return Path(str(_load_audit_config().get("log_path", ""))).expanduser()


def _detect_actor() -> str:
    if os.getenv("HERMES_CRON_SESSION"):
        return "scheduler"
    if os.getenv("HERMES_GATEWAY_SESSION") or os.getenv("HERMES_INTERACTIVE"):
        return "user"
    return "system"


def _rotate_if_needed(path: Path, max_mb: int) -> None:
    if not path.exists():
        return
    try:
        if path.stat().st_size < max_mb * 1024 * 1024:
            return
        rotated = path.with_name(f"{path.name}.{_hermes_now().strftime('%Y%m%d_%H%M%S')}")
        path.rename(rotated)
    except OSError as exc:
        logger.debug("Failed to rotate cron audit log %s: %s", path, exc)


def sanitize_changes(changes: Dict[str, Any]) -> Dict[str, Any]:
    """Return audit-safe update details; never write prompt bodies to disk."""
    safe: Dict[str, Any] = {}
    for key, value in (changes or {}).items():
        if key == "prompt":
            safe[key] = f"<{len(str(value))} chars>"
        else:
            safe[key] = value
    return safe


def audit_event(
    job_id: str,
    job_name: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    actor: Optional[str] = None,
) -> None:
    """Append a cron audit event when audit logging is enabled."""
    cfg = _load_audit_config()
    if not cfg.get("enabled"):
        return

    path = Path(str(cfg.get("log_path", ""))).expanduser()
    if not path:
        return

    entry = {
        "ts": _hermes_now().isoformat(),
        "job_id": job_id,
        "job_name": job_name,
        "action": action,
        "actor": actor or _detect_actor(),
        "details": details or {},
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            _rotate_if_needed(path, int(cfg.get("max_mb") or _DEFAULT_MAX_MB))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - audit logging is best-effort
        logger.warning("Failed to write cron audit log: %s", exc)


def read_audit_entries(
    *,
    limit: int = 50,
    job_id: Optional[str] = None,
    action: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Read recent audit entries, filtering by job id/action when supplied."""
    path = audit_log_path()
    if not path.exists():
        return []

    entries: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                entry = json.loads(text)
            except json.JSONDecodeError:
                continue
            if job_id and entry.get("job_id") != job_id:
                continue
            if action and entry.get("action") != action:
                continue
            entries.append(entry)
    return entries[-max(1, int(limit or 50)):]


def update_details(changes: Dict[str, Any]) -> Dict[str, Any]:
    return {"changes": sanitize_changes(changes)}


def removal_details(reason: str) -> Dict[str, Any]:
    return {"reason": reason}


def completion_details(success: bool, error: Optional[str] = None, delivery_error: Optional[str] = None) -> Dict[str, Any]:
    details: Dict[str, Any] = {"success": success}
    if error:
        details["error"] = error
    if delivery_error:
        details["delivery_error"] = delivery_error
    return details
