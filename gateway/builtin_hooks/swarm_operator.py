"""Shadow-mode gateway hook for the Jeeves swarm operator.

The hook is deliberately observe-only in Sprint 1. When disabled (the default)
it does nothing. When enabled with dry_run=true it creates a SwarmJob, attaches
a deterministic routing plan, and persists JSON/JSONL audit records without
altering gateway response flow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agent.swarm_honcho import persist_swarm_honcho_summary
from agent.swarm_router import route_request
from agent.swarm_state import AuditEvent, SwarmJob
from agent.swarm_store import SwarmStore

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "dry_run": True,
    "max_children": 3,
    "persist_to_honcho": False,
    "honcho_summary_enabled": False,
}


def _get_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _swarm_config(config: Any) -> Dict[str, Any]:
    raw = _get_value(config, "swarm_operator", {}) if config is not None else {}
    if not isinstance(raw, dict):
        raw = {}
    merged = dict(DEFAULT_CONFIG)
    merged.update(raw)
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["dry_run"] = bool(merged.get("dry_run", True))
    try:
        merged["max_children"] = int(merged.get("max_children", 3))
    except (TypeError, ValueError):
        merged["max_children"] = 3
    merged["persist_to_honcho"] = bool(merged.get("persist_to_honcho", False))
    merged["honcho_summary_enabled"] = bool(merged.get("honcho_summary_enabled", False))
    return merged


def handle(event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
    if event_type != "agent:start":
        return
    ctx = context or {}
    cfg = _swarm_config(ctx.get("gateway_config") or ctx.get("config"))
    if not cfg["enabled"]:
        return

    try:
        message = str(ctx.get("message") or "")
        job = SwarmJob.create(
            message,
            platform=str(ctx.get("platform") or ""),
            user_id=str(ctx.get("user_id") or ""),
            chat_id=str(ctx.get("chat_id") or ""),
            session_id=str(ctx.get("session_id") or ""),
            metadata={
                "dry_run": cfg["dry_run"],
                "max_children": cfg["max_children"],
                "persist_to_honcho": cfg["persist_to_honcho"],
                "honcho_summary_enabled": cfg["honcho_summary_enabled"],
                "message_id": ctx.get("message_id"),
            },
        )
        plan = route_request(
            message,
            platform_context={
                "platform": job.platform,
                "user_id": job.user_id,
                "chat_id": job.chat_id,
                "session_id": job.session_id,
            },
            config=cfg,
        )
        job.routing_plan = plan
        job.audit.append(
            AuditEvent(
                "shadow_routed",
                "Swarm operator shadow route recorded.",
                metadata={"mode": plan.mode, "dry_run": cfg["dry_run"]},
            )
        )
        store = ctx.get("swarm_store") or SwarmStore()
        store.save_job(job)
        store.append_event(
            AuditEvent(
                "shadow_job_recorded",
                "Swarm operator shadow job persisted.",
                metadata={
                    "job_id": job.job_id,
                    "mode": plan.mode,
                    "platform": job.platform,
                    "session_id": job.session_id,
                    "dry_run": cfg["dry_run"],
                },
            )
        )
        if cfg["persist_to_honcho"] or cfg["honcho_summary_enabled"]:
            honcho_result = persist_swarm_honcho_summary(
                job,
                enabled=True,
                writer=ctx.get("swarm_honcho_writer"),
            )
            job.metadata["honcho_summary"] = honcho_result
            store.save_job(job)
    except Exception as exc:
        logger.warning("swarm operator shadow hook failed: %s", exc)


__all__ = ["DEFAULT_CONFIG", "handle"]
