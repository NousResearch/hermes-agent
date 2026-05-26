"""Fixed action helpers for Jenny Ops Center.

This module exposes a small hardcoded registry, config gating, approval preflight,
dry-run checks, and the first explicitly gated execute path. Execution is limited to
``read_only_status_probe`` and never accepts shell/command text from the dashboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional


class OpsActionError(ValueError):
    """Raised when a fixed ops action request is invalid."""


ActionError = OpsActionError


@dataclass(frozen=True)
class OpsAction:
    name: str
    title: str
    description: str
    risk_label: str
    verification: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


ACTIONS: dict[str, OpsAction] = {
    "read_only_status_probe": OpsAction(
        name="read_only_status_probe",
        title="Read-only status probe",
        description="Check existing Hermes dashboard/gateway health metadata without changing process state.",
        risk_label="Read-only",
        verification="Return status metadata only; do not restart, write config, send messages, or mutate files.",
    ),
}

_FORBIDDEN_ACTION_NAMES = {
    "shell",
    "command",
    "exec",
    "execute",
    "gateway_restart",
    "restart_gateway",
    "delete",
    "credential_write",
    "cron_create",
    "payment",
    "outreach",
}


_BLOCKED_ACTION_CLASSES = [
    "arbitrary_shell",
    "gateway_restart",
    "cron_mutation",
    "credential_change",
    "public_or_payment_action",
    "messaging_outreach",
]


def action_registry_status(config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Return dashboard-safe fixed-action registry and live config status.

    This endpoint-facing helper is intentionally informational. It does not
    execute, dry-run, create approvals, mutate config, inspect secrets, or call
    service controls.
    """
    ops_cfg = _ops_center_config(config)
    execution_enabled = bool(ops_cfg.get("action_execution_enabled", False))
    allowed_raw = ops_cfg.get("allowed_actions", [])
    allowed_actions = [str(item).strip() for item in allowed_raw] if isinstance(allowed_raw, list) else []
    actions: list[dict[str, Any]] = []
    for action in ACTIONS.values():
        configured_allowed = action.name in allowed_actions
        executable = bool(execution_enabled and configured_allowed and action.name == "read_only_status_probe")
        actions.append({
            **action.to_dict(),
            "configured_allowed": configured_allowed,
            "executable": executable,
            "mutation_scope": "audit_log_only" if executable else "none",
        })
    return {
        "execution_enabled": execution_enabled,
        "allowed_actions": allowed_actions,
        "actions": actions,
        "blocked_action_classes": list(_BLOCKED_ACTION_CLASSES),
        "message": "Only named fixed actions are visible here; arbitrary commands and high-risk action classes remain blocked.",
    }


def get_action(name: str) -> OpsAction:
    action_name = str(name or "").strip()
    if action_name in _FORBIDDEN_ACTION_NAMES:
        raise OpsActionError(f"Unknown ops action: {action_name}")
    try:
        return ACTIONS[action_name]
    except KeyError as exc:
        raise OpsActionError(f"Unknown ops action: {action_name}") from exc


def _load_runtime_config() -> Mapping[str, Any]:
    from hermes_cli.config import load_config

    loaded = load_config() or {}
    if not isinstance(loaded, Mapping):
        return {}
    return loaded


def _ops_center_config(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    source = config if config is not None else _load_runtime_config()
    raw = source.get("ops_center", {}) if isinstance(source, Mapping) else {}
    return raw if isinstance(raw, Mapping) else {}


def preflight_action_config(
    action_name_or_config: str | Mapping[str, Any],
    maybe_action_name: Optional[str] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if isinstance(action_name_or_config, Mapping):
        config = action_name_or_config
        action_name = str(maybe_action_name or "")
        raise_on_denied = True
    else:
        action_name = action_name_or_config
        raise_on_denied = False
    action = get_action(action_name)
    ops_cfg = _ops_center_config(config)
    enabled = bool(ops_cfg.get("action_execution_enabled", False))
    allowed_raw = ops_cfg.get("allowed_actions", [])
    allowed_actions = [str(item).strip() for item in allowed_raw] if isinstance(allowed_raw, list) else []

    if not enabled:
        allowed = False
        reason = "action execution disabled by config"
    elif action.name not in allowed_actions:
        allowed = False
        reason = "action not allowlisted in config"
    else:
        allowed = True
        reason = "action allowed by explicit config"

    result = {
        "action": action.to_dict(),
        "allowed": allowed,
        "reason": reason,
        "execution_enabled": enabled,
        "allowed_actions": allowed_actions,
        "would_execute": False,
    }
    if raise_on_denied and not allowed:
        raise OpsActionError(reason)
    return result


def _parse_iso(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _approval_action_matches(action: OpsAction, approval: Mapping[str, Any]) -> bool:
    candidates = {
        str(approval.get("action_name") or "").strip(),
        str(approval.get("action_key") or "").strip(),
        str(approval.get("fixed_action") or "").strip(),
        str(approval.get("ops_action") or "").strip(),
        str(approval.get("target") or "").strip(),
        str(approval.get("proposed_action") or "").strip(),
    }
    return action.name in candidates


def validate_action_approval(
    action_name: str,
    approval: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    action = get_action(action_name)
    check_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    approval_id = approval.get("id")
    approval_ok = False
    reason = "approval accepted for dry-run preflight"

    if str(approval.get("status") or "").strip() != "approved":
        reason = "approval must be approved"
    elif str(approval.get("risk_label") or "").strip() != action.risk_label:
        reason = "approval risk label does not match action risk"
    else:
        expires_at = _parse_iso(approval.get("expires_at"))
        if expires_at is not None and expires_at <= check_time:
            reason = "approval expired"
        elif not _approval_action_matches(action, approval):
            reason = "approval does not match fixed action"
        else:
            approval_ok = True

    return {
        "approval_id": approval_id,
        "approval_ok": approval_ok,
        "reason": reason,
        "action": action.to_dict(),
        "execution_allowed": False,
        "would_execute": False,
    }


def preflight_approval_for_action(
    approval: Mapping[str, Any],
    action_name: str,
    *,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    result = validate_action_approval(action_name, approval, now=now)
    if not result["approval_ok"]:
        raise OpsActionError(result["reason"])
    return {
        "allowed": True,
        "would_execute": False,
        "execution_allowed": False,
        "reason": result["reason"],
        "action": result["action"],
        "approval_id": result["approval_id"],
    }


def dry_run_action(
    action_name: str,
    approval: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    config_result = preflight_action_config(action_name, config=config)
    approval_result = validate_action_approval(action_name, approval, now=now)
    ok = bool(config_result["allowed"] and approval_result["approval_ok"])
    reason = "dry-run preflight passed" if ok else "; ".join(
        part for part in [config_result["reason"], approval_result["reason"]] if part
    )
    return {
        "ok": ok,
        "reason": reason,
        "action": config_result["action"],
        "config": config_result,
        "approval": approval_result,
        "approval_id": approval.get("id"),
        "execution_allowed": False,
        "would_execute": False,
        "message": "Dry run only — no action executed",
    }



def _read_status_metadata() -> dict[str, Any]:
    """Return existing dashboard/gateway metadata without mutating services.

    This intentionally calls only in-process readers used by the dashboard status
    endpoint. It must not shell out, restart services, write config, or send
    messages.
    """
    try:
        from gateway.status import get_running_pid, read_runtime_status
    except Exception:
        return {
            "gateway_running": None,
            "gateway_state": None,
            "gateway_pid": None,
            "gateway_platforms": {},
        }

    gateway_pid = get_running_pid()
    runtime = read_runtime_status() or {}
    platforms = runtime.get("platforms") if isinstance(runtime, Mapping) else {}
    return {
        "gateway_running": gateway_pid is not None,
        "gateway_state": runtime.get("gateway_state") if isinstance(runtime, Mapping) else None,
        "gateway_pid": gateway_pid,
        "gateway_platforms": platforms if isinstance(platforms, Mapping) else {},
        "gateway_updated_at": runtime.get("updated_at") if isinstance(runtime, Mapping) else None,
        "gateway_exit_reason": runtime.get("exit_reason") if isinstance(runtime, Mapping) else None,
    }


def execute_read_only_status_probe(
    action_name: str,
    approval: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
    audit: Optional[Callable[[str, Mapping[str, Any], str], None]] = None,
    actor: str = "dashboard",
) -> dict[str, Any]:
    """Execute the single approved read-only status probe.

    The only side effect permitted here is append-only audit logging via the
    provided callback. The probe itself is read-only metadata collection.
    """
    if str(action_name or "").strip() != "read_only_status_probe":
        raise OpsActionError("only read_only_status_probe can be executed from the Ops Center")

    config_result = preflight_action_config(action_name, config=config)
    if not config_result["allowed"]:
        raise OpsActionError(config_result["reason"])

    approval_result = preflight_approval_for_action(approval, action_name, now=now)
    action = config_result["action"]
    approval_id = approval.get("id")

    if audit is not None:
        audit("execute_requested", approval, actor)

    try:
        status = _read_status_metadata()
        platforms = status.get("gateway_platforms") or {}
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gateway_running": status.get("gateway_running"),
            "gateway_state": status.get("gateway_state"),
            "gateway_pid": status.get("gateway_pid"),
            "gateway_platform_count": len(platforms) if isinstance(platforms, Mapping) else 0,
            "gateway_updated_at": status.get("gateway_updated_at"),
            "gateway_exit_reason": status.get("gateway_exit_reason"),
            "action_execution_enabled": config_result["execution_enabled"],
            "allowed_actions": config_result["allowed_actions"],
            "approval_status": approval.get("status"),
        }
    except Exception:
        if audit is not None:
            audit("execute_failed", approval, actor)
        raise

    if audit is not None:
        audit("execute_completed", approval, actor)

    return {
        "ok": True,
        "executed": True,
        "action": action,
        "approval": approval_result,
        "approval_id": approval_id,
        "mutation_scope": "audit_log_only",
        "result": result,
        "message": "Read-only status probe completed; no service, config, cron, messaging, credential, file, or public action was changed.",
    }
