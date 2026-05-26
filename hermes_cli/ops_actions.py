"""Fixed action dry-run helpers for Jenny Ops Center.

This module intentionally does not execute actions. It exposes a small registry,
config gating, and approval preflight checks so the dashboard can show what would
be allowed before any future execution route is separately approved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


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
