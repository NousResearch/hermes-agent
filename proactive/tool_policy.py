from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PolicyLevel(Enum):
    AUTO_ALLOW = "AUTO_ALLOW"
    CONFIRM_FIRST = "CONFIRM_FIRST"
    DENY = "DENY"


@dataclass(frozen=True)
class ToolPolicy:
    auto_allow: set[str] = field(default_factory=set)
    confirm_first: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    level: PolicyLevel
    reason: str


DEFAULT_POLICY = ToolPolicy(
    auto_allow={"read_obsidian", "write_audit_log", "write_proactive_state", "status_check", "create_local_report", "delegate_low_risk"},
    confirm_first={"external_message", "production_change", "deploy", "delete_data", "money_movement", "secret_change"},
    deny={"leak_secrets", "bypass_approval", "disable_audit", "unauthorized_financial_access", "unconfirmed_trade"},
)


def _items(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(item).strip() for item in value if str(item).strip()}


def load_tool_policy(path: str | Path | None = None) -> ToolPolicy:
    if not path:
        return DEFAULT_POLICY
    policy_path = Path(path)
    if not policy_path.exists():
        return DEFAULT_POLICY
    data = _parse_policy_yaml(policy_path.read_text(encoding="utf-8"))
    return ToolPolicy(
        auto_allow=_items(data.get("AUTO_ALLOW")),
        confirm_first=_items(data.get("CONFIRM_FIRST")),
        deny=_items(data.get("DENY")),
    )


def _parse_policy_yaml(text: str) -> dict[str, list[str]]:
    data: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith("- "):
            current = line[:-1].strip()
            data.setdefault(current, [])
            continue
        if current and line.startswith("- "):
            data[current].append(line[2:].strip())
    return data


def decide_action(action: str, policy: ToolPolicy | None = None) -> PolicyDecision:
    active = policy or DEFAULT_POLICY
    if action in active.deny:
        return PolicyDecision(action=action, level=PolicyLevel.DENY, reason="Action is denied by tool policy.")
    if action in active.auto_allow:
        return PolicyDecision(action=action, level=PolicyLevel.AUTO_ALLOW, reason="Action is auto-allowed by tool policy.")
    if action in active.confirm_first:
        return PolicyDecision(action=action, level=PolicyLevel.CONFIRM_FIRST, reason="Action requires KJ approval.")
    return PolicyDecision(action=action, level=PolicyLevel.CONFIRM_FIRST, reason="Unknown actions require confirmation by default.")
