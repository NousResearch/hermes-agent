"""Safe action controls for the graphical Context Cockpit.

The cockpit is read-only. Controls may only copy commands, refresh the local
page, or open fixed read-only URLs owned by the cockpit server.
"""

from __future__ import annotations

from typing import Any, Dict, List

CONTROL_ACTION_TYPES = {
    "copy_command",
    "refresh_status",
    "open_readonly_url",
    "run_readonly_fixed_helper",
}

SAFE_COPY_COMMANDS = {
    "/lcm status": "Inspect LCM state from Hermes without changing it.",
    "/compress --preview": "Preview compression impact only; no compaction runs.",
    "/usage": "Inspect current session usage and cost summary.",
    "/model": "Inspect the current model/provider selection only.",
    "/status": "Inspect Hermes status text from the native slash surface.",
}

FIXED_OPEN_PATHS = {
    "/api/status": "Open the local cockpit JSON proof payload.",
    "/operator-guide": "Open the local read-only Context Cockpit operator guide.",
}


def _control(
    control_id: str,
    label: str,
    description: str,
    risk_class: str,
    action_type: str,
    behavior: str,
    *,
    allowed: bool,
    mode: str,
    command: str | None = None,
    url: str | None = None,
    disabled_reason: str | None = None,
    recommended: bool = False,
) -> Dict[str, Any]:
    if action_type not in CONTROL_ACTION_TYPES:
        raise ValueError(f"unsupported control action type: {action_type}")
    if action_type == "copy_command" and allowed and command not in SAFE_COPY_COMMANDS:
        raise ValueError(f"unsafe copy command rejected: {command}")
    if action_type == "open_readonly_url" and url not in FIXED_OPEN_PATHS:
        raise ValueError(f"unsafe open path rejected: {url}")
    return {
        "id": control_id,
        "label": label,
        "description": description,
        "risk_class": risk_class,
        "action_type": action_type,
        "behavior": behavior,
        "mode": mode,
        "allowed": bool(allowed),
        "command": command,
        "url": url,
        "disabled_reason": disabled_reason,
        "recommended": bool(recommended),
    }


def build_action_controls(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    recommended = str(status.get("command") or "")
    controls = [
        _control(
            "refresh-now",
            "Refresh Now",
            "Re-fetch the local cockpit status payload immediately.",
            "read-only local",
            "refresh_status",
            "Refreshes this browser view only. No Hermes command runs.",
            allowed=True,
            mode="refresh",
        ),
        _control(
            "open-json",
            "Open JSON Status",
            FIXED_OPEN_PATHS["/api/status"],
            "read-only local",
            "open_readonly_url",
            "Opens a fixed localhost JSON endpoint in a new tab.",
            allowed=True,
            mode="open",
            url="/api/status",
        ),
        _control(
            "open-guide",
            "Open Operator Guide",
            FIXED_OPEN_PATHS["/operator-guide"],
            "read-only local",
            "open_readonly_url",
            "Opens a fixed localhost guide page rendered from the repo/operator doc.",
            allowed=True,
            mode="open",
            url="/operator-guide",
        ),
    ]

    for command, description in SAFE_COPY_COMMANDS.items():
        controls.append(
            _control(
                f"copy-{command.strip('/').replace(' ', '-').replace('--', '')}",
                f"Copy {command}",
                description,
                "copy-only",
                "copy_command",
                "Copies the exact slash command to the clipboard. Never executes it.",
                allowed=True,
                mode="copy",
                command=command,
                recommended=command == recommended,
            )
        )

    if recommended and recommended not in SAFE_COPY_COMMANDS:
        controls.append(
            _control(
                "blocked-recommended",
                f"Blocked: {recommended}",
                "The cockpit can show this recommendation, but executing it from the browser is intentionally forbidden.",
                "blocked future",
                "copy_command",
                "Disabled placeholder only. No clipboard write and no execution path are exposed here.",
                allowed=False,
                mode="blocked",
                command=recommended,
                disabled_reason="Execution controls for mutating commands are intentionally blocked in the browser cockpit.",
                recommended=True,
            )
        )

    return controls
