"""Best-effort CLI human-intervention notifications.

This module is intentionally dependency-light and fail-closed: notification
failures must never change approval, sudo, clarify, or computer_use semantics.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli.config import load_config

_LAST_NOTIFIED: dict[tuple[str, str], float] = {}

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([^\s'\"]+)"), r"\1***"),
    (re.compile(r"(?i)(authorization\s*:\s*basic\s+)([^\s'\"]+)"), r"\1***"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd)\s*=\s*([^\s'\"&;]+)"), r"\1=***"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd)\s*:\s*([^\s'\"&;]+)"), r"\1:***"),
    (re.compile(r"(?i)([?&](?:api[_-]?key|token|secret|password|passwd|pwd)=)([^\s'\"&;]+)"), r"\1***"),
)


def _human_cfg() -> dict[str, Any]:
    cfg = load_config() or {}
    notifications = cfg.get("notifications") if isinstance(cfg, dict) else None
    human = notifications.get("human_intervention") if isinstance(notifications, dict) else None
    if not isinstance(human, dict):
        return {}
    return human


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _redact_and_truncate(text: str, max_chars: int = 120) -> str:
    """Return a redacted, single-line preview no longer than max_chars + ellipsis."""
    value = str(text or "").replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value).strip()
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    if max_chars <= 0:
        return ""
    if len(value) > max_chars:
        return value[: max(0, max_chars - 1)].rstrip() + "…"
    return value


_REMOTE_ACTION_TEMPLATES: dict[str, str] = {
    "deny": "/iv deny {code}",
    "extend": "/iv extend {code} 15",
    "status": "/iv status {code}",
}


def _compose_message(
    kind: str,
    title: str,
    message: str,
    *,
    session_key: str = "",
    timeout_seconds: int | None = None,
    remote_code: str = "",
    remote_actions: list[str] | None = None,
    risk_level: str = "",
    risk_explanation: str = "",
    approve_tier: str = "",
    approve_token: str = "",
) -> str:
    lines = [f"{title}", "请回到 CLI 处理；此通知不会远程批准本地命令。"]
    if kind:
        lines.append(f"类型: {kind}")
    if risk_level:
        lines.append(f"风险: {risk_level}")
    if timeout_seconds is not None:
        lines.append(f"超时: {timeout_seconds}s")
    if session_key:
        lines.append(f"会话: {session_key}")
    if risk_explanation:
        lines.append("危险解释:")
        lines.append(risk_explanation)
    if remote_code and remote_actions:
        action_lines = [
            _REMOTE_ACTION_TEMPLATES[action].format(code=remote_code)
            for action in remote_actions
            if action in _REMOTE_ACTION_TEMPLATES
        ]
        remote_approve_allowed = approve_tier in ("one_tap", "typed_confirm")
        approve_line = ""
        if approve_tier == "one_tap":
            approve_line = f"/iv approve {remote_code}"
        elif approve_tier == "typed_confirm":
            approve_line = f"/iv approve {remote_code} {approve_token}"
        # Approve is the primary action when allowed: list it first.
        command_lines = ([approve_line] if approve_line else []) + action_lines
        if command_lines:
            lines.append("可远程操作:")
            lines.extend(command_lines)
            if remote_approve_allowed:
                if approve_tier == "typed_confirm":
                    lines.append("注意：高风险命令需输入确认令牌。")
                else:
                    lines.append("注意：远程批准等价于本地“仅此一次”。")
            else:
                lines.append("注意：当前阶段不支持远程批准。批准请回到 CLI。")
    preview = str(message or "").strip()
    if preview:
        lines.append(preview)
    return "\n".join(lines)


def _write_log(kind: str, title: str, message: str, *, severity: str, session_key: str, timeout_seconds: int | None) -> None:
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "severity": severity,
        "title": title,
        "message": message,
        "session_key": session_key,
        "timeout_seconds": timeout_seconds,
    }
    with (log_dir / "human-intervention.log").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _send_bell() -> None:
    sys.stderr.write("\a")
    sys.stderr.flush()


def _send_desktop(title: str, message: str) -> None:
    notify_send = shutil.which("notify-send")
    if notify_send:
        subprocess.run(
            [notify_send, title, message],
            timeout=3,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _send_gateway_message(args: dict[str, Any]) -> str:
    try:
        from hermes_cli.send_cmd import _load_hermes_env
        _load_hermes_env()
    except Exception:
        pass

    from tools.send_message_tool import send_message_tool

    return send_message_tool(args)


def _send_gateway_targets(targets: list[str], body: str) -> None:
    def worker() -> None:
        for target in targets:
            try:
                _send_gateway_message({"action": "send", "target": target, "message": body})
            except Exception:
                pass

    thread = threading.Thread(
        target=worker,
        name="hermes-human-intervention-gateway-notify",
        daemon=True,
    )
    thread.start()


def notify_human_intervention(
    kind: str,
    title: str,
    message: str,
    *,
    session_key: str = "",
    severity: str = "warning",
    dedupe_key: str = "",
    timeout_seconds: int | None = None,
    remote_code: str = "",
    remote_actions: list[str] | None = None,
    risk_level: str = "",
    risk_explanation: str = "",
    approve_tier: str = "",
    approve_token: str = "",
) -> None:
    """Emit a best-effort human-intervention notification and never raise."""
    try:
        cfg = _human_cfg()
        if not cfg.get("enabled", False):
            return

        channels = _as_list(cfg.get("channels"))
        if "log" not in channels:
            channels.append("log")
        targets = _as_list(cfg.get("gateway_targets"))
        if targets and "gateway" not in channels:
            channels.append("gateway")
        if not channels:
            channels = ["log"]

        cooldown = float(cfg.get("cooldown_seconds", 20) or 0)
        key = (str(kind or "unknown"), str(dedupe_key or message or title or "default"))
        now = time.monotonic()
        previous = _LAST_NOTIFIED.get(key)
        if previous is not None and cooldown > 0 and now - previous < cooldown:
            return
        _LAST_NOTIFIED[key] = now

        preview_chars = int(cfg.get("command_preview_chars", 120) or 120)
        include_preview = bool(cfg.get("include_command_preview", True))
        preview = _redact_and_truncate(message, preview_chars) if include_preview else ""
        safe_title = _redact_and_truncate(title, 100)
        body = _compose_message(
            str(kind or "unknown"),
            safe_title,
            preview,
            session_key=session_key,
            timeout_seconds=timeout_seconds,
            remote_code=remote_code,
            remote_actions=remote_actions,
            risk_level=risk_level,
            risk_explanation=risk_explanation,
            approve_tier=approve_tier,
            approve_token=approve_token,
        )

        if "bell" in channels:
            try:
                _send_bell()
            except Exception:
                pass
        if "desktop" in channels:
            try:
                _send_desktop(safe_title, preview)
            except Exception:
                pass
        if "gateway" in channels and targets:
            try:
                _send_gateway_targets(targets, body)
            except Exception:
                pass
        if "log" in channels:
            try:
                _write_log(
                    str(kind or "unknown"),
                    safe_title,
                    preview,
                    severity=str(severity or "warning"),
                    session_key=str(session_key or ""),
                    timeout_seconds=timeout_seconds,
                )
            except Exception:
                pass
    except Exception:
        return
