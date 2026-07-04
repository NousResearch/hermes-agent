"""Read local healer status and print startup warnings for CLI/TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, IO, Optional

from hermes_constants import get_hermes_home


def healer_status_path() -> Path:
    return get_hermes_home() / "healer" / "status.json"


def load_healer_status() -> Optional[dict[str, Any]]:
    path = healer_status_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def format_healer_warnings(status: dict[str, Any]) -> list[str]:
    overall = str(status.get("overall") or "unknown")
    if overall == "healthy":
        return []

    lines: list[str] = []
    summary = status.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(summary.strip())

    components = status.get("components")
    if isinstance(components, dict):
        for name, comp in components.items():
            if not isinstance(comp, dict):
                continue
            health = comp.get("health")
            if health not in {"failed", "circuit_open", "degraded"}:
                continue
            issue = comp.get("issue") or health
            extras: list[str] = []
            if comp.get("circuit_open"):
                extras.append("circuit OPEN")
            if comp.get("in_cooldown"):
                until = comp.get("cooldown_until")
                if until:
                    extras.append(f"cooldown until {until}")
                else:
                    extras.append("in cooldown")
            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"  - {name}: {issue}{suffix}")

    if status.get("user_alert_pending"):
        alert_path = status.get("user_alert_path") or (get_hermes_home() / "healer" / "USER_ALERT.txt")
        lines.append(f"  - alert pending: {alert_path}")

    return lines


def print_healer_startup_warnings(stream: Optional[IO[str]] = None) -> None:
    """Print a colored stderr banner when healer reports degraded/failed state."""
    import sys

    out = stream or sys.stderr
    status = load_healer_status()
    if not status:
        return

    lines = format_healer_warnings(status)
    if not lines:
        return

    overall = status.get("overall", "unhealthy")
    out.write(f"\n\033[33m[!] Hermes healer: {overall}\033[0m\n")
    for line in lines:
        if line.startswith("  "):
            out.write(f"\033[33m{line}\033[0m\n")
        else:
            out.write(f"\033[33m  {line}\033[0m\n")
    out.write(f"\033[2m  Details: {healer_status_path()}\033[0m\n\n")
