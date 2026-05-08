"""
Restart awareness: persistent activity tracker for gateway restarts.

Writes and reads a JSON activity file so the agent can recover context
after an unplanned or intentional gateway restart.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from hermes_constants import get_hermes_home


def _activity_path() -> Path:
    return get_hermes_home() / "state" / "current_activity.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_activity(
    current_task: str,
    files_modified: Optional[List[str]] = None,
    last_action: Optional[str] = None,
    next_expected_step: Optional[str] = None,
    mode: str = "simple",
) -> None:
    """Write current activity state to disk."""
    path = _activity_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "current_task": current_task,
        "files_modified": files_modified or [],
        "last_action": last_action or "",
        "next_expected_step": next_expected_step or "",
        "mode": mode,
        "updated_at": _now(),
    }
    path.write_text(json.dumps(data, indent=2))


def read_activity() -> Optional[Dict]:
    """Read the last known activity state, or None if none exists."""
    path = _activity_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear_activity() -> None:
    """Clear the activity file after a clean handoff."""
    path = _activity_path()
    if path.exists():
        path.unlink()


def _compute_staleness(updated_at_str: Optional[str]) -> tuple[bool, str]:
    """Return (is_stale, age_text) given an ISO timestamp string."""
    if not updated_at_str:
        return False, ""
    try:
        updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - updated_at
        is_stale = age > timedelta(minutes=30)
        total_minutes = age.total_seconds() // 60
        if total_minutes >= 1440:
            age_text = f"{int(total_minutes // 1440)}d {int((total_minutes % 1440) // 60)}h old"
        elif total_minutes >= 60:
            age_text = f"{int(total_minutes // 60)}h {int(total_minutes % 60)}m old"
        else:
            age_text = f"{int(total_minutes)}m old"
        return is_stale, age_text
    except Exception:
        return False, ""


def build_handoff(activity: Dict) -> str:
    """Build the handoff text to inject on first message after restart."""
    mode = activity.get("mode", "simple")
    updated_at_str = activity.get("updated_at")
    is_stale, age_text = _compute_staleness(updated_at_str)

    handoff_lines = []
    if mode == "verbose":
        if is_stale:
            handoff_lines.append(
                f"[Restart handoff \u2014 STALE ({age_text})] Activity below is from {updated_at_str or 'unknown time'}. "
                "Do NOT auto-execute the next step. Summarize the task and ask the user whether to continue."
            )
        else:
            handoff_lines.append(
                "[Restart handoff \u2014 FRESH] Agent restarted while working on the task below. Resume immediately."
            )
        handoff_lines.append(f"Task: {activity.get('current_task', '?')}")
        if activity.get("files_modified"):
            handoff_lines.append(f"Files touched: {', '.join(activity['files_modified'])}")
        if activity.get("last_action"):
            handoff_lines.append(f"Last action: {activity['last_action']}")
        if activity.get("next_expected_step"):
            handoff_lines.append(f"Next step: {activity['next_expected_step']}")
    else:
        if is_stale:
            handoff_lines.append(
                f"[Restart handoff \u2014 STALE ({age_text})] "
                "Back after gateway restart. Last recorded activity is old \u2014 ask the user what to do next."
            )
        else:
            handoff_lines.append(
                "[Restart handoff \u2014 FRESH] Back after gateway restart."
            )
    return "\n".join(handoff_lines)
