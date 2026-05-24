"""Credential rotation mechanism for the agent credential pool.

Provides time-based credential rotation that cycles through multiple
API keys/providers, reducing the blast radius of a leaked key.

Config
------
```yaml
security:
  credential_rotation_enabled: true
  credential_rotation_interval_hours: 24
  credential_rotation_notify: true  # notify user on rotation
```

When enabled, the agent automatically rotates through available credentials
in the pool at the configured interval.  Rotation state is persisted in
``~/.hermes/credential_rotation.json``.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_ROTATION_STATE_FILE = "credential_rotation.json"
_DEFAULT_INTERVAL_HOURS = 24


def _get_rotation_state_path() -> Path:
    """Return the path to the rotation state file."""
    return get_hermes_home() / _ROTATION_STATE_FILE


def _load_rotation_state() -> dict[str, Any]:
    """Load the current rotation state."""
    path = _get_rotation_state_path()
    if not path.exists():
        return {
            "current_index": 0,
            "last_rotation": None,
            "rotation_count": 0,
            "total_rotations": 0,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "current_index": 0,
            "last_rotation": None,
            "rotation_count": 0,
            "total_rotations": 0,
        }


def _save_rotation_state(state: dict[str, Any]) -> None:
    """Persist the rotation state."""
    path = _get_rotation_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Failed to save rotation state: %s", e)


def should_rotate(interval_hours: int = _DEFAULT_INTERVAL_HOURS) -> bool:
    """Check whether it's time to rotate credentials.

    Parameters
    ----------
    interval_hours:
        Hours between rotations.

    Returns
    -------
    bool
        True if rotation should occur.
    """
    state = _load_rotation_state()
    if state.get("last_rotation") is None:
        return True  # Never rotated yet

    try:
        last = datetime.fromisoformat(state["last_rotation"])
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return elapsed >= interval_hours
    except (ValueError, TypeError):
        return True


def rotate_credentials(
    pool: list[dict[str, Any]],
    interval_hours: int = _DEFAULT_INTERVAL_HOURS,
    notify: bool = True,
) -> dict[str, Any]:
    """Rotate to the next credential in the pool.

    Parameters
    ----------
    pool:
        List of credential dicts (from credential pool).
    interval_hours:
        Hours between rotations.
    notify:
        Whether to log a notification about the rotation.

    Returns
    -------
    dict
        The new active credential, plus rotation metadata.
    """
    if not pool:
        return {"error": "No credentials in pool"}

    state = _load_rotation_state()
    current_index = state.get("current_index", 0)
    next_index = (current_index + 1) % len(pool)

    state["current_index"] = next_index
    state["last_rotation"] = datetime.now(timezone.utc).isoformat()
    state["rotation_count"] = state.get("rotation_count", 0) + 1
    state["total_rotations"] = state.get("total_rotations", 0) + 1
    state["active_provider"] = pool[next_index].get("provider", "unknown")
    state["active_model"] = pool[next_index].get("model", "unknown")

    _save_rotation_state(state)

    if notify:
        logger.info(
            "Credential rotation #%d: switched to %s/%s (was index %d, now %d)",
            state["total_rotations"],
            state["active_provider"],
            state["active_model"],
            current_index,
            next_index,
        )

    return {
        "rotated": True,
        "credential": pool[next_index],
        "rotation_count": state["total_rotations"],
        "next_rotation_hours": interval_hours,
    }


def get_rotation_status() -> dict[str, Any]:
    """Get the current rotation status for diagnostics."""
    state = _load_rotation_state()
    return {
        "current_index": state.get("current_index", 0),
        "last_rotation": state.get("last_rotation", "never"),
        "total_rotations": state.get("total_rotations", 0),
        "active_provider": state.get("active_provider", "unknown"),
        "active_model": state.get("active_model", "unknown"),
    }


def reset_rotation() -> None:
    """Reset rotation state (start from index 0)."""
    _save_rotation_state({
        "current_index": 0,
        "last_rotation": None,
        "rotation_count": 0,
        "total_rotations": 0,
    })
    logger.info("Credential rotation state reset")
