"""Pending queue helpers for background skill evolution changes."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


VALID_EVOLUTION_MODES = {"auto", "confirm", "readonly"}


def normalize_evolution_mode(value: Any) -> str:
    """Return a supported evolution mode, defaulting to auto."""
    if not isinstance(value, str):
        return "auto"
    mode = value.strip().lower()
    if mode in VALID_EVOLUTION_MODES:
        return mode
    return "auto"


def get_evolution_mode(config: dict[str, Any] | None = None) -> str:
    """Read skills.evolution_mode from config, defaulting to auto."""
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()

    skills_config = config.get("skills") if isinstance(config, dict) else None
    if not isinstance(skills_config, dict):
        return "auto"
    return normalize_evolution_mode(skills_config.get("evolution_mode"))


def pending_queue_path() -> Path:
    """Return the profile-aware pending queue path."""
    return get_hermes_home() / "skills" / ".evolution_pending.json"


def _empty_queue() -> dict[str, list[dict[str, Any]]]:
    return {"changes": []}


def _read_queue() -> dict[str, list[dict[str, Any]]]:
    path = pending_queue_path()
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _empty_queue()

    changes = data.get("changes") if isinstance(data, dict) else None
    if not isinstance(changes, list):
        return _empty_queue()
    return {"changes": [change for change in changes if isinstance(change, dict)]}


def _write_queue(queue: dict[str, list[dict[str, Any]]]) -> None:
    path = pending_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2, sort_keys=True))


def queue_pending_change(action: str, name: str, payload: dict) -> dict:
    """Append a background skill evolution change to the pending queue."""
    change = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "name": name,
        "payload": payload if isinstance(payload, dict) else {},
        "origin": "background_review",
        "status": "pending",
    }

    queue = _read_queue()
    queue["changes"].append(change)
    _write_queue(queue)
    return {"success": True, **change}


def list_pending_changes() -> list[dict]:
    """Return pending changes, tolerating absent or malformed queue files."""
    return _read_queue()["changes"]


def approve_pending_change(change_id: str, apply_func: Callable[[dict], dict]) -> dict:
    """Apply and remove a pending change only when the apply callback succeeds."""
    queue = _read_queue()
    for index, change in enumerate(queue["changes"]):
        if change.get("id") != change_id:
            continue

        apply_result = apply_func(change)
        if not isinstance(apply_result, dict):
            apply_result = {"success": False, "error": "Apply function returned a non-dict result"}
        if not apply_result.get("success"):
            return {
                "success": False,
                "change_id": change_id,
                "apply_result": apply_result,
            }

        del queue["changes"][index]
        _write_queue(queue)
        return {
            "success": True,
            "applied_change_id": change_id,
            "apply_result": apply_result,
        }

    return {"success": False, "error": f"Pending change not found: {change_id}"}


def reject_pending_change(change_id: str) -> dict:
    """Remove a pending change without applying it."""
    queue = _read_queue()
    for index, change in enumerate(queue["changes"]):
        if change.get("id") != change_id:
            continue

        del queue["changes"][index]
        _write_queue(queue)
        return {"success": True, "rejected_change_id": change_id}

    return {"success": False, "error": f"Pending change not found: {change_id}"}
