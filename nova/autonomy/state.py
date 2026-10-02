"""Where an action's earned state is recorded: ``autonomy_state.yaml`` in the bundle.

Written only by NOVA — when an administrator confirms a promotion, and when a demotion
trigger fires — through the bundle writer, so every change is validated, applied and
audited like any other bundle edit. Kept apart from ``policy.yaml`` so a machine write
never rewrites the file a person wrote (and its comments), and so "who changed this
action's state, when and why" is one small file's history.

::

    actions:
      send_external_email:
        state: graduated
        model_version: jev-1.13.0
        changed_at: "2026-10-02T12:00:00Z"
        changed_by: priya-ops
        reason: promotion confirmed (52 decisions, 100% agreement)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

STATE_FILE = "autonomy_state.yaml"

HEADER = ("# Written by NOVA when a promotion is confirmed or an action is demoted.\n"
          "# Do not edit by hand: promotion is earned in the Control Centre, not declared.\n")


def load_states(root: Path) -> dict[str, dict[str, Any]]:
    """The recorded state per action. Empty when the file is absent or unreadable.

    Unreadable reads as empty, which means every action is supervised: a damaged state
    file must never be the reason an action runs without a person.
    """
    path = Path(root) / STATE_FILE
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, yaml.YAMLError):
        return {}
    actions = (loaded or {}).get("actions") if isinstance(loaded, Mapping) else None
    return {str(k): dict(v) for k, v in (actions or {}).items() if isinstance(v, Mapping)}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def with_change(states: Mapping[str, Mapping[str, Any]], action: str, *, state: str, model_version: str,
                actor: str, reason: str) -> dict[str, Any]:
    """The whole state document with one action changed, ready to write."""
    updated = {name: dict(entry) for name, entry in states.items()}
    entry = {"state": state, "changed_at": now(), "changed_by": actor, "reason": reason}
    if state == "graduated":
        entry["model_version"] = model_version
    previous = updated.get(action, {})
    if state == "supervised" and previous.get("state") == "graduated":
        entry["demoted_from_model"] = previous.get("model_version", "")
    updated[action] = entry
    return {"actions": updated}
