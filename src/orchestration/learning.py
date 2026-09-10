"""Persist lightweight reflex hints from failure trajectories."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import get_hermes_home


def _reflex_path() -> Path:
    root = get_hermes_home() / "orchestration"
    root.mkdir(parents=True, exist_ok=True)
    return root / "reflex_hints.jsonl"


def record_failure(
    *,
    orchestration_id: str,
    task_id: str,
    error: str,
    trajectory_snippet: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Append a reflex record; returns suggested prompt refinement text."""

    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "orchestration_id": orchestration_id,
        "task_id": task_id,
        "error": error[:4000],
        "trajectory_snippet": (trajectory_snippet or "")[:4000],
        "meta": dict(metadata or {}),
    }
    path = _reflex_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    refine = (
        f"When coordinating `{task_id}`, avoid repeating: {error[:280]}. "
        "Add explicit checkpoints and narrower tool scopes."
    )
    return refine


def load_recent_hints(limit: int = 10) -> list[dict[str, Any]]:
    """Best-effort tail read for prompt stitching."""

    path = _reflex_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for raw in lines[-limit:]:
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out
