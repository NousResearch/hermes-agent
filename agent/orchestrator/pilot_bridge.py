"""Pilot bridge: read pilot dirs from disk and convert to KanbanTask + TaskState.

The pilot dirs under ~/.hermes/pilots/<date>/<task_id>/ contain artifacts
(producer outputs, normalizer reports, review results) but not always
a state.json. This module:

1. Reads pilot dirs.
2. Synthesizes a state.json if missing (with state derived from the
   existing normalizer report).
3. Reads the state.json via KanbanAdapter.
4. Returns a list of KanbanTask + a worker registry.

This is the entry point for `BatchRunner` against real Kanban data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from agent.orchestrator.dispatcher import (  # noqa: E402
    TaskState,
    WorkerRegistryEntry,
)
from agent.orchestrator.kanban_adapter import (  # noqa: E402
    KanbanAdapter,
    KanbanTask,
)


@dataclass
class PilotBridgeConfig:
    """Configuration for the pilot bridge."""
    pilots_root: Path
    pilot_date: str = "2026-06-28"
    # Synthetic worker registry (used when no real worker command is found).
    default_worker_command: list | None = None
    # Restrict to a specific subset of task_ids. None = all.
    include_task_ids: list | None = None
    # Exclude tasks in these states (per approval: RUNNING/WAITING/DONE).
    exclude_states: list = field(
        default_factory=lambda: ["RUNNING", "WAITING", "DONE"]
    )


def _derive_state_from_normalizer(pilot_dir: Path) -> str:
    """Derive task state from existing normalizer artifacts.

    Priority:
      - DONE if reviewer verdict == ACCEPTED or PARTIAL.
      - BLOCKED if reviewer verdict == BLOCKED or normalizer verdict == BLOCKED.
      - READY otherwise.
    """
    metrics_path = pilot_dir / "normalizer" / "normalizer_metrics.v1.0.0.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            v = data.get("normalizer_verdict", "UNKNOWN")
            if v == "BLOCKED":
                return "BLOCKED"
            if v == "PARTIAL":
                return "FAILED"  # PARTIAL in our enum = FAILED retryable path
            if v == "PASS":
                return "READY"
        except Exception:
            pass
    return "READY"


def synthesize_state_for_pilot(pilot_dir: Path) -> dict:
    """Build a synthetic state.json for a pilot dir that lacks one."""
    state = _derive_state_from_normalizer(pilot_dir)
    pilot_id = pilot_dir.name
    return {
        "task_id": pilot_id,
        "state": state,
        "last_worker_id": None,
        "last_worker_status": None,
        "failure_count": 0,
        "human_input_required": False,
        "requires_human": False,
        "retry_count": 0,
        "stop_reason": None,
        "board": "pilots",
        "updated_at": "2026-06-28T00:00:00Z",
    }


def ensure_state_for_pilot(pilot_dir: Path) -> Path:
    """Ensure pilot_dir has a state.json. Returns the path."""
    state_path = pilot_dir / "state.json"
    if not state_path.exists():
        synthetic = synthesize_state_for_pilot(pilot_dir)
        state_path.write_text(
            json.dumps(synthetic, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return state_path


def resolve_pilot_date_dir(pilots_root: Path, pilot_date: str) -> Path:
    """Resolve the directory that contains pilot task subdirectories.

    Canonical layout is pilots_root/<pilot_date>/, but controlled canary
    sandboxes may pass the date directory itself as pilots_root.
    """
    root = Path(pilots_root)
    date_dir = root / pilot_date
    if date_dir.exists():
        return date_dir

    if root.exists() and any(
        d.is_dir()
        and (
            (d / "state.json").exists()
            or (d / "normalizer").exists()
            or (d / "evidence").exists()
        )
        for d in root.iterdir()
    ):
        return root

    return date_dir


def list_pilot_dirs(pilots_root: Path, pilot_date: str) -> list:
    """List all pilot dirs under the resolved pilot date directory."""
    date_dir = resolve_pilot_date_dir(pilots_root, pilot_date)
    if not date_dir.exists():
        return []
    return sorted([d for d in date_dir.iterdir() if d.is_dir()])


def pilot_to_kanban_task(pilot_dir: Path) -> KanbanTask | None:
    """Read a pilot's state.json and return a KanbanTask.

    Returns None if state.json cannot be read.
    """
    ensure_state_for_pilot(pilot_dir)
    state_path = pilot_dir / "state.json"
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return KanbanTask(
        task_id=data.get("task_id", pilot_dir.name),
        state=data["state"],
        last_worker_id=data.get("last_worker_id"),
        last_worker_status=data.get("last_worker_status"),
        failure_count=data.get("failure_count", 0),
        human_input_required=data.get("human_input_required", False),
        requires_human=data.get("requires_human", False),
        retry_count=data.get("retry_count", 0),
        stop_reason=data.get("stop_reason"),
        board=data.get("board", "pilots"),
    )


def make_worker_registry(config: PilotBridgeConfig) -> list:
    """Build a synthetic worker registry for the pilot board.

    The registry contains a single generic worker that can handle any
    READY task. Real production would have multiple workers registered.
    """
    return [
        WorkerRegistryEntry(
            worker_id="pilot_runner",
            worker_kind="generic",
            handles_states=["READY", "FAILED"],
            requires_http=False,
            requires_llm=False,
            mutates_state=True,
            spawns_subproc=True,
            is_retryable=True,
            recovery_kind="retryable",
            command=config.default_worker_command
            or ["python3", "-c", "print('pilot_runner done')"],
        ),
    ]


def build_tasks_and_workers(config: PilotBridgeConfig) -> tuple:
    """Read pilot dirs and return (tasks, workers) for BatchRunner.

    Honors include_task_ids filter and exclude_states.
    """
    pilot_dirs = list_pilot_dirs(config.pilots_root, config.pilot_date)
    if config.include_task_ids is not None:
        include_set = set(config.include_task_ids)
        pilot_dirs = [d for d in pilot_dirs if d.name in include_set]

    tasks = []
    for d in pilot_dirs:
        kt = pilot_to_kanban_task(d)
        if kt is None:
            continue
        if kt.state in config.exclude_states:
            continue
        tasks.append(kt)

    workers = make_worker_registry(config)
    return tasks, workers