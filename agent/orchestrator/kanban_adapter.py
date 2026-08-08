"""Kanban adapter for the orchestrator (read/convert/snapshot/rollback).

The adapter talks to a local Kanban representation: a directory of
pilot dirs, each with a `state.json` describing the task state. This
mirrors the production Kanban but is local-only (no real HTTP / DB).

Mutations are gated by an explicit scope list. The adapter enforces:
- Pre-snapshot before any mutation.
- Post-snapshot after mutation.
- Diff scope: only the target task may change. Anything else triggers
  rollback to the pre-snapshot.

This is the ONLY place that mutates Kanban state. DecisionEngine and
the dispatcher itself do not mutate anything.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VALID_STATES = {"READY", "RUNNING", "WAITING", "BLOCKED", "FAILED", "DONE"}


@dataclass
class KanbanTask:
    """A single task in the Kanban."""
    task_id: str
    state: str
    last_worker_id: str | None = None
    last_worker_status: str | None = None
    failure_count: int = 0
    human_input_required: bool = False
    requires_human: bool = False
    retry_count: int = 0
    stop_reason: str | None = None
    board: str = "default"

    def __post_init__(self):
        if self.state not in VALID_STATES:
            raise ValueError(
                f"task {self.task_id}: invalid state {self.state!r}"
            )


class KanbanAdapter:
    """Reads + writes Kanban tasks to a local directory.

    Layout: <board_root>/<task_id>/state.json

    The adapter is the ONLY writer. Callers (handlers) request changes
    via apply_change(); the adapter validates scope and rolls back if
    anything outside the expected scope changed.
    """

    def __init__(self, board_root: Path) -> None:
        self.board_root = Path(board_root)

    # ----- reads -----

    def list_tasks(self) -> list:
        if not self.board_root.exists():
            return []
        out = []
        for task_dir in sorted(self.board_root.iterdir()):
            if not task_dir.is_dir():
                continue
            state_path = task_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                out.append(self._dict_to_task(data, task_dir.name))
            except Exception:
                continue
        return out

    def get_task(self, task_id: str) -> KanbanTask | None:
        state_path = self.board_root / task_id / "state.json"
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return self._dict_to_task(data, task_id)
        except Exception:
            return None

    def ensure_task(self, task: KanbanTask) -> Path:
        """Create a task if it doesn't exist. Returns the state.json path."""
        task_dir = self.board_root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        state_path = task_dir / "state.json"
        if not state_path.exists():
            state_path.write_text(
                json.dumps(self._task_to_dict(task), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return state_path

    # ----- mutations with scope enforcement -----

    def apply_change(
        self,
        task_id: str,
        new_state: dict,
        *,
        allowed_task_ids: Iterable[str] | None = None,
    ) -> dict:
        """Apply a state change to one task with scope enforcement.

        Steps:
        1. Take pre-snapshot (dict of task_id -> content).
        2. Write new state for the target task.
        3. Take post-snapshot.
        4. Diff the snapshots: only the target task should differ.
        5. If anything outside the allowed scope changed → rollback.

        Args:
            task_id: which task to mutate.
            new_state: full new state dict for the task.
            allowed_task_ids: set of task IDs allowed to differ. Default = {task_id}.

        Returns:
            Dict with keys 'pre' and 'post' (snapshots) plus 'changed' (set of task_ids).
        """
        if allowed_task_ids is None:
            allowed_task_ids = {task_id}

        pre = self._snapshot()
        target_state_path = self.board_root / task_id / "state.json"
        target_state_path.parent.mkdir(parents=True, exist_ok=True)
        target_state_path.write_text(
            json.dumps(new_state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        post = self._snapshot()

        # Diff: which task_ids changed?
        changed = self._diff_tasks(pre, post)
        unexpected = changed - set(allowed_task_ids)
        if unexpected:
            # Rollback.
            self._restore(pre)
            raise ScopeViolation(
                f"mutation changed tasks outside allowed scope: {unexpected}"
            )
        return {"pre": pre, "post": post, "changed": changed}

    # ----- internal helpers -----

    def _task_to_dict(self, task: KanbanTask) -> dict:
        return {
            "task_id": task.task_id,
            "state": task.state,
            "last_worker_id": task.last_worker_id,
            "last_worker_status": task.last_worker_status,
            "failure_count": task.failure_count,
            "human_input_required": task.human_input_required,
            "requires_human": task.requires_human,
            "retry_count": task.retry_count,
            "stop_reason": task.stop_reason,
            "board": task.board,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def _dict_to_task(self, data: dict, task_id: str) -> KanbanTask:
        return KanbanTask(
            task_id=data.get("task_id", task_id),
            state=data["state"],
            last_worker_id=data.get("last_worker_id"),
            last_worker_status=data.get("last_worker_status"),
            failure_count=data.get("failure_count", 0),
            human_input_required=data.get("human_input_required", False),
            requires_human=data.get("requires_human", False),
            retry_count=data.get("retry_count", 0),
            stop_reason=data.get("stop_reason"),
            board=data.get("board", "default"),
        )

    def _snapshot(self) -> dict:
        """Snapshot the entire board: {task_id: state.json content}."""
        if not self.board_root.exists():
            return {}
        out = {}
        for task_dir in sorted(self.board_root.iterdir()):
            if not task_dir.is_dir():
                continue
            state_path = task_dir / "state.json"
            if not state_path.exists():
                continue
            out[task_dir.name] = state_path.read_text(encoding="utf-8")
        return out

    def _restore(self, snapshot: dict) -> None:
        """Restore the entire board from a snapshot."""
        if not self.board_root.exists():
            return
        for task_dir in self.board_root.iterdir():
            if task_dir.is_dir():
                shutil.rmtree(task_dir, ignore_errors=True)
        for task_id, content in snapshot.items():
            task_dir = self.board_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "state.json").write_text(content, encoding="utf-8")

    def _diff_tasks(self, pre: dict, post: dict) -> set:
        """Return set of task_ids whose state.json content changed."""
        changed = set()
        all_ids = set(pre.keys()) | set(post.keys())
        for tid in all_ids:
            if pre.get(tid) != post.get(tid):
                changed.add(tid)
        return changed


class ScopeViolation(Exception):
    """Raised when a Kanban mutation changed tasks outside the allowed scope."""