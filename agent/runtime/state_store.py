"""Append-only runtime state store.

RuntimeStateStore is a small, opt-in recovery primitive for future runtime and
self-improvement layers. It complements existing SessionDB and shadow-git
checkpoints instead of replacing them.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.runtime.state import RuntimeSessionState
from utils import atomic_json_write


_STATE_LOG = "runtime-state.jsonl"
_INDEX_FILE = "runtime-state.index.json"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RuntimeStateCheckpoint(BaseModel):
    """One persisted runtime state checkpoint."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(default_factory=lambda: uuid4().hex)
    step_id: str
    reason: str
    state: RuntimeSessionState
    created_at: datetime = Field(default_factory=_utc_now)


class RuntimeStateStore:
    """Single-process JSONL store for runtime state checkpoints.

    Consistency model:
    - checkpoint records are append-only JSONL lines;
    - `runtime-state.index.json` is atomically replaced after each append;
    - readers use the index first, so a corrupt trailing JSONL line does not
      hide the most recent fully indexed checkpoint.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / _STATE_LOG
        self.index_path = self.root / _INDEX_FILE

    def append(self, state: RuntimeSessionState, *, step_id: str, reason: str) -> RuntimeStateCheckpoint:
        checkpoint = RuntimeStateCheckpoint(step_id=step_id, reason=reason, state=state)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(checkpoint.model_dump_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        index = self._load_index()
        index[checkpoint.checkpoint_id] = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "step_id": checkpoint.step_id,
            "reason": checkpoint.reason,
            "created_at": checkpoint.created_at.isoformat(),
        }
        index["_latest"] = checkpoint.checkpoint_id
        atomic_json_write(self.index_path, index)
        return checkpoint

    def list(self) -> list[RuntimeStateCheckpoint]:
        return self._read_all()

    def latest(self) -> RuntimeStateCheckpoint | None:
        latest_id = self._load_index().get("_latest")
        return self.read(latest_id) if latest_id else None

    def read(self, checkpoint_id: str | None) -> RuntimeStateCheckpoint | None:
        if not checkpoint_id:
            return None
        for checkpoint in self._read_all():
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {}
        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _read_all(self) -> list[RuntimeStateCheckpoint]:
        if not self.log_path.exists():
            return []
        checkpoints: list[RuntimeStateCheckpoint] = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    checkpoints.append(RuntimeStateCheckpoint.model_validate_json(line))
                except ValueError:
                    # A crash during append can leave a corrupt trailing line.
                    # Keep earlier complete checkpoints available.
                    continue
        return checkpoints
