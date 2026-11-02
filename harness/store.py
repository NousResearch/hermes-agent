"""Harness persistence on top of the session database.

No new files, no schema change: every harness record is a JSON document in
``state_meta`` under a ``harness:`` key prefix (``SessionDB.get_meta`` /
``set_meta`` / ``list_meta_prefix``). WAL, repair, profiles, and temp-home
test isolation all come from the existing session store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .state import Checkpoint, ExecutionState, FeatureState, KnowledgeItem, Task

_PREFIX = "harness:"
_TASKS = _PREFIX + "task:"
_FEATURES = _PREFIX + "feature:"
_EXEC = _PREFIX + "exec:"
_OBS = _PREFIX + "obs:"
_CHECKPOINTS = _PREFIX + "checkpoint:"
_KNOWLEDGE = _PREFIX + "knowledge:"
_EVENTS = _PREFIX + "event:"
_SEQ = _PREFIX + "event_seq"
_BUDGETS = _PREFIX + "budget:"
_CONTEXTS = _PREFIX + "context:"


def _doc_key(prefix: str, doc_id: str) -> str:
    return f"{prefix}{doc_id}"


class HarnessStore:
    """Persist harness records through a SessionDB's meta KV."""

    def __init__(self, session_db) -> None:
        self._db = session_db

    @classmethod
    def open(cls, db_path: Optional[Path | str] = None) -> "HarnessStore":
        from hermes_state import SessionDB

        if db_path is None:
            return cls(SessionDB())
        return cls(SessionDB(db_path=Path(db_path)))

    def close(self) -> None:
        self._db.close()

    # -- generic docs ----------------------------------------------------

    def _put(self, key: str, doc: Dict[str, Any]) -> None:
        self._db.set_meta(key, json.dumps(doc, sort_keys=True))

    def _get(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self._db.get_meta(key)
        return json.loads(raw) if raw is not None else None

    def _scan(self, prefix: str) -> List[Tuple[str, Dict[str, Any]]]:
        out = []
        for key, raw in self._db.list_meta_prefix(prefix):
            try:
                out.append((key, json.loads(raw)))
            except ValueError:
                continue
        out.sort(key=lambda item: item[0])
        return out

    # -- tasks / features / execution ------------------------------------

    def save_task(self, task: Task) -> None:
        self._put(_doc_key(_TASKS, task.id), task.to_dict())

    def load_task(self, task_id: str) -> Optional[Task]:
        doc = self._get(_doc_key(_TASKS, task_id))
        return Task.from_dict(doc) if doc else None

    def list_tasks(self) -> List[Task]:
        return [Task.from_dict(doc) for _, doc in self._scan(_TASKS)]

    def save_feature(self, feature: FeatureState) -> None:
        self._put(_doc_key(_FEATURES, feature.id), feature.to_dict())

    def features_for_task(self, task_id: str) -> List[FeatureState]:
        return [
            FeatureState.from_dict(doc)
            for _, doc in self._scan(_FEATURES)
            if doc.get("task_id") == task_id
        ]

    def save_execution(self, state: ExecutionState) -> None:
        self._put(_doc_key(_EXEC, state.task_id), state.to_dict())

    def load_execution(self, task_id: str) -> Optional[ExecutionState]:
        doc = self._get(_doc_key(_EXEC, task_id))
        return ExecutionState.from_dict(doc) if doc else None

    # -- observations (per-tool records, not just events) --------------------

    def save_observation(self, task_id: str, doc: Dict[str, Any]) -> None:
        obs_id = doc.get("id") or "obs"
        self._put(_doc_key(_OBS, f"{task_id}:{obs_id}"), dict(doc, task_id=task_id))

    def observations_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        return [doc for _, doc in self._scan(_OBS + task_id + ":")]

    # -- events (append-only log) -----------------------------------------

    def append_event(self, kind: str, payload: str) -> int:
        raw = self._db.get_meta(_SEQ)
        seq = int(raw) + 1 if raw is not None else 1
        self._put(
            _doc_key(_EVENTS, f"{seq:010d}"),
            {"seq": seq, "kind": kind, "payload": payload},
        )
        self._db.set_meta(_SEQ, str(seq))
        return seq

    def list_events(self, after: int = 0) -> List[Dict[str, Any]]:
        return [doc for _, doc in self._scan(_EVENTS) if doc.get("seq", 0) > after]

    def task_terminal_outcome(self, task_id: str) -> Optional[str]:
        """Latest terminal TASK_* marker for the task, replayed from the log."""
        last: Optional[str] = None
        for doc in self.list_events():
            payload = doc.get("payload", "")
            if doc.get("kind") == "TASK" and payload.endswith(":" + task_id):
                last = payload
        if not last:
            return None
        marker = last.split(":", 1)[0]
        if marker in ("TASK_COMPLETED", "TASK_FAILED", "TASK_BUDGET_EXHAUSTED"):
            return marker[len("TASK_") :]
        return None

    # -- checkpoints / knowledge / budgets / contexts ---------------------

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._put(_doc_key(_CHECKPOINTS, checkpoint.id), checkpoint.to_dict())

    def checkpoints_for_task(self, task_id: str) -> List[Checkpoint]:
        return [
            Checkpoint.from_dict(doc)
            for _, doc in self._scan(_CHECKPOINTS)
            if doc.get("task_id") == task_id
        ]

    def latest_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        found = self.checkpoints_for_task(task_id)
        return found[-1] if found else None

    def save_knowledge(self, item: KnowledgeItem) -> None:
        self._put(_doc_key(_KNOWLEDGE, item.id), item.to_dict())

    def list_knowledge(self) -> List[KnowledgeItem]:
        return [KnowledgeItem.from_dict(doc) for _, doc in self._scan(_KNOWLEDGE)]

    def save_budget(self, budget_id: str, doc: Dict[str, Any]) -> None:
        self._put(_doc_key(_BUDGETS, budget_id), doc)

    def load_budget(self, budget_id: str) -> Optional[Dict[str, Any]]:
        return self._get(_doc_key(_BUDGETS, budget_id))

    def save_context(self, context_id: str, doc: Dict[str, Any]) -> None:
        self._put(_doc_key(_CONTEXTS, context_id), doc)

    def load_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        return self._get(_doc_key(_CONTEXTS, context_id))
