"""Lightweight JSONL tracing for Hermes orchestration runs.

This intentionally avoids a hosted tracing dependency.  It gives the gateway,
CLI, delegate_task, Kanban workflows, and future dashboard one stable local run
ledger to point at.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


@dataclass
class OrchestrationTrace:
    run_id: str
    session_id: str
    path: Path
    workflow_name: str = ""
    created_at: float = field(default_factory=time.time)

    @classmethod
    def start(
        cls,
        session_id: str,
        *,
        root_dir: Optional[Path] = None,
        workflow_name: str = "",
        run_id: Optional[str] = None,
    ) -> "OrchestrationTrace":
        rid = run_id or f"orch-{uuid.uuid4().hex[:12]}"
        root = Path(root_dir) if root_dir else get_hermes_home() / "runs"
        root.mkdir(parents=True, exist_ok=True)
        trace = cls(
            run_id=rid,
            session_id=session_id or "",
            path=root / f"{rid}.jsonl",
            workflow_name=workflow_name or "",
        )
        trace.record("run_start", workflow_name=trace.workflow_name)
        return trace

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {
            "ts": time.time(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event": event,
        }
        if self.workflow_name:
            row["workflow_name"] = self.workflow_name
        row.update({k: _json_safe(v) for k, v in fields.items()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        return row


def get_or_create_trace(agent: Any, *, workflow_name: str = "") -> OrchestrationTrace:
    trace = getattr(agent, "_orchestration_trace", None)
    if isinstance(trace, OrchestrationTrace):
        return trace
    trace = OrchestrationTrace.start(
        getattr(agent, "session_id", "") or "",
        workflow_name=workflow_name,
    )
    setattr(agent, "_orchestration_trace", trace)
    return trace


def record_agent_event(agent: Any, event: str, **fields: Any) -> Optional[dict[str, Any]]:
    if agent is None:
        return None
    trace = getattr(agent, "_orchestration_trace", None)
    if not isinstance(trace, OrchestrationTrace):
        # Avoid making every normal Hermes turn noisy.  Tracing auto-starts only
        # when orchestration mode marks the session or a workflow explicitly does.
        if not getattr(agent, "_orchestration_trace_enabled", False):
            return None
        trace = get_or_create_trace(agent)
    return trace.record(event, **fields)
