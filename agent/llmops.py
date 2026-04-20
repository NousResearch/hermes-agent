from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

from hermes_state import SessionDB


@dataclass
class RunEnvelope:
    run_id: str
    session_id: str
    task_id: str
    platform: str = ""
    thread_id: str = ""
    workflow: str = "conversation"
    backend: str = "legacy"
    model: str = ""
    tool_names: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RunRecorder:
    """Thin durable recorder for LLMOps run events."""

    def __init__(self, session_db: Optional[SessionDB], envelope: RunEnvelope):
        self.session_db = session_db
        self.envelope = envelope

    def record(self, event_type: str, **payload: Any) -> Dict[str, Any]:
        event_payload = dict(payload)
        if self.session_db and self.envelope.session_id:
            self.session_db.append_event(
                session_id=self.envelope.session_id,
                event_type=event_type,
                payload=event_payload,
            )
        return event_payload

    def metadata(self, **extra: Any) -> Dict[str, Any]:
        data = self.envelope.to_dict()
        data.update(extra)
        return data


def build_run_envelope(
    *,
    session_id: Optional[str],
    task_id: Optional[str],
    platform: Optional[str],
    thread_id: Optional[str],
    workflow: Optional[str],
    backend: Optional[str],
    model: Optional[str],
    tool_names: Iterable[str] | None,
) -> RunEnvelope:
    return RunEnvelope(
        run_id=str(uuid.uuid4()),
        session_id=session_id or "",
        task_id=task_id or "",
        platform=platform or "",
        thread_id=thread_id or "",
        workflow=workflow or "conversation",
        backend=backend or "legacy",
        model=model or "",
        tool_names=list(tool_names or []),
    )
