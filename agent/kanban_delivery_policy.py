"""Canonical pre-delivery containment for unresolved architecture gates.

The policy is deliberately keyed from the server-side board state and task id.
It never trusts model output or a caller supplied ``approved`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class ArchitectureDeliveryPolicy:
    gate_id: str
    state: str
    _buffer: list[str] = field(default_factory=list, repr=False)

    @property
    def withholding(self) -> bool:
        return self.state not in {"human_approved"}

    @property
    def receipt(self) -> str:
        return f"Architecture approval pending; output withheld (gate {self.gate_id})."

    def buffer(self, text: object) -> None:
        if isinstance(text, str) and text:
            self._buffer.append(text)

    def stream_delta(self, text: str) -> Optional[str]:
        if self.withholding:
            self.buffer(text)
            return None
        return text

    def interim(self, text: str) -> Optional[str]:
        if self.withholding:
            self.buffer(text)
            return None
        return text

    def final(self, text: object) -> object:
        if self.withholding:
            self.buffer(text)
            return self.receipt
        return text


def policy_for_current_kanban_task() -> Optional[ArchitectureDeliveryPolicy]:
    """Resolve the current worker's policy from canonical persistence.

    Failure is fail-open only when no Kanban task is active; a live task with a
    readable unresolved gate always resolves to a withholding policy.
    """
    task_id = os.environ.get("HERMES_KANBAN_TASK")
    if not task_id:
        return None
    try:
        from hermes_cli.kanban_db import connect, get_architecture_gate_for_task
        with connect() as conn:
            gate = get_architecture_gate_for_task(conn, task_id)
            if gate is None or gate.enforcement_mode != "enforce":
                return None
            return ArchitectureDeliveryPolicy(gate_id=gate.gate_id, state=gate.state)
    except Exception:
        # The delivery policy must not make non-Kanban chat unusable on a
        # transient board lookup failure. The database domain guard remains the
        # authority for all protected mutations.
        return None
