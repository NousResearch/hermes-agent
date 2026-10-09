"""One turn-wide live-tool budget shared through delegated descendants."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Optional


@dataclass(frozen=True)
class ToolExecutionBudgetDecision:
    allowed: bool
    used_tool_executions: int
    max_tool_executions: Optional[int]
    exhausted: bool
    became_exhausted: bool = False


class TurnResourceBudget:
    """Thread-safe process-local aggregate budget."""

    def __init__(self, max_tool_executions: Optional[int]) -> None:
        if max_tool_executions is not None:
            if isinstance(max_tool_executions, bool) or not isinstance(max_tool_executions, int):
                raise TypeError("max_tool_executions must be an int or None")
            if max_tool_executions <= 0:
                raise ValueError("max_tool_executions must be positive or None")
        self.max_tool_executions = max_tool_executions
        self._used_tool_executions = 0
        self._lock = threading.Lock()

    @property
    def used_tool_executions(self) -> int:
        with self._lock:
            return self._used_tool_executions

    @property
    def exhausted(self) -> bool:
        with self._lock:
            maximum = self.max_tool_executions
            return maximum is not None and self._used_tool_executions >= maximum

    def snapshot(self) -> ToolExecutionBudgetDecision:
        with self._lock:
            maximum = self.max_tool_executions
            exhausted = maximum is not None and self._used_tool_executions >= maximum
            return ToolExecutionBudgetDecision(
                not exhausted, self._used_tool_executions, maximum, exhausted
            )

    def try_consume_tool_execution(
        self, *, tool_name: str = "", tool_call_id: str = ""
    ) -> ToolExecutionBudgetDecision:
        del tool_name, tool_call_id
        with self._lock:
            maximum = self.max_tool_executions
            if maximum is not None and self._used_tool_executions >= maximum:
                return ToolExecutionBudgetDecision(
                    False, self._used_tool_executions, maximum, True
                )
            self._used_tool_executions += 1
            exhausted = maximum is not None and self._used_tool_executions >= maximum
            return ToolExecutionBudgetDecision(
                True, self._used_tool_executions, maximum, exhausted, exhausted
            )


class _RemoteTurnResourceBudget:
    """Common cached view for authority-mediated budget implementations."""

    def __init__(self, max_tool_executions: Optional[int]) -> None:
        self.max_tool_executions = max_tool_executions
        self._used_tool_executions = 0
        self._exhausted = False
        self._lock = threading.Lock()

    @property
    def used_tool_executions(self) -> int:
        with self._lock:
            return self._used_tool_executions

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._exhausted

    def snapshot(self) -> ToolExecutionBudgetDecision:
        with self._lock:
            return ToolExecutionBudgetDecision(
                not self._exhausted,
                self._used_tool_executions,
                self.max_tool_executions,
                self._exhausted,
            )

    def _consume(self, *, maximum: int, tool_name: str, tool_call_id: str) -> dict:
        raise NotImplementedError

    def try_consume_tool_execution(
        self, *, tool_name: str = "", tool_call_id: str = ""
    ) -> ToolExecutionBudgetDecision:
        with self._lock:
            if self._exhausted:
                return ToolExecutionBudgetDecision(
                    False, self._used_tool_executions, self.max_tool_executions, True
                )
            result = self._consume(
                maximum=self.max_tool_executions or 0,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
            self._used_tool_executions = int(result["used_tool_executions"])
            self._exhausted = bool(result["exhausted"])
            return ToolExecutionBudgetDecision(
                bool(result["allowed"]),
                self._used_tool_executions,
                self.max_tool_executions,
                self._exhausted,
                bool(result.get("became_exhausted")),
            )


class AdmissionTurnResourceBudget(_RemoteTurnResourceBudget):
    """Owner-ledger budget for an in-process One Gateway admission."""

    def __init__(
        self, db: Any, *, epoch: int, admission_id: str, generation: int,
        max_tool_executions: Optional[int],
    ) -> None:
        super().__init__(max_tool_executions)
        self.db = db
        self.epoch = epoch
        self.admission_id = admission_id
        self.generation = generation

    def _consume(self, *, maximum: int, tool_name: str, tool_call_id: str) -> dict:
        from hermes_state_runtime import consume_admission_tool_execution

        return consume_admission_tool_execution(
            self.db,
            epoch=self.epoch,
            admission_id=self.admission_id,
            generation=self.generation,
            max_tool_executions=maximum,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )


class WorkerTurnResourceBudget(_RemoteTurnResourceBudget):
    """Authority-mediated budget used by a managed worker process."""

    def __init__(self, store: Any, max_tool_executions: Optional[int]) -> None:
        super().__init__(max_tool_executions)
        self.store = store

    def _consume(self, *, maximum: int, tool_name: str, tool_call_id: str) -> dict:
        return self.store.consume_turn_tool_execution(
            max_tool_executions=maximum,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )


def build_turn_resource_budget(agent: Any):
    """Create a root budget; delegated children inherit the same handle."""
    from tools.budget_config import configured_max_tool_executions

    maximum = configured_max_tool_executions(getattr(agent, "max_iterations", None))
    session_db = getattr(agent, "_session_db", None)

    try:
        from agent.runtime_session_store import RuntimeSessionStore
        if isinstance(session_db, RuntimeSessionStore):
            return WorkerTurnResourceBudget(session_db, maximum)
    except Exception:
        pass

    try:
        from gateway.session_ingress import current_admission_resource_budget
        scope = current_admission_resource_budget()
    except Exception:
        scope = None
    if scope is not None:
        return AdmissionTurnResourceBudget(
            scope["db"],
            epoch=scope["epoch"],
            admission_id=scope["admission_id"],
            generation=scope["generation"],
            max_tool_executions=maximum,
        )
    return TurnResourceBudget(maximum)
