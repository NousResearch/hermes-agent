"""Minimal execution port owned by the Engineering architecture."""

from __future__ import annotations

from typing import Protocol

from .models import TurnRequest, TurnResult


class AgentRuntimeError(Exception):
    """Raised when a runtime adapter cannot honor the execution contract.

    Expected turn outcomes, including failure and interruption, are represented
    by ``TurnResult``. Exceptions are reserved for adapter or infrastructure
    failures that prevent a valid result from being returned.
    """


class AgentRuntime(Protocol):
    """Execute one agent turn without exposing runtime implementation details."""

    def run_turn(self, request: TurnRequest) -> TurnResult: ...
