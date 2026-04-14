#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class OrchestratorRuntimeRequest:
    command: str
    request: dict[str, Any]
    state_db: Path
    retention_max_events: int


@dataclass
class OrchestratorRuntimeContext:
    envelope: OrchestratorRuntimeRequest
    connection: sqlite3.Connection | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorRuntimeResult:
    packet: dict[str, Any]
    exit_code: int = 0


class OrchestratorRuntimeBehavior(Protocol):
    def before(self, context: OrchestratorRuntimeContext) -> None: ...

    def after(self, context: OrchestratorRuntimeContext, result: OrchestratorRuntimeResult) -> None: ...

    def on_error(self, context: OrchestratorRuntimeContext, error: Exception) -> None: ...


class OrchestratorRuntimeHandler(Protocol):
    def __call__(self, context: OrchestratorRuntimeContext) -> OrchestratorRuntimeResult: ...


@dataclass(frozen=True)
class _HandlerRegistration:
    handler: OrchestratorRuntimeHandler
    requires_db: bool


class OrchestratorRuntimeMediator:
    """Mediator-ready command dispatcher with runtime-isolated handler execution.

    Each dispatch call executes exactly one handler in an isolated runtime context.
    Handlers can opt into a short-lived SQLite connection scoped to the call.
    """

    def __init__(self, *, connect_db: Callable[[Path], sqlite3.Connection]) -> None:
        self._connect_db = connect_db
        self._handlers: dict[str, _HandlerRegistration] = {}
        self._behaviors: list[OrchestratorRuntimeBehavior] = []

    def register(self, command: str, handler: OrchestratorRuntimeHandler, *, requires_db: bool = False) -> None:
        token = str(command or "").strip()
        if not token:
            raise ValueError("command_required")
        if token in self._handlers:
            raise ValueError(f"duplicate_handler:{token}")
        self._handlers[token] = _HandlerRegistration(handler=handler, requires_db=requires_db)

    def add_behavior(self, behavior: OrchestratorRuntimeBehavior) -> None:
        self._behaviors.append(behavior)

    def dispatch(self, envelope: OrchestratorRuntimeRequest) -> OrchestratorRuntimeResult:
        registration = self._handlers.get(envelope.command)
        if registration is None:
            raise ValueError(f"unregistered_command:{envelope.command}")

        context = OrchestratorRuntimeContext(envelope=envelope)
        for behavior in self._behaviors:
            behavior.before(context)

        try:
            with self._runtime_scope(context, requires_db=registration.requires_db):
                result = registration.handler(context)
            for behavior in reversed(self._behaviors):
                behavior.after(context, result)
            return result
        except Exception as exc:
            for behavior in reversed(self._behaviors):
                behavior.on_error(context, exc)
            raise

    @contextlib.contextmanager
    def _runtime_scope(self, context: OrchestratorRuntimeContext, *, requires_db: bool):
        if not requires_db:
            yield
            return

        con = self._connect_db(context.envelope.state_db)
        context.connection = con
        try:
            yield
        finally:
            context.connection = None
            con.close()
