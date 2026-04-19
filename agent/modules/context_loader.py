"""Enterprise Context Loader — Hermes input-pipeline submodule C§1.2.

Receives a UserMessage and an IdentityPacket, then assembles a bounded
ContextPackage (≤N tokens) for the downstream Conversation Interpreter.
Sources include: session history, memory-engine recall, active tasks,
and company-scoped data for enterprise mode.

Phase-3 build plan reference: §C§1 table, row 2.
Wire-up to the central Hermes entrypoint is task C§1.9 (not this file).

Event emitted: ``hermes.context.assembled``
Emission mechanism: stdout JSON line (single-line, newline-terminated).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.modules.identity import IdentityPacket

# ---------------------------------------------------------------------------
# I/O types
# ---------------------------------------------------------------------------

#: Default token budget for the assembled context package.
DEFAULT_TOKEN_BUDGET = 8_192


class UserMessage(BaseModel):
    """Raw inbound message before any processing."""

    text: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str
    turn_index: int = 0


class ContextPackage(BaseModel):
    """Bounded context snapshot delivered to the Conversation Interpreter.

    ``token_estimate`` is a best-effort rough count; exact counting is
    delegated to the model-layer during prompt assembly.
    """

    session_history: list[dict[str, Any]] = Field(default_factory=list)
    memory_snippets: list[str] = Field(default_factory=list)
    active_tasks: list[dict[str, Any]] = Field(default_factory=list)
    company_context: Optional[dict[str, Any]] = None
    token_estimate: int = 0
    token_budget: int = DEFAULT_TOKEN_BUDGET


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


def _emit(event: str, payload: dict) -> None:
    """Write a single-line JSON event to stdout.

    Replace with an @agrv/hermes-events call in C§1.9 when the shared
    event bus is wired into this workspace.
    """
    line = json.dumps({"event": event, **payload}, default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Submodule entry point
# ---------------------------------------------------------------------------


def assemble_context(
    message: UserMessage,
    identity: IdentityPacket,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> ContextPackage:
    """Assemble a bounded ContextPackage for the current turn.

    Stub implementation: returns an empty package with metadata populated.
    Full implementation (memory-engine recall, session-DB history hydration,
    enterprise company-context fetch) is deferred to C§1.9.

    Emits ``hermes.context.assembled`` on completion.
    """
    package = ContextPackage(
        session_history=[],
        memory_snippets=[],
        active_tasks=[],
        company_context={"company_id": identity.company_id}
        if identity.company_id
        else None,
        token_estimate=0,
        token_budget=token_budget,
    )

    _emit(
        "hermes.context.assembled",
        {
            "session_id": message.session_id,
            "user_id": identity.user_id,
            "token_estimate": package.token_estimate,
            "token_budget": package.token_budget,
            "history_turns": len(package.session_history),
            "memory_snippets": len(package.memory_snippets),
        },
    )

    return package
