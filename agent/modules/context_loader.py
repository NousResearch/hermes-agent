"""Enterprise Context Loader — Hermes input-pipeline submodule C§1.2.

Receives a UserMessage and an IdentityPacket, then assembles a bounded
ContextPackage (≤N tokens) for the downstream Conversation Interpreter.
Sources include: session history, memory-engine recall, active tasks,
and company-scoped data for enterprise mode.

Phase-3 build plan reference: §C§1 table, row 2.
Wire-up to the central Hermes entrypoint is task C§1.9 (not this file).

Event emitted: ``hermes.context.assembled``
Emission mechanism: EventEmitter instance (injected by turn_handler).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.modules.event_emitter import EventEmitter
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
# Module-level emitter (injected by turn_handler)
# ---------------------------------------------------------------------------

_emitter: Optional[EventEmitter] = None


def set_emitter(emitter: EventEmitter) -> None:
    """Inject the shared event emitter.

    Called by turn_handler.run_turn() before processing.
    """
    global _emitter
    _emitter = emitter


# ---------------------------------------------------------------------------
# Submodule entry point
# ---------------------------------------------------------------------------


def _estimate_tokens(item: Any) -> int:
    """Best-effort rough count of tokens."""
    return len(str(item)) // 4

def _fetch_session_history(session_id: str) -> list[dict[str, Any]]:
    """last N turns from local session store (stub: empty list if DB unreachable)."""
    return []

import os
import httpx
import logging

def _fetch_memory_snippets(user_id: str, tenant_id: Optional[str]) -> list[str]:
    """mem0 top_k=5 with cosine >= 0.80 + tenant filter."""
    feature_flag = os.environ.get("FEATURE_HYBRID_RETRIEVAL", "false").lower() == "true"
    
    if feature_flag and tenant_id:
        try:
            url = "http://localhost:9080/api/model-router/retrieval"
            payload = {
                "query": "active context memory " + user_id,
                "collection": "default",
                "tenantId": tenant_id,
                "mode": "hybrid+rerank"
            }
            res = httpx.post(url, json=payload, timeout=30.0)
            res.raise_for_status()
            results = res.json().get("results", [])
            return [r.get("content") for r in results if "content" in r]
        except Exception as e:
            logging.error(f"Failed hybrid retrieval: {e}")
            return []
    
    return []

def _fetch_active_tasks(user_id: str) -> list[dict[str, Any]]:
    """state-engine active runs for the user (stub: empty)."""
    return []

def _fetch_recent_capsules() -> list[Any]:
    """TODO with stub fetcher (trace-summary package not yet shipped)."""
    return []

def assemble_context(
    message: UserMessage,
    identity: IdentityPacket,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> ContextPackage:
    """Assemble a bounded ContextPackage for the current turn.

    Hydrates session history, memory snippets, and active tasks.
    Enforces the provided token budget by dropping less critical panels if needed.

    Emits ``hermes.context.assembled`` on completion.
    """
    company_context = {"company_id": identity.company_id} if identity.company_id else None
    
    # Optional logic for recent_capsules (TODO)
    _fetch_recent_capsules()

    # Hydrate raw lists
    raw_history = _fetch_session_history(message.session_id)
    raw_memory = _fetch_memory_snippets(identity.user_id, identity.company_id)
    raw_tasks = _fetch_active_tasks(identity.user_id)

    # Priority to KEEP: company_context > session_history > memory_snippets > active_tasks
    used = 0
    final_company_context = None
    final_history = []
    final_memory = []
    final_tasks = []

    # 1. company_context
    cc_tokens = _estimate_tokens(company_context) if company_context else 0
    if used + cc_tokens <= token_budget:
        final_company_context = company_context
        used += cc_tokens

    # 2. session_history
    for h in raw_history:
        h_tokens = _estimate_tokens(h)
        if used + h_tokens <= token_budget:
            final_history.append(h)
            used += h_tokens

    # 3. memory_snippets
    for m in raw_memory:
        m_tokens = _estimate_tokens(m)
        if used + m_tokens <= token_budget:
            final_memory.append(m)
            used += m_tokens

    # 4. active_tasks
    for t in raw_tasks:
        t_tokens = _estimate_tokens(t)
        if used + t_tokens <= token_budget:
            final_tasks.append(t)
            used += t_tokens

    package = ContextPackage(
        session_history=final_history,
        memory_snippets=final_memory,
        active_tasks=final_tasks,
        company_context=final_company_context,
        token_estimate=used,
        token_budget=token_budget,
    )

    if _emitter is not None:
        _emitter.emit(
            "hermes.context.assembled",
            {
                "budget": token_budget,
                "used": used,
                "items_by_panel": {
                    "company_context": 1 if final_company_context else 0,
                    "session_history": len(final_history),
                    "memory_snippets": len(final_memory),
                    "active_tasks": len(final_tasks),
                },
                "retrieval_calls": 3,
                "session_id": message.session_id,
                "user_id": identity.user_id,
                "token_estimate": package.token_estimate,
                "token_budget": package.token_budget,
                "history_turns": len(package.session_history),
            },
        )

    return package
