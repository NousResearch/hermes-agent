"""Reactive same-role merge for the MoA aggregator request (#112358, last atom).

The aggregator request deliberately ends ``user(task), user(guidance)`` on iteration 1 of a
turn: the guidance is its own trailing message so every earlier message stays byte-stable and
the provider prefix cache keeps growing (``moa_loop._attach_reference_guidance``). Strict-
alternation chat templates (llama.cpp / vLLM Jinja templates, Mistral, some OpenRouter routes)
400 on that adjacency. Merging proactively for everyone would re-introduce the divergence the
split shape removed and only move the 400, so the merge is reactive and destination-scoped:

* a 400 classified ``FailoverReason.role_alternation`` → retry ONCE with adjacent same-role
  messages merged;
* the destination (``base_url``/provider + model) is remembered on the facade for the rest of
  the session, so later iterations pre-merge for it and never pay the 400 again;
* destinations that accepted the split shape are never touched — their prefix stays byte-stable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def destination_key(runtime: dict[str, Any]) -> tuple[str, str]:
    """``(route, model)`` identity of an aggregator destination: the base_url when the slot
    resolved one (two providers can share a model id), else the provider slug."""
    route = str(runtime.get("base_url") or runtime.get("provider") or "").strip().rstrip("/")
    return route, str(runtime.get("model") or "").strip()


def merge_same_role_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``messages`` with adjacent user turns folded into one (the only same-role adjacency
    the aggregator request produces); other rows are shared, never mutated. Returns the input
    object itself when nothing merged so callers can detect a no-op."""
    from agent.agent_runtime_helpers import _UNMERGEABLE, _merge_user_content

    merged: list[dict[str, Any]] = []
    changed = False
    for message in messages:
        prev = merged[-1] if merged else None
        content: Any = _UNMERGEABLE
        if prev is not None and prev.get("role") == "user" and message.get("role") == "user":
            content = _merge_user_content(prev.get("content", ""), message.get("content", ""))
        if content is _UNMERGEABLE:
            merged.append(message)
            continue
        merged[-1] = {**prev, "content": content}
        changed = True
    return merged if changed else messages


def is_role_alternation_rejection(exc: Exception, runtime: dict[str, Any]) -> bool:
    """True when the aggregator destination rejected the request for adjacent same-role messages."""
    from agent.error_classifier import FailoverReason, classify_api_error

    try:
        classified = classify_api_error(
            exc, provider=str(runtime.get("provider") or ""), model=str(runtime.get("model") or ""),
            base_url=str(runtime.get("base_url") or ""),
        )
    except Exception:  # pragma: no cover - classification must never mask the original error
        return False
    return classified.reason is FailoverReason.role_alternation
