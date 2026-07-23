"""Session-local capability tracking for structured reasoning replay."""

from __future__ import annotations

from typing import Any

from agent.backend_identity import (
    BackendIdentity,
    FailureScope,
    should_skip_candidate,
)


def current_backend_identity(
    agent: Any, *, model: Any = None
) -> BackendIdentity:
    """Build the normalized identity for the agent's active deployment."""
    return BackendIdentity.build(
        provider=getattr(agent, "provider", None),
        model=getattr(agent, "model", None) if model is None else model,
        base_url=getattr(agent, "base_url", None),
    )


def reasoning_details_rejected_for_current_backend(agent: Any) -> bool:
    """Return whether this exact model deployment rejected replay metadata."""
    candidate = current_backend_identity(agent)
    rejected = getattr(agent, "_reasoning_details_rejected_backends", set())
    return any(
        should_skip_candidate(candidate, failed, FailureScope.MODEL)
        for failed in rejected
    )


def remember_reasoning_details_rejection(
    agent: Any, *, model: Any = None
) -> tuple[BackendIdentity, bool]:
    """Remember a schema rejection for the active model deployment.

    Returns ``(identity, learned)`` where ``learned`` is false when an
    equivalent deployment was already recorded.
    """
    identity = current_backend_identity(agent, model=model)
    # A model-scoped negative capability is useful only when all three axes
    # identify one concrete deployment.  Never create a wildcard-like entry
    # that could suppress replay for an unrelated model or endpoint.
    if not (identity.provider and identity.model and identity.base_url):
        return identity, False
    rejected = getattr(agent, "_reasoning_details_rejected_backends", None)
    if rejected is None:
        rejected = set()
        agent._reasoning_details_rejected_backends = rejected
    if any(
        should_skip_candidate(identity, failed, FailureScope.MODEL)
        for failed in rejected
    ):
        return identity, False
    rejected.add(identity)
    return identity, True


def outbound_messages_contain_reasoning_details(api_kwargs: Any) -> bool:
    """Check the actual outbound Chat Completions payload for the field."""
    if not isinstance(api_kwargs, dict):
        return False
    messages = api_kwargs.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict) and "reasoning_details" in message
        for message in messages
    )


__all__ = [
    "current_backend_identity",
    "outbound_messages_contain_reasoning_details",
    "reasoning_details_rejected_for_current_backend",
    "remember_reasoning_details_rejection",
]
