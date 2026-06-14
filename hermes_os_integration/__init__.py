"""Hermes OS integration helpers for using Hermes Agent as a worker runtime."""

from .contracts import (
    AgentRequest,
    AgentResponse,
    DelegationRequest,
    RuntimeStatus,
    validate_agent_request,
    validate_agent_response,
)

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "DelegationRequest",
    "RuntimeStatus",
    "validate_agent_request",
    "validate_agent_response",
]
