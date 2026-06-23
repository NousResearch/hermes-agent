"""Knowledge routing package for durable Hermes knowledge writes.

The package separates pure routing policy from backend adapters and tool
wrappers. The core router is deterministic and can be tested without network or
profile state.
"""

from knowledge.router import KnowledgeRouter, route_knowledge_write
from knowledge.types import (
    DuplicatePolicy,
    KnowledgeDestination,
    KnowledgeWriteRequest,
    KnowledgeWriteResult,
    RouteAction,
    RouteDecision,
)

__all__ = [
    "DuplicatePolicy",
    "KnowledgeDestination",
    "KnowledgeRouter",
    "KnowledgeWriteRequest",
    "KnowledgeWriteResult",
    "RouteAction",
    "RouteDecision",
    "route_knowledge_write",
]
