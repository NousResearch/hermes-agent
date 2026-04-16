"""
Core module for Hermes Agent.

Contains core functionality including:
- trace_system_v2: Enhanced trace system with three-level indexing
"""

from .trace_system_v2 import (
    TraceSystemV2,
    TraceEvent,
    TraceSession,
    EventType,
    EventPriority,
    IntelligentCompressor,
    SessionBoundaryDetector,
    TraceStreamContext,
    AsyncStorageManager,
    get_trace_system,
    record_event,
    create_trace_callbacks
)

__all__ = [
    "TraceSystemV2",
    "TraceEvent", 
    "TraceSession",
    "EventType",
    "EventPriority",
    "IntelligentCompressor",
    "SessionBoundaryDetector",
    "TraceStreamContext",
    "AsyncStorageManager",
    "get_trace_system",
    "record_event",
    "create_trace_callbacks"
]