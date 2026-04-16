"""
Database module for Hermes Agent.

Contains database functionality including:
- trace_manager_v2: Database operations for enhanced trace system
- trace_schema_v2: Database schema for trace system V2
- migrations: Database migration scripts
"""

from .trace_manager_v2 import TraceManagerV2, get_trace_manager

__all__ = [
    "TraceManagerV2",
    "get_trace_manager"
]