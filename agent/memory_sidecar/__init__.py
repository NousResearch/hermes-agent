"""Hermes memory sidecar primitives.

MVP surface for structured transcript observations backed by a dedicated
SQLite/FTS5 sidecar database.
"""

from .models import (
    OBSERVATION_TYPES,
    PRIVACY_STATES,
    ContextQuery,
    ContextResult,
    IngestSource,
    Observation,
    ObservationFile,
    SessionFact,
    validate_observation_type,
    validate_privacy_status,
)
from .retrieval import format_context_result, merge_context_results
from .store import MemorySidecarStore

__all__ = [
    "OBSERVATION_TYPES",
    "PRIVACY_STATES",
    "ContextQuery",
    "ContextResult",
    "IngestSource",
    "MemorySidecarStore",
    "Observation",
    "format_context_result",
    "merge_context_results",
    "ObservationFile",
    "SessionFact",
    "validate_observation_type",
    "validate_privacy_status",
]
