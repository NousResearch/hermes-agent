"""Control Room — the shared keyboard-first control surface contract.

Domain contract (DTOs, attention ranking, action envelope) shared by the
CLI, Ink TUI, native Hermes Desktop and Kensei Dashboard. Pure module set:
no I/O, no gateway imports.
"""

from .attention import rank_attention, severity_for_kind
from .contract import (
    SNAPSHOT_VERSION,
    ActionTarget,
    AgentRow,
    AttentionItem,
    AttentionKind,
    AttentionSeverity,
    Capabilities,
    ControlRoomAction,
    ControlRoomActionResult,
    ControlRoomError,
    ControlRoomSnapshot,
    ErrorCode,
    MessageRow,
    SnapshotCounts,
    SourceMeta,
    SystemSummary,
    TaskRow,
    unavailable_attention,
    unavailable_source,
    unavailable_system,
)

__all__ = [
    "SNAPSHOT_VERSION",
    "ActionTarget",
    "AgentRow",
    "AttentionItem",
    "AttentionKind",
    "AttentionSeverity",
    "Capabilities",
    "ControlRoomAction",
    "ControlRoomActionResult",
    "ControlRoomError",
    "ControlRoomSnapshot",
    "ErrorCode",
    "MessageRow",
    "SnapshotCounts",
    "SourceMeta",
    "SystemSummary",
    "TaskRow",
    "rank_attention",
    "severity_for_kind",
    "unavailable_attention",
    "unavailable_source",
    "unavailable_system",
]
