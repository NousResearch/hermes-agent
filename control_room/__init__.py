"""Control Room — the shared keyboard-first control surface contract.

Domain contract (DTOs, attention ranking, action envelope), snapshot service,
plain-text renderer, action router/executors and creation flows shared by the
CLI, Ink TUI, native Hermes Desktop and Kensei Dashboard. Pure module set:
no I/O, no gateway imports (executors isolate backend access behind
injectable boundaries).
"""

from .actions import ControlRoomActionRouter
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
from .executors import default_executors
from .flows import NewAgentRunFlow, NewMessageFlow, NewTaskFlow
from .service import ControlRoomService
from .text import (
    attention_status_line,
    normalize_section,
    render_home,
    render_section,
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
    "ControlRoomActionRouter",
    "ControlRoomError",
    "ControlRoomService",
    "ControlRoomSnapshot",
    "ErrorCode",
    "MessageRow",
    "NewAgentRunFlow",
    "NewMessageFlow",
    "NewTaskFlow",
    "SnapshotCounts",
    "SourceMeta",
    "SystemSummary",
    "TaskRow",
    "attention_status_line",
    "default_executors",
    "normalize_section",
    "rank_attention",
    "render_home",
    "render_section",
    "severity_for_kind",
    "unavailable_attention",
    "unavailable_source",
    "unavailable_system",
]
