"""Typed gateway monitoring events.

Content-free service-health and redacted diagnostic events for the gateway
daemon. These are the only event shapes the monitoring plane emits: no
prompts, messages, tool args/results, session history, or usage analytics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


def _now_ns() -> int:
    return time.time_ns()


@dataclass(slots=True)
class GatewayHealthEvent:
    """Content-free gateway health snapshot or lifecycle event."""

    name: str
    gateway_state: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    exit_reason: Optional[str] = None
    restart_requested: Optional[bool] = None
    active_agents: int = 0
    gateway_busy: bool = False
    gateway_drainable: bool = False
    platform_count: int = 0
    fatal_platform_count: int = 0
    profile: Optional[str] = None
    install_id: Optional[str] = None
    version: Optional[str] = None
    supervision_mode: Optional[str] = None
    pid: Optional[int] = None
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "gateway_health", **asdict(self)}


@dataclass(slots=True)
class GatewayDiagnosticEvent:
    """Redacted gateway diagnostic event for operator-owned observability."""

    name: str
    subsystem: str
    error_class: str = "unknown"
    error_code: Optional[str] = None
    redacted_message: Optional[str] = None
    platform: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    profile: Optional[str] = None
    version: Optional[str] = None
    severity: str = "warning"
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "gateway_diagnostic", **asdict(self)}


__all__ = [
    "GatewayHealthEvent",
    "GatewayDiagnosticEvent",
]
