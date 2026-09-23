"""Typed gateway monitoring events.

Content-free service-health and redacted diagnostic events for the gateway
daemon — the only shapes the monitoring plane emits: no prompts, messages,
tool args/results, session history, or usage analytics. Field order is wire
order (``asdict``); never reorder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar, Dict, Optional


class _MonitoringEvent:
    __slots__ = ()
    EVENT: ClassVar[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"event": self.EVENT, **asdict(self)}


@dataclass(slots=True)
class GatewayHealthEvent(_MonitoringEvent):
    """Content-free gateway health snapshot or lifecycle event."""
    EVENT: ClassVar[str] = "gateway_health"
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
    ts_ns: int = field(default_factory=time.time_ns)


@dataclass(slots=True)
class GatewayDiagnosticEvent(_MonitoringEvent):
    """Redacted gateway diagnostic event for operator-owned observability."""
    EVENT: ClassVar[str] = "gateway_diagnostic"
    name: str
    subsystem: str
    error_class: str = "unknown"
    error_code: Optional[str] = None
    platform: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    profile: Optional[str] = None
    version: Optional[str] = None
    severity: str = "warning"
    ts_ns: int = field(default_factory=time.time_ns)
    source_logger: Optional[str] = None


@dataclass(slots=True)
class CronExecutionEvent(_MonitoringEvent):
    """Content-free durable cron execution lifecycle projection."""
    EVENT: ClassVar[str] = "cron_execution"
    status: str
    job_key: str
    source: str = "unknown"
    duration_ms: Optional[int] = None
    delivery_outcome: Optional[str] = None
    error_class: Optional[str] = None
    ts_ns: int = field(default_factory=time.time_ns)


__all__ = ["GatewayHealthEvent", "GatewayDiagnosticEvent", "CronExecutionEvent"]
