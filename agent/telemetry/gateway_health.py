"""Gateway health and diagnostics signal producer.

This module keeps the plane narrow: service health monitoring plus
redacted operational diagnostics. It reuses the existing gateway runtime-status
contract and emits content-free metrics/events. No prompts, messages, tool args,
session history, audit records, or product analytics belong here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.telemetry.events import GatewayDiagnosticEvent, GatewayHealthEvent


@dataclass(frozen=True, slots=True)
class GatewayMetric:
    name: str
    value: int | float
    attributes: Dict[str, str]


@dataclass(frozen=True, slots=True)
class GatewayHealthSnapshot:
    metrics: List[GatewayMetric]
    events: List[GatewayHealthEvent | GatewayDiagnosticEvent]


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+\-/]+=*", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]+|sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,})\b")
_SECRET_LITERAL_RE = re.compile(r"\*{3,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?!\w)")

_RUNNING_PLATFORM_STATES = {"running", "connected", "ok", "ready"}
_FATAL_PLATFORM_STATES = {"fatal", "degraded", "error", "failed"}


def _allowed_logger(name: str) -> bool:
    return name == "gateway" or name.startswith("gateway.")


def redact_gateway_message(message: Any) -> str:
    """Redact gateway diagnostic free text for customer-owned export."""
    text = str(message or "")
    try:
        from agent.telemetry.redaction import redact_for_export
        redacted = redact_for_export(text, content_mode="pii") or ""
    except Exception:
        redacted = "[redaction-unavailable]"
    redacted = _BEARER_RE.sub("[redacted]", redacted)
    redacted = _TOKEN_RE.sub("[redacted]", redacted)
    redacted = _SECRET_LITERAL_RE.sub("[redacted]", redacted)
    redacted = re.sub(r"\bBearer\s+\[[^\]]+\]", "[redacted]", redacted, flags=re.IGNORECASE)
    redacted = _EMAIL_RE.sub("[email]", redacted)
    redacted = _PHONE_RE.sub("[phone]", redacted)
    return redacted[:500]


def classify_gateway_error(raw: Any) -> str:
    s = str(raw or "").lower()
    if any(k in s for k in ("auth", "token", "unauthorized", "forbidden", "401", "403")):
        return "auth_failed"
    if "rate" in s and "limit" in s:
        return "rate_limited"
    if "timeout" in s or "timed out" in s:
        return "timeout"
    if any(k in s for k in ("network", "connection", "dns", "socket")):
        return "network_error"
    if any(k in s for k in ("config", "missing", "invalid")):
        return "invalid_config"
    if "startup" in s:
        return "startup_failed"
    if "fatal" in s:
        return "platform_fatal"
    return "unknown"


def subsystem_for_logger(logger_name: str) -> str:
    if logger_name.startswith("gateway.platforms."):
        parts = logger_name.split(".")
        if len(parts) >= 3 and parts[2]:
            return f"platform.{parts[2]}"
    if logger_name.startswith("gateway.platforms"):
        return "platform"
    if logger_name.startswith("gateway"):
        return "gateway"
    return "gateway"


def platform_for_subsystem(subsystem: str) -> Optional[str]:
    if subsystem.startswith("platform."):
        return subsystem.split(".", 1)[1] or None
    return None


def _parse_active_agents(raw: Any) -> int:
    try:
        from gateway.status import parse_active_agents
        return parse_active_agents(raw)
    except Exception:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0


def _derive_busy(gateway_running: bool, gateway_state: Any, active_agents: Any) -> bool:
    try:
        from gateway.status import derive_gateway_busy
        return derive_gateway_busy(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
            active_agents=active_agents,
        )
    except Exception:
        return bool(gateway_running and gateway_state == "running" and _parse_active_agents(active_agents) > 0)


def _derive_drainable(gateway_running: bool, gateway_state: Any) -> bool:
    try:
        from gateway.status import derive_gateway_drainable
        return derive_gateway_drainable(gateway_running=gateway_running, gateway_state=gateway_state)
    except Exception:
        return bool(gateway_running and gateway_state == "running")


def _base_attrs(*, profile: str, install_id: str, version: str, supervision_mode: str) -> Dict[str, str]:
    return {
        "hermes.profile": str(profile),
        "service.instance.id": str(install_id),
        "service.version": str(version),
        "hermes.supervision_mode": str(supervision_mode),
    }


def _metric(name: str, value: int | float, attrs: Dict[str, str], **extra: str) -> GatewayMetric:
    out = dict(attrs)
    for key, val in extra.items():
        if val is not None:
            out[key] = str(val)
    return GatewayMetric(name=name, value=value, attributes=out)


def build_gateway_health_snapshot(
    runtime: Optional[dict[str, Any]],
    *,
    gateway_running: bool,
    profile: str,
    install_id: str,
    version: str,
    supervision_mode: str = "unknown",
) -> GatewayHealthSnapshot:
    """Convert gateway_state.json-compatible runtime state into P0 signals."""
    runtime = runtime or {}
    gateway_state = runtime.get("gateway_state")
    active_agents = _parse_active_agents(runtime.get("active_agents", 0))
    busy = _derive_busy(gateway_running, gateway_state, active_agents)
    drainable = _derive_drainable(gateway_running, gateway_state)
    platforms = runtime.get("platforms") if isinstance(runtime.get("platforms"), dict) else {}
    base = _base_attrs(profile=profile, install_id=install_id, version=version, supervision_mode=supervision_mode)

    metrics: list[GatewayMetric] = [
        _metric("hermes.gateway.up", 1 if gateway_running else 0, base),
        _metric("hermes.gateway.active_agents", active_agents, base),
        _metric("hermes.gateway.busy", 1 if busy else 0, base),
        _metric("hermes.gateway.drainable", 1 if drainable else 0, base),
        _metric("hermes.gateway.restart_requested", 1 if runtime.get("restart_requested") else 0, base),
    ]
    if gateway_state:
        metrics.append(_metric("hermes.gateway.state", 1, base, **{"hermes.gateway.state": str(gateway_state)}))

    fatal_count = 0
    events: list[GatewayHealthEvent | GatewayDiagnosticEvent] = []
    for platform, pdata in platforms.items():
        pdata = pdata if isinstance(pdata, dict) else {}
        state = str(pdata.get("state") or "unknown").lower()
        raw_error = pdata.get("error_code") or pdata.get("error_message")
        error_code = classify_gateway_error(raw_error)
        is_up = state in _RUNNING_PLATFORM_STATES
        is_degraded = state in _FATAL_PLATFORM_STATES
        if is_degraded:
            fatal_count += 1
        metrics.append(_metric(
            "hermes.platform.up",
            1 if is_up else 0,
            base,
            **{"hermes.platform": str(platform), "hermes.platform.state": state},
        ))
        metrics.append(_metric(
            "hermes.platform.degraded",
            1 if is_degraded else 0,
            base,
            **{"hermes.platform": str(platform), "hermes.platform.state": state, "hermes.error_code": error_code},
        ))
        if is_degraded:
            events.append(GatewayDiagnosticEvent(
                name="platform.fatal",
                subsystem=f"platform.{platform}",
                platform=str(platform),
                error_code=error_code,
                error_class=classify_gateway_error(error_code or pdata.get("error_message")),
                redacted_message=redact_gateway_message(pdata.get("error_message")),
                profile=profile,
                version=version,
                severity="error" if state == "fatal" else "warning",
            ))

    events.insert(0, GatewayHealthEvent(
        name="gateway.health_snapshot",
        gateway_state=str(gateway_state) if gateway_state is not None else None,
        active_agents=active_agents,
        gateway_busy=busy,
        gateway_drainable=drainable,
        platform_count=len(platforms),
        fatal_platform_count=fatal_count,
        profile=profile,
        install_id=install_id,
        version=version,
        supervision_mode=supervision_mode,
        pid=_coerce_pid(runtime.get("pid")),
    ))
    return GatewayHealthSnapshot(metrics=metrics, events=events)


def _safe_profile() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return str(get_active_profile_name() or "default")
    except Exception:
        return "default"


def _safe_version() -> str:
    try:
        from hermes_cli import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def emit_runtime_status_transition(previous: Optional[dict[str, Any]], current: dict[str, Any]) -> None:
    """Emit immediate content-free gateway events for runtime status changes.

    Called by gateway.status.write_runtime_status after persisting the new status.
    Fully fail-open: failures never affect gateway status writes.
    """
    try:
        from agent.telemetry import emitter
        out: list[GatewayHealthEvent | GatewayDiagnosticEvent] = []
        profile = _safe_profile()
        version = _safe_version()
        old_gateway_state = str((previous or {}).get("gateway_state")) if (previous or {}).get("gateway_state") is not None else None
        new_gateway_state = str(current.get("gateway_state")) if current.get("gateway_state") is not None else None
        if old_gateway_state != new_gateway_state and new_gateway_state:
            out.append(GatewayHealthEvent(
                name="gateway.lifecycle",
                gateway_state=new_gateway_state,
                old_state=old_gateway_state,
                new_state=new_gateway_state,
                exit_reason=current.get("exit_reason"),
                restart_requested=bool(current.get("restart_requested")),
                active_agents=_parse_active_agents(current.get("active_agents", 0)),
                profile=profile,
                version=version,
                pid=_coerce_pid(current.get("pid")),
            ))
            if new_gateway_state == "startup_failed":
                out.append(GatewayDiagnosticEvent(
                    name="gateway.startup_failed",
                    subsystem="gateway",
                    error_class=classify_gateway_error(current.get("exit_reason") or "startup_failed"),
                    error_code=classify_gateway_error(current.get("exit_reason") or "startup_failed"),
                    redacted_message=redact_gateway_message(current.get("exit_reason") or "startup failed"),
                    profile=profile,
                    version=version,
                    severity="error",
                ))
            if new_gateway_state == "stopped":
                out.append(GatewayHealthEvent(
                    name="gateway.exit",
                    gateway_state=new_gateway_state,
                    old_state=old_gateway_state,
                    new_state=new_gateway_state,
                    exit_reason=current.get("exit_reason"),
                    restart_requested=bool(current.get("restart_requested")),
                    active_agents=_parse_active_agents(current.get("active_agents", 0)),
                    profile=profile,
                    version=version,
                    pid=_coerce_pid(current.get("pid")),
                ))

        old_platforms_raw = (previous or {}).get("platforms")
        new_platforms_raw = current.get("platforms")
        old_platforms = old_platforms_raw if isinstance(old_platforms_raw, dict) else {}
        new_platforms = new_platforms_raw if isinstance(new_platforms_raw, dict) else {}
        for platform, pdata in new_platforms.items():
            pdata = pdata if isinstance(pdata, dict) else {}
            prev_raw = old_platforms.get(platform, {})
            prev = prev_raw if isinstance(prev_raw, dict) else {}
            old_state = str(prev.get("state")) if prev.get("state") is not None else None
            new_state = str(pdata.get("state")) if pdata.get("state") is not None else None
            if old_state == new_state or not new_state:
                continue
            error_code = classify_gateway_error(pdata.get("error_code") or pdata.get("error_message"))
            severity = "error" if new_state.lower() in {"fatal", "failed", "error"} else "warning"
            out.append(GatewayDiagnosticEvent(
                name="platform.state_change",
                subsystem=f"platform.{platform}",
                platform=str(platform),
                old_state=old_state,
                new_state=new_state,
                error_code=error_code,
                error_class=error_code,
                redacted_message=redact_gateway_message(pdata.get("error_message")),
                profile=profile,
                version=version,
                severity=severity,
            ))
            if new_state.lower() in _FATAL_PLATFORM_STATES:
                out.append(GatewayDiagnosticEvent(
                    name="platform.fatal",
                    subsystem=f"platform.{platform}",
                    platform=str(platform),
                    error_code=error_code,
                    error_class=error_code,
                    redacted_message=redact_gateway_message(pdata.get("error_message")),
                    profile=profile,
                    version=version,
                    severity=severity,
                ))
        for ev in out:
            emitter.emit(ev)
    except Exception:
        logging.getLogger(__name__).debug("gateway runtime status transition emit failed", exc_info=True)


def _coerce_pid(raw: Any) -> Optional[int]:
    try:
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


class GatewayDiagnosticLogHandler(logging.Handler):
    """Allowlisted warning/error bridge for gateway-owned diagnostics."""

    def __init__(self, *, profile: str = "default", version: str = "unknown") -> None:
        super().__init__(level=logging.WARNING)
        self.profile = profile
        self.version = version

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.WARNING:
                return
            if not _allowed_logger(record.name):
                return
            subsystem = subsystem_for_logger(record.name)
            message = record.getMessage()
            event = GatewayDiagnosticEvent(
                name=f"gateway.log.{record.levelname.lower()}",
                subsystem=subsystem,
                platform=platform_for_subsystem(subsystem),
                error_class=classify_gateway_error(message),
                redacted_message=redact_gateway_message(message),
                profile=self.profile,
                version=self.version,
                severity=record.levelname.lower(),
            )
            from agent.telemetry import emitter
            emitter.get_emitter().emit(event)
        except Exception:
            logging.getLogger(__name__).debug("gateway diagnostic emit failed", exc_info=True)


__all__ = [
    "GatewayMetric",
    "GatewayHealthSnapshot",
    "GatewayDiagnosticLogHandler",
    "build_gateway_health_snapshot",
    "classify_gateway_error",
    "redact_gateway_message",
]
