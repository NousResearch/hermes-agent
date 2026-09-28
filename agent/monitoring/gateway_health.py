"""Gateway health and diagnostics signal producer.

Keeps the plane narrow: service health plus redacted operational diagnostics,
derived from the existing gateway runtime-status contract. No prompts,
messages, tool args, session history, audit records, or product analytics.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent.monitoring import emitter
from agent.monitoring.events import GatewayDiagnosticEvent, GatewayHealthEvent
from agent.monitoring.redaction import redact_bounded

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GatewayMetric:
    name: str
    value: int | float
    attributes: Dict[str, str]


@dataclass(frozen=True, slots=True)
class GatewayHealthSnapshot:
    metrics: List[GatewayMetric]
    events: List[GatewayHealthEvent | GatewayDiagnosticEvent]


_RUNNING_PLATFORM_STATES = {"running", "connected", "ok", "ready"}
_FATAL_PLATFORM_STATES = {"fatal", "degraded", "error", "failed"}
_KNOWN_GATEWAY_STATES = _RUNNING_PLATFORM_STATES | _FATAL_PLATFORM_STATES | {"starting", "draining", "stopping", "stopped", "startup_failed", "unknown"}
_KNOWN_PLATFORM_STATES = _RUNNING_PLATFORM_STATES | _FATAL_PLATFORM_STATES | {"connecting", "disconnected", "disabled", "paused", "retrying", "unknown"}
_SUPERVISION_MODES = {"systemd", "s6", "container", "launchd", "manual", "unknown"}
_SOURCE_LOGGER_RE = re.compile(r"^gateway(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def source_logger_for_export(name: Any) -> Optional[str]:
    """Return a bounded source-controlled gateway logger name for OTLP scope."""
    value = str(name or "")
    return value if len(value) <= 128 and _SOURCE_LOGGER_RE.fullmatch(value) else None


def _contains_any(*needles: str) -> Callable[[str], bool]:
    return lambda text: any(needle in text for needle in needles)


# Ordered (predicate, class) rules; first match wins, so auth outranks rate-limit etc.
_GATEWAY_ERROR_RULES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (_contains_any("auth", "token", "unauthorized", "forbidden", "401", "403"), "auth_failed"),
    (lambda s: "rate" in s and "limit" in s, "rate_limited"),
    (_contains_any("timeout", "timed out"), "timeout"),
    (_contains_any("network", "connection", "dns", "socket", "connect call failed", "failed to connect", "cannot connect",
                   "unreachable", "name resolution"), "network_error"),
    (_contains_any("config", "missing", "invalid"), "invalid_config"),
    (_contains_any("startup"), "startup_failed"),
    (_contains_any("fatal"), "platform_fatal"),
)


def classify_gateway_error(raw: Any) -> str:
    s = str(raw or "").lower()
    return next((label for match, label in _GATEWAY_ERROR_RULES if match(s)), "unknown")


def classify_exit_reason(raw: Any, *, state: Any, restart_requested: bool) -> Optional[str]:
    """Reduce free-form shutdown text to a bounded operational class."""
    if restart_requested:
        return "restart_requested"
    state_name = str(state or "").lower()
    if raw is None and state_name != "startup_failed":
        return None
    classified = classify_gateway_error(raw)
    if state_name == "startup_failed":
        return classified if classified != "unknown" else "startup_failed"
    text = str(raw or "").lower()
    if "signal" in text or "sigterm" in text or "sigint" in text:
        return "signal"
    if state_name == "stopped" and any(word in text for word in ("shutdown", "stop")):
        return "planned_stop"
    return classified


def _bounded_state(raw: Any, *, allowed: set[str]) -> str:
    state = str(raw or "unknown").lower()
    return state if state in allowed else "unknown"


def _optional_state(raw: Any, *, allowed: set[str]) -> Optional[str]:
    """``_bounded_state`` that preserves "absent" (None) instead of coercing to unknown."""
    return None if raw is None else _bounded_state(raw, allowed=allowed)


def _safe_metric_value(raw: Any, *, limit: int = 128) -> str:
    return redact_bounded(raw, limit=limit, empty="unknown", unavailable="unknown")


def _safe_instance_id(raw: Any) -> str:
    """Return a stable opaque instance key without exporting the source ID."""
    value = str(raw or "unknown").encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(value).hexdigest()[:24]}"


def subsystem_for_logger(logger_name: str) -> str:
    parts = logger_name.split(".")
    if parts[:2] == ["gateway", "relay"]:
        return "platform.relay"
    if parts[:2] == ["gateway", "platforms"] and len(parts) >= 3 and parts[2]:
        return f"platform.{parts[2]}"
    return "platform" if logger_name.startswith("gateway.platforms") else "gateway"


def platform_for_subsystem(subsystem: str) -> Optional[str]:
    return (subsystem.split(".", 1)[1] or None) if subsystem.startswith("platform.") else None


def _coerce_pid(raw: Any) -> Optional[int]:
    try:
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _gateway_status(name: str, fallback: Callable[[], Any], /, **kwargs: Any) -> Any:
    """Prefer ``gateway.status.<name>`` (the runtime-status contract); fall back to the local approximation."""
    try:
        import gateway.status as status
        return getattr(status, name)(**kwargs)
    except Exception:
        return fallback()


def _parse_active_agents(raw: Any) -> int:
    def fallback() -> int:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    return _gateway_status("parse_active_agents", fallback, raw=raw)


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _platforms_of(runtime: Optional[dict[str, Any]]) -> dict[str, Any]:
    return _dict_or_empty((runtime or {}).get("platforms"))


def build_gateway_health_snapshot(
    runtime: Optional[dict[str, Any]], *, gateway_running: bool, profile: str, install_id: str, version: str,
    supervision_mode: str = "unknown",
) -> GatewayHealthSnapshot:
    """Convert gateway_state.json-compatible runtime state into P0 signals."""
    runtime = runtime or {}
    gateway_state = _bounded_state(runtime.get("gateway_state"), allowed=_KNOWN_GATEWAY_STATES)
    active_agents = _parse_active_agents(runtime.get("active_agents", 0))
    running = gateway_running and gateway_state == "running"
    busy = _gateway_status(
        "derive_gateway_busy", lambda: bool(running and _parse_active_agents(active_agents) > 0),
        gateway_running=gateway_running, gateway_state=gateway_state, active_agents=active_agents,
    )
    drainable = _gateway_status(
        "derive_gateway_drainable", lambda: bool(running), gateway_running=gateway_running, gateway_state=gateway_state
    )
    platforms = _platforms_of(runtime)
    mode = str(supervision_mode or "unknown").lower()
    base = {
        "service.instance.id": _safe_instance_id(install_id),
        "service.version": _safe_metric_value(version, limit=64),
        "hermes.supervision_mode": mode if mode in _SUPERVISION_MODES else "unknown",
    }

    def metric(name: str, value: int | float, **extra: str) -> GatewayMetric:
        attrs = dict(base)
        for key, val in extra.items():
            if val is not None:
                attrs[key] = _safe_metric_value(val)
        return GatewayMetric(name=name, value=value, attributes=attrs)

    metrics: list[GatewayMetric] = [
        metric("hermes.gateway.up", int(bool(gateway_running))),
        metric("hermes.gateway.active_agents", active_agents),
        metric("hermes.gateway.busy", int(bool(busy))),
        metric("hermes.gateway.drainable", int(bool(drainable))),
        metric("hermes.gateway.restart_requested", int(bool(runtime.get("restart_requested")))),
        metric("hermes.gateway.state", 1, **{"hermes.gateway.state": gateway_state}),
    ]
    fatal_count = 0
    events: list[GatewayHealthEvent | GatewayDiagnosticEvent] = []
    # classify_* is idempotent on its own labels, so error_class == error_code downstream.
    for platform, pdata in platforms.items():
        pdata = _dict_or_empty(pdata)
        state = _bounded_state(pdata.get("state"), allowed=_KNOWN_PLATFORM_STATES)
        error_code = classify_gateway_error(pdata.get("error_code") or pdata.get("error_message"))
        is_degraded = state in _FATAL_PLATFORM_STATES
        fatal_count += is_degraded
        pattrs = {"hermes.platform": str(platform), "hermes.platform.state": state}
        metrics.append(metric("hermes.platform.up", int(state in _RUNNING_PLATFORM_STATES), **pattrs))
        metrics.append(metric("hermes.platform.degraded", int(is_degraded), **pattrs, **{"hermes.error_code": error_code}))
        if is_degraded:
            events.append(GatewayDiagnosticEvent(
                name="platform.fatal", subsystem=f"platform.{platform}", platform=str(platform),
                error_code=error_code, error_class=error_code, profile=profile, version=version,
                severity="error" if state == "fatal" else "warning",
            ))
    events.insert(0, GatewayHealthEvent(
        name="gateway.health_snapshot", gateway_state=gateway_state, active_agents=active_agents, gateway_busy=busy,
        gateway_drainable=drainable, platform_count=len(platforms), fatal_platform_count=fatal_count, profile=profile,
        install_id=install_id, version=version, supervision_mode=supervision_mode, pid=_coerce_pid(runtime.get("pid")),
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


def _lifecycle_events(
    previous: Optional[dict[str, Any]], current: dict[str, Any], *, profile: str, version: str
) -> list[GatewayHealthEvent | GatewayDiagnosticEvent]:
    """Gateway-level transition events: lifecycle, startup_failed diagnostic, exit."""
    old_state = _optional_state((previous or {}).get("gateway_state"), allowed=_KNOWN_GATEWAY_STATES)
    new_state = _optional_state(current.get("gateway_state"), allowed=_KNOWN_GATEWAY_STATES)
    if old_state == new_state or not new_state:
        return []
    restart_requested = bool(current.get("restart_requested"))

    def health(name: str) -> GatewayHealthEvent:
        return GatewayHealthEvent(
            name=name, gateway_state=new_state, old_state=old_state, new_state=new_state,
            exit_reason=classify_exit_reason(current.get("exit_reason"), state=new_state, restart_requested=restart_requested),
            restart_requested=restart_requested, active_agents=_parse_active_agents(current.get("active_agents", 0)),
            profile=profile, version=version, pid=_coerce_pid(current.get("pid")),
        )

    out: list[GatewayHealthEvent | GatewayDiagnosticEvent] = [health("gateway.lifecycle")]
    if new_state == "startup_failed":
        error_class = classify_gateway_error(current.get("exit_reason") or "startup_failed")
        out.append(GatewayDiagnosticEvent(
            name="gateway.startup_failed", subsystem="gateway", error_class=error_class, error_code=error_class,
            profile=profile, version=version, severity="error",
        ))
    if new_state == "stopped":
        out.append(health("gateway.exit"))
    return out


def _platform_events(
    previous: Optional[dict[str, Any]], current: dict[str, Any], *, profile: str, version: str
) -> list[GatewayDiagnosticEvent]:
    """Per-platform state_change diagnostics, plus platform.fatal when the new state is fatal."""
    old_platforms = _platforms_of(previous)
    out: list[GatewayDiagnosticEvent] = []
    for platform, pdata in _platforms_of(current).items():
        pdata = _dict_or_empty(pdata)
        prev = _dict_or_empty(old_platforms.get(platform, {}))
        old_state = _optional_state(prev.get("state"), allowed=_KNOWN_PLATFORM_STATES)
        new_state = _optional_state(pdata.get("state"), allowed=_KNOWN_PLATFORM_STATES)
        if old_state == new_state or not new_state:
            continue
        error_code = classify_gateway_error(pdata.get("error_code") or pdata.get("error_message"))
        common: dict[str, Any] = dict(
            subsystem=f"platform.{platform}", platform=str(platform), error_code=error_code, error_class=error_code,
            profile=profile, version=version, severity="error" if new_state in {"fatal", "failed", "error"} else "warning",
        )
        out.append(GatewayDiagnosticEvent(name="platform.state_change", old_state=old_state, new_state=new_state, **common))
        if new_state in _FATAL_PLATFORM_STATES:
            out.append(GatewayDiagnosticEvent(name="platform.fatal", **common))
    return out


def emit_runtime_status_transition(previous: Optional[dict[str, Any]], current: dict[str, Any]) -> None:
    """Emit immediate content-free gateway events for runtime status changes.  Called by
    gateway.status.write_runtime_status after persisting; fully fail-open."""
    try:
        ctx = dict(profile=_safe_profile(), version=_safe_version())
        for ev in _lifecycle_events(previous, current, **ctx) + _platform_events(previous, current, **ctx):
            emitter.emit(ev)
    except Exception:
        logger.debug("gateway runtime status transition emit failed", exc_info=True)


class GatewayDiagnosticLogHandler(logging.Handler):
    """Allowlisted warning/error bridge for gateway-owned diagnostics."""

    def __init__(self, *, profile: str = "default", version: str = "unknown") -> None:
        super().__init__(level=logging.WARNING)
        self.profile = profile
        self.version = version

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.WARNING or not (record.name == "gateway" or record.name.startswith("gateway.")):
                return
            subsystem = subsystem_for_logger(record.name)
            error_class = classify_gateway_error(record.getMessage())
            emitter.get_emitter().emit(GatewayDiagnosticEvent(
                name=f"gateway.log.{record.levelname.lower()}", subsystem=subsystem,
                source_logger=source_logger_for_export(record.name), platform=platform_for_subsystem(subsystem),
                error_class=error_class, error_code=error_class, profile=self.profile, version=self.version,
                severity=record.levelname.lower(),
            ))
        except Exception:
            logger.debug("gateway diagnostic emit failed", exc_info=True)


__all__ = [
    "GatewayMetric", "GatewayHealthSnapshot", "GatewayDiagnosticLogHandler",
    "build_gateway_health_snapshot", "classify_gateway_error", "source_logger_for_export",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def redact_gateway_message(message: Any) -> str:
    """Redact gateway diagnostic free text for operator-owned export.

    Single scrub path: everything goes through
    ``agent.monitoring.redaction.redact_for_export`` (unconditional
    secrets + PII), then is length-bounded.
    """
    try:
        from agent.monitoring.redaction import redact_for_export
        redacted = redact_for_export(str(message or "")) or ""
    except Exception:
        redacted = "[redaction-unavailable]"
    return redacted[:500]
# ---- END PLUGIN-COMPAT ----
