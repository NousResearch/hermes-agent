"""Opt-in runtime-path diagnosis for ``hermes doctor --runtime``.

The probe resolves and constructs the selected profile's real primary runtime, but it is
persistence-disabled before dispatch. It makes exactly one low-level streaming request and
never enters the conversation loop, so no session, history, fallback, title, or delivery
side effects can be produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from time import perf_counter, time
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit
import uuid

from hermes_cli.doctor_report import _section, check_fail, check_ok, check_warn


logger = logging.getLogger(__name__)
_LOCAL_SLOW_MS = 1_000.0
_PROVIDER_SLOW_MS = 2_000.0


@dataclass
class RuntimePhase:
    name: str
    status: str
    elapsed_ms: float
    error_class: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }
        if self.error_class:
            result["error_class"] = self.error_class
        return result


@dataclass
class RuntimeReport:
    phases: list[RuntimePhase] = field(default_factory=list)
    resolved_runtime: dict[str, str] = field(default_factory=dict)
    status: str = "ok"
    failed_phase: Optional[str] = None
    pre_dispatch_ms: Optional[float] = None
    provider_ttfb_ms: Optional[float] = None
    provider_total_ms: Optional[float] = None
    hermes_first_chunk_ms: Optional[float] = None
    likely_bottleneck: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": self.status,
            "resolved_runtime": dict(self.resolved_runtime),
            "phases": [phase.to_dict() for phase in self.phases],
            "timings": {
                key: round(value, 3) if value is not None else None
                for key, value in (
                    ("pre_dispatch_ms", self.pre_dispatch_ms),
                    ("provider_ttfb_ms", self.provider_ttfb_ms),
                    ("provider_total_ms", self.provider_total_ms),
                    ("hermes_first_chunk_ms", self.hermes_first_chunk_ms),
                )
            },
            "likely_bottleneck": self.likely_bottleneck,
        }
        if self.failed_phase:
            result["failed_phase"] = self.failed_phase
            failed = next((p for p in self.phases if p.name == self.failed_phase), None)
            result["error_class"] = failed.error_class if failed else "RuntimeDiagnosticError"
        return result


def _safe_base_url(value: Any) -> str:
    """Return runtime identity without URL credentials, query parameters, or fragments."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    except (TypeError, ValueError):
        return "(invalid URL)"


def _record_phase(
    report: RuntimeReport,
    name: str,
    operation: Callable[[], Any],
    *,
    slow_ms: float = _LOCAL_SLOW_MS,
) -> Any:
    started = perf_counter()
    try:
        value = operation()
    except Exception as exc:
        elapsed = (perf_counter() - started) * 1_000
        report.phases.append(RuntimePhase(name, "fail", elapsed, type(exc).__name__))
        if report.failed_phase is None:
            report.failed_phase = name
        report.status = "fail"
        return None
    elapsed = (perf_counter() - started) * 1_000
    report.phases.append(RuntimePhase(name, "slow" if elapsed >= slow_ms else "ok", elapsed))
    return value


def _model_identity(config: dict[str, Any]) -> tuple[str, str]:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    return str(model_cfg.get("provider") or "auto"), str(model_cfg.get("default") or "")


def _runtime_identity(runtime: dict[str, Any], model: str) -> dict[str, str]:
    return {
        "provider": str(runtime.get("provider") or ""),
        "requested_provider": str(runtime.get("requested_provider") or runtime.get("provider") or ""),
        "model": model,
        "api_mode": str(runtime.get("api_mode") or ""),
        "base_url": _safe_base_url(runtime.get("base_url")),
    }


def _build_agent(runtime: dict[str, Any], model: str):
    from run_agent import AIAgent

    agent = AIAgent(
        model=model,
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        requested_provider=runtime.get("requested_provider"),
        api_mode=runtime.get("api_mode"),
        acp_command=runtime.get("command"),
        acp_args=runtime.get("args"),
        credential_pool=runtime.get("credential_pool"),
        max_iterations=1,
        max_tokens=8,
        quiet_mode=True,
        save_trajectories=False,
        skip_background_review=True,
        session_db=None,
        session_id=f"doctor-runtime-{uuid.uuid4().hex}",
        platform="doctor",
    )
    agent._persist_disabled = True
    agent._session_db = None
    agent._end_session_on_close = False
    agent.suppress_status_output = True
    agent._api_max_retries = 1
    return agent


def _minimal_request(agent: Any, system_prompt: str) -> tuple[float, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Reply exactly OK. Do not call tools."},
    ]
    api_kwargs = agent._build_api_kwargs(messages)
    started_perf = perf_counter()
    started_wall = time()
    agent._last_api_first_chunk_at = None
    agent._interruptible_streaming_api_call(api_kwargs)
    finished = perf_counter()
    first_chunk_at = getattr(agent, "_last_api_first_chunk_at", None)
    ttfb_ms = (
        max(0.0, float(first_chunk_at) - started_wall) * 1_000
        if first_chunk_at is not None
        else (finished - started_perf) * 1_000
    )
    return ttfb_ms, (finished - started_perf) * 1_000


def run_runtime_diagnostic(
    *,
    config_override: dict[str, Any] | None = None,
    runtime_override: dict[str, Any] | None = None,
) -> RuntimeReport:
    """Run the profile-scoped startup + one-request probe with per-phase isolation.

    Isolation diagnostics supply an already-resolved runtime so the ephemeral profile can
    use the source credential in memory without copying an auth file or secret to disk.
    """
    report = RuntimeReport()
    config = _record_phase(
        report,
        "profile_config_resolution",
        lambda: config_override if config_override is not None else __import__(
            "hermes_cli.config", fromlist=["load_config_readonly"]
        ).load_config_readonly(),
    )
    if config is None:
        return report
    requested_provider, model = _model_identity(config)

    def _resolve_runtime():
        if runtime_override is not None:
            return runtime_override
        from hermes_cli.runtime_provider import resolve_runtime_provider
        return resolve_runtime_provider(requested=requested_provider, target_model=model)

    runtime = _record_phase(report, "provider_resolution", _resolve_runtime)
    if runtime is None:
        return report
    report.resolved_runtime = _runtime_identity(runtime, model)

    _record_phase(
        report,
        "plugin_hook_initialization",
        lambda: __import__("hermes_cli.plugins", fromlist=["discover_plugins"]).discover_plugins(),
    )

    def _mcp_ready():
        from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build
        ensure_mcp_discovery_before_agent_build(
            logger=logger, single_query=True, thread_name="doctor-runtime-mcp-discovery"
        )

    _record_phase(report, "mcp_initialization", _mcp_ready)
    agent = _record_phase(report, "agent_tool_context_initialization", lambda: _build_agent(runtime, model))
    if agent is None:
        report.pre_dispatch_ms = sum(p.elapsed_ms for p in report.phases)
        return report

    try:
        prompt = _record_phase(
            report,
            "prompt_construction",
            lambda: agent._build_system_prompt(
                "Runtime diagnostic only. Reply exactly OK and do not call tools."
            ),
        )
        report.pre_dispatch_ms = sum(p.elapsed_ms for p in report.phases)
        if prompt is None:
            return report

        provider_result = _record_phase(
            report,
            "provider_first_chunk",
            lambda: _minimal_request(agent, prompt),
            slow_ms=_PROVIDER_SLOW_MS,
        )
        if provider_result is not None:
            report.provider_ttfb_ms, report.provider_total_ms = provider_result
            report.hermes_first_chunk_ms = report.pre_dispatch_ms + report.provider_ttfb_ms
            report.likely_bottleneck = (
                "local_pre_dispatch"
                if report.pre_dispatch_ms > report.provider_ttfb_ms
                else "provider_or_network"
            )
        return report
    finally:
        _record_phase(report, "agent_resource_cleanup", agent.close)


def render_runtime_report(report: RuntimeReport) -> None:
    """Render the human-readable form without exposing credentials or raw exceptions."""
    _section("Runtime Path (opt-in, one minimal inference request)")
    identity = report.resolved_runtime
    if identity:
        print(
            "  Resolved: "
            f"provider={identity['provider'] or '(none)'} "
            f"model={identity['model'] or '(none)'} "
            f"api_mode={identity['api_mode'] or '(none)'} "
            f"base_url={identity['base_url'] or '(default)'}"
        )
    reporters = {"ok": check_ok, "slow": check_warn, "fail": check_fail}
    for phase in report.phases:
        detail = f"({phase.elapsed_ms:.1f} ms)"
        if phase.error_class:
            detail = f"{detail} {phase.error_class}"
        reporters[phase.status](phase.name.replace("_", " ").title(), detail)
    if report.provider_ttfb_ms is not None:
        print(f"  Pre-dispatch total: {report.pre_dispatch_ms:.1f} ms")
        print(f"  Provider first chunk: {report.provider_ttfb_ms:.1f} ms")
        print(f"  Hermes first chunk: {report.hermes_first_chunk_ms:.1f} ms")
        print(f"  Likely bottleneck: {report.likely_bottleneck.replace('_', ' ')}")
