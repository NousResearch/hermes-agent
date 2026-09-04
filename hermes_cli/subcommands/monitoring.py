"""``hermes monitoring`` subcommand parser.

Gateway monitoring control and inspection. ``status`` shows whether the
gateway health & diagnostics export is enabled, where it points, and the
redaction posture.

The handler is injected to avoid importing ``main`` (mirrors the insights
subcommand).
"""

from __future__ import annotations

from typing import Any, Callable

_JSON_METRIC_ATTRIBUTE_KEYS = frozenset({
    "service.instance.id",
    "service.version",
    "hermes.supervision_mode",
    "hermes.gateway.state",
    "hermes.platform",
    "hermes.platform.state",
    "hermes.error_code",
})


def _metric_payload(metric: Any) -> dict[str, Any]:
    """Project an existing content-free metric into the CLI JSON contract."""
    return {
        "name": str(metric.name),
        "value": metric.value,
        "attributes": {
            key: value
            for key, value in metric.attributes.items()
            if key in _JSON_METRIC_ATTRIBUTE_KEYS
        },
    }


def build_monitoring_status_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Build a machine-readable local health snapshot without reading logs."""
    from agent.monitoring import otlp_exporter

    mon_raw = config.get("monitoring") if isinstance(config, dict) else None
    mon: dict[str, Any] = mon_raw if isinstance(mon_raw, dict) else {}
    health_raw = mon.get("gateway_health_export")
    health_cfg: dict[str, Any] = health_raw if isinstance(health_raw, dict) else {}
    export_raw = mon.get("export")
    export_cfg: dict[str, Any] = export_raw if isinstance(export_raw, dict) else {}
    otlp_raw = export_cfg.get("otlp")
    otlp: dict[str, Any] = otlp_raw if isinstance(otlp_raw, dict) else {}
    health_enabled = bool(health_cfg.get("enabled"))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "content_free": True,
        "export": {
            "health_enabled": health_enabled,
            "metrics_enabled": health_enabled
            and bool(health_cfg.get("metrics_enabled", True)),
            "diagnostic_events_enabled": health_enabled
            and bool(
                health_cfg.get("diagnostic_events_enabled", True),
            ),
            "warning_error_events_enabled": health_enabled
            and bool(
                health_cfg.get("warning_error_events_enabled", True),
            ),
            "otlp_configured": bool(otlp.get("enabled") and otlp.get("endpoint")),
            "otlp_sdk_available": bool(otlp_exporter.is_available()),
        },
        "health": {
            "gateway": {"status": "unavailable", "metrics": []},
            "cron": {"status": "unavailable", "metrics": []},
        },
    }

    try:
        from agent.monitoring.gateway_health import build_gateway_health_snapshot
        from gateway.status import (
            get_running_pid,
            read_runtime_status,
            resolve_gateway_liveness,
        )
        from hermes_cli import __version__

        runtime = read_runtime_status() or {}
        liveness = resolve_gateway_liveness(
            runtime=runtime,
            use_cache=False,
            pid_probe=lambda: get_running_pid(cleanup_stale=False),
        )
        if not (liveness.probe_error and not liveness.running):
            snapshot = build_gateway_health_snapshot(
                runtime,
                gateway_running=bool(liveness.running),
                profile="default",
                install_id=str(mon.get("install_id") or "unknown"),
                version=str(__version__),
            )
            payload["health"]["gateway"] = {
                "status": "available",
                "metrics": [_metric_payload(metric) for metric in snapshot.metrics],
            }
    except Exception:
        payload["health"]["gateway"]["status"] = "error"

    try:
        from agent.monitoring.cron_health import build_cron_health_snapshot

        snapshot = build_cron_health_snapshot()
        payload["health"]["cron"] = {
            "status": "available",
            "metrics": [_metric_payload(metric) for metric in snapshot.metrics],
        }
    except Exception:
        payload["health"]["cron"]["status"] = "error"

    return payload


def build_monitoring_parser(subparsers, *, cmd_monitoring: Callable) -> None:
    """Attach the ``monitoring`` subcommand (with actions) to ``subparsers``."""
    p = subparsers.add_parser(
        "monitoring",
        help="Inspect gateway monitoring (health & diagnostics export)",
        description=(
            "Gateway monitoring: service health metrics plus redacted "
            "diagnostics, exported over OTLP to an operator-configured "
            "endpoint. Content-free by construction — no prompts, messages, "
            "tool args/results, or usage analytics. Configure under "
            "monitoring.* in config.yaml."
        ),
    )
    sub = p.add_subparsers(dest="monitoring_action")

    status = sub.add_parser(
        "status",
        help="Show monitoring settings, export state, and redaction posture",
    )
    status.add_argument(
        "--json",
        action="store_true",
        help="Print content-free local health and export status as JSON",
    )

    p.set_defaults(func=cmd_monitoring)
