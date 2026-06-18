# SPDX-License-Identifier: MIT
"""Tracer-provider bootstrap for the observability/otel plugin.

Two ways to get a tracer, in priority order:

1. **genai_otel** — if the ``genai-otel-instrument`` library is installed, use
   it. It is a reference OTel GenAI emitter and brings token/cost/latency
   conventions plus optional on-prem GPU/energy metrics for free. Preferred.

2. **plain OTel SDK** — fall back to a vanilla ``TracerProvider`` + OTLP HTTP
   exporter so the plugin still works when ``genai_otel`` is not present. This
   keeps the contribution's hard dependency surface minimal (OTel SDK only).

A stable ``service.name`` is the single most important resource attribute: any
OTel backend uses it to group the agent's telemetry. Default is
``agent.coding.hermes``; override via config or ``OTEL_SERVICE_NAME`` /
``OTEL_EXPORTER_OTLP_ENDPOINT`` env vars.

This module performs no filesystem I/O — it exports over the network only — so
``get_hermes_home()`` from ``hermes_constants`` is intentionally not used here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "agent.coding.hermes"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"
INSTRUMENTATION_SCOPE = "hermes.observability.otel"


def build_tracer(
    *,
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    user_id: str | None = None,
    capture_content: bool = False,
) -> Any:
    """Return an OpenTelemetry ``Tracer`` for the plugin.

    Resolves ``service_name`` / ``otlp_endpoint`` from arguments first, then
    standard OTel env vars, then sane on-prem defaults. Never raises on
    transport problems — observability must not crash the agent; on failure we
    log and return a no-op tracer.
    """
    service_name = (
        service_name
        or os.environ.get("OTEL_SERVICE_NAME")
        or DEFAULT_SERVICE_NAME
    )
    otlp_endpoint = (
        otlp_endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or DEFAULT_OTLP_ENDPOINT
    )

    tracer = _try_genai_otel(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        user_id=user_id,
        capture_content=capture_content,
    )
    if tracer is not None:
        return tracer
    return _build_plain_otel(service_name=service_name, otlp_endpoint=otlp_endpoint)


def _add_resource_attr(key: str, value: str) -> None:
    """Append ``key=value`` to OTEL_RESOURCE_ATTRIBUTES if not already present.

    Resource attributes are read by the OTel SDK at provider-init time, so this
    must run before the tracer provider is built. Avoids depending on a
    library-specific kwarg for identity attributes like ``user.id``.
    """
    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    if f"{key}=" in existing:
        return
    pair = f"{key}={value}"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
        f"{existing},{pair}" if existing else pair
    )


def _try_genai_otel(
    *,
    service_name: str,
    otlp_endpoint: str,
    user_id: str | None,
    capture_content: bool,
) -> Any | None:
    """Preferred path: instrument via genai-otel-instrument if available."""
    try:
        import genai_otel  # type: ignore[import-not-found]
    except ImportError:
        logger.info(
            "genai-otel-instrument not installed; using plain OTel SDK path"
        )
        return None

    try:
        # genai_otel.instrument() configures the global tracer provider + OTLP
        # exporter with the GenAI conventions. ``user_id`` is passed via
        # OTEL_RESOURCE_ATTRIBUTES rather than a kwarg, so it is not forwarded
        # here. We keep the call to documented-stable keys only.
        instrument_kwargs: dict[str, Any] = {
            "service_name": service_name,
            "endpoint": otlp_endpoint,
            "enable_content_capture": capture_content,
        }
        if user_id:
            # Best-effort: surface the user on every span's resource without
            # depending on a kwarg the library may not accept.
            _add_resource_attr("user.id", user_id)
        genai_otel.instrument(**instrument_kwargs)
        from opentelemetry import trace

        return trace.get_tracer(INSTRUMENTATION_SCOPE)
    except Exception:
        logger.warning(
            "genai-otel-instrument present but instrument() failed; "
            "falling back to plain OTel SDK",
            exc_info=True,
        )
        return None


def _build_plain_otel(*, service_name: str, otlp_endpoint: str) -> Any:
    """Fallback path: vanilla OTel SDK + OTLP HTTP exporter."""
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": service_name,
                "gen_ai.system": "hermes",
            }
        )
        provider = TracerProvider(resource=resource)
        # OTLP HTTP traces endpoint convention is <endpoint>/v1/traces.
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint.rstrip("/") + "/v1/traces"
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        # Only set the global provider if one is not already configured, to
        # avoid clobbering a host application's tracing setup.
        current = trace.get_tracer_provider()
        if not isinstance(current, TracerProvider):
            trace.set_tracer_provider(provider)
            active = provider
        else:
            active = current
        logger.info(
            "OTel SDK tracer configured: service.name=%s endpoint=%s",
            service_name,
            otlp_endpoint,
        )
        return active.get_tracer(INSTRUMENTATION_SCOPE)
    except Exception:
        logger.error(
            "failed to configure OTel SDK tracer; observability disabled",
            exc_info=True,
        )
        return _noop_tracer()


def _noop_tracer() -> Any:
    """Return a tracer from the API's default no-op provider.

    Guarantees the plugin never crashes the agent even if the SDK is missing.
    """
    from opentelemetry import trace

    return trace.get_tracer(INSTRUMENTATION_SCOPE)


# ---------------------------------------------------------------------------
# OTel *logs* pipeline (separate from the spans pipeline above).
#
# The TraceVerse agent-coding dashboard is built from OTel **log records** (the
# `claude_code.*` event model), not spans. Claude Code / Codex / Copilot emit
# log events, so they populate the dashboard; a spans-only agent does not.
# build_logger() gives the plugin a LoggerProvider + OTLP **log** exporter so it
# can ALSO emit dashboard-shaped `agent.coding.*` log records alongside its
# GenAI spans — making Hermes a first-class citizen of the `/agent-coding` view.
#
# Identity (session.id / user.id / organization.id / team.id) is placed on the
# OTel **resource** so it matches the `resource.*` shape the dashboard aggregates
# on (by_user / by_org leaderboards, session cardinality).
# ---------------------------------------------------------------------------


def _resource_attrs_from_env(service_name: str, user_id: str | None) -> dict:
    """Build the resource attribute dict for the logs pipeline.

    Reads OTEL_RESOURCE_ATTRIBUTES (k=v,k=v) for team.id / organization.id /
    deployment.environment / user.id, then overlays explicit args. The same
    identity the spans resource carries, surfaced as proper resource keys so the
    dashboard's resource.* aggregations work without relying on the ingest
    pipeline's attribute->resource promotion.
    """
    attrs: dict[str, Any] = {
        "service.name": service_name,
        "gen_ai.system": "hermes",
    }
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                attrs[k] = v
    if user_id:
        attrs["user.id"] = user_id
    return attrs


def _flush_logger_provider(provider: Any) -> None:
    """Best-effort flush+shutdown of the logs provider at interpreter exit."""
    try:
        provider.force_flush()
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        provider.shutdown()
    except Exception:  # pragma: no cover - defensive
        pass


def build_logger(
    *,
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    user_id: str | None = None,
) -> Any:
    """Return an OpenTelemetry ``Logger`` for emitting dashboard log records.

    Never raises on transport/SDK problems — observability must not crash the
    agent; on failure we log and return ``None`` (callers treat ``None`` as
    "logs disabled" and simply skip log emission, keeping spans working).
    """
    service_name = (
        service_name
        or os.environ.get("OTEL_SERVICE_NAME")
        or DEFAULT_SERVICE_NAME
    )
    otlp_endpoint = (
        otlp_endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or DEFAULT_OTLP_ENDPOINT
    )
    try:
        from opentelemetry._logs import get_logger, set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create(_resource_attrs_from_env(service_name, user_id))
        provider = LoggerProvider(resource=resource)
        # OTLP HTTP logs endpoint convention is <endpoint>/v1/logs.
        exporter = OTLPLogExporter(endpoint=otlp_endpoint.rstrip("/") + "/v1/logs")
        provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        # Flush on interpreter exit. The span pipeline is flushed by Hermes'
        # shutdown path, but this LoggerProvider is ours alone — without an
        # explicit flush, late-turn records (api_response / tool_result emitted
        # just before the CLI exits) can be dropped by the batch processor.
        import atexit

        atexit.register(_flush_logger_provider, provider)
        # Only adopt as the global provider if none is set, mirroring the tracer
        # path — don't clobber a host application's logging setup.
        try:
            set_logger_provider(provider)
        except Exception:  # pragma: no cover - provider already set
            pass
        logger.info(
            "OTel SDK logger configured: service.name=%s endpoint=%s",
            service_name,
            otlp_endpoint,
        )
        return get_logger(INSTRUMENTATION_SCOPE, logger_provider=provider)
    except Exception:
        logger.warning(
            "failed to configure OTel SDK logger; dashboard log records disabled "
            "(spans still emit)",
            exc_info=True,
        )
        return None
