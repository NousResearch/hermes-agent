"""OTLP exporter and tracer provider setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import OTelConfig


@dataclass
class OTelRuntime:
    provider: Any
    tracer: Any


def create_runtime(config: OTelConfig, *, span_exporter: Any = None) -> OTelRuntime:
    """Create an isolated provider without changing OTel's global provider."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = span_exporter or _create_otlp_exporter(config)
    provider = TracerProvider(
        resource=Resource.create({"service.name": config.service_name})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return OTelRuntime(
        provider=provider,
        tracer=provider.get_tracer("hermes-agent.observability.otel"),
    )


def _create_otlp_exporter(config: OTelConfig) -> Any:
    kwargs: dict[str, Any] = {}
    if config.endpoint:
        kwargs["endpoint"] = config.endpoint
    if config.headers:
        kwargs["headers"] = config.headers

    if config.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    return OTLPSpanExporter(**kwargs)
