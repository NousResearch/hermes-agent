"""Knowledge Discovery source adapters.

Production source adapters for the EvidencePackEngine. Each adapter is a
self-contained provider primitive matching the engine source contract:

    provider(query, *, max_hits: int, observed_at: str) -> list[KnowledgeHitV2]

Adapters are stateless and free of side effects. They hold no filesystem,
network, thread, or process handles, and require no close(). They depend
only on injected read-only state accessors so the engine composition layer
can wire the SessionDB / state_meta backing without any adapter reaching
into hermes_cli or hermes_state internals.

The ``audit_sink`` adapter is the production seam between the engine's
in-memory audit emit and the process monitoring emitter. It accepts an
explicitly injected emitter (no singleton lookup, no hermes_cli /
gateway / tui_gateway import), borrows the reference for its lifetime,
delegates a single canonical-shape payload per accepted event, and
swallows every internal failure so an audit anomaly can never affect
the EvidencePack result.
"""

from agent.executive.knowledge_discovery.adapters.audit_sink import (
    ALLOWED_GATE_TYPE,
    ALLOWED_SEVERITY,
    EVENT,
    IDENTIFIER_FIELDS,
    OUTPUT_FIELD_ORDER,
    SCHEMA_VERSION,
    SOURCE,
    EvidencePackMonitoringAuditSink,
    MonitoringEmitterLike,
)

__all__ = [
    # audit sink
    "EvidencePackMonitoringAuditSink",
    "MonitoringEmitterLike",
    "ALLOWED_GATE_TYPE",
    "ALLOWED_SEVERITY",
    "EVENT",
    "SCHEMA_VERSION",
    "SOURCE",
    "OUTPUT_FIELD_ORDER",
    "IDENTIFIER_FIELDS",
]