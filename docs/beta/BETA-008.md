# BETA-008 — End-to-end orchestration

Beta now uses the existing `delegate_task` entrypoint as a thin orchestration
boundary. A top-level goal in `agent.mode: beta` is routed through the
specialist registry, delegated as one parallel read-only batch, validated,
consolidated, and returned to the parent agent. Hermes mode keeps the original
`delegate_task` path unchanged.

The deterministic acceptance scenario diagnoses a slow PostgreSQL request
with DBA, infrastructure, and monitoring evidence. A dependent QA review runs
when findings conflict or a recommendation is high risk. High-risk
recommendations produce an exact approval request; the orchestrator records no
impactful execution itself.

Specialist timeouts remain visible as partial failures while valid sibling
evidence is preserved.
