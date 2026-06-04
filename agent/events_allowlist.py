# GENERATED — do not edit
# Source: packages/hermes-events/src/allowlist.ts
# Regenerate: npm run sync:events-allowlist  (or python scripts/sync_events_allowlist.py)

EVENT_ALLOWLIST = [
    "hermes.identity.bootstrap",
    "hermes.context.assembled",
    "hermes.interp.done",
    "hermes.intent.classified",
    "hermes.mission.compiled",
    "hermes.route.dispatched",
    "hermes.response.shaped",
    "hermes.summary.emitted",
    "mission.submitted",
    "specialist.dispatch.started",
    "specialist.dispatch.completed",
    "specialist.dispatch.failed",
    "specialist.degraded",
    "write.payload.created",
    "write.payload.updated",
    "write.payload.published",
    "write.medusa.created",
    "write.medusa.updated",
    "write.twenty.created",
    "write.twenty.updated",
    "approval.requested",
    "approval.granted",
    "approval.denied",
    "approval.timeout",
    "baumbad.research.completed",
    "baumbad.editorial-plan.produced",
    "baumbad.briefing.approved",
    "baumbad.draft.written",
    "baumbad.qa.decision",
    "baumbad.handoff.packaged",
]

def is_allowed_event(event_name: str) -> bool:
    return event_name in EVENT_ALLOWLIST
