# Responsibility execution

Status: implemented in the codebase; see [implementation map and validation limits](implementation.md).

## Decision

Preserve the established employee responsibility-execution behavior. This
includes file-defined triggers, run-context assembly, model-facing instructions,
handoff updates, reporting and completion/stop semantics. Local deployment is
not a reason to redesign that contract.

For scheduled work, retain responsibility ownership of declarations, fresh
charter/handoff context for new runs, scope and reporting instructions, guard
behavior, malformed-edit handling, one-shot completion and rearming, and
retirement when declarations or packages are removed or archived. Preserve the
reference behavior for finite work reaching its stated completion condition.

For webhook-driven work, preserve the applicable declaration, payload framing,
run/session and reporting contracts. Public reachability and persistence on a
local machine require a concrete implementation design; parity must not be
claimed merely because native webhook ingress exists.

## Native implementation

Use existing Hermes cron, delivery, file and gateway owners directly. Port
applicable employee logic into those owners or focused native modules, not a
hosted runtime with local adapters. Keep ordinary native jobs distinct from
file-owned responsibility jobs so CLI/UI edits cannot silently create a second
source of truth.

Adapt paths and removed-capability references according to the agreed
[layout](local-layout.md), [tool surface](tool-surface.md) and
[guides](guides.md). Preserve model-visible instructions and feedback otherwise.

Compare exact trigger and run-assembly paths during implementation. Local
sleep, shutdown, restart and missed-run behavior follow native Hermes, as
agreed in [the runtime lifecycle](local-layout.md#runtime-lifecycle). Adapt
reference instructions accordingly rather than promising hosted availability.
Webhook ingress uses the native Hermes server; public reachability is user
configuration, not a new tunnel-management feature.

## Verification when implemented

Exercise a real declaration through native scheduling and delivery, including
handoff updates, duplicate prevention after restart, malformed edits, completed
one-shots, finite completion, archive/restore and profile isolation. Compare the
actual model-bound run context and results with the reference contract. Test
webhook dispatch separately when its local ingress contract is implemented.

## Webhook mechanism

Decision: retain native ingress while preserving the employee webhook
mechanism. The employee creates and edits responsibility-owned YAML files under
`webhooks/` with ordinary file tools; it does not switch to a separate native
subscription command or model tool as its authoring interface.

Preserve applicable declaration validation, stable route identity, returned URL
and repair feedback, reconciliation on edit/delete/archive/restore, payload
framing, stream/buffering and deduplication behavior, and run/reporting semantics.
Implement missing behavior directly in native owners. Do not equate using the
native HTTP server with accepting weaker delivery or declaration semantics.

Expose usable webhook URLs only when the configured ingress supports them;
do not fabricate public reachability. Native availability during shutdown or
sleep still applies. No hosted webhook application or managed tunnel is added.
