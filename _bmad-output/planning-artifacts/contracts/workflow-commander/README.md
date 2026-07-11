# Workflow Commander Shared Contract Package

Status: Story 1.3a command, provider binding, and delivery status contracts are present.
Status: Story 1.3b workflow event envelope and callback rejection contracts are present.
Status: Story 1.3c Project Work Item identity, Phase Task identity, and materialization contracts are present.
This package contains the shared JSON schemas and examples that Archon producer stories and Hermes consumer stories use before runtime implementation begins.
Story 1.3a owns workflow command envelopes, Workflow Provider Binding payloads, and workflow delivery status results only.
Story 1.3c owns materialization, Project Work Item identity, and Phase Task identity examples only.
This package does not implement Hermes storage, Hermes materialization runtime, phase-task persistence, gates, workflow event receipts, reconciliation, or UI surfaces.

## Schema Version Values

- Command envelope examples use `workflow-command-envelope.v1`.
- Workflow Provider Binding examples use `workflow-provider-binding.v1`.
- Workflow delivery status examples use `workflow-delivery-status.v1`.
- Workflow event envelope examples use `workflow-event-envelope.v1`.
- Callback rejection examples use `workflow-callback-rejection.v1`.
- Project Work Item identity examples use `workflow-project-work-identity.v1`.
- Phase Task identity examples use `workflow-phase-task-identity.v1`.
- Materialization examples use `workflow-materialization-case.v1`.

## Canonical Files

```
schemas/workflow-command-envelope.schema.json
schemas/workflow-provider-binding.schema.json
schemas/workflow-delivery-status.schema.json
schemas/workflow-event-envelope.schema.json
schemas/project-work-identity.schema.json
schemas/phase-task-identity.schema.json
schemas/materialization-case.schema.json
examples/providers/archon/commands/start-success.json
examples/providers/archon/commands/status-success.json
examples/providers/archon/commands/approve-success.json
examples/providers/archon/commands/reject-success.json
examples/providers/archon/commands/resume-success.json
examples/providers/archon/commands/retry-success.json
examples/providers/archon/commands/cancel-success.json
examples/providers/archon/commands/binding-create-success.json
examples/providers/archon/commands/binding-status-success.json
examples/providers/archon/commands/binding-rotate-success.json
examples/providers/archon/commands/binding-disable-success.json
examples/providers/archon/commands/error-malformed-request.json
examples/providers/archon/commands/error-timeout.json
examples/providers/archon/commands/error-schema-mismatch.json
examples/providers/archon/commands/error-unexpected-exit.json
examples/providers/archon/commands/error-unexpected-state.json
examples/providers/archon/bindings/create-request.json
examples/providers/archon/bindings/update-request.json
examples/providers/archon/bindings/create-success.json
examples/providers/archon/bindings/update-success.json
examples/providers/archon/bindings/rotate-success.json
examples/providers/archon/bindings/disable-success.json
examples/providers/archon/bindings/remove-success.json
examples/providers/archon/bindings/status-valid.json
examples/providers/archon/bindings/status-missing.json
examples/providers/archon/bindings/status-stale.json
examples/providers/archon/bindings/status-disabled.json
examples/providers/archon/bindings/status-rotated.json
examples/providers/archon/bindings/status-conflicting.json
examples/providers/archon/bindings/error-malformed-request.json
examples/providers/archon/delivery/healthy.json
examples/providers/archon/delivery/delayed.json
examples/providers/archon/delivery/retrying.json
examples/providers/archon/delivery/failed.json
examples/providers/archon/delivery/duplicated.json
examples/providers/archon/delivery/terminal-failure.json
examples/providers/archon/delivery/reconciliation-pending.json
examples/workflow-events/workflow-completed.json
examples/workflow-events/workflow-completed-redelivery.json
examples/workflow-events/workflow-failed.json
examples/workflow-events/approval-requested.json
examples/workflow-events/delivery-failed.json
examples/workflow-events/artifact-event.json
examples/providers/archon/events/workflow-run-started.json
examples/providers/archon/events/workflow-completed.json
examples/providers/archon/events/workflow-completed-redelivery.json
examples/providers/archon/events/workflow-failed.json
examples/providers/archon/events/approval-requested.json
examples/providers/archon/events/delivery-failed.json
examples/providers/archon/events/artifact-event.json
examples/callback-rejections/bad-signature.json
examples/callback-rejections/stale-timestamp.json
examples/callback-rejections/duplicate-event-id.json
examples/callback-rejections/wrong-binding.json
examples/callback-rejections/unknown-project.json
examples/callback-rejections/schema-mismatch.json
examples/callback-rejections/wrong-profile-secret.json
examples/callback-rejections/wrong-codebase.json
examples/callback-rejections/unsupported-provider.json
examples/materialization/new-story.json
examples/materialization/unchanged-story.json
examples/materialization/changed-story.json
examples/materialization/missing-sprint-status.json
examples/materialization/malformed-sprint-status.json
examples/materialization/duplicate-phase-task-prevention.json
validate_contracts.py
```

## Command Envelope Rules

Command results are provider output from Archon to Hermes.
Every command envelope declares `schemaVersion`, `intendedProducer`, `intendedConsumer`, `owningSubproject`, `provider`, `command`, `correlationId`, `issuedAt`, and `success`.
Successful workflow commands include `workflowRunRef` when the command operates on a workflow run.
Successful binding commands include `bindingRef` when the command operates on provider-side binding state.
Failed command results include a machine-readable `error` with stable `code`, `category`, `retryable`, and structured `details`.
Hermes must not parse display text, stdout text, stderr text, or prose diagnostics to understand these examples.

## Workflow Provider Binding Rules

Workflow Provider Binding payloads use generic controller identity fields `provider` and `name`.
Binding payload examples do not use JSON object keys named `profile`, `agent_name`, `agent`, or `agent_provider`.
Request-shaped create and update payloads identify Hermes as producer and Archon as consumer.
Result-shaped, status-shaped, and error-shaped payloads identify Archon as producer and Hermes as consumer.
The owning subproject for persisted Workflow Provider Binding state remains `archon`.
Binding examples may use `${WORKFLOW_ENGINE_REPOSITORY_PATH}` as a portable placeholder for `projectRef.repositoryPath`.
That placeholder is contract fixture data, not an implicit validator environment variable.
Runtime implementations must resolve the path from their own project configuration and shared fixtures must not persist host-specific absolute paths such as `/Users/...` or `/home/...`.

## Delivery Status Rules

The canonical external delivery status values are `healthy`, `delayed`, `retrying`, `failed`, `duplicated`, `terminal-failure`, and `reconciliation-pending`.
The canonical value for reconciliation wait state is `reconciliation-pending`.
Older local planning names `waiting-for-reconciliation` and `waiting_for_reconciliation` are compatibility aliases that consumers may normalize to `reconciliation-pending`, but new examples must not emit those aliases.
Delivery status examples keep workflow execution independent from event delivery by setting `blockingWorkflowExecution` to `false`.

## Workflow Event Envelope Rules

Workflow event envelopes are provider callbacks from Archon to Hermes.
Every event envelope declares `schemaVersion`, `intendedProducer`, `intendedConsumer`, `owningSubproject`, `provider`, `eventId`, `eventType`, `occurredAt`, `bindingRef`, `workflowRunRef`, `projectRef`, `profileRoute`, `idempotencyKey`, `signature`, and `payload`.
The canonical event type values are `workflow.run.started`, `workflow.run.completed`, `workflow.run.failed`, `workflow.approval.requested`, `workflow.delivery.failed`, and `workflow.artifact.recorded`.
Event envelopes carry profile route metadata so Hermes can reject callbacks that arrive through the wrong profile, binding, provider, project, or codebase before mutating workflow state.
Event envelopes carry signature verification metadata through `signature` fields and `profileRoute.secretRef`, but examples must never contain raw shared secrets or raw signature header values.
Signature algorithm, signature header name, timestamp header name, canonicalization, and replay-window duration remain deferred shared security decisions.
Fixtures use deterministic placeholder strings such as `${WORKFLOW_EVENT_SIGNATURE_ALGORITHM}` instead of final policy values.
The generic `examples/workflow-events/` fixtures model shared event semantics, while `examples/providers/archon/events/` fixtures model Archon-specific provider output.
The `workflow-completed-redelivery.json` fixtures reuse the original completed event id and idempotency key with a later delivery attempt, proving redelivery stability without permitting duplicate side effects.
The `workflow.delivery.failed` examples set `deliveryOnly` and `mutationIntent` so consumers do not confuse delivery health with typed project-work mutation.

## Callback Rejection Rules

Callback rejection fixtures describe fail-closed Hermes behavior for invalid provider callbacks.
Each rejection fixture declares `diagnosticCategory` and `expectedMutationBehavior`.
The expected mutation behavior for these fixtures is no state mutation, no workflow run mutation, no binding mutation, no event ledger mutation, no Project Work Item mutation, no Phase Task mutation, no HILT Gate mutation, no workflow reference mutation, no comment mutation, no Story Status History mutation, and a diagnostic emission.
Required rejection reasons are `bad-signature`, `stale-timestamp`, `duplicate-event-id`, `wrong-binding`, `unknown-project`, `schema-mismatch`, `wrong-profile-secret`, `wrong-codebase`, and `unsupported-provider`.
The `schema-mismatch` fixture intentionally contains an embedded event envelope that fails the workflow event schema.
All other rejection fixtures keep the embedded event envelope structurally valid and rely on semantic rejection evidence.
The `stale-timestamp` fixture references the deferred replay-window decision instead of hard-coding a final clock-skew duration.

## Project Work Item Identity Rules

Project Work Item identity belongs to Hermes runtime project-work materialization.
It is derived from percent-encoded bound project cwd, BMAD artifact path, and BMAD epic or story identity components.
It must not include phase kind, provider run id, workflow run id, workflow event id, GitHub PR number, HILT gate state, current Kanban status, target BMAD status, verification result, or mutable title text.
Changed or unchanged source data for the same BMAD story must resolve to the same Project Work Item identity.
The canonical identity schema is `schemas/project-work-identity.schema.json`.
The canonical schema version is `workflow-project-work-identity.v1`.

## Phase Task Identity Rules

Phase Task identity belongs to Hermes runtime phase-task materialization.
It is derived from Project Work Item identity plus stable phase kind.
The v1 combined story workflow uses `story_implementation` as the phase kind.
The combined workflow name is recorded separately as `bmad-create-and-dev-story`.
The workflow name must not replace the stable phase kind in identity inputs.
Repeated materialization for the same Project Work Item and phase kind must reuse the same Phase Task identity.
The canonical identity schema is `schemas/phase-task-identity.schema.json`.
The canonical schema version is `workflow-phase-task-identity.v1`.

## Materialization Rules

Materialization fixtures are Hermes-to-Hermes behavior contracts for interpreting BMAD source artifacts into Project Work Items and Phase Tasks.
The canonical wrapper schema is `schemas/materialization-case.schema.json`.
The canonical schema version is `workflow-materialization-case.v1`.
`examples/materialization/new-story.json` creates one Project Work Item and one linked Phase Task.
`examples/materialization/unchanged-story.json` resolves the same Project Work Item identity and reuses the same Phase Task identity without creating duplicates.
`examples/materialization/changed-story.json` updates the same Project Work Item identity and reuses the same Phase Task identity when mutable source fields change.
`examples/materialization/missing-sprint-status.json` fails closed before Project Work Item or Phase Task mutation.
`examples/materialization/malformed-sprint-status.json` fails closed before Project Work Item or Phase Task mutation.
`examples/materialization/duplicate-phase-task-prevention.json` proves repeated story-level materialization resolves to one Phase Task identity and one persisted Phase Task.
Successful materialization examples include project binding reference, bound cwd placeholder, BMAD artifact path, BMAD story identity, Project Work Item identity, Phase Task identity, and expected mutation behavior.
Failure materialization examples include diagnostic category, recovery guidance, checked-before-mutation evidence, and explicit no-mutation expectations.
Fixtures use `${WORKFLOW_ENGINE_REPOSITORY_PATH}` as a portable placeholder and must not include host-specific absolute paths.
Downstream persistence stories for Project Binding, project-work, phase-task, gate, workflow event receipt, or reconciliation state must carry migration, uniqueness, and idempotency test expectations from these examples before they are marked implementation-ready.
Each materialization fixture carries concrete `migrationExpectation`, `uniquenessExpectation`, and `idempotencyExpectation` objects so downstream stories inherit record, identity-key, rerun-case, and required-test expectations instead of only boolean readiness flags.
If a downstream story depends on one of these shared examples and the relevant example or readiness expectation is missing, that downstream story is not implementation-ready.

## Validation

Run this command from the Archon subproject root:

```bash
python3 _bmad-output/planning-artifacts/contracts/workflow-commander/validate_contracts.py
```

The validator uses only Python standard-library JSON parsing and explicit invariant checks.
It checks strict UTF-8 JSON parseability with duplicate-key and non-standard constant rejection, schema and example conformance against the shipped JSON schemas, required metadata fields, command envelope success and error shape, binding vocabulary, forbidden binding keys, delivery status vocabulary, event envelope routing metadata, project identity consistency, per-event payload fields, redelivery idempotency stability, workspace-relative file URI targets, callback rejection semantics, Project Work Item identity inputs, percent-encoded Project Work Item identity key derivation, Phase Task identity derivation, materialization identity stability, duplicate Phase Task prevention, fail-closed materialization source and recovery safety, concrete downstream readiness expectations, required JSON files as regular non-empty files, no unexpected JSON examples outside the canonical fixture list, no raw secret or raw signature material in schemas or examples including nested array string values, no hard-coded deferred signature policy values, no host-specific absolute paths in schemas or examples including nested array string values and common local temp prefixes, no symlinks, and local package validation without parent workspace traversal.

## Readiness Rule

Archon producer stories that emit provider command, Workflow Provider Binding, delivery status, or workflow event payloads must validate against these fixtures.
Hermes consumer stories that parse provider command, Workflow Provider Binding, delivery status, workflow event, or callback rejection payloads must validate against the same fixtures.
Hermes materialization and persistence stories that create or update Project Binding, Project Work Item, Phase Task, gate, workflow event receipt, or reconciliation records must validate their migration, uniqueness, and idempotency expectations against the Story 1.3c identity and materialization examples.
Downstream stories must not claim runtime Hermes materialization, storage, phase-task, gate, or reconciliation implementation readiness from this package slice.
