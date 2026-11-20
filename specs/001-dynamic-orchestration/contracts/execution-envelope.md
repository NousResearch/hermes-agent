# Contract: ExecutionEnvelope v1

Status: PROPOSED.

`ExecutionEnvelopeV1{schema_version:"execution-envelope/v1",execution_id,task_id,attempt_id,decision_id,route_id,reservation_id?,effort:E0..E4,verification:V0..V4,workdir?,allowed_context_refs[],allowed_tools[],permissions[],timeout,budget,trace_context?,telemetry_consent_id?,requested_identity,resolved_identity?,effective_identity?,dispatch_state,result_evidence_refs[],normalized_error?,reconciliation_state}`.

Adapter may materialize only persisted TaskEnvelope/RouteDecision authorization. It cannot choose another route/recompute policy/review. It returns bounded requested/resolved/effective attestation, result/error and reservation reconciliation; unsupported controls are unattested.

V0 schema, V1 artifact, V2 independent validator, V3 independent high-capability review, V4 human/external authority. Same execution/route/constrained pool cannot satisfy required independence. For a fallback, dispatch is permitted only after `QualityCompensationPlanV1` enforcement has passed; verification evaluates its acceptance thresholds, evidence references, review independence and declared escalation.

## Telemetry activation gate
Telemetry is default-disabled. It may emit no event unless a persisted, user-facing consent record exists with `telemetry_consent_id`, explicit scope, activation timestamp and revocation state. Future activation MUST expose the config gate, setup prompt and `hermes tools` toggle required by repository policy; this phase does not claim any are implemented. Consent scope is limited to this orchestration event family and cannot authorize prompts, context, secrets, headers, credentials, raw tool/provider output or unbounded error bodies. After active consent only, events use a bounded schema of IDs, canonicalization/policy versions, normalized route metadata and bounded/sanitized reason codes. Telemetry failure cannot block dispatch, verification or reconciliation.

Legacy executes current behavior; shadow creates evidence without new dispatch; canary is feature-gated; rollback restores legacy and releases holds.
