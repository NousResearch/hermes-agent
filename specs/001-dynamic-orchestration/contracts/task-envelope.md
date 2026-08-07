# Contract: TaskEnvelope v1

Status: PROPOSED.

## Shape
`{schema_version, task_id, root_task_id?, objective, deliverables[], capabilities_required[], tools_allowed[], permissions_required[], context{classification,max_tokens?,allowed_sources[]}, privacy{classification,outbound_allowed,retention}, risk{level,reversibility,impact}, effort:E0..E4, budget{currency,soft_cap?,hard_cap?,paid_allowed}, verification{minimum:V0..V4,independent_required,human_gate_required}, policy_version, audited_model_justification?}`

## Invariants
- Capability/tool/permission identifiers are explicit/normalized; unknown required item fails closed.
- E0 has no model route. E1–E4 specify work intensity, not model/provider identity.
- Any model hint needs audited policy/version/reason/evidence/author/expiry; otherwise it is rejected.
- Context/privacy/budget/review fields feed deterministic policy, not prompt advice.
- Envelope excludes prompts/secrets/raw private payloads and is immutable after decision creation.

## Acceptance
A private browser task with paid cap and V3 review is representable without naming any model/provider; policy chooses an eligible route.
