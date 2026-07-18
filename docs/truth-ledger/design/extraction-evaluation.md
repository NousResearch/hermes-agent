# Truth Ledger admission and extraction evaluation design

Date: 2026-07-17
Task: t_8556d347
Plan anchor: `/Users/hermes/.hermes/hermes-agent/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md` (T4)

## 1) Purpose and non-goals

This document freezes the admission policy and evaluation protocol for Truth Ledger extraction before implementation work begins.

Goals:
- Maximize precision for durable, atomic facts.
- Require abstention when evidence is weak or non-durable.
- Enforce zero secret leakage into candidates, ledger events, views, logs, and error artifacts.
- Keep extraction outputs reviewable and deterministic enough for repeatable scoring.

Non-goals:
- Maximizing recall at the expense of precision.
- Automatic promotion into curated memory or GBrain.
- Persisting raw transcripts or conversation history.
- Using inferred identity labels as canonical speaker identity.

## 2) Frozen vocabulary

### 2.1 Fact kinds (small controlled set)

Allowed kinds:
- `preference`
- `identity`
- `decision`
- `constraint`
- `environment`
- `workflow`
- `project`
- `relationship`
- `deadline`
- `correction`
- `lesson`
- `commitment`

Canonical compact enum form:
`preference identity decision constraint environment workflow project relationship deadline correction lesson commitment`

No additional kinds are valid for admission in this phase.

### 2.2 Scope values

Allowed scopes:
- `user`
- `agent`
- `project`
- `world`

### 2.3 Evidence types

Allowed evidence types:
- `user_stated`
- `tool_verified`
- `joint_decision`
- `assistant_inferred`

Policy freeze:
- `assistant_inferred` candidates are not admitted to active ledger state in this phase.
- `assistant_inferred` can be emitted only to review/quarantine streams if needed for later human review.

### 2.4 Operations and admission status

Ledger operations:
- `assert`
- `confirm`
- `supersede`
- `retract`

Extraction-only non-write status:
- `NONE` (explicit abstention / no admissible fact)

`NONE` is not a ledger operation and must not produce a ledger event.

## 3) Atomicity and logical identity contract

A candidate is admissible only if it is atomic:
- One independently searchable claim per candidate.
- No conjunctions that hide multiple claims in one record.
- Stable key granularity: one `scope + subject + key` identity per candidate.

Logical identity:
- Entity identity = `scope + subject + key`.
- Value/version identity = immutable `fact_id`.
- Event identity = immutable `event_id` per append operation.

Practical rule:
- If a statement contains two different durable claims, split into two candidates or return `NONE` until it can be split safely.

## 4) Source and status semantics

### 4.1 Source-of-truth hierarchy

Candidate strength order:
1. `tool_verified`
2. `user_stated`
3. `joint_decision`
4. `assistant_inferred` (review-only; not active)

When evidence types conflict in one turn, stronger evidence dominates for the same logical identity.

### 4.2 Status semantics by operation

- `assert`: first admissible version for a logical identity.
- `confirm`: re-observation of an already active fact with no value change.
- `supersede`: new fact version replacing prior active version for same logical identity.
- `retract`: invalidates a prior fact from active view without deleting historical events.
- `NONE`: extractor abstains; no event.

### 4.3 Contradictions and correction handling

Contradiction pattern:
- New value conflicts with current active value for same logical identity.
- Default resolution: `supersede` (not `assert`) if current fact exists and new claim passes admission gates.

Correction pattern:
- If user/tool explicitly marks earlier statement as wrong ("I was wrong", "that was incorrect", tool disproves), produce either:
  - `supersede` with corrected value, or
  - `retract` when corrected value is unknown or should be removed.

Ambiguous contradiction:
- If timing/source identity cannot be disambiguated safely, emit `NONE` and route for review.

## 5) Admission policy (hard gates)

A candidate is admitted only when all gates pass:

1. Durability: likely useful beyond current turn.
2. Atomicity: one claim, one logical identity.
3. Evidence: allowed evidence type with sufficient support.
4. Scope/kind validity: in frozen vocab.
5. Identity safety: stable speaker attribution available for user-scoped facts.
6. Sensitivity safety: no credentials/disallowed sensitive payloads.
7. Duplicate control: not semantic duplicate unless operation is `confirm`.
8. Non-noise class: not world trivia, assistant promise, chatter, or ephemeral progress update.

Failing any hard gate => `NONE`.

## 6) Do-not-remember and redaction policy

### 6.1 Do-not-remember semantics

If user intent indicates memory refusal (examples: "don’t remember this", "off the record", "forget that"), extraction must default to `NONE` for that content scope unless required for immediate safety/compliance handling.

If a previously admitted fact is explicitly withdrawn with do-not-remember intent:
- emit `retract` for affected logical identity,
- do not create replacement value unless the user provides a new durable permitted value.

### 6.2 Sensitive-content policy

Sensitive detections include (non-exhaustive):
- API keys/tokens/secrets
- passwords/passphrases
- OAuth/session cookies
- private keys/PEM blocks
- connection strings with secrets
- credential-like high-entropy strings tied to access contexts

Policy:
- Default action is reject (`NONE`) for candidates containing sensitive material.
- Do not partially preserve secret-bearing values.
- Preserve only minimal non-sensitive structural metadata when needed for diagnostics.
- Secret leakage tolerance for evaluation is absolute zero.

## 7) Evidence minimization and provenance constraints

Store minimum provenance needed for audit/replay only:
- `profile`, `platform`, `session_id`, `turn_id`, `task_id`
- stable speaker/channel/thread identifiers when available
- extractor schema/model provenance

Do not persist:
- raw `conversation_history`
- full source text excerpts
- chain-of-thought / hidden reasoning
- wholesale tool output dumps

If required provenance fields for safe attribution are missing (especially speaker identity for `user` scope), candidate may be review-only at most and is not admitted to active projection.

## 8) Abstention rubric (`NONE` expected)

Abstention is preferred over speculative admission. `NONE` is the expected outcome for many turns.

Return `NONE` when any condition is true:
- No durable user/tool/joint fact appears.
- Statement is temporary execution chatter.
- Claim is bundled and cannot be split safely.
- Identity attribution is unstable/unknown.
- Sensitive content appears in the candidate.
- Evidence is inferred-only for active admission.
- Duplicate/no-op with no meaningful confirmation need.

No-fact turns are a first-class class in evaluation and must score >=95% correctness.

## 9) Evaluation design

### 9.1 Dataset protocol (sanitized >=100-turn fixture set)

Minimum fixture corpus:
- At least 100 labeled turns.
- Sanitized synthetic/representative data only.
- No private transcripts, secrets, or personal raw conversation dumps.

Recommended class balance (minimum):
- 35 no-fact turns (expected `NONE`)
- 20 straightforward admissible facts (`assert`)
- 10 confirmations (`confirm`)
- 15 contradictions/corrections (`supersede` or `retract`)
- 10 identity-ambiguous/missing-speaker turns (expected abstention/review-only)
- 10 sensitive-content turns (expected reject with zero leakage)

Each fixture record must include:
- `fixture_id`
- sanitized turn text (or abstracted turn payload)
- expected class (`NONE` or operation)
- expected structured fields (`scope`, `kind`, `key`, optional value constraints)
- leakage expectation (always false)
- notes for adjudication

### 9.2 Metrics

Primary gate metrics:
- Precision (admitted facts): >=95%
- No-fact abstention correctness: >=95%
- Secret leakage rate: 0%

Secondary metrics (reporting, non-overriding):
- Recall (admissible truths found)
- Duplicate rate
- Correction/contradiction resolution accuracy
- Operation confusion matrix (`assert/confirm/supersede/retract/NONE`)

Policy freeze:
- Recall cannot override precision gate.
- Failing leakage gate is automatic fail regardless of other metrics.
- No automatic promotion regardless of metric success.

### 9.3 Scoring method

For each fixture:
1. Run deterministic pre-gates + extraction schema pass.
2. Compare predicted class to expected class.
3. For admitted classes, compare required structured fields.
4. Run leakage scanner over candidate payloads and artifacts.
5. Accumulate confusion matrix + gate metrics.

Pass/fail:
- PASS only if all primary metrics meet targets and leakage is zero.
- Otherwise REQUEST_CHANGES with failure buckets and remediation tests.

## 10) Examples and counterexamples (sanitized)

Admit (`assert`):
- "Use concise responses by default."
  - scope=`user`, kind=`preference`, key=`response.style`, value=`concise`

Admit (`confirm`):
- "Still prefer concise responses." where same logical identity already active.

Admit (`supersede`):
- "Actually, use detailed responses for engineering tasks." replacing prior concise default for that key.

Admit (`retract`):
- "Ignore my earlier timezone preference; don't store it." with no replacement value.

Abstain (`NONE`):
- "Running tests now..."
- "I think maybe this could work" (speculation without durable commitment)
- "Remember this token sk_live_xxx" (sensitive)
- Group chat line with unknown speaker identity for user-scoped preference

## 11) Traceability to acceptance requirements

- Narrow policy with frozen vocabulary: Sections 2, 5
- Atomicity/source/status semantics: Sections 3, 4
- assert/confirm/supersede/retract/NONE semantics: Sections 2.4, 4.2, 8
- Abstention and no-fact correctness target: Sections 8, 9
- Corrections/contradictions: Section 4.3
- Do-not-remember + redaction: Section 6
- Scope + evidence minimization: Sections 2.2, 7
- Sanitized >=100-turn protocol: Section 9.1
- Acceptance gates (precision, abstention, zero leakage, no auto-promotion): Section 9.2

## 12) Implementation guardrails for downstream tasks

- Keep deterministic admission/redaction gates before any model-dependent path.
- Treat `NONE` as success for no-fact turns.
- Reject rather than guess on missing attribution metadata.
- Preserve append-only event history; never hard-delete as part of admission/evaluation flow.
- Maintain review-gated boundary for any future promotion into curated memory/GBrain.
