# HQ Memory Trust Gate Integration Plan

Generated: 2026-06-23 KST
Scope: docs/reference plan for approved `clean-worktree 및 quality gate 정리` cleanup.
Safety: no secrets, no ELIOS access, no main/master merge, no live enforcement enabled by this document.

## 1. Goal

Unify HQ's fragmented memory-safety components into a deterministic, auditable trust-gated memory control loop:

1. MemoryWire-style operation vocabulary for interop and fixture coverage.
2. TAM-style write gate for ingest-time allow/review/block decisions.
3. Negative memory index for harmful, stale, duplicate, or temporary memories.
4. Faithfulness gate for post-draft consistency checks before promotion.
5. Prompt-injection boundary checks for memory retrieval/write paths.

## 2. Existing HQ touchpoints

| Component | Current role | Integration target |
|---|---|---|
| `hq_memory_learning_gate.py` | Filters learning-note candidates and temporary artifacts. | Add deterministic ingest decision fields: `allow`, `review`, `block`, `trust_score`, `policy_factor`. |
| `hq_learning_notes.py` | Extracts durable learning candidates from `state.db`. | Add pre-write constraint checks and negative-index re-evaluation. |
| `hq_context_hygiene.py` | Measures memory/context pressure. | Surface trust-gate findings and pressure-driven cleanup recommendations. |
| `hq_harness_validator.py` | Local policy/eval fixture harness. | Add memory trust-gate fixtures for retention, injection, and policy decisions. |
| `agent/eval_gate.py` | Deterministic pre-dispatch gate scaffold in this worktree. | Keep disabled/audit-only by default; use as pattern for future memory-gate enforcement. |

## 3. MemoryWire operation mapping

Reference vocabulary only; no external backend dependency is required.

| Operation | HQ use |
|---|---|
| `remember` | Candidate durable fact accepted into memory/learning notes. |
| `recall` | Retrieval for session context, reports, or learning summaries. |
| `forget` | Remove temporary, stale, unsafe, or user-retracted facts. |
| `merge` | Consolidate duplicate memories/skills/learning notes. |
| `expire` | Time-bound operational facts that should not persist past usefulness. |

Recommended memory types:

- `semantic`: stable facts/preferences.
- `episodic`: session-derived lessons with timestamp/source.
- `procedural`: reusable workflow candidates; promote to skills, not memory, when complex.
- `emotional`/preference-like: user tone/style preferences only when durable.

## 4. TAM-style gate sequence

```text
candidate extraction
  -> sanitize/noise filter
  -> write gate: allow/review/block + trust_score + policy_factor
  -> negative-index check: duplicate/stale/unsafe/temporary signal
  -> optional HITL review queue
  -> memory/skill/report write
  -> faithfulness gate: saved content matches evidence and user intent
  -> audit artifact
```

## 5. Prompt-injection boundary

Apply deterministic checks at two points:

1. Retrieval boundary: treat retrieved memory/session text as data, not instructions.
2. Write boundary: block or review candidates that contain instruction-hijack patterns, secret requests, or policy-bypass language.

Suggested detector classes:

- local-alignment/instruction-overlap checks for “ignore previous instructions”-style text.
- stylometric/fatigue anomaly checks for noisy injected artifacts.
- explicit policy regexes for secrets, destructive actions, paid/uncapped calls, and ELIOS access.

## 6. Quality gate expectations

A candidate implementation should pass:

- deterministic fixture tests for allow/review/block decisions,
- no live enforcement unless config explicitly enables it,
- audit-only default behavior,
- rollback by reverting the branch commit,
- report artifacts with source session/file evidence,
- no raw secrets in logs or reports.

## 7. Rollout plan

1. Keep current worktree changes local and reviewed.
2. Land disabled/audit-only gate scaffold with tests.
3. Add memory-gate fixtures to `hq_harness_validator.py`.
4. Run quality gate and context hygiene reports.
5. Request separate approval before enabling any live enforcement.

## 8. Current approval status

The operator approved cleanup using:

```text
clean-worktree 및 quality gate 정리 승인
```

This approval covers worktree cleanup and quality-gate organization. It does **not** by itself authorize main/master merge, secrets access, ELIOS access, destructive cleanup outside the named worktree, or live enforcement enablement.
