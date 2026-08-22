# Canonical Memory Continuity Pattern (CLOE_CANONICAL_DECISION_CONTINUITY_AND_OPERATIONAL_LEARNING_MEMORY_V1)

Established 2026-08-10 (PR #754, merged b6b6abb). Governed structured-memory + authority-resolution
for cross-session decision continuity and reusable operational learning. Reuses the EXISTING memory
components (memory-events store, campaign-memory-schema, cloe-memory-append, campaign-memory-reader,
decision-records, cognitive-context-engine) — does NOT build another memory database.

## Module map (new files, lah-openclaw-mvp/src)

| Module | Role | Phases |
|--------|------|--------|
| `services/canonical-memory-classes.js` | Enums: AUTHORITY_CLASSES (CURRENT_STATE_REFERENCE, OPERATOR_DECISION, VALIDATED_OPERATING_RULE, CAMPAIGN_LEARNING, OBSERVATION, HYPOTHESIS), SOURCE_AUTHORITIES (9), VALIDATION_STATES, RESOLUTION_STATES; subject/scope builders; scopeMatches; TYPE_TO_AUTHORITY | 1,2,10 |
| `brain/canonical-memory-resolver.js` | PURE deterministic resolver: resolveCanonicalContinuity → bounded continuity object | 3,4,10 |
| `brain/canonical-continuity-retriever.js` | readCanonicalMemoryRecords, retrieveCanonicalContinuity, retrieveCampaignPlaybook (field-contract evolution) | 3,6,8,12 |
| `services/canonical-memory-write-seam.js` | Governed production write: writeCanonicalMemoryRecord (fail-closed gate) | 13 |
| `services/canonical-learning-promotion.js` | classifyLearningCandidate + promoteValidatedLearning (progressive learning) | 5,9 |
| `brain/canonical-session-continuity.js` | persistSessionContinuity / retrieveSessionContinuity (rolling continuity, no raw transcript) | 7 |
| `brain/canonical-memory-observability.js` | getMemoryObservability (read-only diagnostics) | 16 |

## Injection points (minimal, existing surfaces)

| File | Change | Phase |
|------|--------|-------|
| `brain/cognitive-context-engine.js` | `canonical_continuity` collector in COLLECTOR_REGISTRY; force-include when session object present OR campaign_memory/memory collectors selected | 6 |
| `services/readonly-operator-cli-client.js` | After evidence dossier (~L1365): push `canonical_continuity_v1` available_item (non-blocking, dynamic import) | 6 |
| `services/cloe-canonical-business-context.js` | Before generateGovernedMicrotestProposal (~L308): retrieveCampaignPlaybook → push `campaign_playbook_memory_v1` item (non-blocking, fail-open) | 12,8 |

## Resolver semantics (proven decisions — do not "simplify" these)

- **SINGULAR vs CUMULATIVE split (critical).** OPERATOR_DECISION + CURRENT_STATE_REFERENCE are
  SINGULAR: one canonical per subject (selectCanonicalPerSubject picks highest-authority + newest).
  VALIDATED_OPERATING_RULE + CAMPAIGN_LEARNING are CUMULATIVE: ALL effective rules apply. A naive
  per-subject top-1 selection silently drops valid rules (test D caught this).
- **Scope filter is conditional:** scopeMatches(r, q) — empty record scope → generic (applies
  everywhere); empty QUERY scope → true (subject filter is the primary precision axis). Only a
  caller-provided query scope restricts. Without this, subject-exact queries wrongly exclude scoped
  records (test F caught it).
- **Conflict classification trap:** a default/absent resolution_state (null/'unresolved') is NOT a
  conflict. Only explicit `conflict_state='conflicted'` or `resolution_state='conflicted'` counts.
  The write seam normalizes resolution_state to null by default; if the resolver treats 'unresolved'
  as conflict, every written record is classified conflicted and never effective (write-seam tests
  caught this).
- **Newer low-authority NEVER overrides older operator decision** — observation/hypothesis go to
  historical_evidence, not effective.
- **Supersession:** explicit supersedes_record_id chains + explicit resolution_state=superseded/
  historical; superseded excluded from effective, retained in `superseded` for audit. Stale records
  (expires_at passed OR data_freshness=stale) also demoted to superseded with stale flag.
- **Staleness:** only expires_at/TTL or explicit stale freshness signal demotes. Do NOT auto-stale
  provider-scoped values without TTL (would invent expiration).

## Write seam gating (double flag — the trap)

`writeCanonicalMemoryRecord` gate: `mode='write'` AND `env.CLOE_CANONICAL_MEMORY_WRITE==='true'`.
BUT appendMemoryFact internally ALSO requires `MEMORY_APPEND_WRITE=true`. When the canonical gate is
open you MUST force `appendEnv = { ...env, MEMORY_APPEND_WRITE: 'true' }` (same pattern as the
evidence route) — otherwise receipts return `read_only_dryrun` / `write_activated:false` despite
the gate being open. Also: call `normalizeCampaignMemoryMetadata(metadata, { requireCampaignId: false })`
— canonical records are not campaign-bound; the default requireCampaignId=true rejects them.

## Learning promotion (progressive, governed)

- AUTO_PROMOTE only for: deterministic authoritative events (CAMPAIGN_CREATED, APPROVAL_GRANTED/
  REJECTED, CAMPAIGN_RECONCILED, CONVERSION/REVENUE_OBSERVED), operator explicit approval of a named
  decision, operator explicit correction (carries supersedes_record_id), provider readback,
  validated outcome pattern (>=3 comparable campaigns AND validated).
- OPERATOR_CONFIRM_REQUIRED (candidate) for ambiguous rules and 2+ comparable patterns — never
  silently persisted. OBSERVATION_ONLY for single-campaign outcome (comparable_count=1).
- Progressive ladder: 1 campaign → OBSERVATION; N comparable → candidate pattern; validated → rule.

## Pitfalls (session-verified)

- **TDZ shadowing:** `const recordSubject = recordSubject(record)` inside the resolver throws
  "Cannot access 'recordSubject' before initialization" — the local const shadows the helper
  function. Rename the local (`recordSubjectOf`) or the helper.
- **Stash-based pre-existing-failure proof:** `git stash push -- <modified files>` → run tests →
  `git stash pop`. Identical failures with/without the delta = pre-existing on main. Proved
  cloe-affiliate-lot8, brain-ask-route ×2, brain-ask-route-grounded, readonly-operator-cli-client
  hang were all pre-existing, zero new regressions.
- **Store pollution by tests:** `data/memory-events/` is NOT gitignored (only `data/export/` is).
  Existing tests (campaign-memory.test.js) write there when MEMORY_APPEND_WRITE=true is set. Clean
  `data/memory-events/*.json` before commit; it is a runtime store, never commit it.
- **Cognitive-context-engine collector must be SYNC** (buildCognitiveContextPack calls collectors
  synchronously) — readCanonicalMemoryRecords + resolveCanonicalContinuity are sync; do not make
  the collector async.
- **Behavioral simulation receipt written OUTSIDE the repo** (e.g. `~/cloe-self-audit-evidence/
  <MISSION>/`) + validated from repo root: VALID first pass (27 scenarios). Keeps the worktree clean
  for push; validator resolves diff paths from cwd, not receipt location. Dynamic changed_files via
  `git diff --name-only <before>..HEAD` — never hardcode.
- **Router multi-receipt reconciliation (Gate 0):** a mission mentioning "memory" in prose can
  alias-hit cartelogic-v2 even when the mission implements memory INSIDE openclaw-runtime. The
  mission-name line `CLOE_CANONICAL_...` resolves correctly to openclaw-runtime; reconcile receipts,
  explicit REPOSITORY AUTHORITY wins.

## Test matrix (test/canonical-memory-continuity.test.js — 27 tests, 27/27 PASS)

A cross-session continuity · B supersession + low-authority-not-overriding · C long-conversation
(>12 messages via continuity marker) · D campaign learning auto-recovery · E scope isolation ·
F provider staleness · G chain A→B→C · H schema evolution (synthetic new required field exposed
without redesign) · I safety (no promotion of unvalidated claims, fail-closed seam) · OurDream
regression as DATA not hardcode · observability · write seam end-to-end · context integration.

## OurDream regression (Phase 11)

Lessons stored as ordinary memory records (offer authority, required fields, creation≠activation,
reference-campaign-not-authority, MISSING_AUTHORITY≠TECHNICAL_DEFECT, exact-offer-no-fallback).
NEVER hardcode offer 10138/OurDream as a runtime special case — fixture data only.
