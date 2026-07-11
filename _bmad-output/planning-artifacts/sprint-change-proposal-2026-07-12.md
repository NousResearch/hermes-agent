# Sprint Change Proposal: Workflow Commander Readiness Corrections

**Date:** 2026-07-12
**Project:** hermes-agent
**Requested by:** kevin
**Change trigger:** `implementation-readiness-report-2026-07-12.md` marked the handoff `NEEDS WORK`.
**Mode:** Automated batch correction. User approval for in-scope planning-artifact corrections was supplied by the workflow invocation.

## 1. Issue Summary

The latest readiness evaluator found two issues:

1. Major: provider-dependent Hermes story completion still requires validated compatible external Archon producer output.
2. Minor: Story 2.3 contains a non-blocking future-adapter validation note that belongs with Story 3.4a or must be explicitly non-blocking.

The local PRD, architecture, epics, UX contract, tracker, and contract fixtures are otherwise aligned. This is a completion-gate and story-scope clarification, not a Workflow Commander MVP scope change.

## 2. Impact Analysis

**Epic impact:** Epics 2 through 5 remain valid. No epic is removed, added, renumbered, or resequenced.

**Story impact:** Provider-dependent stories keep their Archon dependency records. Story 2.3 remains independently implementable with generic cwd enforcement and a minimal provider-action test double. Story 3.4a owns real provider-adapter cwd validation.

**Artifact conflicts:** No PRD, architecture, UX, or tracker conflict was found. The tracker already contains all 30 current story keys and warns that provider-dependent stories cannot be marked done from local fixtures alone.

**Technical impact:** No production code changed. Provider-dependent completion claims remain blocked until compatible external Archon producer output is supplied and validated against the local contract package.

## 3. Recommended Approach

Use **Direct Adjustment with an explicit external blocker**.

The in-scope planning corrections are:

- Make the provider-dependent completion gate explicit in the local epics and isolated handoff contract.
- Keep unsupported external producer proof out of the local handoff rather than inventing evidence.
- Move the Story 2.3 future-adapter validation note into Story 3.4a's real-adapter validation text.

Rollback and MVP review are not justified because the planning package is otherwise consistent.

**Effort:** Low planning correction.
**Risk:** Low. The changes affect planning and handoff text only.
**Timeline impact:** Hermes-side implementation can proceed for locally satisfiable stories. Provider-dependent done claims remain blocked until the external Archon validation evidence is available.

## 4. Detailed Change Proposals

### Provider-Dependent Completion Gate

OLD:

```text
Provider-dependent stories already kept dependency records, but the local handoff did not enumerate the external producer output families that remain missing before completion claims.
```

NEW:

```text
The epics and isolated handoff contract state that local fixtures are Hermes-side readiness evidence only.
Provider-dependent stories must not be marked done until compatible external Archon producer output is supplied and validated for provider binding lifecycle output, workflow command output, workflow event output, and delivery/outbox status output.
```

Rationale: Local schemas and examples support Hermes implementation, but they do not prove external Archon producer compatibility.

### Story 2.3 And Story 3.4a

OLD:

```text
Story 2.3 integration validation said Story 3.4a repeats cwd validation with the real provider adapter.
```

NEW:

```text
Story 2.3 now keeps only generic cwd enforcement and minimal provider-action test-double validation.
Story 3.4a's integration validation now explicitly owns real provider-adapter cwd validation through the Story 2.3 cwd guard.
```

Rationale: The real provider adapter is introduced in Epic 3, so adapter-specific validation belongs in Story 3.4a while Story 2.3 remains independently implementable.

## 5. Checklist Summary

| Checklist Area | Status | Finding |
| --- | --- | --- |
| Trigger and context | Done | Latest readiness report supplied the trigger and evidence. |
| Epic impact | Done | Epics remain valid; no resequencing or scope reduction required. |
| Artifact conflict analysis | Done | No PRD, architecture, UX, or tracker mismatch remains. |
| Path forward | Done | Direct Adjustment selected for planning text; external validation remains blocked. |
| Proposal components | Done | Specific epics and handoff edits are captured above. |
| Final handoff | Done with blocker | Provider-dependent stories remain gated by compatible Archon producer output. |

## 6. Implementation Handoff

**Scope classification:** Minor planning correction with blocked external validation.

**Developer agent responsibilities:**

- Use `_bmad-output/implementation-artifacts/sprint-status.yaml` as the current implementation tracker.
- Do not mark provider-dependent Hermes stories done from local fixtures alone.
- Validate compatible external Archon producer output before completion claims for provider-dependent stories.
- Implement Story 2.3 without requiring the real Archon adapter.
- Implement Story 3.4a with real provider-adapter cwd validation through the Story 2.3 cwd guard.

**Missing facts for external validation:**

- Captured Archon provider binding lifecycle output.
- Captured Archon workflow command output.
- Captured Archon workflow event output.
- Captured Archon delivery and outbox status output.

**Success criteria:**

- Provider-dependent completion gating is explicit and unsupported external evidence is not invented.
- Story 2.3 no longer contains the future real-adapter validation note in integration validation.
- Story 3.4a owns real provider-adapter cwd validation.
