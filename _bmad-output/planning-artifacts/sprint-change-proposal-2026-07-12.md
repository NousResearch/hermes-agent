# Sprint Change Proposal: Workflow Commander Readiness Corrections

**Date:** 2026-07-12
**Project:** hermes-agent
**Requested by:** kevin
**Change trigger:** `implementation-readiness-report-2026-07-12.md` marked the handoff `NOT READY`.
**Mode:** Automated batch correction. User approval for in-scope planning-artifact corrections was supplied by the workflow invocation.
**Correct Course result:** BLOCKED for missing external Archon producer evidence; supported planning corrections were applied.

## 1. Issue Summary

The latest readiness evaluator found two issues:

1. Major: provider-dependent Hermes story completion still requires validated compatible external Archon producer output.
2. Minor: Story 2.3 contains a non-blocking future-adapter validation note that belongs with Story 3.4a or must be explicitly non-blocking.

The local PRD, architecture, epics, UX contract, tracker, and contract fixtures are otherwise aligned. This is a completion-gate and story-scope clarification, not a Workflow Commander MVP scope change.

The major issue cannot be fully resolved from local project facts because this handoff contains no captured external Archon producer runtime output. Unsupported completion proof remains unchanged and is not invented.

## 2. Impact Analysis

**Epic impact:** Epics 2 through 5 remain valid. No epic is removed, added, renumbered, or resequenced.

**Story impact:** Provider-dependent stories keep their Archon dependency records. Story 2.3 remains independently implementable with generic cwd enforcement and a minimal provider-action test double. Story 3.4a owns real provider-adapter cwd validation.

**Artifact conflicts:** No PRD, architecture, UX, or tracker conflict was found. The tracker already contains all 30 current story keys and warns that provider-dependent stories cannot be marked done from local fixtures alone.

**Technical impact:** No production code changed. Provider-dependent completion claims remain blocked until compatible external Archon producer output is supplied and validated against the local contract package.

## 3. Recommended Approach

Use **Direct Adjustment with an explicit external blocker**.

The in-scope planning corrections are:

- Make the provider-dependent completion gate and current external-evidence absence explicit in the local epics.
- Keep unsupported external producer proof out of the local handoff rather than inventing evidence.
- Remove future real-adapter wording from Story 2.3 and keep real provider-adapter cwd validation in Story 3.4a.

Rollback and MVP review are not justified because the planning package is otherwise consistent.

**Effort:** Low planning correction.
**Risk:** Low. The changes affect planning and handoff text only.
**Timeline impact:** Hermes-side implementation can proceed for locally satisfiable stories. Provider-dependent done claims remain blocked until the external Archon validation evidence is available.
**Blocked facts:** captured external Archon producer output for provider binding lifecycle, workflow command, workflow event, and delivery/outbox status families.

## 4. Detailed Change Proposals

### Provider-Dependent Completion Gate

OLD:

```text
Provider-dependent stories stated that local fixtures were not proof of external Archon producer compatibility, but the epics did not state the current evidence status in the Provider Completion Gate section.
```

NEW:

```text
The epics and isolated handoff contract state that local fixtures are Hermes-side readiness evidence only.
Provider-dependent stories must not be marked done until compatible external Archon producer output is supplied and validated for provider binding lifecycle output, workflow command output, workflow event output, and delivery/outbox status output.
The epics now also state that no captured external Archon producer runtime output is present in this isolated handoff.
```

Rationale: Local schemas and examples support Hermes implementation, but they do not prove external Archon producer compatibility.

### Story 2.3 And Story 3.4a

OLD:

```text
Implementation Scope: ... minimal provider-action cwd port/test double used only to prove cwd propagation before real provider adapters exist.
Blocking behavior: ... Real provider-adapter evidence is not required to complete Story 2.3.
Acceptance Criteria: ... without requiring the Archon provider adapter to exist yet.
```

NEW:

```text
Story 2.3 keeps only generic cwd enforcement and minimal provider-action test-double validation.
Provider-specific adapter evidence is explicitly out of scope for Story 2.3.
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
| Proposal components | Done | Specific epics edits are captured above. |
| Final handoff | BLOCKED for external validation | Batch-mode approval covers supported planning edits, but the major evidence gap requires missing external producer output. |

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
- Story 2.3 no longer contains future real-adapter validation wording.
- Story 3.4a owns real provider-adapter cwd validation.

## 7. Corrections Applied

| Artifact | Change | Reason |
| --- | --- | --- |
| `_bmad-output/planning-artifacts/epics.md` | Added current external evidence status to the Provider Completion Gate. | Addresses the major readiness finding without inventing missing Archon producer output. |
| `_bmad-output/planning-artifacts/epics.md` | Tightened Story 2.3 wording so it owns generic cwd enforcement and the minimal provider-action test double only. | Addresses the minor readiness concern by keeping provider-specific adapter validation with Story 3.4a. |
| `_bmad-output/planning-artifacts/sprint-change-proposal-2026-07-12.md` | Refreshed this proposal with the latest `NOT READY` trigger, BLOCKED result, missing facts, and applied corrections. | Completes the Correct Course handoff for the current readiness run. |
