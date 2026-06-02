---
id: phase2-d4-review-integration-deferred
title: Phase 2 D4 — requesting-code-review integration (DEFERRED by default)
status: deferred-pending-explicit-approval
executor: none-pending-jc-approval
parallel_safe: false
base_branch: docs/ua-flywheel-phase1-phase2-plan
allowed_files: []
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/
  - skills/code-analysis/
  - .hermesignore
depends_on:
  - phase2-d2-code-scan-skill
  - phase2-d3-validation-gate-skill
  - explicit-jc-approval-required
verification: []
risk: medium
---

# Phase 2 D4 — requesting-code-review Integration (DEFERRED)

## Context & Intent

**Default status: DEFERRED.** This deliverable is **not included** in the standard Phase 2 approval package. It will only execute if JC explicitly includes D4 in the Phase 2 approval scope. The recommended approval package defaults D4=deferred.

**Why this file exists.** To document the deferred integration scope so that, if JC later approves it, a coder subagent has a complete execution specification without ambiguity. It records what WOULD be built, what files WOULD be touched, and what tests WOULD be required — but nothing is implemented until explicit approval.

**Authority docs.** `.plans/phase-2-flywheel-ua-integration.md` (§D4: Integration with `requesting-code-review`) describes the optional integration. The Phase 2 approval scope explicitly lists D4 as "execute it only if approval explicitly includes D4."

**Intent (if approved):** Extend the existing `requesting-code-review` skill to optionally invoke the code-scan pipeline (D2) and validation-gate (D3) on changed files during PR/diff review. The scan must be opt-in (user explicitly requests it) and must not slow down normal review when not requested.

**Non-goals (always):** No dashboard. No React UI. No auto-injection of scan results into every review. No new runtime dependencies. No tree-sitter/WASM. No SQLite store. No CLI command. No always-on scanning.

## Implementation Details

### Scope (only if JC approves)

This deliverable modifies the **existing** `requesting-code-review` skill. No new files are created — the skill is extended in-place.

### Proposed changes to requesting-code-review (outline only)

1. **Optional pre-scan step:** When user requests code analysis as part of review (e.g., "review this PR with code analysis"), the skill invokes `scan_project.py` on the target directory, optionally with a `--changed` flag (if Phase 3 incremental analysis is available) or by filtering scan output to match the diff file list.

2. **Validation gate integration:** After scan, optionally invoke the validation-gate skill (D3) on any graph-like output.

3. **Context enrichment:** Feed scan results and validation verdicts into the review context as structured data (not hallucinated).

4. **Performance guard:** If the scan + validation step exceeds a timeout (e.g., 10s for the script execution portion), skip the enriched review and fall back to normal review flow with a note explaining why.

### Files that would be touched (if approved)

| File | Change type | Notes |
|---|---|---|
| `skills/software-development/requesting-code-review/SKILL.md` | Modify | Add optional scan+validation step |
| `skills/software-development/requesting-code-review/` | May add reference files | If needed, ≤20 lines |

### Constraints (if approved)

- Only activates when user explicitly requests code analysis during review.
- Does not slow down normal review when code-scan is not requested.
- Existing `requesting-code-review` tests must still pass.
- SKILL.md line budget: the modified skill must still be ≤80 lines total (may require condensing existing content).
- If line budget cannot be maintained, D4 must be split into a separate sub-skill or deferred to Phase 3.

### Why D4 is deferred by default

1. The existing `requesting-code-review` skill is a production-critical artifact. Modifying it introduces regression risk.
2. The skill's current line budget may not accommodate the additional scan+validation steps without exceeding 80 lines.
3. The value of scan-enriched code review is unproven — Phase 2 D1/D2/D3 must first demonstrate value in isolation.
4. If JC approves after D1/D2/D3 ship, there will be more data to inform the integration design.

## Complexity Tier

**T2 (if approved)** — Modifies an existing production skill. Line budget management required. Integration with Phase 2 D2/D3 artifacts. Estimated 6–8 subagent iterations once approved. Higher risk than D1/D2/D3 due to modification of an existing skill.

## Execution Engine

**Executor:** `delegate-coder` (only if JC approves this deliverable).

**Process (if approved):**
1. Coder subagent reads existing `requesting-code-review` SKILL.md to understand current structure.
2. Coder subagent extends the skill with optional scan+validation steps, maintaining ≤80 total lines.
3. Hermes verifies line budget, existing test compatibility.
4. Reviewer subagent validates: spec compliance, scope guardrails, no regression to existing review flow.
5. Hermes presents evidence to JC for commit approval.

**Until JC approves:** This bead's executor is `none`. No subagent dispatch. No file modifications.

## Required Inline Context

### Existing skill location

- Path: `skills/software-development/requesting-code-review/SKILL.md`
- Current state: Must be read during implementation to understand structure, line count, and integration points.

### Preserved unrelated WIP — DO NOT TOUCH

```
tools/skills_sync.py                 # stashed unrelated WIP
tests/tools/test_skills_sync.py      # stashed unrelated WIP
```

### Scope guardrails (if approved)

- Only files listed in "Files that would be touched" may be modified.
- `requesting-code-review` existing behavior must be preserved when scan is not requested.
- No changes to Phase 1 scripts, Phase 2 D1/D2/D3 files, or any forbidden files.

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 1 code-scan scripts | prerequisite | Completed |
| D1: extract_imports.py | prerequisite | Must be implemented |
| D2: code-scan skill | prerequisite | Must be implemented |
| D3: validation-gate skill | prerequisite | Must be implemented |
| **JC explicit approval** | **approval gate** | **NOT granted — this is deferred** |

## Test Obligations

### If approved (outline only)

| Check | Method | Pass criteria |
|---|---|---|
| Line budget | `wc -l skills/software-development/requesting-code-review/SKILL.md` | ≤80 |
| Existing tests | `pytest` targeting requesting-code-review tests | No regression |
| Scan opt-in behavior | Manual test: review without requesting code analysis | No scan invoked, review completes normally |
| Scan requested behavior | Manual test: review with explicit code analysis request | Scan results included in review context |
| Timeout fallback | Artificially slow scan (mock) | Falls back to normal review with explanatory note |

### RED/GREEN/FULL evidence (if approved)

- **RED:** Existing tests fail due to new integration
- **GREEN:** Existing tests pass, new behavior verified
- **FULL:** All checks pass, reviewer approves, no regression

## Verification Command

**Not applicable while deferred.** No verification commands to run. When/if JC approves D4, the above test obligations become the verification plan.

## Approval Evidence

**Not applicable while deferred.**

If JC approves D4 in the future, the evidence bundle will include:
1. Line count of modified SKILL.md (≤80)
2. Existing test pass confirmation (no regression)
3. Opt-in behavior verification
4. Scope guardrail check
5. Reviewer verdict

**Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
This deliverable is DEFERRED — no implementation work until explicit approval.
```

---

> **Bead status: DEFERRED.** This file documents what would be built if JC approves D4.
> **No subagent has been dispatched. No files have been modified.**
> **Revisit only after D1, D2, D3 ship and JC explicitly approves D4 inclusion.**
