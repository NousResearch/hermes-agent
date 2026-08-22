# Verdict Supersession / Truth-Correction Pattern

Use when a previous mission's verdict is found to be overcertified — the evidence does not support the claimed verdict level. This pattern governs correcting the official record without erasing history.

## When to Use

- A previous mission declared `CERTIFIED` but the evidence was weaker than claimed
- Candidate outcomes used `REMOVAL_REVALIDATED` without distinguishing clean absence from behavioral equivalence
- A report had a material factual error (commit count, hash mismatch, claimed-but-unexecuted operation)
- Adapted tests were presented as independent proof of preserved capability
- Clean structural absence was presented as behavioral preservation

## Core Distinction: Clean Absence vs Behavioral Equivalence

This is the most important distinction in a truth-correction mission. They are NOT the same:

| State | Meaning | Example |
|-------|---------|---------|
| **CLEAN_ABSENCE** | File is gone; no broken imports, no crashes, adapted tests pass | Acceptance tool adapted to gracefully handle missing module |
| **BEHAVIOR_PRESERVED** | The original operator capability still exists through a canonical replacement | A replacement tool produces the same output as the removed one |
| **CAPABILITY_INTENTIONALLY_DEPRECATED** | The capability is intentionally removed; no replacement exists; operator is informed | Deprecation stub with alternative tooling recommendation |

A module can be absent without broken imports while still representing a lost operator capability. **Do not classify clean absence as revalidated removal.**

## Candidate Classification Framework

When re-evaluating a previous mission's candidate removals, use these evidence-based classifications:

| Classification | Requires |
|----------------|----------|
| `REMOVAL_BEHAVIORALLY_PROVEN` | Structural cleanliness AND behavioral equivalence via replacement implementation |
| `REMOVAL_INTENTIONALLY_DEPRECATED_AND_AUTHORIZED` | Deprecation stub/notice + explicit alternative tooling + no replacement needed |
| `REMOVAL_STRUCTURALLY_CLEAN_BUT_BEHAVIOR_UNPROVEN` | No broken imports, adapted tests pass, but behavioral equivalence NOT proven |
| `REMOVAL_REJECTED_OPERATOR_REGRESSION` | Verified operator regression with baseline differential evidence |
| `REMOVAL_REJECTED_CAPABILITY_LOSS` | Capability lost with no replacement and no deprecation authorization |
| `REMOVAL_REJECTED_INSUFFICIENT_EVIDENCE` | Cannot determine status from available evidence |
| `REMOVAL_STATUS_UNRESOLVED` | Status could not be determined |
| `REPAIR_BLOCKED` | Precondition prevents repair |

### Disallowed classification

`REMOVAL_REVALIDATED` must NOT be used unless BOTH structural cleanliness AND meaningful behavioral proof are present. Do not use it for:

- Clean absence alone (that is `REMOVAL_STRUCTURALLY_CLEAN_BUT_BEHAVIOR_UNPROVEN`)
- Deprecation stubs alone (that is `REMOVAL_INTENTIONALLY_DEPRECATED_AND_AUTHORIZED`)
- Adapted tests alone (they prove test coherence, not behavioral equivalence)

## Verdict Supersession Rules

When superseding a previous verdict:

1. **PRESERVE** the historical record — do not erase or overwrite original artifacts
2. **CREATE** corrected artifacts alongside originals (use corrected- prefix for new files)
3. **EXPLAIN** in a supersession report:
   - What was previously claimed
   - Why the claim was too strong
   - What remains valid (certified sub-results)
   - What remains unresolved
   - Why the new verdict is truthful
4. **DISTINGUISH** the mission-level verdict from certified sub-results — a PARTIAL mission may contain fully certified sub-results

### Preserved sub-results pattern

The following types of sub-results typically remain valid when the mission-level verdict is downgraded:

- Truth corrections of individual candidate outcomes (e.g. candidate 2 correction)
- Framework or methodology repairs (e.g. classifyCandidate baseline fix)
- Executed rollback verification (not theoretical)
- Scope-expanded file restoration/adaptation
- Cleanup of untracked artifacts
- No canonical worktree mutation
- No push, merge, deployment, or production mutation

## Commit Count Audit

When a previous report's commit count is inconsistent:

1. Run `git log --oneline --ancestry-path <base_sha>..<final_sha>` to get the exact list
2. Compare with the reported count
3. If they differ:
   - Document the discrepancy as a factual error in the supersession report
   - Correct the count in all structured artifacts
   - Do NOT invent a missing commit — the report may have counted header commits, merge commits, or worktree-creation commits that don't exist in the target branch

## Artifact Correction Requirements

When correcting a previous mission's artifacts, produce at minimum:

1. `partial-verdict-correction-receipt.md` — explains why the verdict changed
2. `previous-certification-supersession-report.md` — documents the supersession
3. `commit-receipt-correction.json` — corrects the commit count
4. `candidate-evidence-reclassification.json` — structured reclassification of all candidates
5. `candidate-evidence-reclassification.md` — detailed markdown reclassification
6. `certified-subresults-and-unresolved-results.md` — lists what remains certified vs unresolved
7. `corrected-aggregate-report.json` — corrected aggregate report
8. `corrected-aggregate-report.md` — corrected aggregate report in Markdown
9. `corrected-certification-receipt.md` — corrected certification receipt
10. `corrected-continuity.json` — corrected continuity JSON
11. `final-tree-provisional-status-report.md` — reports that the final tree is provisional
12. `zero-unrelated-mutation-proof.md` — proves no unrelated mutations
13. `branch-safety-receipt.md` — confirms no push/merge/deploy authorized

Plus a canonical continuity artifact in `docs/mcporter/` with `PARTIAL` in the filename.

## Provisional Tree Declaration

When behavioral equivalence is unproven, the final removal tree is **provisional**. Document explicitly:

```
- SHA <sha> is not authorized for push or merge
- The corrective branch is local-only
- The final removal tree is provisional
- No branch containing the removals may be integrated until a future behavioral-equivalence mission resolves each candidate
```

## Required Structured Fields

The corrected aggregate report must include these fields (or preserve their semantics in the existing schema):

```json
{
  "mission": "...",
  "previous_verdict": "...",
  "corrected_verdict": "...",
  "previous_verdict_superseded": true,
  "integration_authorized": false,
  "push_authorized": false,
  "merge_authorized": false,
  "production_mutation_authorized": false,
  "behavioral_equivalence_fully_proven": false,
  "candidate_2_truth_correction_certified": true,
  "classification_framework_repair_certified": true,
  "executed_rollback_proof_certified": true,
  "explicit_removal_final_tree_certified": false
}
```

## Integration Safety

The branch safety receipt must confirm:

- No push performed
- No merge performed
- No pull request created
- No deployment triggered
- No production mutation
- No Docker mutation
- No systemd mutation
- No history rewriting
- No `git add .` used
- No `git clean` used
- No `git reset --hard` used
- No new candidate removals
- No candidate restoration or re-removal

## Verification Gates

Before certifying a truth-correction mission:

1. All JSON artifacts parse correctly
2. All Markdown artifact paths exist
3. Previous verdict is explicitly superseded in the record
4. Commit count is verified from actual Git history
5. Zero source code was modified (confirmation via git diff)
6. No new candidate mutation occurred
7. Zero unrelated mutation confirmed
8. No push, merge, or production mutation occurred
9. Integration authorization explicitly set to false
10. The next recommended mission is documented
