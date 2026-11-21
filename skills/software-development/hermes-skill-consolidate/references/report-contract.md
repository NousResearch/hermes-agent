# Report Contract

## Planning headings

Return these headings in order:

1. **Hermes Skill Consolidation**
2. **Phase Verdict**
3. **Selected Skills**
4. **Evidence Summary**
5. **Relationship Classification**
6. **Safety Boundary Comparison**
7. **Recommended Structure**
8. **Proposed Changes**
9. **Reference and Dependency Impact**
10. **Rollback Plan**
11. **Verification Plan**
12. **Approval Gate**
13. **Not Verified**

## Allowed phase verdicts

- `NO CHANGE RECOMMENDED`
- `PLAN READY FOR APPROVAL`
- `BLOCKED`
- `APPLIED AND VERIFIED`
- `ROLLED BACK`

Do not use `APPLIED AND VERIFIED` during planning.

## Proposal detail

For each proposed mutation state:

- target path,
- action,
- reason,
- preserved behavior,
- changed behavior,
- safety effect,
- reference impact,
- validation required.

Use `not verified` when evidence is unavailable.

## Approval gate text

The planning report must identify the exact actions that remain blocked pending approval. Do not phrase a recommendation as already authorized.

## Post-apply appendices

After mutation, append:

- **Applied Changes**
- **Verification Evidence**
- **Rollback Status**

List only actions actually verified in live state. A successful write is not equivalent to successful behavior.

## Failure reporting

If cutover or verification fails:

- set the verdict to `ROLLED BACK` only after rollback is confirmed;
- otherwise use `BLOCKED` and identify the partial state;
- never hide failed steps;
- preserve the snapshot until the user decides on cleanup.
