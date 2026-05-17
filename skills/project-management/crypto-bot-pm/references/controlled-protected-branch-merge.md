# Controlled protected-branch PR merge

Session-derived pattern: after crypto_bot S006 had local evidence PASS, PR branch/head sync, refreshed PR body evidence links, and current passing CI, the Operator approved exactly one protected-branch merge for PR #2. The approval required a final read-only preflight and then a single Gitea PR merge endpoint call only.

Reusable sequence:

1. Final read-only preflight before any merge mutation:
   - Confirm PR number, repository, source branch, target branch, and exact source HEAD.
   - Confirm PR is still open before merge.
   - Confirm target is the protected branch explicitly approved, e.g. `main`.
   - Confirm PR head SHA equals the approved validated head.
   - Confirm PR body/evidence links are current.
   - Confirm CI/check evidence is passed and current for the exact head.
   - Run merge-readiness dry-run and verify the only remaining blocker is policy/authority.
2. Perform one merge mutation only:
   - Endpoint: `POST /api/v1/repos/<owner>/<repo>/pulls/<number>/merge`
   - Payload for normal merge: `{ "Do": "merge" }`
   - Use existing auth without printing or storing credential material.
   - Do not push branches, create/update PRs/issues/comments/statuses/checks, start runners/workflows, deploy, inspect secrets, or perform runtime/broker/trading/financial actions unless separately approved.
3. Verify after merge with read-only checks:
   - PR state is `closed`.
   - PR `merged` is true.
   - Merge commit SHA is captured.
   - Remote target branch now points to the merge commit.
   - Source branch remains at the validated source head unless deletion was separately approved.
   - Post-merge PR/CI audit reports `remote_lifecycle_complete: true` and `s006_remote_lifecycle_state: merged`.
4. Repair control-plane semantics if a verifier still treats the merged PR as merely `merge_ready` or stale:
   - A closed merged PR should be classified as `merged`, not as a pending merge candidate.
   - Post-merge audit success should allow exit 0 when `remote_lifecycle_complete` is true, even though `merge_ready` is false because there is nothing left to merge.
   - Add/extend tests for merged lifecycle classification.
5. Re-run readiness after repair:
   - Control-plane self-check should be green.
   - Autonomy readiness should report `ready_for_next_task: true` and `ready_for_local_autonomy: true`.
   - Report any advisory historical warnings separately from blockers.

Reporting checklist:

- Final preflight artifact path and key fields.
- Exact merge endpoint, method, and payload shape.
- Merge response status.
- PR closed/merged state, merged timestamp if available, and merge commit SHA.
- Remote target/source refs after merge.
- Post-merge PR/CI audit artifact and lifecycle fields.
- Control-plane repair commit, if any.
- Explicit non-actions matching the approval boundaries.
