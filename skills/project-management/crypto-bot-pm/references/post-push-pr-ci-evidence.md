# Post-push PR/CI evidence after approved stale-branch repair

Session-derived pattern: Operator approved exactly one controlled remote branch push for an existing crypto_bot S006 PR branch, followed only by read-only PR/CI evidence checks.

Safe sequence after approval:

1. Reconfirm approval scope, local branch, exact local HEAD, clean worktrees, and current remote branch SHA before pushing.
2. Push only the exact approved commit to the exact existing remote source branch, e.g. `git push origin <full-head>:refs/heads/<source-branch>`.
3. Immediately verify with read-only `git ls-remote origin refs/heads/<source-branch>`.
4. Run read-only PR/CI audit against the pushed head.
5. Poll read-only PR/CI audit briefly to distinguish transient `pending` from a terminal blocker.
6. Stop before PR metadata/body/comment/status/check mutation, runner/workflow starts, merge, deploy, runtime, secrets, or broker/trading actions unless a separate exact approval covers that action.

Expected post-push classifications:

- If PR branch now matches the locally validated HEAD but CI is not complete, classify as `pr_created_ci_pending` rather than stale branch mismatch.
- If the PR body still links older completion/PR-evidence artifacts, do not update those links under a branch-push-only approval. Report blockers such as `completion_gate_source_head_mismatch` or `pr_evidence_source_head_mismatch` precisely as stale evidence-link/metadata issues.
- A successful branch push proves only branch synchronization. It does not prove CI readiness, merge readiness, updated PR evidence links, or authority to dispatch the next product task.
- If CI remains pending and runner status is unreadable, the next step is read-only wait/polling or a separately approved runner/CI inspection path, not unilateral runner/workflow action.

Merge-readiness CI evidence normalization:

- `crypto_bot_pr_ci_audit.py` can report `ci_evidence_ready: true` and `ci_state: passed` while `crypto_bot_merge_readiness.py` still rejects that audit JSON as `CI/check evidence is missing`, because merge readiness expects a top-level `statuses`, `checks`, or `check_runs` list plus a matching `source_head`.
- Do not treat that as CI failure. It is an evidence-shape mismatch between two control-plane tools.
- Safe local fix: create a local CI evidence JSON under the Hermes operator state root containing:
  - `source_head`: the exact validated PR HEAD.
  - `read_only_get_only: true`.
  - `statuses`: only latest success/passed status objects for the required contexts, normalized with `state: success` or `status: success`.
  - Optional `ignored_superseded_pending_statuses`: older pending statuses from the same run/context that have been superseded by later success statuses.
- Then pass that local evidence file to `crypto_bot_merge_readiness.py --ci-check-evidence ...`. A valid result has `ci_check_evidence_present: true`, `checks_passed: true`, and `checks_current_for_source_head: true`; merge can still remain blocked by policy.
- This local evidence-file creation is not a Gitea mutation. It must still be reported distinctly from PR metadata/status/check updates.

Reporting checklist:

- Push command shape and exit status.
- Pre-push remote SHA and post-push remote SHA.
- Existing PR number and URL from Gitea output or read-only audit.
- Latest PR/CI audit artifact path.
- `pr_matches_spec`, PR head SHA, `ci_state`, `ci_evidence_ready`, `merge_ready`, and `s006_remote_lifecycle_state`.
- If merge readiness uses normalized CI evidence, report that evidence path and the exact merge-readiness fields: `merge_candidate_validated`, `local_gate_pass`, `pr_source_branch_head_matches_gate`, `ci_check_evidence_present`, `checks_passed`, `checks_current_for_source_head`, and any remaining policy blockers.
- Explicit non-actions: no new PR, no PR metadata/comment/status/check update, no runner/workflow start, no merge, no deploy, no secrets, no runtime/broker/trading actions.
