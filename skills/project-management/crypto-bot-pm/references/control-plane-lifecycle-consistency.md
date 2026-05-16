# Control-plane lifecycle consistency notes

Use this reference when crypto_bot audits disagree about S006/S007A lifecycle, PR existence, CI evidence, or next-task readiness.

## Durable lesson

A native Kanban import audit can be locally healthy (correct board/card/dependency counts) while still holding stale remote lifecycle metadata. If it reports S006 as `pr_absent` but the PR/CI audit discovers the matching Gitea PR by source branch, source HEAD, and target branch, treat this as a Hermes control-plane consistency regression — not as product task readiness and not as a CI-runner problem.

## Correct repair pattern

1. Stop product-task dispatch and runner/CI proposals while audits disagree.
2. Repair the Hermes control-plane tooling first in `/Users/preston/.hermes/hermes-agent`.
3. Prefer a live read-only PR/CI audit payload as the source for S006 remote state when no explicit remote-readiness JSON is supplied to the Kanban import audit.
4. Preserve the distinction between:
   - `pr_exists: true`
   - `ci_evidence_ready: false`
   - `merge_readiness_ready: false`
   - `remote_lifecycle_state: pr_created_ci_pending`
5. Make Kanban classification reflect a valid remote lifecycle block, e.g. `IMPORT_VALID_REMOTE_LIFECYCLE_BLOCKED`, instead of stale `pr_absent`/partial-import states.
6. Add regression coverage so PR/CI payload fields override stale preview-only remote metadata.
7. Re-run the full control-plane validator quartet before any next gate:
   - `python3 tools/crypto_bot_control_plane_self_check.py --format json`
   - `python3 tools/crypto_bot_kanban_import_audit.py --expected-card-count 90 --expected-dependency-count 101 --preview /Users/preston/.local/state/hermes-operator/kanban-import-previews/crypto_bot-preview.json --format json`
   - `python3 tools/crypto_bot_pr_ci_audit.py --repo-root /Users/preston/robinhood/crypto_bot --gitea-url http://127.0.0.1:3005 --owner preston --repo crypto_bot --source-branch hermes/dev13-006-daemon-trust-contract-mapping --source-head 8be208ba317972da03060eb0170a40d2a678aa99 --target-branch main --format json`
   - `python3 tools/crypto_bot_autonomy_readiness.py --format json`
8. Commit the Hermes control-plane repair locally only after targeted tests and validators pass.

## Reporting rule

When this repair succeeds but CI statuses are still missing, report the result as a restored consistency milestone, not as S006 remote completion. The correct next state is usually:

- S006 local evidence: valid.
- S006 PR: exists.
- CI evidence: pending/missing.
- Readiness: `ready_for_local_autonomy: true`, `ready_for_next_task: false`.
- Next action: hold product dispatch and request/prepare only the gated runner or CI evidence recovery path if explicitly approved.

## Pitfall

Do not use a stale Kanban `pr_absent` state as a reason to start product work, and do not jump directly to runner recovery while Kanban audit, PR/CI audit, and readiness disagree. Repair and commit the control-plane consistency first.