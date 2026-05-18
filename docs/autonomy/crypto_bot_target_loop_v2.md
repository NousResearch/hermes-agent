# crypto_bot Target Loop v2

## Purpose

Hermes is the autonomous software-development operator for `crypto_bot`.
Hermes owns planning, task selection, task classification, delegation,
verification, evidence review, local branch commits, and status reporting.
The Operator owns strategy, emergency stop authority, policy changes,
permission expansion, live financial risk, deployments, and protected-branch
promotion.

This target loop replaces per-task Operator approval for ordinary safe
branch-local development with local infrastructure enforcement through
policy/validator gates. The final completion authority is the Hermes-owned
completion evidence contract at
`/Users/preston/.hermes/hermes-agent/docs/autonomy/crypto_bot_completion_evidence_contract.md`
and its read-only gate tool:
`/Users/preston/.hermes/hermes-agent/tools/crypto_bot_completion_gate.py`.
Remote branch, PR, CI, and merge lifecycle work is governed by
`/Users/preston/.hermes/hermes-agent/docs/autonomy/crypto_bot_remote_lifecycle_contract.md`
and `/Users/preston/.hermes/hermes-agent/docs/autonomy/crypto_bot_gitea_ci_pr_target_loop.md`.
That remote layer is staged and disabled for mutation by default.

## Standing Control Loop

1. Hermes loads the managed-project descriptor at
   `/Users/preston/.hermes/hermes-agent/projects/crypto_bot/crypto_bot.project.yaml`.
2. Hermes treats persistent goals and Hermes Kanban as the standing objective
   and lifecycle engine.
3. Hermes reads the `crypto_bot` strategic plan at
   `docs/planning/autoresearch_runpod_to_live_trade/plan.json` and selects the
   next unblocked task that advances the plan.
4. Hermes records durable task-source proof before taking action: either a
   strategic-plan item/session id or a Hermes Kanban card/export path, plus the
   path allowlist source for the selected task.
5. Hermes classifies the task by policy before taking action.
6. Infrastructure gates decide whether the task is auto-executable or
   escalation-required.
7. Hermes delegates bounded work to the correct lane:
   implementation worker, review worker, Codex sidecar audit, local validator,
   or CI-triage reviewer.
8. Workers modify only allowed branch/workspace files and never touch blocked
   surfaces.
9. Hermes runs validators, checks provenance, commits locally only when
   pre-commit evidence passes, renders a final sidecar prompt with
   `/Users/preston/.hermes/hermes-agent/tools/render_crypto_bot_sidecar_audit_prompt.py`, and
   requires a clean post-commit Codex sidecar audit for final branch-local
   completion.
10. Hermes runs `/Users/preston/.hermes/hermes-agent/tools/crypto_bot_completion_gate.py`
   against the exact task/base/branch/HEAD and records the completion-gate JSON
   path.
11. Hermes updates task state, reports `Completed`, and selects another task
   only when the completion gate returns `PASS`. If the gate returns `FAIL` or
   `BLOCKED`, Hermes marks the task `BLOCKED` or `NEEDS_REPAIR`, reports the
   JSON path and blockers, and pauses next-task selection.

## Durable State

- Hermes goals: strategic objective and current operating budget.
- Hermes Kanban: task lifecycle truth for selection, state transitions, and
  blocked/escalated states.
- `crypto_bot` strategic plan: product completion source of truth.
- Gitea: durable repo state, branches, issues, PRs, checks, and evidence once
  write authority is explicitly enabled by policy.
- Local artifact records: validator outputs, Codex audit results, migration
  inventories, and readiness reports.
- PR evidence packets:
  `/Users/preston/.local/state/hermes-operator/pr-evidence/`.

## Worker Lanes

- Planning lane: selects and scopes the next strategic-plan task.
- Implementation lane: performs branch-local code/docs/tests work inside
  allowlisted paths.
- Review lane: inspects diffs, confirms scope, and checks evidence.
- Codex sidecar lane: bounded audit or coding sidecar; never the primary
  decision-maker and never a substitute for failed validators.
- CI-triage lane: reads check evidence and proposes fixes; it does not start
  runners or workflows unless policy later permits.

## Auto-Executable Work

Hermes may execute a task without per-task Operator approval only when all of
these are true:

- The task is branch-local development for the managed `crypto_bot` repository.
- The task writes only allowlisted non-secret paths declared by the selected
  plan item or Hermes policy.
- Docs-only contract paths may mention daemon/service concepts only when the
  exact Markdown path or safe docs pattern is allowlisted by the selected plan
  item or managed-project descriptor. Runtime code, service-start scripts,
  configs, workflows/runners, deploy/GitOps, secrets, broker, trading, and
  financial surfaces remain blocked regardless of docs wording.
- The selected task has durable source evidence: a strategic-plan item/session
  or a Hermes Kanban card/export path. Commissioning or maintenance work outside
  the strategic plan must have a Kanban card/export path before any write.
- The task does not require secrets, live broker/trading/financial APIs,
  runtime services, deployments, protected branches, Gitea writes, workflow
  edits, runner starts, or policy changes.
- The starting worktree is clean.
- Hermes creates or uses a non-protected local branch.
- Required local validators pass.
- `ruff check` may be used on changed Python files, but `ruff format` is
  forbidden.
- Codex sidecar final audit passes when branch-local work is being completed.
  The final audit must run after the local commit on a clean worktree and must
  identify the exact branch and commit hash being reported. A dirty-worktree,
  pre-commit, smoke-test, stale, or different-HEAD audit is not completion
  evidence.
- Hermes completion gate passes for the exact task/base/branch/HEAD and records
  a `hermes.autonomy.crypto_bot_completion_gate.v1` JSON artifact under
  `/Users/preston/.local/state/hermes-operator/completion-gates/`.
- Completion evidence includes branch, commit, changed files, validator
  results, Codex result path, completion-gate JSON path, and non-action proof.

## Escalation-Required Work

Hermes must stop and report the reason before any task that involves:

- Secrets, `.env`, token files, Keychain material, private keys, cookies,
  credential stores, runtime databases, generated credential artifacts, or
  credential-bearing logs.
- Robinhood, broker, exchange, live-market, account, order, position, wallet,
  trading, or financial APIs.
- Runtime service starts, app servers, workers, schedulers, launchd, qmd,
  Docker builds, Kubernetes, Flux, Harbor, OpenBao, RabbitMQ, Redis, Temporal,
  production services, or Gitea runners.
- Deploys, GitOps promotion, protected-branch merge, release publishing, or
  package publication.
- Gitea write authority expansion, issue/PR/comment/label/project/check/status
  mutation, webhook mutation, or repository mutation.
- New tool permissions, policy changes, workflow edits, or runner/workflow
  execution.

## Remote Lifecycle Stages

Local branch-local autonomy remains available when readiness and the completion
gate pass. Remote lifecycle authority is separate and staged:

1. Read-only remote/CI discovery is allowed.
2. PR evidence packet generation is allowed.
3. Controlled remote branch push is escalation-required until explicitly
   enabled by policy.
4. One PR creation pilot is escalation-required until explicitly enabled by
   policy.
5. PR updates, comments, and status mutation are escalation-required.
6. Merge-to-main or protected branch mutation is escalation-required and
   disabled by default.

Hermes may read workflow files and check status when available. Hermes must not
edit workflows, start workflows, start runners, deploy, inspect secrets, call
broker/trading/financial APIs, push, create PRs, mutate Gitea, or merge unless a
future policy explicitly enables the specific authority. A PR may be created
only from a non-protected source branch whose full HEAD exactly matches a
passing local completion gate. PR creation is separate from merge.

## Remote Readiness States

Hermes must keep these states separate:

- `local_evidence_ready`
- `remote_readiness_ready`
- `pr_evidence_ready`
- `ci_evidence_ready`
- `merge_readiness_ready`
- `ready_for_local_autonomy`
- `ready_for_remote_pr_pilot`
- `ready_to_request_controlled_one_pr_pilot`
- `ready_for_merge_autonomy`

`ready_for_local_autonomy` may be true while remote, PR, CI, and merge
readiness are false. `ready_for_remote_pr_pilot` is false unless the local
evidence loop is green, read-only Gitea APIs are reachable, the source
branch/HEAD ties to a passing completion gate, the PR evidence packet validates,
and policy explicitly enables controlled remote branch push and one PR creation.
`ready_to_request_controlled_one_pr_pilot` may become true before push/PR
authority when the local gate, PR packet, read-only remote probe, exact source
branch/head, and target branch safety are aligned; it only permits asking for
explicit Operator approval.
`ready_for_merge_autonomy` defaults false until explicit future policy enables
merge authority and CI/branch-protection requirements are verifiably satisfied.

## Codex Sidecar Role

Codex is an audit and bounded coding sidecar. Hermes remains responsible for
classification, task selection, final evidence review, local commit decisions,
and Operator-facing reporting. A failed or missing Codex audit cannot be
replaced with a Hermes self-audit, PM status packet, stale diagnostic, or
strategic-plan summary.

For branch-local completion, Hermes may use a pre-commit Codex review as an
extra check, but the required final Codex sidecar audit is post-commit,
read-only, external to the product repo, and tied to the reported HEAD.
The final sidecar prompt must not prefill a clean/pass conclusion or `blocked
surfaces: none`. It must require Codex to run and report `git status --short
--branch`, `git rev-parse --abbrev-ref HEAD`, `git rev-parse HEAD`,
`git diff --name-only <base>..<head>`, `git diff --check <base>..<head>`,
command exit codes, changed files, final branch, final full HEAD, and
blocked-surface scan evidence. The prompt must use the canonical full-SHA
base/head range and must keep the audit bounded to the listed read-only
commands plus changed-path blocked-surface scanning.

A sidecar result is rejected if it is missing, empty, stale, diagnostic/smoke
only, names the wrong branch or HEAD, omits the base/head range, omits
`git diff --check`, omits exit codes, omits blocked-surface verification, or
has a final conclusion other than exactly `PASS`. A sidecar `PASS` still cannot
override failed local validators or blocked-surface findings.

## Completion Proof

A task is complete only when Hermes can show:

- Selected strategic-plan item or durable Hermes Kanban card/export path.
- Policy classification and auto-executable or escalation-required decision.
- Branch name and clean starting state.
- Exact files changed.
- Validator commands and results.
- Local commit hash for completed branch-local work.
- Codex sidecar final audit result, when required, showing the same branch and
  commit hash as the local commit evidence.
- Completion-gate JSON path and `PASS` conclusion.
- Confirmation that no blocked surfaces were touched.
- Updated Hermes Kanban/task state and next autonomous action.
- For remote integration completion, a PR evidence packet and, once PR
  authority is enabled, PR URL/ID plus check status evidence.
- For merge completion, a separate merge gate and explicit merge authority.

Do not infer validator success from a clean worktree. Do not treat a
human-readable Telegram report, PM summary, stale diagnostic, prefilled sidecar
result, or old sidecar output as completion evidence. Do not push, create PRs,
mutate Gitea, start services, run workflows/runners, inspect secrets, edit
workflow files, use `ruff format`, or touch broker/trading/financial surfaces
while the evidence loop is unhealthy.

Use strategic-plan session IDs such as `S006` as durable task IDs. Keep branch
aliases such as `dev13-006` separate from invalidated unsupported completion
claims; do not revive an old claim unless a passing gate explicitly names that
claim ID.
