# crypto_bot Completion Evidence Contract

Schema: `hermes.autonomy.crypto_bot_completion_gate.v1`
Evidence issue schema: `hermes.autonomy.crypto_bot_evidence_issue.v1`
Remote readiness schema: `hermes.autonomy.crypto_bot_remote_readiness.v1`
PR evidence schema: `hermes.autonomy.crypto_bot_pr_evidence.v1`
Merge readiness schema: `hermes.autonomy.crypto_bot_merge_readiness.v1`

Hermes may not report a branch-local crypto_bot task as `Completed` unless the
Hermes-owned completion gate passes for the exact task, branch, base ref, and
final HEAD being reported. Telegram text, a clean worktree, a PM summary, or a
Codex sidecar result by itself is not completion evidence.

## Required Gate Inputs

- Managed project id: `crypto_bot`
- Task id or durable task source
- Managed repo path
- Base ref
- Target branch
- Target HEAD/ref
- Optional sidecar prompt/result paths; otherwise Hermes discovers them from
  `/Users/preston/.local/state/hermes-operator/codex-sidecar-audits/`

## Required Gate Commands

The gate runs these read-only commands itself:

```bash
git rev-parse <base>
git rev-parse <head>
git rev-parse refs/heads/<target-branch>
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git diff --name-only <base>..<head>
git diff --check <base>..<head>
```

For changed Python files only, the gate runs `ruff check <files>` when `ruff` is
available. `ruff format` is never allowed. Docs-only changes still require
`git diff --check`.

## Blocked-Surface Scan

The gate records blocked-surface findings with `severity` values:
`block`, `warning`, and `allowed_docs_reference`. Only `block` findings make
the gate `BLOCKED`.

The gate fails or blocks if changed paths touch these critical surfaces:

- `.gitea/workflows`
- secret-like paths such as `.env`, token, key, private-key, cookie, or
  credential files
- runtime databases or logs
- broker, trading, financial, live-market, order, account, position, wallet, or
  Robinhood surfaces
- deployment or GitOps surfaces
- workflow, runner, runtime service-start, daemon, worker, scheduler, or server
  surfaces

Safe docs-only contract references are narrower. A Markdown file under
`docs/contracts/*.md`, `docs/development/*.md`, or `docs/architecture/*.md`
may mention daemon/service concepts only when the selected strategic-plan item
or Hermes managed-project descriptor explicitly allowlists the exact path or a
safe docs pattern, no executable/config/runtime/service-start file changed, and
no secret-looking content was introduced. This docs allowance does not weaken
blocking for runtime code, service-start scripts, workflows/runners,
deploy/GitOps, secrets, broker, trading, or financial surfaces.

## Sidecar Evidence Rules

The sidecar prompt must be generated without a prefilled pass/clean conclusion.
It must require Codex to run and report the required Git commands, exit codes,
changed files, final branch, final full HEAD, `git diff --check`, and
blocked-surface scan evidence.

The canonical sidecar result schema is
`hermes.autonomy.crypto_bot_sidecar_audit.v1`. The result must include a
machine-evidence section with these fields: `Schema`, `Branch observed`,
`Full HEAD observed`, `Base/head range audited`, `Changed files`, `Worktree
status`, `git diff --check exit code`, `Blocked-surface scan`, and
`Final conclusion`.

The completion gate rejects sidecar evidence when it is missing, empty, stale,
diagnostic/smoke-only, names the wrong branch or HEAD, omits the base/head
range, omits `git diff --check`, omits the `git diff --check` exit code, omits
blocked-surface verification, or claims success while the local gate's
`git diff --check` fails. The final sidecar conclusion must be exactly `PASS`;
`FAIL`, `BLOCKED`, missing, or ambiguous conclusions block completion.

## Required Report

Passing and failing gates write JSON reports under:

`/Users/preston/.local/state/hermes-operator/completion-gates/`

Each report records the schema, task id, base ref, target branch, target full
HEAD, repo path, worktree status, changed files, validator commands and exit
codes, `git diff --check` result, Python lint applicability, blocked-surface
scan, sidecar prompt/result paths and hashes, branch/head match, conclusion,
blockers, warnings, timestamp, and a machine-verifiable command transcript
summary.

Valid conclusions are `PASS`, `FAIL`, and `BLOCKED`. Only `PASS` permits Hermes
to report `Completed` or select another autonomous task.

## Remote Integration Evidence

Local branch-local completion is still governed by this completion gate. Remote
integration has additional evidence requirements and never weakens the local
gate.

Remote integration completion requires:

- a local completion-gate report with `PASS`
- a PR evidence packet from
  `/Users/preston/.hermes/hermes-agent/tools/crypto_bot_pr_evidence_contract.py`
- source branch and full SHA matching the completion gate
- target merge base recorded
- target PR changed files matching the completion-gate changed files
- completion gate JSON path and sidecar result path in the proposed PR body
- validators and blocked-surface proof in the proposed PR body
- no secret-looking content in the proposed PR body

When PR authority is explicitly enabled, remote integration evidence must also
include the PR URL/ID and current check/status evidence for the PR source HEAD.

Merge completion requires a separate merge readiness report from
`/Users/preston/.hermes/hermes-agent/tools/crypto_bot_merge_readiness.py` and explicit merge
authority. Merge authority is disabled by default.

## Evidence Issue Registry

Open evidence-loop failures are tracked under:

`/Users/preston/.local/state/hermes-operator/evidence-issues/`

Each issue records `issue_id`, task or claim id, type, status, repo path,
branch, base, bad head, repaired head, gate report path, invalidation reason,
timestamps, evidence paths, and operator/gate basis.

Valid statuses are `active`, `repair_attempted`, `repaired`, `invalidated`, and
`superseded`. Readiness blocks `active` and `repair_attempted` evidence issues.
A dev13-005-style completion failure may be marked `repaired` only when a later
repaired head has a passing completion-gate report over `1227b15..HEAD`; a
failed or missing gate stays `active` or `repair_attempted`. A dev13-006
unsupported completion claim stops blocking only after an explicit
machine-readable invalidation artifact exists, or after a passing gate
explicitly claims the same `claim_id`. A later strategic-plan task such as
`S006` may use `dev13-006` as a branch alias without reviving the invalidated
Telegram-only claim; `task_id`, `session_id`, `branch_alias`, and `claim_id`
remain separate evidence fields.
