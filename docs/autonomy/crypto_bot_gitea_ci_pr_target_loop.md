# crypto_bot Gitea CI PR Target Loop

This target loop is the remote lifecycle extension for
`crypto_bot_target_loop_v2.md`. It is dry-run and read-only by default.

## Loop

1. Hermes confirms local evidence readiness with
   `/Users/preston/.hermes/hermes-agent/tools/crypto_bot_autonomy_readiness.py --format json`.
2. Hermes probes remote readiness with
   `/Users/preston/.hermes/hermes-agent/tools/crypto_bot_remote_readiness.py`.
3. Hermes generates a proposed PR evidence packet with
   `/Users/preston/.hermes/hermes-agent/tools/crypto_bot_pr_evidence_contract.py`.
4. Hermes must run the controlled Gitea PR pilot adapter self-check, then a
   dry-run, then `--create-pr-only --preflight-only` with
   `/Users/preston/.hermes/hermes-agent/tools/crypto_bot_gitea_pr_pilot.py` to prove whether
   an existing remote branch can safely be used for exactly one PR creation
   request.
5. Hermes may request Operator approval for a controlled one-PR pilot only when
   policy explicitly enables the exact remaining action. A partial pilot with a
   remote branch already at the exact approved SHA must request PR creation
   only and must not push again.
6. Hermes reads PR, CI, status, branch protection, workflow, and runner evidence
   when available through read-only APIs.
7. Hermes evaluates merge eligibility only with
   `/Users/preston/.hermes/hermes-agent/tools/crypto_bot_merge_readiness.py`.
8. Hermes never merges until a future merge policy explicitly enables merge
   authority and the merge gate is satisfied.

## Current Default

The default remote lifecycle authority is intentionally narrow:

- read-only remote/CI discovery: allowed
- PR evidence packet generation: allowed
- controlled remote branch push: escalation-required
- one PR creation pilot: escalation-required
- PR updates, comments, statuses, and check mutation: escalation-required
- merge-to-main or protected branch mutation: escalation-required and disabled

Hermes must not use `gh pr create` against local Gitea unless a future
Hermes-owned tool explicitly verifies that the host supports that GitHub CLI
path. The current controlled path is the Hermes-owned Gitea REST adapter and
PR evidence packet.

## Failure Modes

Hermes reports blockers instead of acting when:

- CI is absent.
- a runner is offline or runner status is unreadable.
- workflow files are missing or workflow provenance is unclear.
- workflow execution failed.
- checks are stale, missing, failed, or pending indefinitely.
- the PR branch does not match the completion gate branch.
- the PR HEAD does not match the completion gate full HEAD.
- the target PR diff includes files not covered by the completion gate.
- a protected branch mutation would be attempted.
- a push would include unrelated commits.
- a token is missing, insufficient, or would require secret inspection.
- a Gitea endpoint requires unsupported authentication.
- remote branch persistence succeeded but PR creation failed; this is a partial
  pilot state and not a completed PR pilot.
- the Hermes-owned PR pilot adapter fails before any API call because of
  Python runtime compatibility, CLI drift, import failure, or argument parsing;
  this is a Hermes tool/runtime bug and the pilot remains partial and paused.
- an existing remote source branch already points to the exact approved SHA; a
  retry must not push again.

## Readiness Interpretation

`ready_for_local_autonomy` is governed by the local evidence loop. It remains
available when the local completion gate and readiness verifier are green.

`ready_for_remote_pr_pilot` is false unless:

- local evidence readiness is green,
- read-only Gitea remote and PR APIs are reachable,
- the current source branch and full HEAD match a passing completion gate,
- the PR evidence packet generator validates the target PR diff,
- controlled remote branch push is explicitly policy-enabled, and
- one PR creation pilot is explicitly policy-enabled.

`ready_to_request_controlled_one_pr_pilot` is a weaker request-readiness state:
it may be true when local evidence, PR packet evidence, read-only remote API
reachability, source branch/head, and target branch safety are aligned. It does
not authorize push or PR creation; it only means Hermes may ask the Operator
for one exact controlled pilot.

`ready_for_merge_autonomy` is false unless a future policy enables merge
authority and branch protection, CI, and PR evidence are all verifiably
satisfied.

CI/check evidence remains false until a PR or check/status record exists and is
read for the exact source HEAD. Workflow/runner starts remain blocked and may
not be used to create CI evidence during the pilot.

## Operator Boundary

The exact approval phrase for a future pilot must name the one branch and one
task. A generic approval to "push", "open PRs", "merge", or "use Gitea" is not
valid policy expansion.

For partial-state retries, the approval must explicitly authorize exactly one
PR creation from the already-persisted remote branch/head to `main`, prohibit
additional push, require the Hermes-owned PR pilot adapter, prohibit `gh pr
create`, prohibit PR updates/comments/status/check mutation, prohibit
workflow/runner starts, and prohibit merge.

Hermes must not ask for that retry until the repaired adapter `--self-check`,
dry-run, and `--create-pr-only --preflight-only` all pass with the same command
shape and the same Python runtime used by Hermes. If any adapter failure occurs,
do not retry PR creation again until those three checks pass immediately before
execution.
