# crypto_bot Remote Lifecycle Contract

Schema roots:

- `hermes.autonomy.crypto_bot_remote_readiness.v1`
- `hermes.autonomy.crypto_bot_pr_evidence.v1`
- `hermes.autonomy.crypto_bot_gitea_pr_pilot.v1`
- `hermes.autonomy.crypto_bot_merge_readiness.v1`

This contract extends the branch-local completion evidence loop without
weakening it. Hermes may not create a remote branch, create a PR, update PR
metadata, or consider merge readiness unless the local completion gate for the
exact task, branch, base ref, and full HEAD has already returned `PASS`.

## Authority Stages

1. Read-only remote/CI discovery is allowed.
2. PR evidence packet generation is allowed.
3. Controlled remote branch push is escalation-required until explicitly enabled
   by policy.
4. One PR creation pilot is escalation-required until explicitly enabled by
   policy and must use the Hermes-owned Gitea PR pilot adapter. Hermes must not
   use `gh pr create` for Gitea unless a future tool explicitly verifies that
   the target host supports that path.
5. PR updates, comments, status mutation, and check mutation are
   escalation-required.
6. Merge-to-main and any protected branch mutation are escalation-required and
   disabled by default.

## Controlled Gitea PR Pilot

The only supported controlled PR creation path for the current local Gitea
pilot is:

`/Users/preston/.hermes/hermes-agent/tools/crypto_bot_gitea_pr_pilot.py`

The adapter consumes an existing PR evidence packet, verifies the passing
completion gate, verifies the local branch/head and clean worktree, verifies
the remote source branch already exists at the exact packet SHA, verifies the
target branch exists, reads open PRs through the local Gitea REST API, and
refuses to proceed if an open PR already exists for the same source-to-target
pair.

Dry-run mode is allowed and must not mutate Gitea. Execute mode remains
Operator-approved only. The approval must name the exact task id, source branch,
source SHA, target branch, and PR evidence packet. Execute mode may create at
most one PR through `POST /api/v1/repos/{owner}/{repo}/pulls` with branch-only
`head` for same-repo branches, `base`, `title`, and `body`. It must not push,
update PR metadata, comment, set checks/statuses, start workflows/runners, or
merge.

PR creation authentication, when required, must be supplied by the Operator as
an environment variable name/value in the execution environment. Hermes may
report whether the named environment variable is present, but must not print,
log, or inspect the token value.

A successful remote branch push does not imply PR creation succeeded. A failed
PR creation attempt leaves the pilot in a partial state: the remote branch may
exist, but no PR evidence exists until a PR is observed through read-only Gitea
PR APIs. A retry after partial state requires fresh Operator approval. If the
remote branch already exists at the exact approved SHA, the retry must not push
again.

If the Gitea PR pilot adapter fails before any API call because of Python
runtime compatibility, CLI drift, import failure, or argument parsing, Hermes
must treat that as a Hermes tool/runtime bug rather than a `crypto_bot` product
failure. Hermes must not retry PR creation until the repaired adapter passes
`--self-check`, dry-run, and `--create-pr-only --preflight-only` with the same
command shape and the same Python runtime used by Hermes. After any adapter
failure, Hermes must not retry PR creation again until those checks pass
immediately before execution.
The retry still requires fresh exact Operator approval and does not grant merge
authority, workflow/runner starts, PR updates/comments/status/check mutation, or
any additional push when the remote branch is already exact.

## Required Remote Preconditions

- Local completion gate remains mandatory before remote integration.
- A PR may be created only from a non-protected source branch with a passing
  completion gate.
- The source branch must exactly match the gate's target branch.
- The source full SHA must exactly match the gate's target full HEAD.
- The target merge base must be recorded.
- Changed files from the target merge base must match the completion-gate
  changed files.
- Changed files from the completion gate base must match the completion-gate
  report.
- No direct push to protected branches is allowed.
- No force-push, rebase, squash, amend, cherry-pick, or branch rewrite is
  allowed by this contract.

## PR Evidence Packet

Before a PR exists, Hermes must generate a packet under:

`/Users/preston/.local/state/hermes-operator/pr-evidence/`

The packet and proposed PR body must include:

- task source
- repository path
- target branch and base ref
- source branch and full commit SHA
- target merge base
- changed files
- completion gate JSON path
- Codex sidecar result path
- validators and exit-code evidence
- blocked-surface proof
- proof that the packet did not push, create a PR, run CI, start runners,
  deploy, or merge

Generated PR body text must be scanned for secret-looking content before it is
written. A secret-looking token, credential assignment, private key marker, or
credential-bearing path blocks the packet.

## CI and Check Evidence

CI/check evidence is read-only evidence. Hermes may read workflow files, PR
state, commit status, check state, and actions metadata when available without
credential inspection or mutation.

A PR is not merge-ready if:

- CI is missing.
- CI is stale or for a different source commit.
- CI failed.
- CI is pending indefinitely.
- Gitea check/status APIs are unavailable.
- runner evidence is absent or unreadable.
- workflow evidence is absent.
- the PR branch or HEAD does not match the completion gate.

Workflow files are controlled surfaces. Hermes may read `.gitea/workflows` as
text but must not edit workflows, run workflows, start runners, or use workflow
execution as a side channel for escalation.

CI evidence remains false until a PR/check exists and the relevant check or
status record is read for the exact source HEAD. A remote branch by itself is
not CI evidence.

## Merge Gate

PR creation is separate from merge. Merge-to-main requires a separate merge
readiness gate and explicit future policy enabling. The default state is:

- `merge_authority_enabled: false`
- `merge_ready: false`
- `ready_for_merge_autonomy: false`

The merge gate must verify local gate `PASS`, PR branch/head alignment, current
CI/check success for the source HEAD, branch protection satisfaction, and
explicit merge authority. It must never perform a merge.

## Blocked Surfaces

This remote lifecycle layer does not permit:

- secrets, `.env`, token files, Keychain material, private keys, cookies, or
  credential stores
- broker, Robinhood, exchange, live-market, account, order, position, wallet,
  trading, or financial APIs
- runtime service starts, app servers, workers, schedulers, launchd, qmd,
  Docker builds, Kubernetes, Flux, Harbor, OpenBao, RabbitMQ, Redis, Temporal,
  production services, workflows, or runners
- deploys or GitOps promotion
- Gitea write APIs, issue/PR/comment/label/project/check/status mutation,
  webhook mutation, repository mutation, or branch protection mutation unless a
  future policy explicitly enables the specific authority

## Readiness States

Remote lifecycle readiness is not one boolean:

- `local_evidence_ready`: local branch-local evidence loop is green.
- `remote_readiness_ready`: read-only remote, Gitea, and PR APIs are reachable.
- `pr_evidence_ready`: a proposed PR evidence packet validates locally.
- `ready_for_pr_evidence_packet`: equivalent to `pr_evidence_ready`; the local
  packet is clean and can be used in a later approval request.
- `ci_evidence_ready`: current CI/check evidence for the source HEAD is present
  and passing; this normally remains false before a remote branch, PR, and
  check/status evidence exist.
- `merge_readiness_ready`: merge authority and merge evidence are both
  available.
- `ready_for_local_autonomy`: local task work may continue.
- `ready_for_remote_pr_pilot`: remote branch push and one PR pilot are
  explicitly policy-enabled and evidence is aligned.
- `ready_to_request_controlled_one_pr_pilot`: local gate, PR packet, source
  branch/head, and read-only remote evidence are aligned enough to ask the
  Operator for one exact pilot. This state does not itself authorize push or PR
  creation.
- `ready_for_merge_autonomy`: merge authority is explicitly enabled and all
  merge gates are satisfied.

`ready_for_local_autonomy` may be true while every remote, PR, CI, and merge
readiness state is false. `ready_for_merge_autonomy` remains false without
current CI/check evidence and explicit merge authority.
