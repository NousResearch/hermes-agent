# crypto_bot Hooks Policy

## Purpose

Hooks are a mechanical guardrail layer for crypto_bot autonomy. They support the
Hermes-native `/goal`, Kanban, and worker-lane loop by enforcing project policy
before tools run and by redacting sensitive output before it reaches model
context.

Hooks begin in audit/dry-run mode. Blocking mode requires explicit validation
and an explicit config change.

## pre_tool_call Blocking Strategy

The guard blocks dangerous terminal and file operations unless a future explicit
mode and approval artifact authorizes the exact surface.

Blocked command families:

- `ruff format`
- `git push`
- `gh pr`, including `gh pr create`
- direct Gitea write calls such as POST, PUT, PATCH, or DELETE to `/api/v1`
- workflow starts, workflow dispatch, runner starts, and runner registration
- edits to `.gitea/workflows`
- merge, rebase, reset, amend, squash, cherry-pick, force-delete, and branch
  rewrite commands
- service starts, launchd starts, Docker/Kubernetes/Flux/Harbor/OpenBao,
  RabbitMQ, Redis, Temporal, qmd, app servers, schedulers, and crypto_bot loops
- `apply_approved_write_plan.py`, `branch_local_writer.py`, and
  `execute_forge_issue_create.py`
- secret reads from `.env`, token files, credential stores, cookies, Keychain
  material, private keys, OAuth files, runtime databases, and
  credential-bearing logs
- Robinhood, broker, exchange, live-market, account, order, position, wallet,
  trading, and financial API commands
- deploy, release, GitOps promotion, and production-service mutation

Read-only Hermes upgrade operations and read-only Gitea GET probes are outside
the product-work block only when the command shape is read-only and the session
scope explicitly authorizes them.

## pre_llm_call Context Strategy

`pre_llm_call` may inject a concise current-state warning:

- crypto_bot dirty worktree status
- local readiness status and blockers
- active Kanban board and card
- blocked remote authority status
- reminder that completion gate PASS is required before done

This hook should be short and deterministic so it does not replace project
docs, readiness tools, or gate output.

## Output Redaction

`transform_terminal_output` and `transform_tool_result` redact sensitive values
from terminal and non-terminal tool results:

- API keys and bearer tokens
- Telegram tokens
- GitHub/Gitea tokens
- OAuth tokens
- private-key blocks
- cookie/header credential values
- known secret-like environment assignments

Redaction never marks a command safe; it only limits blast radius if output
contains sensitive material.

## Compatibility

- Plugin hooks are preferred for CLI and gateway compatibility.
- Shell hooks can be used as a fallback through `~/.hermes/config.yaml`.
- Gateway event hooks are not sufficient for tool gating because they run at
  gateway lifecycle boundaries rather than tool-call boundaries.
- Blocking mode must be tested with `hermes hooks test` or plugin-level unit
  tests before live enablement.

## Live Enablement

The source guard lives under `/Users/preston/.hermes/hermes-agent/hooks/crypto_bot_policy_guard`.
It defaults to observe mode. It can block in tests by setting explicit test
mode, but this session does not install or enable a live blocking hook.
