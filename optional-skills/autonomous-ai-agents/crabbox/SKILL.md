---
name: crabbox
description: Run isolated agent and remote test sandboxes safely.
version: 1.2.0
author: Yossi Eliaz (@zozo123), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [coding-agent, sandbox, cloud, delegation, remote-test]
    related_skills: [claude-code, codex, opencode, blackbox, honcho, github-pr-workflow]
    entrypoint: ./scripts/crabbox.sh
prerequisites:
  cli: [bash]
---

# Crabbox Skill

Use the bundled wrapper to delegate autonomous work to an isolated agent sandbox or run an existing checkout on remote compute. The wrapper normalizes a small safety-focused command surface; it does not make every backend capability interchangeable.

## When to Use

- Build, review, or refine a pull request in an isolated cloud microVM.
- Fan out independent coding tasks across repositories.
- Run a dirty checkout or expensive test suite on remote compute.
- Isolate model-generated or third-party code from the local machine.
- Avoid it for read-only planning, issue triage, or local log summarization; use Hermes `delegate_task` workers or the relevant MCP server instead.
- Avoid it when a local coding-agent skill can finish the task cheaply and safely.

For detailed build, review, triage, and remote-run recipes, read [references/orchestration-patterns.md](references/orchestration-patterns.md).

## Prerequisites

- Bash on Linux or macOS. Windows users must run inside WSL2.
- The wrapper at `scripts/crabbox.sh`.
- One supported backend:
  - `islo` (default) for an autonomous agent that returns a PR.
  - `crabbox` from openclaw/crabbox for syncing a checkout and running commands remotely.
- The GitHub CLI for workflows that inspect issues, branches, reviews, or PRs.

Install and authenticate the selected backend before launching work:

```bash
# Agent-capable default
curl -fsSL https://islo.dev/install.sh | sh
islo login

# Remote run/test backend
brew install openclaw/tap/crabbox
export CRABBOX_BACKEND=crabbox
```

Never print backend or GitHub credentials. Pass required secrets through the backend's supported credential or environment mechanism.

## How to Run

Invoke the wrapper through Hermes `terminal` from the skill directory:

```bash
WRAPPER="./scripts/crabbox.sh"
"$WRAPPER" backends
"$WRAPPER" help
```

The wrapper is self-describing:

```bash
"$WRAPPER" skill
```

For islo, inspect the live machine-readable surface before composing an unfamiliar command:

```bash
"$WRAPPER" schema use
```

The crabbox backend has no schema endpoint. Use its per-command `--json`, `providers --json`, and `config show --json` surfaces when machine-readable output is needed.

## Quick Reference

| Wrapper command | islo | crabbox | Purpose |
|---|---|---|---|
| `new NAME ...` | `use NAME ...` | `run --id NAME ...` | Start or reuse work |
| `list` | `ls` | `list` | List boxes |
| `status NAME` | positional name | `--id NAME` | Inspect one box |
| `rm NAME -f` | `rm NAME -f` | `stop --id NAME` | Remove one box |
| `schema CMD` | supported | refused with guidance | Discover flags |
| `skill` | local | local | Print this `SKILL.md` |
| `backends` | local | local | Compare mappings and capabilities |

For crabbox, the wrapper injects `--id NAME` for `new`, `status`, `rm`, `pause`, `resume`, `stop`, `ssh`, `share`, and `ports`. It deliberately does not inject `--id` for `logs`, because crabbox logs are addressed by run ID.

## Procedure

### 1. Choose the backend

Use islo when the requested outcome is a branch or PR produced by an autonomous agent. Use crabbox when the checkout and diff already exist and the requested outcome is command output, test evidence, or a reproduced failure.

```bash
export CRABBOX_BACKEND=islo      # default, agent -> PR
# or
export CRABBOX_BACKEND=crabbox   # checkout -> remote command
```

Do not pass `--agent` or `--task` to crabbox. The wrapper rejects those flags instead of silently starting a non-agent run.

### 2. Inspect and verify

```bash
"$WRAPPER" backends
"$WRAPPER" doctor
"$WRAPPER" status
```

For islo, run `"$WRAPPER" init` once in a repository before the first agent task. Confirm GitHub authentication before any workflow expected to open or update a PR.

### 3. Name work uniquely

Include the repository, issue or PR number, agent, and a timestamp. Record every name created by the current session.

```bash
BOX="build-api-42-claude-$(date +%s)"
```

Never remove a box merely because its name looks familiar. List first and match it against the current session's record.

### 4. Launch work

Agent-to-PR with islo:

```bash
"$WRAPPER" new "$BOX" \
  --source github://ORG/REPO \
  --workdir REPO \
  --agent claude \
  --task "Implement issue #42, run the repository tests, and open a PR."
```

Remote test/run with crabbox:

```bash
export CRABBOX_BACKEND=crabbox
"$WRAPPER" new ci-box -- bash -lc 'npm ci && npm test'
```

The `--` separator is mandatory for remote commands. Prefer one-shot commands with parseable exit codes over interactive shells.

### 5. Monitor the outcome

Use `status` for liveness and machine-readable output where supported. For agent work, treat the expected branch or PR as the completion signal; islo status alone does not prove the agent finished successfully.

For crabbox, capture the run ID when logs are needed. Query logs with that run ID, not the box name.

### 6. Report and clean up

Report the backend, box name, task, relevant command result, test evidence, and PR URL when one exists. If the task times out, report the box name and the exact status or logs command needed to continue.

Before cleanup:

```bash
"$WRAPPER" list
"$WRAPPER" rm "$BOX" -f
```

The wrapper refuses a missing name, wildcard, `all`, `-a`, and `--all`. For crabbox it strips the islo-compatible `-f` because `stop` defines no force flag. Remove only boxes created by the current session.

## Pitfalls

- `crabbox cleanup` is a fleet-wide garbage-collection sweep, not single-box teardown. Use `rm NAME`, which maps to `crabbox stop --id NAME`.
- A crabbox run without `--id` is ephemeral and auto-releases. Create a durable box with `crabbox warmup --slug NAME` or `crabbox run --keep --slug NAME`.
- Crabbox `logs` uses a `run_<hex>` ID and may require a coordinator. Direct-provider runs should capture stdout, stderr, and timing evidence during `run`.
- Crabbox pause/resume support depends on the provider. Surface the provider error; do not assume success.
- Only the workhorse verbs, schema capability, and box addressing are normalized. Other verbs pass through and may differ by backend.
- Backend boxes may not match production dependencies or credentials. Call out skipped integration tests in the PR or report.
- Parallel agents can produce competing branches or merge conflicts. Leave final selection and conflict resolution explicit.
- Native Windows is unsupported; use WSL2.

## Verification

Run the repository's focused test and the wrapper's local checks:

```bash
scripts/run_tests.sh tests/skills/test_crabbox_skill.py
bash -n optional-skills/autonomous-ai-agents/crabbox/scripts/crabbox.sh
optional-skills/autonomous-ai-agents/crabbox/scripts/crabbox.sh help
optional-skills/autonomous-ai-agents/crabbox/scripts/crabbox.sh skill
```

Verify that:

- Skills Hub search and inspect return `official/autonomous-ai-agents/crabbox`.
- The emitted skill text exactly matches `SKILL.md`.
- Stubbed islo and crabbox commands receive the expected verb and box addressing.
- Wildcard cleanup and agent flags on a non-agent backend fail before invoking the backend.
