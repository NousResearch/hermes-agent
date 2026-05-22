#!/usr/bin/env bash
# crabbox.sh — single-file Hermes skill + cloud-sandbox CLI wrapper.
#
# This file is both the skill (run `./crabbox.sh skill` to print the full
# markdown body, frontmatter and all) and the tool Hermes invokes to provision
# and drive sandboxes. One artifact, one path, one install.
#
# Skill loader hint: the frontmatter below is mirrored verbatim inside the
# `cmd_skill` heredoc, so static scanners that read the top of the file *and*
# loaders that execute `./crabbox.sh skill` both get the same metadata.
#
# ---
# name: crabbox
# description: "Delegate work to crabbox.sh — a single-file Hermes skill and CLI wrapper that provisions cloud sandboxes for AI-generated code. Each sandbox is a dedicated, hardware-isolated environment (microVM) running Cursor or Claude on a chosen GitHub repo. Backend-agnostic (default: islo.dev). Use for autonomous build, code review, refinement, parallel planning, issue triage, and log analysis."
# version: 1.0.0
# author: Hermes Agent
# license: MIT
# platforms: [linux, macos]
# metadata:
#   hermes:
#     tags: [Coding-Agent, Sandbox, microVM, Cloud, PR-Automation, Code-Review, Cross-Repo, Cursor, Claude, Delegation, Security-Isolation, Single-File-Skill]
#     related_skills: [claude-code, codex, opencode, blackbox, honcho, github-pr-workflow]
#     entrypoint: ./crabbox.sh
# prerequisites:
#   cli: [crabbox.sh, gh]
# ---

set -euo pipefail

CRABBOX_VERSION="1.0.0"
CRABBOX_BACKEND="${CRABBOX_BACKEND:-islo}"

cmd_skill() {
  cat <<'CRABBOX_SKILL_MARKDOWN'
---
name: crabbox
description: "Delegate work to crabbox.sh — a single-file Hermes skill and CLI wrapper that provisions cloud sandboxes for AI-generated code. Each sandbox is a dedicated, hardware-isolated environment (microVM) running Cursor or Claude on a chosen GitHub repo. Backend-agnostic (default: islo.dev). Use for autonomous build, code review, refinement, parallel planning, issue triage, and log analysis."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Coding-Agent, Sandbox, microVM, Cloud, PR-Automation, Code-Review, Cross-Repo, Cursor, Claude, Delegation, Security-Isolation, Single-File-Skill]
    related_skills: [claude-code, codex, opencode, blackbox, honcho, github-pr-workflow]
    entrypoint: ./crabbox.sh
prerequisites:
  cli: [crabbox.sh, gh]
---

# crabbox.sh — Hermes Orchestration Guide

**crabbox.sh** is a single-file Hermes skill *and* CLI wrapper. Reading the file gives you the skill documentation; executing it provisions sandboxes. One artifact, one path, one install.

Under the hood `crabbox.sh` delegates to a configurable sandbox backend (`CRABBOX_BACKEND`, default `islo`). The backend leases a hardware-isolated sandbox (microVM — *"Isolation | Hardware-level — dedicated microVM"* per the islo docs), clones a chosen GitHub repo into it, optionally starts a `claude` or `cursor` agent against a `--task`, and streams output back. The agent inside the sandbox can read, write, run shells, and open PRs — but it cannot exfiltrate credentials or escape the VM.

**Mental model.** Hermes's `delegate_task` spawns *in-process* children that share the parent's container. `crabbox.sh new` spawns *isolated sandboxes* with their own kernel and an agent inside. Reach for crabbox when the work is heavy enough to deserve its own machine, parallel enough to deserve fan-out, cross-repo enough that one Hermes session shouldn't try, or **untrusted enough** that hardware isolation matters (running model-generated code, evaluating third-party agents, executing diffs from external contributors).

**Platforms.** macOS and Linux only (Windows users: run inside WSL2, which presents as Linux). The wrapper is portable POSIX shell; the backend's CLI talks to its API; the actual sandbox runs on the backend's infrastructure.

## Discovery and invocation

- **Path:** `optional-skills/autonomous-ai-agents/crabbox/crabbox.sh`
- **Self-describe:** `./crabbox.sh skill` prints this whole document, frontmatter included.
- **Help:** `./crabbox.sh help` lists subcommands and global flags.
- **Schema (machine-readable):** `./crabbox.sh schema [COMMAND]` defers to the backend's own schema. **Hermes should call `./crabbox.sh schema <cmd>` before composing any invocation it isn't certain about** — the backend CLI is the source of truth, not this skill.

## Prerequisites

- **Install crabbox.sh:** copy `optional-skills/autonomous-ai-agents/crabbox/crabbox.sh` somewhere on PATH and `chmod +x`. The script has no runtime deps beyond a POSIX shell and the configured backend.
- **Install backend** (default `islo`): `curl -fsSL https://islo.dev/install.sh | sh`, then `islo login` (browser OAuth; tokens go to OS keychain — macOS Keychain, Linux Secret Service; falls back to `~/.islo/auth.json`). To swap backends: `export CRABBOX_BACKEND=<other-cli>`.
- **`gh` CLI:** required for parsing GitHub URLs and polling for PRs (`gh auth status`).
- **Verify:** `./crabbox.sh doctor` (backend health) and `./crabbox.sh status` (auth + config + sandbox state).
- **Project config (one-time per repo):** `./crabbox.sh init` writes a backend config file in cwd with sandbox name, image, sources, tools (rust/cargo/node/etc., detected interactively).

## CLI Surface

| Command | Purpose |
|---------|---------|
| `crabbox.sh skill` | Print the full skill markdown (this document) |
| `crabbox.sh help` | Subcommand index and global flags |
| `crabbox.sh version` | crabbox.sh version + active backend |
| `crabbox.sh login [--tool ...]` | Authenticate or connect an integration (github / anthropic / claude / gitlab / slack) |
| `crabbox.sh logout [--tool ...]` | Clear credentials or disconnect an integration |
| `crabbox.sh init` | Setup wizard — create backend config, detect tools |
| `crabbox.sh add [TOOL]` | Add setup scripts (interactive detection if no tool given) |
| `crabbox.sh new NAME [flags] [-- COMMAND]` | Lease/use a sandbox; shell, exec, or run an agent (the workhorse) |
| `crabbox.sh list` | List sandboxes |
| `crabbox.sh status [NAME]` | Show auth/config status, or detailed sandbox state |
| `crabbox.sh pause NAME` | Snapshot state, free resources |
| `crabbox.sh resume NAME` | Resume a paused sandbox |
| `crabbox.sh stop NAME` | Stop a running sandbox without removing it |
| `crabbox.sh rm NAME -f` | Remove a sandbox (no recycle bin) |
| `crabbox.sh logs NAME [SESSION_ID] [--type agent\|exec\|interactive] [--follow] [--tail N] [--since 1h]` | Investigate logs by mode |
| `crabbox.sh doctor` | Backend system health check |
| `crabbox.sh port-forward NAME PORT` | Forward a local port into the sandbox |
| `crabbox.sh share / shares / unshare` | Manage shareable URLs for sandbox ports |
| `crabbox.sh ssh NAME` | SSH into the sandbox (setup or proxy) |
| `crabbox.sh snapshot ...` | Manage VM snapshots |
| `crabbox.sh schema [COMMAND]` | Machine-readable backend schema (use this from Hermes!) |
| `crabbox.sh update` | Update the backend CLI |

**Global flags** are passed through to the backend unchanged. `--output {table,json,plain}` and `--no-color` are honored by the default `islo` backend. **Always pass `--output json` from Hermes** when consuming output programmatically.

## The `crabbox.sh new` workhorse

`crabbox.sh new` does five jobs depending on flags. Hermes should compose the right form for the task at hand.

```bash
# 1. Open an interactive shell (last resort for Hermes — prefer exec form)
crabbox.sh new my-sandbox

# 2. Exec a one-shot command (preferred for Hermes — exit code is parseable)
crabbox.sh new my-sandbox -- echo 'Hello from sandbox'
crabbox.sh new my-sandbox -- bash -c 'npm install && npm test'

# 3. Provision a sandbox with one or more repos cloned in
crabbox.sh new build-issue-42 \
  --source github://incredibuild/ibguard:main \
  --workdir ibguard

# 4. Multi-repo fan-out (multiple --source repeats)
crabbox.sh new cross-repo-task \
  --source github://incredibuild/api \
  --source github://incredibuild/frontend

# 5. Start an agent against a task (the delegation form)
crabbox.sh new refactor-auth \
  --source github://incredibuild/ibguard \
  --workdir ibguard \
  --agent claude \
  --task "refactor the auth module to use the new token schema; open a PR"
```

**Flags passed through to the backend:** `--agent {claude,cursor}` · `--source github://...[:branch]` (repeatable) · `--task "..."` (requires `--agent`) · `--workdir` (relative to `/workspace` or absolute) · `--image` · `--env`/`-e KEY=VALUE` (repeatable) · `--gateway-profile` · `--snapshot NAME` · `--new-session` · `--session`/`-s NAME` · `--list-sessions` · `--run-as-user`.

**Trailing args after `--`** are the command to run inside the sandbox (omit for interactive shell).

## Cleanup safety (read first)

- **Never use wildcards** with `crabbox.sh rm`. There is no recycle bin.
- **Never remove sandboxes you didn't create.** Always `crabbox.sh list --output json` first and confirm the name matches a sandbox spawned in the current Hermes session before deleting.
- After every fan-out, only `rm` the exact `$SANDBOX` names this session created. Other Hermes sessions or other users may have parallel sandboxes you'd be destroying.
- For long-lived dev sandboxes use `crabbox.sh pause` / `crabbox.sh resume` (state preserved, resources freed). For ephemeral fan-out work use `crabbox.sh rm -f` after the PR is confirmed.

## Patterns

### Pattern 1 — Build a feature or fix a bug (autonomous PR)

When the user says *"implement issue #42"*, *"fix this bug in repo X"*, or *"add a /health endpoint to these three repos"*:

```bash
ORG=incredibuild; REPO=ibguard; ISSUE_NUM=42
TS=$(date +%s)
SANDBOX="build-${REPO}-${ISSUE_NUM}-cursor-${TS}"
BRANCH_NAME="feat/issue-${ISSUE_NUM}-${TS}"
ISSUE_TITLE=$(gh issue view "$ISSUE_NUM" -R "$ORG/$REPO" --json title -q .title 2>/dev/null || echo "issue-${ISSUE_NUM}")
BUILD_PROMPT="Implement issue #${ISSUE_NUM} (${ISSUE_TITLE}) on branch ${BRANCH_NAME}. Read the issue, explore the codebase, plan briefly, branch from default, implement in the repo's existing style, run tests, commit with explicit file paths, and open a PR with 'Closes #${ISSUE_NUM}'."

# Pre-flight: existing PRs / branches for this issue
gh pr list -R $ORG/$REPO --search "linked:$ORG/$REPO#$ISSUE_NUM" --json number,title,state,url
gh api "repos/$ORG/$REPO/branches" --jq '.[].name' | grep -iE "(^|[^0-9])${ISSUE_NUM}([^0-9]|$)" || true

# Launch
crabbox.sh new "$SANDBOX" \
  --source "github://${ORG}/${REPO}" \
  --workdir "$REPO" \
  --agent cursor \
  --task "$BUILD_PROMPT" \
  --output json
```

The `BUILD_PROMPT` should instruct the sandbox agent to: read the issue (`gh issue view`), explore the codebase (deps, conventions, test patterns), plan briefly, branch from default, implement in the repo's existing style, run tests (`go test`/`npm test`/`pytest`/`cargo test`), commit with explicit file paths (never `git add -A`), open a PR with `Closes #N`.

**Dual mode** (long PRs, contested designs): launch two sandboxes per task in parallel — one `--agent cursor`, one `--agent claude`. First PR wins; the second adds its perspective as a review comment. Trigger when: PR > 500 lines, > 15 files, > 10 commits, or user requests it.

**Multi-repo builds:** one sandbox per repo via `&` and `wait`, each PR cross-referencing the others. Or one sandbox with multiple `--source` flags if the agent should reason across repos in a single VM.

**Polling for completion** (no reliable agent-finished signal — use the PR as the signal):

```bash
TIMEOUT=$(($(date +%s) + 1800))
DELAY=15           # start small, grow toward 60s
MAX_DELAY=60
while [ $(date +%s) -lt $TIMEOUT ]; do
  PR=$(gh pr list -R $ORG/$REPO --head "$BRANCH_NAME" --json url -q '.[0].url' 2>/dev/null)
  [ -n "$PR" ] && { echo "PR: $PR"; break; }

  # Bail out fast if the sandbox itself died.
  # Status values (default backend): starting|running|paused|stopping|stopped|failed|deleted.
  SANDBOX_STATUS=$(crabbox.sh status "$SANDBOX" --output json 2>/dev/null | jq -r '.status // empty')
  case "$SANDBOX_STATUS" in
    failed|stopped|deleted)
      echo "sandbox $SANDBOX reported status=$SANDBOX_STATUS; aborting poll"; break ;;
  esac

  # +/-20% jitter so parallel pollers don't synchronize.
  JITTER=$(( (RANDOM % (DELAY * 2 / 5 + 1)) - (DELAY / 5) ))
  SLEEP=$(( DELAY + JITTER ))
  [ $SLEEP -lt 1 ] && SLEEP=1
  sleep $SLEEP

  # Exponential-ish backoff capped at MAX_DELAY.
  DELAY=$(( DELAY * 2 ))
  [ $DELAY -gt $MAX_DELAY ] && DELAY=$MAX_DELAY
done
```

After the PR is confirmed: `crabbox.sh rm "$SANDBOX" -f`.

### Pattern 2 — Review a PR

Pasted PR URL → spin a sandbox per PR with the review process injected as `--task`. Sandbox agent: gathers PR context (`gh pr view`/`gh pr diff`/prior reviews/sibling repos), cleans up prior crabbox review artifacts (scorecard + inline comments + dismissed reviews matched by markers), reviews from multiple engineer perspectives (backend, frontend if .tsx/.jsx/.vue/CSS, security, devops, product, cross-repo), runs tests if possible, posts inline-first review (max 10 comments, severity-prefixed, prioritized critical > warning > suggestion > question > nitpick > praise), then a single scorecard comment, then a formal review action mapped from score: `≥4.0 APPROVE` / `3.5–3.9 COMMENT` / `<3.5 REQUEST_CHANGES`. Draft PRs always COMMENT only. Large PRs (>2000 lines or >30 files): focus on highest-risk files, skip nitpick/praise, note "Partial review — focused on highest-risk files" in scorecard.

### Pattern 3 — Refine (address PR review comments)

```bash
SANDBOX="refine-${REPO}-${PR_NUM}-cursor-${TS}"
crabbox.sh new "$SANDBOX" \
  --source "github://${ORG}/${REPO}" --workdir "$REPO" \
  --agent cursor \
  --task "Read PR #${PR_NUM} on ${ORG}/${REPO}. Fetch all unresolved review comments via 'gh api repos/$ORG/$REPO/pulls/$PR_NUM/comments'. Address each in a focused commit. Push to the same branch. Reply on each thread with a one-line summary. Do not merge. Do not resolve threads — leave that for the reviewer."
```

### Pattern 4 — Parallel Planner (runs locally, not in a sandbox)

Planning is read-only — use Hermes `delegate_task` workers, not `crabbox.sh new`. Spawn 5 parallel children:

| Worker | Goal | Sources |
|--------|------|---------|
| Requirements Analyst | What needs to be built | Issue body, Linear, Slack threads |
| Architecture Analyst | Where/how changes go | Codebase, dep graph, existing patterns |
| Risk & Dependencies | What could go wrong | Cross-repo search, CI configs, security surface |
| Prior Art | What's been tried before | PRs, issues, git history, abandoned branches |
| Operations | Deploy, monitor, runbook | CI/CD, Dockerfiles, monitoring, infra-as-code |

Merge into `./plans/${PLAN_ID}.md`: executive summary, requirements (P0/P1/P2/out-of-scope), architecture (current/target/files-to-modify), implementation phases (T-shirt sized), risk register, decision points, prior art, operations checklist, Linear-ready task table.

Depth: `quick` (~30s) / `deep` (~2min, default) / `exhaustive` (~5min, includes cross-repo cloning). Total time = slowest worker (parallel).

End by suggesting concrete next steps — Pattern 1 (Build) for the first phase, creating Linear issues from the breakdown, or resolving the Open Questions table.

### Pattern 5 — Triage Linear issues (read-only)

Pull issues via the Linear MCP server (`hermes mcp add linear-server` if not present) → enrich from Slack via the Slack MCP if Slack URLs are in the body → cross-reference codebase via up to 3 parallel `delegate_task` workers (one per repo group) using `git log --grep="ISL-NNN"`, `gh pr list --search "ISL-NNN"`, and reading source files. Classify: FIXED / STALE / ACTIVE / NEEDS_INFO / BLOCKED / IN_PROGRESS, with confidence HIGH/MEDIUM/LOW, and a suggested action (Close / Prioritize / Assign / Investigate / Design / Defer). Output a triage report grouped by recommended action with concrete code evidence (file paths, line numbers, commit hashes). **No writes to Linear, GitHub, Slack, or filesystem.**

### Pattern 6 — Log analysis (local first)

Logs are best read locally — fetch via `terminal()` (gcloud, `gh run view`, `kubectl logs`, Logfire MCP) and summarize: error patterns, time clusters, affected users/services, suspected root cause, recommended next step. If the root cause points to a code fix, hand off to Pattern 1 (Build). If it points to a Linear issue, hand off to Pattern 5 (Triage). For an *agent* sandbox's logs, use `crabbox.sh logs SANDBOX --type agent --follow`; for exec stream, `--type exec`. Don't conflate the read and the write.

### Pattern 7 — Skill autoresearch

Search the local install (`ls ~/.hermes/skills`, `ls $HERMES_INSTALL/optional-skills`), the [Skills Hub](https://agentskills.io) (Hermes is compatible with the agentskills.io standard), and the Hermes repo (`gh search code "..." -R NousResearch/hermes-agent --filename SKILL.md`). Report which skills exist (with paths/links), the gap, and a recommendation: install existing, port from another agent platform, or write new. If write, propose a `feat(skills): add X` PR and offer to scaffold against this skill's single-file conventions (frontmatter at top in shell comments, mirrored in a `cmd_skill` heredoc).

## Sandbox best practices

- **Unique sandbox names** with a timestamp suffix (`${prefix}-${repo}-${id}-${agent}-${ts}`) so parallel work doesn't collide.
- **`crabbox.sh init` once per project** to drop the backend config — the setup wizard interferes with `--task` if no config exists.
- **`-- COMMAND` for one-shots**, interactive shell only when truly interactive (debugging). Hermes can parse `--output json` exit codes from exec form; PTY shells are noise to parse.
- **`--env` over `--gateway-profile` for casual env vars**; reserve `--gateway-profile` for repeatable network/credential policy.
- **`--snapshot NAME`** to restore from a known-good VM state (faster cold start, shared base image across runs).
- **`crabbox.sh pause` between bursts of work**, not `crabbox.sh stop` — pause snapshots and frees resources, resume is fast.
- **`crabbox.sh logs --type exec --since 1h --output json`** for post-hoc debugging without polluting the orchestrator's context.
- **Sandbox test environments** lack production parity — integration tests needing real DBs, external APIs, or org-private credentials will skip; the sandbox agent should call this out in the PR description.
- **Merge conflicts** between parallel build sandboxes targeting the same repo — leave for the human reviewer; don't auto-resolve from the orchestrator.

## Reporting

After every fan-out, Hermes reports back:

- Sandbox name(s) and a link to the backend's web dashboard (default: `https://app.islo.dev`). The exact per-sandbox deep-link path isn't part of the wrapper contract — report the dashboard root plus the sandbox name and let the user navigate.
- What each sandbox is working on
- PR URL(s) once available — or sandbox names + the `crabbox.sh logs SANDBOX --type agent --tail 50` hint if the timeout fires

For long fan-outs from the messaging gateway, schedule a notification on PR creation and let the user keep their TUI free.

## Integration credentials

`crabbox.sh login --tool {github,gitlab,slack,claude,anthropic}` connects an integration via OAuth — credentials are injected into sandboxes whose gateway-profile permits them. Never echo these tokens in Hermes output. The sandbox agent uses `gh` (auto-authed via the integration) for GitHub work; mirror this for GitLab/Slack/etc.

## Known limitations

1. **No reliable completion API** — poll for the PR on the expected branch; `crabbox.sh status` is for liveness, not done-ness; `crabbox.sh logs --type agent --follow` for liveness signal.
2. **`--agent claude` doesn't pin Opus vs Sonnet** — controlled by Anthropic integration settings, not the CLI flag.
3. **Sandbox lacks production parity** — flag skipped integration tests in PRs.
4. **No backend config in cwd** races the setup wizard with `--task` startup. Fix: `crabbox.sh init` first.
5. **Wildcard `crabbox.sh rm`** has no safety net. Always `crabbox.sh list --output json` first; remove only names this session created.
6. **Cross-org `gh` search** sees only what the auth token can access — private repos in other orgs won't appear.
7. **Large plans spanning 5+ repos** lose fidelity in one mega-plan; run separate plans per repo and merge phase dependencies manually.
8. **Native Windows is unsupported** — use WSL2.

## Why single-file?

- **One install path** — copy `crabbox.sh` and you have the skill, the CLI, and the docs.
- **One discovery path** — Hermes finds the script at `optional-skills/autonomous-ai-agents/crabbox/crabbox.sh`; loaders that prefer SKILL.md can `./crabbox.sh skill > SKILL.md` to materialize one.
- **Backend-agnostic** — switch the underlying sandbox by setting `CRABBOX_BACKEND` (default `islo`). Any CLI exposing `use|ls|status|logs|rm|schema|doctor` is a drop-in backend.
- **Auditable** — the wrapper is a few dozen lines of POSIX shell. Reviewable in one screen, no opaque transport layer.

## Related

- `claude-code` skill — delegate to Claude Code CLI on the local machine (no cloud sandbox)
- `codex` skill — same, for OpenAI Codex
- `blackbox`, `honcho` — sibling delegation skills under `optional-skills/autonomous-ai-agents/`
- `github-pr-workflow` skill — PR mechanics that don't need a sandbox
- Hermes `delegate_task` tool — in-process subagent fan-out (read-only investigation, planning)

crabbox.sh's value over local-CLI delegation is *parallel cloud execution in hardware-isolated microVMs with PRs as the return value*. Use local-CLI delegation when work is small, trusted, or read-only; use crabbox when the work is heavy, parallel, cross-repo, or needs microVM-grade hardware isolation (running model-generated code, evaluating untrusted contributors, executing third-party diffs).

## Self-documentation hook

When in doubt about a flag, **don't guess** — call `crabbox.sh schema <command>` from a `terminal()` and parse the JSON. The backend ships its own schema for AI agents; this skill describes patterns, not flag minutiae.
CRABBOX_SKILL_MARKDOWN
}

ensure_backend() {
  if ! command -v "$CRABBOX_BACKEND" >/dev/null 2>&1; then
    cat >&2 <<EOF
crabbox.sh: backend '$CRABBOX_BACKEND' not found on PATH.

Install the default backend with:
  curl -fsSL https://islo.dev/install.sh | sh
  islo login

Or set CRABBOX_BACKEND to a different sandbox CLI that exposes the
'use|ls|status|logs|rm|schema|doctor' verbs.
EOF
    exit 127
  fi
}

# Verb mapping: crabbox.sh -> backend CLI. Anything not remapped here is
# forwarded verbatim, so backend-specific verbs (snapshot, ssh, share, etc.)
# pass through without crabbox.sh needing to know about them.
forward() {
  ensure_backend
  exec "$CRABBOX_BACKEND" "$@"
}

usage() {
  cat <<EOF
crabbox.sh v$CRABBOX_VERSION — single-file Hermes skill + sandbox wrapper

Usage:
  crabbox.sh skill                            Print the full skill markdown (frontmatter + 7 patterns)
  crabbox.sh new NAME [flags] [-- CMD]        Lease a sandbox (backend: $CRABBOX_BACKEND use)
  crabbox.sh list [flags]                     List sandboxes
  crabbox.sh status [NAME]                    Show sandbox / auth status
  crabbox.sh logs NAME [flags]                Stream sandbox logs
  crabbox.sh rm NAME -f                       Remove a sandbox (no recycle bin)
  crabbox.sh schema [COMMAND]                 Print backend command schema (JSON)
  crabbox.sh doctor                           Backend system health check
  crabbox.sh pause|resume|stop NAME           Sandbox lifecycle
  crabbox.sh login|logout [--tool ...]        Auth + integration management
  crabbox.sh init|add [TOOL]                  Project setup
  crabbox.sh port-forward|ssh|share|snapshot  Passthrough to backend
  crabbox.sh update                           Update the backend CLI
  crabbox.sh version                          Print crabbox + backend version
  crabbox.sh help                             This message

Environment:
  CRABBOX_BACKEND    sandbox CLI to wrap (default: islo). Anything exposing
                     'use|ls|status|logs|rm|schema|doctor' is a drop-in backend.

The skill documentation lives inside this file. Run 'crabbox.sh skill' to read
the seven patterns Hermes uses to delegate Build / Review / Refine / Plan /
Triage / Log-analysis / Skill-autoresearch work to crabbox sandboxes.
EOF
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  skill|--skill)               cmd_skill ;;
  help|--help|-h|"")            usage ;;
  version|--version)           ensure_backend
                                echo "crabbox.sh $CRABBOX_VERSION (backend: $CRABBOX_BACKEND $("$CRABBOX_BACKEND" --version 2>/dev/null | head -1))" ;;
  new)                          forward use "$@" ;;
  list)                         forward ls "$@" ;;
  *)                            forward "$cmd" "$@" ;;
esac
