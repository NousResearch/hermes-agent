#!/usr/bin/env bash
# crabbox.sh — single-file Hermes skill + cloud-sandbox CLI wrapper.
#
# This file is both the skill (run `./crabbox.sh skill` to print the full
# markdown body, frontmatter and all) and the tool Hermes invokes to provision
# and drive sandboxes. One artifact, one path, one install.
#
# It is backend-agnostic: `CRABBOX_BACKEND` selects which sandbox CLI to drive.
# Two backends are wired in by name (their verb spellings and capabilities
# differ, so the wrapper normalizes them in load_backend):
#   * islo     — agent-capable cloud microVMs (default). Provision → run an
#                agent against --task → get a PR back.   https://islo.dev
#   * crabbox  — openclaw/crabbox, "warm a box, sync the diff, run the suite":
#                rsync your dirty checkout to a leased box and run a command.
#                Remote edit-save-run; no autonomous agent.  https://crabbox.sh
#
# Skill loader hint: the frontmatter below is mirrored verbatim inside the
# `cmd_skill` heredoc, so static scanners that read the top of the file *and*
# loaders that execute `./crabbox.sh skill` both get the same metadata.
#
# ---
# name: crabbox
# description: "Delegate work to crabbox.sh — a single-file Hermes skill and CLI wrapper that drives a cloud sandbox backend for AI work. Backend-agnostic (CRABBOX_BACKEND): the default `islo` backend provisions hardware-isolated microVMs that run Cursor or Claude on a GitHub repo and return a PR; the `crabbox` backend (openclaw/crabbox) syncs your dirty checkout to a leased box and runs the suite remotely. Use for autonomous build, code review, refinement, parallel planning, issue triage, log analysis, and remote test/run."
# version: 1.1.0
# author: Hermes Agent
# license: MIT
# platforms: [linux, macos]
# metadata:
#   hermes:
#     tags: [Coding-Agent, Sandbox, microVM, Cloud, PR-Automation, Code-Review, Cross-Repo, Cursor, Claude, Delegation, Security-Isolation, Single-File-Skill, Remote-Test, Backend-Agnostic]
#     related_skills: [claude-code, codex, opencode, blackbox, honcho, github-pr-workflow]
#     entrypoint: ./crabbox.sh
# prerequisites:
#   cli: [crabbox.sh, gh]
# ---

set -euo pipefail

CRABBOX_VERSION="1.1.0"
CRABBOX_BACKEND="${CRABBOX_BACKEND:-islo}"

cmd_skill() {
  cat <<'CRABBOX_SKILL_MARKDOWN'
---
name: crabbox
description: "Delegate work to crabbox.sh — a single-file Hermes skill and CLI wrapper that drives a cloud sandbox backend for AI work. Backend-agnostic (CRABBOX_BACKEND): the default `islo` backend provisions hardware-isolated microVMs that run Cursor or Claude on a GitHub repo and return a PR; the `crabbox` backend (openclaw/crabbox) syncs your dirty checkout to a leased box and runs the suite remotely. Use for autonomous build, code review, refinement, parallel planning, issue triage, log analysis, and remote test/run."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Coding-Agent, Sandbox, microVM, Cloud, PR-Automation, Code-Review, Cross-Repo, Cursor, Claude, Delegation, Security-Isolation, Single-File-Skill, Remote-Test, Backend-Agnostic]
    related_skills: [claude-code, codex, opencode, blackbox, honcho, github-pr-workflow]
    entrypoint: ./crabbox.sh
prerequisites:
  cli: [crabbox.sh, gh]
---

# crabbox.sh — Hermes Orchestration Guide

**crabbox.sh** is a single-file Hermes skill *and* CLI wrapper. Reading the file gives you the skill documentation; executing it drives a cloud sandbox backend. One artifact, one path, one install.

`crabbox.sh` itself does no networking — it normalizes a small set of verbs and forwards to the backend selected by `CRABBOX_BACKEND` (default `islo`). Different backends spell their verbs differently and have different capabilities, so the wrapper maps the workhorse verbs (`new`/`list`/`rm`) and gates capability-specific flags. **Run `./crabbox.sh backends` to see what's wired in and `./crabbox.sh schema <cmd>` for the live flag surface.**

## Backends

| Backend | Paradigm | `new`→ | agent→PR? | `schema`? | Install |
|---------|----------|--------|-----------|-----------|---------|
| **islo** (default) | Provision a hardware-isolated microVM, clone a repo, run `claude`/`cursor` against `--task`, **return a PR**. | `use` | ✅ yes | ✅ yes | `curl -fsSL https://islo.dev/install.sh \| sh` then `islo login` |
| **crabbox** ([openclaw/crabbox](https://github.com/openclaw/crabbox)) | "Warm a box, sync the diff, run the suite": rsync your dirty checkout to a leased box, run a command remotely, stream output, release. **No autonomous agent.** | `run` | ❌ no | ❌ no (falls back to `--help`) | `brew install openclaw/tap/crabbox` ([crabbox.sh](https://crabbox.sh)) |

**Pick by the work, not by the brand.** Use **islo** when you want a machine to go off and *produce a PR on its own* (Patterns 1–3). Use **openclaw/crabbox** when *you* hold the diff and want cloud-grade compute for the edit-save-run loop — running a heavy suite, reproducing a flake on a "beast" box, or testing a PR with `--fresh-pr` (Pattern 8). Patterns that pass `--agent`/`--task` require an **agent-capable** backend; crabbox.sh refuses those flags on a non-agent backend with a clear message instead of silently producing nothing.

**Other backends.** Any CLI works if you teach `load_backend` its verb names: add a `case` arm declaring its `new`/`list`/`rm` spellings and whether it has agent/schema support. Unknown backends are assumed islo-style.

**Mental model.** Hermes's `delegate_task` spawns *in-process* children that share the parent's container. `crabbox.sh new` (islo backend) spawns *isolated sandboxes* with their own kernel and an agent inside. Reach for an islo sandbox when work is heavy enough to deserve its own machine, parallel enough to deserve fan-out, cross-repo enough that one Hermes session shouldn't try, or **untrusted enough** that hardware isolation matters (running model-generated code, evaluating third-party agents, executing diffs from external contributors). Reach for crabbox when you already have the change and just need a bigger/cleaner box to run it on.

**Platforms.** macOS and Linux (Windows users: run inside WSL2). The wrapper is portable shell; the backend's CLI talks to its own service.

## Discovery and invocation

- **Path:** `optional-skills/autonomous-ai-agents/crabbox/crabbox.sh`
- **Self-describe:** `./crabbox.sh skill` prints this whole document, frontmatter included.
- **Backends:** `./crabbox.sh backends` lists wired-in backends, their verb maps, and capabilities.
- **Help:** `./crabbox.sh help` lists subcommands and global flags.
- **Schema (machine-readable):** `./crabbox.sh schema [COMMAND]` defers to the backend's own schema (islo). Backends without a schema verb (crabbox) fall back to `<command> --help`. **Hermes should call `./crabbox.sh schema <cmd>` before composing any invocation it isn't certain about** — the backend CLI is the source of truth, not this skill.

## Prerequisites

- **Install crabbox.sh:** copy `optional-skills/autonomous-ai-agents/crabbox/crabbox.sh` somewhere on PATH and `chmod +x`. No runtime deps beyond a shell and the configured backend.
- **Install a backend:** see the table above. Default `islo`: `curl -fsSL https://islo.dev/install.sh | sh`, then `islo login` (browser OAuth; tokens go to the OS keychain, falling back to `~/.islo/auth.json`). Swap with `export CRABBOX_BACKEND=crabbox`.
- **`gh` CLI:** required for parsing GitHub URLs and polling for PRs (`gh auth status`).
- **Verify:** `./crabbox.sh doctor` (backend health) and `./crabbox.sh status` (auth + config + sandbox state).
- **Project config (one-time per repo):** `./crabbox.sh init`.

## CLI Surface

| Command | Purpose |
|---------|---------|
| `crabbox.sh skill` | Print the full skill markdown (this document) |
| `crabbox.sh backends` | List wired-in backends, verb maps, and capabilities |
| `crabbox.sh help` | Subcommand index and global flags |
| `crabbox.sh version` | crabbox.sh version + active backend |
| `crabbox.sh login [--tool ...]` | Authenticate or connect an integration |
| `crabbox.sh logout [--tool ...]` | Clear credentials or disconnect an integration |
| `crabbox.sh init` | Setup wizard — create backend config, detect tools |
| `crabbox.sh add [TOOL]` | Add setup scripts (interactive detection if no tool given) |
| `crabbox.sh new NAME [flags] [-- COMMAND]` | Lease/use a box; shell, exec, or (islo) run an agent (the workhorse) |
| `crabbox.sh list` | List boxes/sandboxes |
| `crabbox.sh status [NAME]` | Show auth/config status, or detailed box state |
| `crabbox.sh pause NAME` | Snapshot state, free resources |
| `crabbox.sh resume NAME` | Resume a paused box |
| `crabbox.sh stop NAME` | Stop a running box without removing it |
| `crabbox.sh rm NAME -f` | Remove a box (no recycle bin; wildcard-guarded) |
| `crabbox.sh logs NAME [...]` | Investigate logs by mode |
| `crabbox.sh doctor` | Backend system health check |
| `crabbox.sh ssh NAME` | SSH into the box |
| `crabbox.sh share / ports / snapshot ...` | Passthrough to backend (verb spelling is backend-specific) |
| `crabbox.sh schema [COMMAND]` | Machine-readable backend schema, or `--help` fallback |
| `crabbox.sh update` | Update the backend CLI |

`crabbox.sh` normalizes only the three workhorse verbs (`new`/`list`/`rm`) and the `schema` capability across backends; everything else forwards verbatim, so a verb a given backend doesn't have will surface that backend's own error.

**Global flags** pass through unchanged. `--output {table,json,plain}` is honored by `islo`; crabbox uses `--json`. **From Hermes, prefer machine-readable output** (`--output json` on islo, `--json` on crabbox) when consuming results programmatically.

## The `crabbox.sh new` workhorse

`crabbox.sh new NAME` maps to the backend's lease/run verb (`islo use` / `crabbox run`). What it does depends on the flags and the backend.

```bash
# Exec a one-shot command (preferred for Hermes — exit code is parseable; works on any backend)
crabbox.sh new my-box -- bash -c 'npm install && npm test'

# islo: provision a sandbox with one or more repos cloned in
crabbox.sh new build-issue-42 \
  --source github://incredibuild/ibguard:main \
  --workdir ibguard

# islo: multi-repo fan-out (repeat --source)
crabbox.sh new cross-repo-task \
  --source github://incredibuild/api \
  --source github://incredibuild/frontend

# islo: start an agent against a task (the delegation form — REQUIRES an agent-capable backend)
crabbox.sh new refactor-auth \
  --source github://incredibuild/ibguard --workdir ibguard \
  --agent claude \
  --task "refactor the auth module to use the new token schema; open a PR"
```

**islo flags** passed through: `--agent {claude,cursor}` · `--source github://...[:branch]` (repeatable) · `--task "..."` (requires `--agent`) · `--workdir` · `--image` · `--env`/`-e KEY=VALUE` (repeatable) · `--gateway-profile` · `--snapshot NAME` · `--new-session` · `--session`/`-s NAME` · `--list-sessions` · `--run-as-user`. **`--agent`/`--task` on a non-agent backend are rejected by crabbox.sh** — use Pattern 8 instead.

**Trailing args after `--`** are the command to run inside the box (omit for an interactive shell — a last resort for Hermes; prefer the exec form).

## Cleanup safety (enforced, not just advised)

- **`crabbox.sh rm` refuses wildcards / `--all` / a missing name** and exits non-zero, so a glob can't sweep boxes you didn't create. It maps to the backend's removal verb (`islo rm` / `crabbox cleanup`).
- **Always `crabbox.sh list` first** and confirm the name matches a box spawned in the current Hermes session before deleting. Other sessions/users may have parallel boxes.
- For long-lived dev boxes use `crabbox.sh pause` / `resume` (state preserved, resources freed). For ephemeral fan-out, `crabbox.sh rm NAME -f` after the PR is confirmed.

## Patterns

> Patterns 1–3 (autonomous PR creation) require an **agent-capable** backend (`islo`). Patterns 4–7 run locally via `delegate_task` (no sandbox). Pattern 8 is crabbox-native and needs no agent.

### Pattern 1 — Build a feature or fix a bug (autonomous PR) · *agent backend*

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

# Launch (islo)
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
  # Status values (islo): starting|running|paused|stopping|stopped|failed|deleted.
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

### Pattern 2 — Review a PR · *agent backend*

Pasted PR URL → spin a sandbox per PR with the review process injected as `--task`. Sandbox agent: gathers PR context (`gh pr view`/`gh pr diff`/prior reviews/sibling repos), cleans up prior crabbox review artifacts (scorecard + inline comments + dismissed reviews matched by markers), reviews from multiple engineer perspectives (backend, frontend if .tsx/.jsx/.vue/CSS, security, devops, product, cross-repo), runs tests if possible, posts inline-first review (max 10 comments, severity-prefixed, prioritized critical > warning > suggestion > question > nitpick > praise), then a single scorecard comment, then a formal review action mapped from score: `≥4.0 APPROVE` / `3.5–3.9 COMMENT` / `<3.5 REQUEST_CHANGES`. Draft PRs always COMMENT only. Large PRs (>2000 lines or >30 files): focus on highest-risk files, skip nitpick/praise, note "Partial review — focused on highest-risk files" in scorecard.

### Pattern 3 — Refine (address PR review comments) · *agent backend*

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

### Pattern 8 — Remote test/run on a leased box (crabbox-native, no agent)

When *you* already hold the change (a dirty worktree, a fix in progress, a flake to reproduce) and just want cloud-grade compute for the edit-save-run loop, use the `crabbox` backend. It rsyncs your checkout to a leased box and runs the command there, exiting with the remote exit code — nothing is committed and no agent is involved.

```bash
export CRABBOX_BACKEND=crabbox

# Run the suite on a big remote box against your local dirty checkout
crabbox.sh new ci-box -- pnpm test:changed

# Reproduce a flake on a beefy class, capturing JUnit evidence
crabbox.sh new flake-repro -- go test -run TestThatFlakes -count=20 ./...

# Test a PR fresh (crabbox clones the PR for you), then release
crabbox.sh new pr-smoke -- bash -c 'pnpm install --frozen-lockfile && pnpm test:integration'
crabbox.sh list
crabbox.sh rm pr-smoke -f
```

This is complementary to Patterns 1–3: islo sends an agent to *write* a PR; crabbox gives you a remote box to *run* code you already have. When a Build sandbox can't reproduce an environment-specific failure, hand the diff to a crabbox box with production-like tooling.

## Best practices

- **Unique box names** with a timestamp suffix (`${prefix}-${repo}-${id}-${agent}-${ts}`) so parallel work doesn't collide.
- **`crabbox.sh init` once per project** to drop the backend config — the wizard can interfere with `--task` startup if no config exists.
- **`-- COMMAND` for one-shots**, interactive shell only when truly interactive (debugging). Hermes can parse machine-readable exit codes from the exec form; PTY shells are noise to parse.
- **`--env` over `--gateway-profile` for casual env vars** (islo); reserve `--gateway-profile` for repeatable network/credential policy.
- **`--snapshot NAME`** (islo) to restore from a known-good VM state (faster cold start, shared base image).
- **`crabbox.sh pause` between bursts**, not `stop` — pause snapshots and frees resources, resume is fast.
- **Sandbox/box environments lack production parity** — integration tests needing real DBs, external APIs, or org-private credentials will skip; the sandbox agent should call this out in the PR description.
- **Merge conflicts** between parallel build sandboxes targeting the same repo — leave for the human reviewer; don't auto-resolve from the orchestrator.

## Reporting

After every fan-out, Hermes reports back:

- Box name(s) and a link to the backend's web dashboard (islo: `https://app.islo.dev`). The exact per-box deep-link path isn't part of the wrapper contract — report the dashboard root plus the name.
- What each box is working on.
- PR URL(s) once available — or box names + the `crabbox.sh logs NAME --type agent --tail 50` hint if the timeout fires.

For long fan-outs from the messaging gateway, schedule a notification on PR creation and let the user keep their TUI free.

## Integration credentials

`crabbox.sh login --tool {github,gitlab,slack,claude,anthropic}` connects an integration via OAuth — credentials are injected into boxes whose gateway-profile permits them. Never echo these tokens in Hermes output. The sandbox agent uses `gh` (auto-authed via the integration) for GitHub work; mirror this for GitLab/Slack/etc.

## Known limitations

1. **No reliable completion API (islo)** — poll for the PR on the expected branch; `status` is for liveness, not done-ness; `logs --type agent --follow` for a liveness signal.
2. **`--agent claude` doesn't pin Opus vs Sonnet** — controlled by Anthropic integration settings, not the CLI flag.
3. **`--agent`/`--task` need an agent-capable backend** — crabbox.sh rejects them on `crabbox`; switch to `islo` (or another agent backend).
4. **Box environments lack production parity** — flag skipped integration tests in PRs.
5. **No backend config in cwd** races the setup wizard with `--task` startup. Fix: `crabbox.sh init` first.
6. **`crabbox.sh rm`** is wildcard-guarded but still has no recycle bin. Always `crabbox.sh list` first; remove only names this session created.
7. **Cross-org `gh` search** sees only what the auth token can access — private repos in other orgs won't appear.
8. **Only `new`/`list`/`rm`/`schema` are normalized across backends** — other verbs forward verbatim, so a verb a backend lacks surfaces that backend's own error. Run `crabbox.sh backends` / `schema <cmd>` when unsure.
9. **Native Windows is unsupported** — use WSL2.

## Why single-file?

- **One install path** — copy `crabbox.sh` and you have the skill, the CLI, and the docs.
- **One discovery path** — Hermes finds the script at `optional-skills/autonomous-ai-agents/crabbox/crabbox.sh`; loaders that prefer SKILL.md can `./crabbox.sh skill > SKILL.md` to materialize one.
- **Backend-agnostic** — `CRABBOX_BACKEND` selects the sandbox CLI; verb names and capabilities are declared in `load_backend` (islo and openclaw/crabbox ship in the box).
- **Auditable** — the wrapper is a screenful of shell: a backend table, three guards, one dispatch. No opaque transport layer.

## Related

- `claude-code` skill — delegate to Claude Code CLI on the local machine (no cloud sandbox)
- `codex` skill — same, for OpenAI Codex
- `blackbox`, `honcho` — sibling delegation skills under `optional-skills/autonomous-ai-agents/`
- `github-pr-workflow` skill — PR mechanics that don't need a sandbox
- Hermes `delegate_task` tool — in-process subagent fan-out (read-only investigation, planning)

crabbox.sh's value over local-CLI delegation is *cloud execution in isolated boxes* — with islo, parallel agents that return PRs; with openclaw/crabbox, a remote box for your own dirty checkout. Use local-CLI delegation when work is small, trusted, or read-only.

## Self-documentation hook

When in doubt about a flag, **don't guess** — call `crabbox.sh schema <command>` and parse the output. islo ships a machine-readable schema for AI agents; crabbox falls back to `--help`. This skill describes patterns, not flag minutiae.
CRABBOX_SKILL_MARKDOWN
}

# --- Backend registry -------------------------------------------------------
# Normalize the three workhorse verbs and capability flags per backend. Every
# other verb forwards verbatim. To add a backend, append a case arm; unknown
# backends fall back to the islo-style contract.
load_backend() {
  # islo-style defaults
  BK_NEW="use"; BK_LIST="ls"; BK_RM="rm"; BK_AGENT="yes"; BK_SCHEMA="yes"
  BK_INSTALL='curl -fsSL https://islo.dev/install.sh | sh   # then: islo login'
  case "$CRABBOX_BACKEND" in
    islo) : ;;
    crabbox)
      BK_NEW="run"; BK_LIST="list"; BK_RM="cleanup"; BK_AGENT="no"; BK_SCHEMA="no"
      BK_INSTALL='brew install openclaw/tap/crabbox   # docs: https://crabbox.sh' ;;
    *) : ;;  # unknown backend: assume islo-style verbs/capabilities
  esac
}

# Translate a canonical workhorse verb to the active backend's spelling.
to_backend_verb() {
  case "$1" in
    new)  printf '%s' "$BK_NEW" ;;
    list) printf '%s' "$BK_LIST" ;;
    rm)   printf '%s' "$BK_RM" ;;
    *)    printf '%s' "$1" ;;
  esac
}

ensure_backend() {
  if ! command -v "$CRABBOX_BACKEND" >/dev/null 2>&1; then
    cat >&2 <<EOF
crabbox.sh: backend '$CRABBOX_BACKEND' not found on PATH.

Install it:
  $BK_INSTALL

Known backends (set CRABBOX_BACKEND to choose):
  islo      agent-capable cloud microVMs (default)   https://islo.dev
  crabbox   openclaw remote run/test boxes           https://crabbox.sh
Other CLIs work too — declare their verb names in load_backend().
EOF
    exit 127
  fi
}

forward() {
  ensure_backend
  exec "$CRABBOX_BACKEND" "$@"
}

# Refuse --agent/--task on a backend that can't run an autonomous agent.
guard_agent_flags() {
  [ "$BK_AGENT" = yes ] && return 0
  local a
  for a in "$@"; do
    case "$a" in
      --agent|--task|--agent=*|--task=*)
        cat >&2 <<EOF
crabbox.sh: backend '$CRABBOX_BACKEND' does not run autonomous agents.
It syncs your working tree to a remote box and runs a command (edit-save-run
on cloud compute) — there is no --agent/--task -> PR flow.

  * Remote test/run (this backend):   crabbox.sh new my-box -- pnpm test
  * Autonomous agent -> PR:            CRABBOX_BACKEND=islo crabbox.sh new ... --agent claude --task "..."

See 'crabbox.sh skill' Pattern 8 (crabbox-native) vs Patterns 1-3 (agent).
EOF
        exit 2 ;;
    esac
  done
}

# Enforce the cleanup-safety contract the skill documents: no wildcard/bulk
# removes, and a name is required. (-a/--all are matched before generic flags.)
guard_rm() {
  local a have_name=no
  for a in "$@"; do
    case "$a" in
      --all|-a|all)
        echo "crabbox.sh: refusing bulk remove ('$a'). Name exactly one box; run 'crabbox.sh list' first." >&2
        exit 2 ;;
      *'*'*|*'?'*|'~'*)
        echo "crabbox.sh: refusing glob in box name ('$a'). Name exactly one box; run 'crabbox.sh list' first." >&2
        exit 2 ;;
      -*) : ;;            # other flags (e.g. -f) are fine
      *)  have_name=yes ;;
    esac
  done
  if [ "$have_name" = no ]; then
    echo "crabbox.sh: rm needs an explicit box name. Run 'crabbox.sh list' first." >&2
    exit 2
  fi
}

cmd_backends() {
  cat <<EOF
crabbox.sh backends — set CRABBOX_BACKEND to choose (current: $CRABBOX_BACKEND)

  islo     [default] agent-capable cloud microVMs. Provision a sandbox, clone a
           repo, run claude/cursor against --task, return a PR.
           verbs: new->use  list->ls    rm->rm       agent: yes   schema: yes
           install: curl -fsSL https://islo.dev/install.sh | sh && islo login

  crabbox  openclaw/crabbox — "warm a box, sync the diff, run the suite". Rsync
           your dirty checkout to a leased box and run a command. No agent/PR.
           verbs: new->run  list->list  rm->cleanup  agent: no    schema: no
           install: brew install openclaw/tap/crabbox   (https://crabbox.sh)

Patterns needing --agent/--task (Build/Review/Refine) require an agent-capable
backend (islo). Remote test/run (Pattern 8) works on crabbox. Add a backend by
declaring its verb names and capabilities in load_backend().
EOF
}

usage() {
  cat <<EOF
crabbox.sh v$CRABBOX_VERSION — single-file Hermes skill + sandbox wrapper
active backend: $CRABBOX_BACKEND  (new->$(to_backend_verb new), list->$(to_backend_verb list), rm->$(to_backend_verb rm))

Usage:
  crabbox.sh skill                            Print the full skill markdown (frontmatter + patterns)
  crabbox.sh backends                         List wired-in backends + capabilities
  crabbox.sh new NAME [flags] [-- CMD]        Lease a box (backend: $CRABBOX_BACKEND $(to_backend_verb new))
  crabbox.sh list [flags]                     List boxes
  crabbox.sh status [NAME]                    Show box / auth status
  crabbox.sh logs NAME [flags]                Stream box logs
  crabbox.sh rm NAME -f                       Remove a box (wildcard-guarded; no recycle bin)
  crabbox.sh schema [COMMAND]                 Backend schema (JSON), or --help fallback
  crabbox.sh doctor                           Backend system health check
  crabbox.sh pause|resume|stop NAME           Box lifecycle
  crabbox.sh login|logout [--tool ...]        Auth + integration management
  crabbox.sh init|add [TOOL]                  Project setup
  crabbox.sh ssh|share|ports|snapshot ...     Passthrough to backend
  crabbox.sh update                           Update the backend CLI
  crabbox.sh version                          Print crabbox + backend version
  crabbox.sh help                             This message

Environment:
  CRABBOX_BACKEND    sandbox CLI to drive (default: islo; also: crabbox).
                     Verb names + capabilities are declared in load_backend().

Run 'crabbox.sh skill' for the orchestration patterns and 'crabbox.sh backends'
to compare islo (agent->PR) vs openclaw/crabbox (remote run/test).
EOF
}

load_backend

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  skill|--skill)               cmd_skill ;;
  backends)                    cmd_backends ;;
  help|--help|-h|"")           usage ;;
  version|--version)           ensure_backend
                               echo "crabbox.sh $CRABBOX_VERSION (backend: $CRABBOX_BACKEND $("$CRABBOX_BACKEND" --version 2>/dev/null | head -1))" ;;
  new)                         guard_agent_flags "$@"
                               forward "$(to_backend_verb new)" "$@" ;;
  list)                        forward "$(to_backend_verb list)" "$@" ;;
  rm)                          guard_rm "$@"
                               forward "$(to_backend_verb rm)" "$@" ;;
  schema)
    if [ "$BK_SCHEMA" = yes ]; then
      forward schema "$@"
    else
      ensure_backend
      echo "crabbox.sh: backend '$CRABBOX_BACKEND' has no machine-readable schema; showing --help instead." >&2
      if [ $# -gt 0 ]; then
        sub="$(to_backend_verb "$1")"; shift
        exec "$CRABBOX_BACKEND" "$sub" "$@" --help
      else
        exec "$CRABBOX_BACKEND" --help
      fi
    fi ;;
  *)                           forward "$cmd" "$@" ;;
esac
