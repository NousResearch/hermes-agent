---
name: commandcode
description: "Delegate coding to CommandCode CLI (features, PR review)."
version: 1.1.0
author: James Drake (jbdrak) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, CommandCode, Autonomous, Refactoring, Code-Review]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# CommandCode CLI

Use [CommandCode](https://commandcode.ai) as an autonomous coding worker orchestrated by Hermes terminal/process tools. CommandCode is a "coding agent that continuously learns your taste of writing code" — provider-hosted, with a TUI and CLI.

## Prerequisites

- **Install:** `npm install -g commandcode` (or check `~/.npm-global/bin/commandcode`)
- **Auth:** run `commandcode auth login` or set `COMMANDCODE_API_KEY` env var
- **Verify:** `commandcode status` should show `Authenticated`
- **Git repo required** for coding tasks (read, edit, commit, push)
- **Default model:** `deepseek/deepseek-v4-pro` via `~/.commandcode/config.json`
- **PTY mode:** Always pair `background=true, notify_on_complete=true, pty=true` for TUI sessions

## When to Use

- User explicitly asks to use CommandCode
- You want an external coding agent to implement/refactor/review code with persistent taste memory across sessions
- You need long-running coding sessions with progress checks
- You want parallel task execution in isolated worktrees (built-in `--worktree` flag)
- You have CommandCode Go credits active and want to spend them on a real coding job

## Prerequisites

- CommandCode installed: `npm i -g commandcode` (or check `~/.npm-global/bin/commandcode`)
- Auth configured: `commandcode auth login` or set `COMMANDCODE_API_KEY` env var
- Verify: `commandcode status` should show `Authenticated`
- Git repository for code tasks (recommended)
- `pty=true` for interactive TUI sessions

## Default Model

CommandCode's own config (`~/.commandcode/config.json`) sets `"model": "deepseek/deepseek-v4-pro"` — so running `commandcode` without `-m` already uses V4 Pro. The `-m` flag in examples below is explicit for clarity but technically optional.

## Binary Resolution (Important)

Shell environments may resolve different CommandCode binaries. If behavior differs between your terminal and Hermes, check:

```
terminal(command="which -a commandcode")
terminal(command="commandcode --version")
```

If needed, pin an explicit binary path:

```
terminal(command="$HOME/.npm-global/bin/commandcode \"First prompt here\" -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

## One-Shot Tasks

**For real coding work (multi-step, tool calls, edits):** use `commandcode "message"` — it starts the TUI with your prompt pre-loaded. The agent reads files, makes edits, runs commands. Do **not** use `-p` for coding tasks — `-p` is single-turn and CANNOT run the tool loop (Read/Edit/Bash). It only generates a one-shot text response and exits. If you need a one-shot text answer (no edits, no tool calls), `-p` is fine.

```
# Real coding task — starts TUI with initial message
terminal(command="commandcode \"Add retry logic to API calls and update tests\" -t -m deepseek/deepseek-v4-pro --auto-accept", workdir="~/project", background=true, notify_on_complete=true, pty=true)
# Returns session_id — poll it
```

```
# One-shot text answer (NO tool calls, just a reply)
terminal(command="commandcode -p 'What does this regex do: /^foo[0-9]+$/'", workdir="~/project")
```

You can ALSO pass an initial message with flags:

```
terminal(command="commandcode 'Refactor the auth module' --max-turns 30 -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

## Interactive Sessions (Background)

The TUI is the primary mode for any task that needs tool calls. **You CANNOT reliably submit follow-up messages from outside the TUI** — `process(action="submit", ...)` puts text in the input box but the Enter key isn't sent through the PTY (the TUI uses its own keypress handler that doesn't react to raw stdin). The agent only runs after YOU submit a real Enter. Workaround: always start with an initial message via the CLI arg.

```
terminal(command="commandcode \"First prompt goes here\"", workdir="~/project", background=true, notify_on_complete=true, pty=true)
# Returns session_id
```

Monitor progress:

```
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")
```

Exit cleanly with Ctrl+C (`\x03`):

```
process(action="write", session_id="<id>", data="\x03")
# Or just kill the process
process(action="kill", session_id="<id>")
```

### Auto-Accept Mode (Skip Permission Prompts)

For unattended runs where you don't want to babysit tool-use approvals:

```
terminal(command="commandcode \"Run the test suite and fix any failures\" --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

Note: `--auto-accept` works in TUI mode but the agent may still appear to "hang" while deliberating. Watch for the spinner states (`Sketching…`, `Conjuring…`, `Sculpting…`, `Threading…`, `Composing…`, `Channeling…`, `Deliberating…`) — if token count freezes AND no tool calls happen for >30s, the agent may be waiting on a permission prompt that `--auto-accept` should have approved. Kill and retry with `--yolo` if you trust the work.

Or skip permissions entirely with `--yolo` (use sparingly — bypasses ALL safety prompts):

```
terminal(command="commandcode \"Bulk-format all Python files\" --yolo -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

### Worktree Isolation

CommandCode has a built-in `--worktree` flag for isolated work. The CLI creates and manages the worktree for you:

```
terminal(command="commandcode \"Fix issue #101 and open a PR\" -w issue-101 --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

Omit the name and it auto-generates one. Use `#PR` to attach to an existing PR's branch.

## Resuming Sessions

CommandCode auto-persists sessions to disk. Resume with:

```
terminal(command="commandcode -c", workdir="~/project", background=true, notify_on_complete=true, pty=true)  # Continue last
terminal(command="commandcode -r 'fix-auth-bug'", workdir="~/project", background=true, notify_on_complete=true, pty=true)  # By name
terminal(command="commandcode -r ses_abc123", workdir="~/project", background=true, notify_on_complete=true, pty=true)  # By session-id prefix
```

Fork an existing session (resume a copy without touching the original):

```
terminal(command="commandcode -r 'fix-auth-bug' --fork-session", workdir="~/project")
```

Skip persistence for a one-off in-memory run:

```
terminal(command="commandcode \"Quick regex test\" --no-session --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

## Common Flags

| Flag | Use |
|------|-----|
| `-p "query"` | Non-interactive print mode, output and exit |
| `--max-turns N` | Cap agent turns (default 100) |
| `--output-format json` | Machine-readable NDJSON output |
| `-m, --model <name>` | Force specific model (e.g. `deepseek/deepseek-v4-pro`, `minimax-m3`) |
| `--effort <level>` | Reasoning effort: `low`, `medium`, `high` (per model) |
| `-c, --continue` | Continue last session |
| `-r, --resume <id\|name>` | Resume by session-id prefix or display name |
| `--fork-session` | Fork an existing session into a new one |
| `--no-session` | Don't persist session to disk |
| `-n, --name <name>` | Set the session display name |
| `-w, --worktree [name]` | Isolated managed worktree |
| `--permission-mode <mode>` | `standard`, `plan`, `auto-accept` |
| `--auto-accept` | Skip tool-use permission prompts |
| `--yolo` | Bypass ALL permission prompts (alias for `--dangerously-skip-permissions`) |
| `--plan` | Start in plan mode (no edits, just propose) |
| `--add-dir <path>` | Add directory to workspace context |
| `--skill <path>` | Load extra skills (path to skill dir) |
| `--no-skills` | Skip skill discovery |
| `--mod <path>` | Load a mod file/directory for this session |
| `--ide-setup` | Connect IDE to share open file/selection |
| `--no-auto-update` | Disable background CLI updates for this run |
| `--list-models` | List models available for use |
| `-t, --trust` | Auto-trust project (skip initial permission prompt) |

## Plan Mode

For review-before-edit workflows:

```
terminal(command="commandcode \"Design a refactor for the auth module. Output a plan, do not edit yet.\" --plan --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

Then resume without `--plan` to actually apply changes.

## Procedure

1. Verify tool readiness:
   - `terminal(command="commandcode --version")`
   - `terminal(command="commandcode status")`
2. For one-shot text answers, use `commandcode -p '...'` (no pty needed). For coding work (any tool calls), use TUI mode with an initial message: `commandcode "your prompt" -t -m deepseek/deepseek-v4-pro --auto-accept`, with `background=true, notify_on_complete=true, pty=true`. The `-t` flag is required to auto-approve the initial "Do you trust the files" prompt.
3. Monitor long tasks with `process(action="poll"|"log")`.
4. If the TUI is asking for a permission prompt that `--auto-accept` didn't catch, kill and restart with `--yolo` (only if you trust the work).
5. Exit with `process(action="kill", session_id="<id>")`.
6. Summarize file changes, test results, and next steps back to user.

## PR Review Workflow

CommandCode doesn't have a built-in `pr` command like OpenCode. Review PRs in a temporary clone for isolation:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && commandcode \"Review this PR vs main. Report bugs, security risks, test gaps, and style issues.\" -w pr-review --auto-accept -m deepseek/deepseek-v4-pro", background=true, notify_on_complete=true, pty=true)
```

Or fetch the diff directly:

```
terminal(command="commandcode \"Review this diff and flag issues\" -f <(git diff origin/main) --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

## Parallel Work Pattern

Use `--worktree` for natural isolation, or separate workdirs:

```
terminal(command="commandcode \"Fix issue #101 and commit\" -w issue-101 --auto-accept -m deepseek/deepseek-v4-pro", workdir="/tmp/repo", background=true, notify_on_complete=true, pty=true)
terminal(command="commandcode \"Add parser regression tests and commit\" -w issue-102 --auto-accept -m deepseek/deepseek-v4-pro", workdir="/tmp/repo", background=true, notify_on_complete=true, pty=true)
process(action="list")
```

The `-w` flag prevents two sessions from stomping each other in the same repo.

## Session & Cost Management

List past sessions:

```
terminal(command="commandcode --help  # No built-in 'session list' — check ~/.commandcode/sessions/ for transcripts")
```

Check current user / credit balance:

```
terminal(command="commandcode whoami")
```

## Pricing Context (Go $1 Plan)

Plan: $1/mo → $10 in credits. Open-source models only (no Claude/GPT/Grok). Rolling limits: $3/5h, $6/week. DeepSeek V4 Pro is permanently 75% off on CommandCode, so $10 goes ~4x further than API retail.

## Pitfalls

- **TUI doesn't auto-exit — `notify_on_complete` won't fire** — CommandCode's TUI stays open waiting for input after completing all work. The process never ends, so background `notify_on_complete` is never triggered. **Fix:** Append an exit instruction to every TUI prompt:
  ```
  commandcode "your prompt... After committing and pushing, type /exit to close the session." --yolo ...
  ```
  Or build a polling + kill pattern into your workflow (poll every 60s, when output shows "Worked for" + no active indicators, kill the process).

- **`-p` is single-turn only** — it CANNOT run the tool loop (Read/Edit/Bash). Use it for one-shot text answers (regex explanations, doc lookups). For any task that needs to actually read files / make edits / run commands, use TUI mode with an initial message: `commandcode "your prompt"`.
- **`-t` is required for unattended TUI runs** — the initial "Do you trust the files in this folder?" prompt blocks indefinitely without it. `--auto-accept` does NOT cover this prompt; you need both `-t --auto-accept`. Without `-t`, the worker sits at the trust dialog forever.
- **`--auto-accept` does NOT skip per-edit permission prompts** — every Edit/Write/Bash tool call will still trigger a "Do you want to make this edit?" dialog. For truly unattended runs you need `--yolo` (which also auto-approves the trust prompt — no need for `-t` if you have `--yolo`). So: `commandcode "your prompt" --yolo -m deepseek/deepseek-v4-pro` for fire-and-forget automation, `commandcode "your prompt" -t -m deepseek/deepseek-v4-pro --auto-accept` if you want to review each edit.
- **External TUI submission doesn't work** — `process(action="submit", data="...")` puts text in the input box but the Enter key isn't sent through the PTY. The TUI's keypress handler ignores raw stdin writes. Always start with an initial CLI arg message.
- **The agent can appear stuck while deliberating** — spinner states like `Sketching…`, `Conjuring…`, `Sculpting…`, `Threading…`, `Composing…`, `Channeling…`, `Deliberating…` mean the model is reasoning. Token count freezing AND no tool calls for >30s may mean the agent is waiting on a permission prompt that `--auto-accept` should have approved. Kill and retry with `--yolo` if you trust the work.
- **TUI sessions need `pty=true`** (always pair `background=true, notify_on_complete=true, pty=true`). The `-p` mode does NOT need pty.
- PATH mismatch can select the wrong CommandCode binary. Always `which -a commandcode` first.
- `--yolo` is genuinely dangerous — bypasses ALL safety prompts. Use `--auto-accept` for routine unattended runs.
- `--no-session` cannot be resumed. If you think you might want to continue, don't use it.
- `-w` creates a managed worktree in the repo. If the worktree already exists with that name, behavior depends on CommandCode version — use a unique name.
- `--max-turns` defaults to 100. On hard tasks with many file reads, you can hit this cap and exit code 8.
- If CommandCode appears stuck in interactive mode, inspect logs before killing:
  - `process(action="log", session_id="<id>")`
- Avoid sharing one working directory across parallel CommandCode sessions without `--worktree`.

## Verification

Smoke test (text only, no tool calls needed — `-p` mode is fine here):

```
terminal(command="commandcode -p 'Respond with exactly: COMMANDCODE_SMOKE_OK' --max-turns 5")
```

For verifying the full TUI + tool loop works end-to-end, run a real coding task:

```
terminal(command="commandcode \"Make a one-line change to README.md and report what you did\" -m deepseek/deepseek-v4-pro --auto-accept", workdir="~/project", background=true, notify_on_complete=true, pty=true)
# Wait ~30s, then check the worktree for the actual edit
```

Success criteria:
- `-p` smoke: Output includes `COMMANDCODE_SMOKE_OK`, exits cleanly
- TUI coding smoke: Worktree has the actual file change, agent reports what it did

## Rules

1. Prefer `commandcode "your prompt" --yolo -m deepseek/deepseek-v4-pro` (TUI mode) for unattended fire-and-forget automation — it runs the tool loop and skips ALL permission prompts. For runs where you want to review each edit, use `commandcode "your prompt" -t -m deepseek/deepseek-v4-pro --auto-accept` instead. Use `-p` ONLY for plain text answers (no tool calls).
2. Always start with an initial CLI arg message — never rely on `process(action="submit", ...)` to send a follow-up.
3. Always scope CommandCode sessions to a single repo/workdir.
4. Use `--worktree` (not manual `git worktree add`) for parallel work.
5. For long tasks, provide progress updates from `process` logs.
6. Report concrete outcomes (files changed, tests, remaining risks).
7. Exit interactive sessions with `process(action="kill")`.
8. Use `--yolo` for unattended runs (skips ALL prompts, including trust + per-edit). Use `-t --auto-accept` if you want to review each edit manually. Reserve plain `--auto-accept` for runs where you'll be watching.