---
name: commandcode
description: "Delegate coding and PR review to a hosted CLI agent."
version: 1.2.0
author: James Drake (jbdrak) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, CommandCode, Autonomous, Refactoring, Code-Review]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# CommandCode Skill

Orchestrate [CommandCode](https://commandcode.ai), a provider-hosted autonomous coding agent CLI, from Hermes via `terminal` and `process`. Use it to hand multi-step coding work, refactors, and PR review to an external worker that runs on its own model and credit budget.

It does not manage Hermes itself, and it does not replace the `claude-code`, `codex`, or `opencode` skills — CommandCode has its own CLI surface, model set, and permission model.

## When to Use

- The user explicitly asks for CommandCode
- You want an external agent to implement, refactor, or review code in a repository
- You need a long-running coding session you can poll for progress
- You want parallel tasks in isolated worktrees (built-in `--worktree`)
- You have CommandCode credits to spend on real coding work

Don't use it for one-shot text answers you could answer directly, or for changes to the Hermes tree that must run Hermes's own test suite.

## Prerequisites

- **Install:** `npm install -g commandcode` (or check `~/.npm-global/bin/commandcode`)
- **Auth:** `commandcode auth login`, or set `COMMANDCODE_API_KEY`
- **Verify:** `commandcode status` reports `Authenticated`
- **Git repo required** for anything that reads, edits, or commits
- **Default model:** `deepseek/deepseek-v4-pro`, from `~/.commandcode/config.json`, so `-m` is optional
- **PTY:** pair `pty=true` with every TUI call; `-p` mode does not need it

Sessions and transcripts persist under `~/.commandcode/sessions/`; `commandcode whoami` shows the current account and credit balance.

## How to Run

Real coding work must run in TUI mode with the prompt passed as a CLI argument. `-p` is single-turn and cannot run the tool loop, so it will not read files, edit, or execute anything.

```
# Coding task — TUI with an initial message, unattended
terminal(command="commandcode \"Add retry logic to the API client and update its tests\" -t --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
# Returns a session_id — poll it with process(action="poll")
```

```
# One-shot text answer only (no tool calls)
terminal(command="commandcode -p 'Explain this regex: ^foo[0-9]+$'", workdir="~/project")
```

The TUI does not exit when the work is done, so `notify_on_complete` never fires by itself. Append an exit instruction to the prompt:

```
terminal(command="commandcode \"Fix issue #101, commit, then type /exit to close the session.\" --yolo -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

Resume earlier work with `-c` (last session), `-r '<name-or-id-prefix>'`, or `--fork-session` to branch a copy.

## Quick Reference

| Flag | Use |
|------|-----|
| `-p "query"` | Non-interactive print mode; text only, no tool calls |
| `--max-turns N` | Cap agent turns (default 100) |
| `--output-format json` | Machine-readable NDJSON output |
| `-m, --model <name>` | Pin a model (e.g. `deepseek/deepseek-v4-pro`) |
| `--effort <level>` | Reasoning effort: `low`, `medium`, `high` |
| `-c, --continue` | Continue the last session |
| `-r, --resume <id\|name>` | Resume by session-id prefix or display name |
| `--fork-session` | Fork an existing session into a new one |
| `--no-session` | Do not persist the session |
| `-n, --name <name>` | Name the session |
| `-w, --worktree [name]` | Managed isolated worktree |
| `--permission-mode <mode>` | `standard`, `plan`, `auto-accept` |
| `--auto-accept` | Skip tool-use permission prompts |
| `--yolo` | Bypass all prompts, including trust and per-edit |
| `--plan` | Start in plan mode; propose without editing |
| `-t, --trust` | Auto-trust the project directory |
| `--add-dir <path>` | Add a directory to the workspace context |
| `--skill <path>` / `--no-skills` | Load or skip extra skills |
| `--no-auto-update` | Disable background CLI self-update |
| `--list-models` | List available models |

## Procedure

1. Confirm readiness with `terminal(command="commandcode --version")` and `terminal(command="commandcode status")`.
2. Pin the binary if resolution looks ambiguous: `terminal(command="which -a commandcode")`.
3. Launch TUI work with `background=true, notify_on_complete=true, pty=true`, passing the prompt as an argument.
4. Poll with `process(action="poll", session_id="<id>")` and read output with `process(action="log", session_id="<id>")`.
5. Send follow-up input with `process(action="submit", session_id="<id>", data="yes")`, which writes the data plus a newline and therefore does act as Enter. A full-screen TUI that installs its own keypress handler can still ignore raw stdin; if the follow-up does not take, relaunch with the prompt as an argument instead.
6. Close out with `process(action="kill", session_id="<id>")`, then report changed files, test results, and remaining risk.

### Parallel work

Give each task its own worktree so two sessions cannot stomp each other:

```
terminal(command="commandcode \"Fix issue #101 and commit\" -w issue-101 --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
terminal(command="commandcode \"Add parser regression tests and commit\" -w issue-102 --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
process(action="list")
```

### PR review

Review inside a clone so the working tree is untouched:

```
terminal(command="git clone https://github.com/user/repo.git repo-review", workdir="~/project")
terminal(command="commandcode \"Review this diff against origin/main. Report bugs, security risks, test gaps, and style issues.\" --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project/repo-review", background=true, notify_on_complete=true, pty=true)
```

### Plan mode

```
terminal(command="commandcode \"Design a refactor for the auth module. Output a plan, do not edit.\" --plan --auto-accept -m deepseek/deepseek-v4-pro", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

Resume the same session without `--plan` to apply the changes.

## Pitfalls

- `-p` is single-turn only. It cannot read files, edit, or run commands, so it is wrong for anything that is not a plain text answer.
- The TUI never exits on its own, so `notify_on_complete` will not fire. Append `type /exit to close the session` to the prompt, or poll and kill manually.
- `-t` is required for unattended runs. The initial "Do you trust the files in this folder?" prompt blocks indefinitely without it, and `--auto-accept` does not cover that prompt.
- `--auto-accept` does not skip per-edit prompts. Every edit still asks for approval. Use `--yolo` for fire-and-forget runs, which also covers the trust prompt, so `-t` is unnecessary alongside it.
- A stalled spinner (`Sketching…`, `Conjuring…`, `Deliberating…`, and similar) with a frozen token count and no tool calls for 30s usually means a permission prompt is waiting. Kill and relaunch with `--yolo` if you trust the work.
- PATH can resolve a different CommandCode binary than the one you expect. Check `which -a commandcode` before trusting version-specific behaviour.
- `--no-session` cannot be resumed. Do not use it if you may want to continue the session later.
- `-w` creates a managed worktree. Use a unique name; behaviour on an existing name varies by version.
- `--max-turns` defaults to 100. Hitting the cap exits with code 8.
- Never run two sessions in the same working directory without `--worktree`.
- `--yolo` bypasses every safety prompt. Reserve it for work you trust, and review the diff before committing.

## Verification

Text-only smoke test (`-p` is correct here):

```
terminal(command="commandcode -p 'Respond with exactly: COMMANDCODE_SMOKE_OK' --max-turns 5")
```

Full tool-loop smoke test:

```
terminal(command="commandcode \"Make a one-line change to README.md and report what you did\" -m deepseek/deepseek-v4-pro --auto-accept", workdir="~/project", background=true, notify_on_complete=true, pty=true)
```

Success criteria:

- [ ] `-p` smoke prints `COMMANDCODE_SMOKE_OK` and exits cleanly
- [ ] TUI smoke leaves the real file change in the worktree and reports what it did
- [ ] `commandcode status` reported `Authenticated` before any real task started
- [ ] Any session left running was closed with `process(action="kill")`
