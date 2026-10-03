---
name: qodercli
description: "Delegate coding to Qoder CLI (features, PRs, refactors)."
version: 3.0.0
author: explicitcontextualunderstanding
license: MIT
platforms: [linux, macos]
required_environment_variables:
  - name: QODER_PERSONAL_ACCESS_TOKEN
    prompt: Qoder personal access token
    help: Create one at https://qoder.com/settings/tokens (or QODERCN_PERSONAL_ACCESS_TOKEN for China edition)
    required_for: authentication
metadata:
  hermes:
    tags: [Coding-Agent, Qoder, Multi-File, Refactoring, Agentic-Loop, PTY, Automation]
    related_skills: [claude-code, codex, hermes-agent, opencode]
---

# Qoder CLI Skill

Delegate coding tasks to [Qoder CLI](https://docs.qoder.com) through the `terminal` tool. Qoder reads files, writes code, runs shell commands, spawns subagents, and manages git workflows autonomously. It does not replace Hermes for simple lookups or single-file edits.

## When to Use

- Sprawling feature implementations spanning multiple directories
- Deep refactoring that needs dependency mapping across many files
- Batch issue fixing across worktrees
- Repository-wide analysis (audit trails, migration planning)

### Large-repo delegation (80+ files)

For tasks touching 80+ files, put progress checkpoints in the delegation prompt:

> "Process files one at a time. After every 10 files, state your progress
> (Completed N/total). If you lose track, list the directory to recover."

This keeps state alive across the executor's own context compaction. Under 80 files no special guidance is needed.

## Prerequisites

- **Install:** `npm install -g @qoder-ai/qodercli` or `curl -fsSL https://qoder.com/install | bash`
- **Auth:** `qodercli login` (interactive) or set `QODER_PERSONAL_ACCESS_TOKEN`
- **Verify:** `qodercli --version`
- **Platform:** the bundled helper `scripts/qodercli_delegate.sh` is Bash-only, so this skill is gated to `platforms: [linux, macos]`.

### Binary resolution

Resolve `HERMES_QODERCLI_BIN` (absolute path override) first, then PATH, then validate with `qodercli --version`.

```
terminal(command="which -a qodercli && qodercli --version")
```

If PATH resolves the wrong binary, pin it explicitly:

```
terminal(command="HERMES_QODERCLI_BIN=/opt/homebrew/bin/qodercli qodercli -p '...'", workdir="~/project", pty=true)
```

## How to Run

### Mode selection

| Task type | Mode | Why |
|-----------|------|-----|
| Bounded implementation (files known) | **`-p` (print)** | One-shot, no monitoring |
| Repository-wide migration/refactor | **`-p` (print)** | qodercli maps dependencies internally |
| CI/automation/piped input | **`-p` with `-o json`** | Structured output, no PTY needed |
| Genuinely iterative (needs clarification) | `-i` (interactive) | Only for multi-turn dialogue |

**Default to print mode.** Interactive mode needs polling discipline and has a higher stuck-session rate; print mode is one-shot and needs no monitoring at all.

### Print mode (preferred)

Use the bundled helper — it handles preflight, timeout, and error classification:

```
terminal(command="bash scripts/qodercli_delegate.sh 'Add error handling to all API calls in src/routes/' ~/project 300", timeout=360)
```

It prints one JSON object on stdout holding `exit_code`, `error_class`, `files_changed`, `diff_stat`, `output_tail`, `workdir`, `timeout_used`, `git_before` and `git_after`. `error_class` is one of `none`, `timeout`, `auth_failure`, `credit_exhausted`, `permission_blocked`, `network_error`, `unknown_failure`, `usage_error`, `binary_not_found`, `workdir_not_found`. The helper exits 0 on qodercli success, 1 on qodercli failure, and 2 on preflight failure.

**Direct invocation** when you don't need classification:

```
terminal(command="qodercli -p 'Add error handling to all API calls in src/routes/' --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=300)
```

Piped input works too:

```
terminal(command="git diff main...feature | qodercli -p 'Review for bugs and security issues' --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=120)
```

Add `-o json` for programmatic extraction (fields include `session_id`, `result`, `cost`).

### Interactive mode (only when the task needs dialogue)

```
terminal(command="qodercli -i 'Implement the payroll tax engine'", workdir="~/project", background=true, pty=true)
process(action="wait", session_id="<id>", timeout=120)
process(action="log", session_id="<id>")
process(action="write", session_id="<id>", data="\x03")
```

**`pty=true` is mandatory for any `process(action="write")` flow.** Non-PTY background spawns get a closed stdin, so writes fail with "Process stdin not available". Without a PTY you cannot answer the trust dialog or send Ctrl-C — you can only `process(action="kill")`.

**Monitoring patience:** use `process(action="wait", timeout=120)` between checks, never rapid `process(poll)` loops. Multi-file tasks take 60–300s; after 10 `process()` calls check `git diff --stat` for evidence of progress, and only investigate `process(action="log")` if nothing has changed in 5 minutes.

### Folder trust dialog (interactive only)

On first launch in a new directory Qoder prompts for trust. Send `1\n`:

```
terminal(command="qodercli", workdir="~/project", background=true, pty=true)
process(action="write", session_id="<id>", data="1\n")
```

Print mode (`-p`) skips this dialog entirely.

## Quick Reference

| Flag | Effect |
|------|--------|
| `-p, --print` | One-shot mode, exits when done (query is positional) |
| `-i, --prompt-interactive <text>` | Execute prompt, stay interactive |
| `-c, --continue` / `-r, --resume [id]` | Continue most recent session / resume by ID |
| `-m, --model <model>` | Override model |
| `-w, --cwd <dir>` | Set working directory |
| `--worktree [name]` | Start in an isolated git worktree |
| `--permission-mode <mode>` | `default`, `accept_edits`, `bypass_permissions`, `dont_ask`, `auto` |
| `--dangerously-skip-permissions` | Bypass all permission checks |
| `--allowed-tools` / `--disallowed-tools <tool>` | Whitelist / blacklist tools |
| `--agent <name>` / `--mcp-config <config>` | Use a named agent / load MCP servers from JSON |
| `-o, --output-format <fmt>` | `text`, `json`, `stream-json` |
| `--attachment`, `--reasoning-effort`, `--list-sessions`, `--list-models`, `-d/--debug` | Less common; see `qodercli --help` |

Subcommands: `mcp`, `skills`, `hooks`, `agents`, `plugins`, `login`, `commit`, `rollback`, `update`, `status`, `wiki`.

### Model selection

Override the model per invocation with `-m`:

```
terminal(command="qodercli -p 'Refactor src/db/ to SQLAlchemy' -m Qwen3.8-Max-Preview --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=300)
```

Delegation also protects Hermes's own context window: raw file reads happen inside Qoder's workspace, so Hermes sees only the delegation command and the summary result.

## Procedure

1. Verify the binary resolves and `qodercli --version` succeeds.
2. For bounded tasks (the default) use the helper, or `qodercli -p '<scoped prompt>' --permission-mode bypass_permissions`. Set `timeout=300` for one directory, `600` for multi-directory work.
3. For genuinely iterative tasks, start interactive with `background=true, pty=true`, and answer the folder trust dialog if it appears.
4. Monitor with `process(action="wait", timeout=120)` — never rapid polling.
5. For parallel work use `--worktree` or separate directories — never share a cwd.
6. Exit interactive sessions with `\x03` or `process(action="kill")`.
7. Verify results: `git diff --stat`, then run the test suite.

### Parallel worktrees

```
terminal(command="qodercli --worktree feat-a -p 'Implement feature A. Run tests.'", workdir="~/project", background=true, pty=true)
terminal(command="qodercli --worktree feat-b -p 'Implement feature B. Run tests.'", workdir="~/project", background=true, pty=true)
process(action="list")
```

### Session resumption

```
terminal(command="qodercli -c", workdir="~/project", pty=true)
terminal(command="qodercli -r <session-id> --fork-session", workdir="~/project", pty=true)
```

### Cost safeguards

- Never pass open-ended prompts — specify target paths, exact changes, and done-criteria. Tight scope means fewer turns and fewer credits.
- One concern per invocation; split multi-objective work into parallel worktrees.
- Use `--permission-mode bypass_permissions` only for trusted autonomous runs, and kill stalled sessions early.

### Error recovery

Never trust a self-report of success — verify from terminal output:

```
terminal(command="qodercli -p '...' --permission-mode bypass_permissions; echo \"EXIT_CODE=$?\"", workdir="~/project", pty=true, timeout=300)
```

A non-zero exit means qodercli failed regardless of any partial output it produced; the helper's `error_class` names the cause. If it fails, do NOT report success — fix the root cause, then retry or fall back to manual implementation.

### Partial completion and cleanup

When qodercli dies mid-task (credit limit, timeout, crash):

1. Check what landed with `git diff --stat` in the working directory, and judge whether the changes are coherent or half-applied.
2. Choose: **resume** (`qodercli -c`, if credits remain), **salvage** (keep the partial writes, finish manually), or **roll back** (`git checkout -- .`, then retry with a tighter prompt).
3. Kill interactive sessions with `process(action="kill", session_id="<id>")`.
4. Never leave orphans — run `process(action="list")` after any abnormal termination.

## Pitfalls

- **PTY is mandatory for interactive mode.** Qoder hangs without a pseudo-terminal when using `-i` or background sessions. Print mode works without one, but then `process(action="write")` is unavailable.
- **Folder trust blocks silently.** Send `1\n` in new directories (interactive only).
- **`-p` takes a positional query, not `--prompt`.**
- **PATH mismatch** can select the wrong Qoder binary — see Binary resolution above.
- **Parallel sessions need isolation.** A shared cwd causes write conflicts.
- **Auth token expiry.** A 401/403 mid-session means re-run `qodercli login`.
- **Don't echo the token.** qodercli reads `QODER_PERSONAL_ACCESS_TOKEN` itself. Never echo it to validate — use `qodercli --version` or the smoke test below.
- **Spinner means working.** If `process(poll)` or `process(log)` shows only spinner glyphs (⠋⠙⠹⠸⠼⠴) with no text, qodercli is implementing — wait longer, and never poll more than once per 30s. Prefer print mode to skip monitoring entirely.

## Verification

```
terminal(command="qodercli -p 'Respond with exactly: QODER_SMOKE_OK'", workdir="~/project", pty=true, timeout=30)
```

Success: output contains `QODER_SMOKE_OK`, no auth or model errors, exit code 0.

After code tasks: `terminal(command="cd ~/project && git diff --stat && pytest -x -q", timeout=60)`.
