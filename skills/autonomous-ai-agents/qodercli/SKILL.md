---
name: qodercli
description: "Delegate coding to Qoder CLI (features, PRs, refactors)."
version: 2.0.0
author: explicitcontextualunderstanding
license: MIT
platforms: [linux, macos, windows]
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

# Qoder CLI

Delegate coding tasks to [Qoder CLI](https://docs.qoder.com) via the `terminal` tool. Qoder reads files, writes code, runs shell commands, spawns subagents, and manages git workflows autonomously. It does not replace Hermes for simple lookups or single-file edits.

## When to Use

- Sprawling feature implementations spanning multiple directories
- Deep refactoring requiring comprehensive dependency mapping
- Multi-agent cycles with autonomous execution and test-verification loops
- Batch issue fixing across worktrees
- Repository-wide analysis (audit trails, migration planning)

Do NOT use for single-file lookups, basic shell commands, or tasks that fit in one tool call.

## Prerequisites

- **Install:** `npm install -g @qoder-ai/qodercli` or `curl -fsSL https://qoder.com/install | bash`
- **Auth:** `qodercli login` (interactive) or set `QODER_PERSONAL_ACCESS_TOKEN` env var
- **Verify:** `qodercli --version` and `qodercli --list-models`
- **PTY:** always pass `pty=true` for interactive mode (`-i`, background). Print mode (`-p`) works without it, but including it is harmless.

## Binary Resolution (Important)

Resolution order follows the standard Hermes pattern:

1. `HERMES_QODERCLI_BIN` env var (absolute path override)
2. PATH lookup (`which -a qodercli` / `where.exe qodercli` on Windows)
3. Validate: `qodercli --version` must succeed

```
terminal(command="which -a qodercli && qodercli --version")
```

If PATH lookup fails or resolves the wrong binary, set the override or pin explicitly:

```
terminal(command="HERMES_QODERCLI_BIN=/opt/homebrew/bin/qodercli qodercli -p '...'", workdir="~/project", pty=true)
```

## How to Run

### Print mode (one-shot, preferred)

```
terminal(command="qodercli -p 'Add error handling to all API calls in src/routes/' --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=180)
```

Print mode skips interactive dialogs. Use for bounded tasks, CI, and piped input:

```
terminal(command="git diff main...feature | qodercli -p 'Review for bugs and security issues' --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=120)
```

### Interactive mode (multi-turn)

```
terminal(command="qodercli -i 'Implement the payroll tax engine'", workdir="~/project", background=true, pty=true)
process(action="submit", session_id="<id>", data="Use 2026 tax brackets, progressive federal rates")
process(action="poll", session_id="<id>")
process(action="write", session_id="<id>", data="\x03")
```

### Folder trust dialog (interactive mode only)

On first launch in a new directory, Qoder shows a trust prompt. Send `1\n` to accept:

```
terminal(command="qodercli", workdir="~/project", background=true, pty=true)
process(action="write", session_id="<id>", data="1\n")
```

Print mode (`-p`) skips this dialog entirely.

## Quick Reference

| Flag | Effect |
|------|--------|
| `-p, --print` | One-shot mode, exits when done (query is positional arg) |
| `-i, --prompt-interactive <text>` | Execute prompt, stay interactive |
| `-c, --continue` | Continue most recent session |
| `-r, --resume [id]` | Resume session by ID |
| `-m, --model <model>` | Override model |
| `-w, --cwd <dir>` | Set working directory |
| `--worktree [name]` | Start in isolated git worktree |
| `--permission-mode <mode>` | `default`, `accept_edits`, `bypass_permissions`, `dont_ask`, `auto` |
| `--dangerously-skip-permissions` | Bypass all permission checks |
| `--allowed-tools <tool>` | Whitelist tools |
| `--disallowed-tools <tool>` | Blacklist tools |
| `--attachment <file>` | Attach files to prompt |
| `--agent <name>` | Use a named agent |
| `--mcp-config <config>` | Load MCP servers from JSON |
| `-o, --output-format <fmt>` | Output format (text, json) |
| `--reasoning-effort <level>` | Set reasoning effort |
| `--list-sessions` | List sessions |
| `--list-models` | List available models |
| `-d, --debug` | Debug mode |

Subcommands: `mcp`, `skills`, `hooks`, `agents`, `plugins`, `login`, `commit`, `rollback`, `update`, `status`, `wiki`.

### Context window consideration

Models default to a limited context window (e.g., 131k tokens) unless extended context is explicitly configured. When Hermes performs multi-file edits directly, raw file contents accumulate in its prompt context. Delegating to `qodercli` offloads file ingestion to Qoder's own workspace — Hermes sees only the delegation command and the summary result, preserving its context for orchestration and verification.

```
terminal(command="qodercli -p 'Refactor src/db/ to SQLAlchemy' -m <model> --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=300)
```

## Procedure

1. Verify binary resolves (see Binary Resolution above) and `qodercli --version` succeeds.
2. For bounded tasks, use print mode: `qodercli -p '<scoped prompt>' --permission-mode bypass_permissions`.
3. For iterative tasks, start interactive with `background=true, pty=true`.
4. Handle folder trust dialog if needed (`process(action="write", data="1\n")`).
5. Monitor with `process(action="poll"|"log")`.
6. For parallel work, use `--worktree` or separate directories — never share a cwd.
7. Exit interactive sessions with `\x03` or `process(action="kill")`.
8. Verify results: `git diff --stat` and run the test suite.

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

- Never pass open-ended prompts — specify target paths, exact changes, done-criteria.
- One concern per invocation; split multi-objective tasks into parallel worktrees.
- Use `--permission-mode bypass_permissions` for trusted autonomous runs.
- Monitor long tasks; kill stalled sessions early.

## Pitfalls

- **PTY is mandatory for interactive mode.** Qoder hangs without a pseudo-terminal when using `-i` or background sessions. Print mode (`-p`) works without PTY.
- **Folder trust blocks silently.** Send `1\n` in new directories (interactive only; `-p` skips it).
- **`-p` takes a positional query, not `--prompt`.** The flag is `-p`/`--print`; text follows as arg.
- **Don't use `/exit` or `exit`.** Use Ctrl+C (`\x03`) or `process(action="kill")`.
- **PATH mismatch** can select the wrong Qoder binary. See Binary Resolution above.
- **Parallel sessions need isolation.** Shared cwd causes file-write conflicts.
- **Auth token expiry.** 401/403 mid-session means re-run `qodercli login`.
- **Don't echo the token.** `qodercli` reads `QODER_PERSONAL_ACCESS_TOKEN` automatically. Never run `echo $QODER_PERSONAL_ACCESS_TOKEN` for validation — use `qodercli --version` or the smoke test below.
- **Credit drain on vague prompts.** Tight scope = fewer turns = fewer credits.

## Verification

```
terminal(command="qodercli -p 'Respond with exactly: QODER_SMOKE_OK'", workdir="~/project", pty=true, timeout=30)
```

Success: output contains `QODER_SMOKE_OK`, no auth/model errors, exit code 0.

After code tasks: `terminal(command="cd ~/project && git diff --stat && pytest -x -q", timeout=60)`.
