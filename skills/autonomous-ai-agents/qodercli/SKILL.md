---
name: qodercli
description: "Delegate coding to Qoder CLI (features, PRs, refactors)."
version: 2.5.1
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
- **PTY:** pass `pty=true` for interactive foreground (`-i`). Background qodercli auto-switches to pipe mode for NDJSON progress regardless of this flag. Print mode (`-p`) works without it.

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

### Mode selection (important)

| Task type | Mode | Why |
|-----------|------|-----|
| Bounded implementation (files known) | **`-p` (print)** | One-shot, no monitoring, 8x write compression proven in CTA |
| Repository-wide migration/refactor | **`-p` (print)** | qodercli handles dependency mapping internally |
| CI/automation/piped input | **`-p` with `-o json`** | Structured output, no PTY needed |
| Genuinely iterative (needs clarification) | `-i` (interactive) | Only when task requires multi-turn dialogue |

**Default to print mode.** CTA evidence (N=12 sessions): legacy PTY interactive mode had a 40% stuck-session rate (spinner-only polls → premature kill). Plan 7 eliminated this for background tasks via automatic NDJSON structured progress (0% spinner-only, 100% tool-use visibility). Print mode remains simplest for bounded tasks; background mode now has full observability.

### Print mode (one-shot, preferred)

**Simplest: use the delegation wrapper** (handles preflight, timeout, error classification):

```
terminal(command="qodercli-delegate 'Add error handling to all API calls in src/routes/' ~/project 300", timeout=360)
```

Returns structured JSON: `{exit_code, error_class, files_changed, diff_stat, output_tail}`.

**Direct invocation** (if wrapper unavailable):

```
terminal(command="qodercli -p 'Add error handling to all API calls in src/routes/' --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=300)
```

Print mode skips interactive dialogs. Use for bounded tasks, CI, and piped input:

```
terminal(command="git diff main...feature | qodercli -p 'Review for bugs and security issues' --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=120)
```

For programmatic result extraction, use JSON output:

```
terminal(command="qodercli -p 'List all TODO comments in src/' -o json --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=120)
```

JSON output includes structured fields (session_id, result, cost) for downstream processing.

### Interactive mode (advanced — prefer print mode)

Only use when the task genuinely requires multi-turn dialogue with qodercli (e.g., iterative refinement, clarifying questions). Bounded implementation tasks should use print mode.

```
terminal(command="qodercli -i 'Implement the payroll tax engine'", workdir="~/project", background=true, pty=true)
process(action="wait", session_id="<id>", timeout=120)
process(action="log", session_id="<id>")
process(action="write", session_id="<id>", data="\x03")
```

**Monitoring patience (interactive foreground only):**
- Background tasks get automatic NDJSON structured progress — no patience needed (see below).
- For interactive foreground (`-i` without `background=true`): use `process(action="wait", timeout=120)` between checks — NOT rapid `process(poll)` loops.
- Qodercli needs 60–300s for multi-file tasks. Spinner characters (⠋⠙⠹) mean it is actively working. Do NOT kill it.
- Maximum 10 process() calls before checking `git diff --stat` in the working directory for evidence of progress.
- If files are being written, keep waiting. If no file changes after 5 minutes, then investigate with `process(action="log")`.

**NDJSON progress (automatic):** Background qodercli tasks are automatically spawned in pipe mode with `--output-format stream-json`. `process(poll)` returns structured progress like `Tools used: Read (src/auth.py), Edit (src/routes.ts) | Thinking... (2400 chars)` instead of spinner glyphs. You will see exactly which tool qodercli is using — no patience guessing needed.

**Environment friction detection:** Poll output includes a runtime friction index computed from the NDJSON stream (error rate, context velocity, retry density). When the background session hits environment problems (missing packages, broken imports, permission errors), you will see:

- `errors: N/M` — mild friction (some tool calls failing). Monitor — the session may self-recover. No action needed unless it escalates to heavy friction.
- `⚠ Friction: HIGH-ERROR (N/M) | RETRY Bash x4` — heavy friction (session is stuck in fix→fail loops). Act immediately.

**When you see `⚠ Friction` — this is a regime response, not a task decision:**

The friction signal means the *environment* is broken, not the task or the skill. Your response should treat the regime signal: change how you interact with qodercli, not what you ask it to do.

1. The session is burning context on remediation loops — it will likely timeout or produce incomplete work.
2. Kill it: `process(action="kill", session_id="<id>")`
3. Diagnose: check `process(action="log")` output for the specific errors (ModuleNotFoundError, Permission denied, etc.)
4. Fix the environment (install missing deps, fix permissions) BEFORE retrying.
5. **Retry with `-p` (print mode).** Do NOT retry with `-i` or `background=true` — that re-enters the same monitoring loop that friction broke. Print mode is one-shot: no polling, no NDJSON stream, no friction accumulation.

```
terminal(command="qodercli -p '<same task, tighter scope>' --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=300)
```

**Exit-42 pipe conflict:** If qodercli exits with code 42, it means `-i` (interactive) was rejected because stdin is piped. This is expected in background/pipe mode. Fall back to `-p` immediately — do NOT retry with `-i` in a different configuration. The task and prompt are fine; only the mode flag needs changing.

Clean sessions show zero friction overhead — no indicator appears when the environment is healthy.

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
| `-o, --output-format <fmt>` | Output format (text, json, stream-json). Background tasks auto-use stream-json for NDJSON progress. |
| `--reasoning-effort <level>` | Set reasoning effort |
| `--list-sessions` | List sessions |
| `--list-models` | List available models |
| `-d, --debug` | Debug mode |

Subcommands: `mcp`, `skills`, `hooks`, `agents`, `plugins`, `login`, `commit`, `rollback`, `update`, `status`, `wiki`.

### Model Selection (`Qwen3.8-Max-Preview`)

Override the active model to leverage Alibaba Cloud's exclusive 2.4T-parameter flagship:

```
terminal(command="qodercli -p 'Refactor src/db/ to SQLAlchemy' -m Qwen3.8-Max-Preview --permission-mode bypass_permissions", workdir="~/project", pty=true, timeout=300)
```

`Qwen3.8-Max-Preview` defaults to a **131k token context window** on standard invocations but supports up to **1M tokens** when Qoder manages extended repository contexts. Delegating multi-file edits to `qodercli` prevents flooding Hermes's context window by offloading raw file reads to Qoder's workspace — Hermes sees only the delegation command and summary result.

## Procedure

1. Verify binary resolves (see Binary Resolution above) and `qodercli --version` succeeds.
2. For bounded tasks (default), use print mode: `qodercli -p '<scoped prompt>' --permission-mode bypass_permissions`. Set `timeout=300` for single-directory tasks, `timeout=600` for multi-file/multi-directory tasks. Or use `qodercli-delegate` for automatic error handling.
3. For genuinely iterative tasks only, start interactive with `background=true, pty=true`.
4. Handle folder trust dialog if needed (`process(action="write", data="1\n")`).
5. Monitor interactive sessions with `process(action="wait", timeout=120)` — never rapid `process(poll)` loops. Check `git diff --stat` for progress evidence after 10 process() calls.
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

### Error recovery

Never trust the model's self-report of qodercli success. Always verify from the terminal output:

```
terminal(command="qodercli -p '...' --permission-mode bypass_permissions; echo \"EXIT_CODE=$?\"", workdir="~/project", pty=true, timeout=300)
```

Check for these failure patterns in the output:
- `Permission confirmation required` → missing `--permission-mode bypass_permissions`
- `Not logged in` / `Please run /login` → auth token missing or expired
- `402` / `credit` → credit limit exhausted; task incomplete
- Non-zero exit code → qodercli failed regardless of any partial output

If qodercli fails, do NOT report success. Inspect the error, fix the root cause, and retry or fall back to manual implementation.

### Partial completion / session cleanup

When qodercli dies mid-task (credit limit, timeout, crash):

1. Check what was written: `git diff --stat` in the working directory
2. Assess completeness: are the changes coherent or half-applied?
3. Options:
   - **Resume:** `qodercli -c` continues the most recent session (if credits remain)
   - **Salvage:** keep qodercli's partial writes, complete remaining work manually
   - **Rollback:** `git checkout -- .` if changes are incoherent, then retry with a tighter prompt
4. For interactive sessions: `process(action="kill", session_id="<id>")` to clean up the background process
5. Never leave orphaned background processes — check with `process(action="list")` after any abnormal termination

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
- **Spinner means working (interactive foreground only).** If `process(poll)` or `process(log)` shows only spinner characters (⠋⠙⠹⠸⠼⠴) with no meaningful text, qodercli is actively implementing. Do NOT kill it. Wait longer. Background tasks emit structured NDJSON instead — you'll see tool names and thinking state, never bare spinners.
- **Never rapid-poll.** Polling more than once per 30 seconds wastes context on spinner frames. Use `process(action="wait", timeout=120)`. Multi-file implementation takes 2–5 minutes. Prefer print mode to avoid monitoring entirely.
- **Structured progress in background mode.** When qodercli runs as a background process, Hermes automatically uses pipe mode with `--output-format stream-json`. Poll output shows structured events (`Tools used: Read (file.py) | Thinking... | Completed (success, N turns, Ns)`) instead of spinner glyphs. No action needed — this is automatic.

## Verification

```
terminal(command="qodercli -p 'Respond with exactly: QODER_SMOKE_OK'", workdir="~/project", pty=true, timeout=30)
```

Success: output contains `QODER_SMOKE_OK`, no auth/model errors, exit code 0.

After code tasks: `terminal(command="cd ~/project && git diff --stat && pytest -x -q", timeout=60)`.
