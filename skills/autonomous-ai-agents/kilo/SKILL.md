---
name: kilo
description: "Delegate coding to Kilo CLI (features, PR review)."
version: 1.0.0
author: Kilo Team
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Kilo, Autonomous, Refactoring, Code-Review]
    related_skills: [claude-code, codex, hermes-agent, opencode]
---

# Kilo CLI

Use [Kilo](https://kilo.ai/cli) as an autonomous coding worker orchestrated by Hermes terminal/process tools. Kilo is an open-source AI coding agent (a fork of OpenCode) with a CLI and web console, 500+ provider models, and built-in agents (Code, Plan, Ask, Debug, Review).

## When to Use

- User explicitly asks to use Kilo
- You want an external coding agent to implement/refactor/review code
- You need long-running coding sessions with progress checks
- You want parallel task execution in isolated workdirs/worktrees
- You want fully autonomous runs for CI/CD pipelines

## Prerequisites

- Kilo installed: `npm i -g @kilocode/cli`, `brew install Kilo-Org/tap/kilo`, `curl -fsSL https://kilo.ai/cli/install | bash`, or AUR `paru -S kilo-bin`
- Auth configured: `kilo auth login` (Kilo account recommended — no API keys required to start) or set provider env vars / `kilo auth login <provider>`
- Verify: `kilo auth list` should show at least one credential
- Git repository for code tasks (recommended)
- `pty=true` for interactive (`-i`) and background sessions

## Binary Resolution (Important)

Shell environments may resolve different Kilo binaries. If behavior differs between your terminal and Hermes, check:

```
terminal(command="which -a kilo")
terminal(command="kilo --version")
```

If needed, pin an explicit binary path:

```
terminal(command="$HOME/.kilo/bin/kilo run '...'", workdir="~/project", pty=true)
```

## One-Shot Tasks

Use `kilo run` for bounded, non-interactive tasks (no pty needed):

```
terminal(command="kilo run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

Attach context files with `-f`:

```
terminal(command="kilo run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

Show model thinking with `--thinking`:

```
terminal(command="kilo run 'Debug why tests fail in CI' --thinking", workdir="~/project")
```

Force a specific model in `provider/model` form:

```
terminal(command="kilo run 'Refactor auth module' --model anthropic/claude-sonnet-4", workdir="~/project")
```

Machine-readable output for parsing:

```
terminal(command="kilo run 'List all functions in src/' --format json", workdir="~/project")
```

## Autonomous Mode (CI/CD)

Kilo's headline feature for pipelines. `--auto` disables all permission prompts and auto-approves every action, and tracks `task` child-session permission requests so spawned subagent work proceeds unattended:

```
terminal(command="kilo run --auto 'run tests and fix any failures'", workdir="~/project")
```

Only use `--auto` in trusted environments — it lets the agent execute any action without confirmation. For ad-hoc one-shots that still want to avoid prompts, `--dangerously-skip-permissions` auto-approves permission requests once — but it still strips the approval gate, so use it only in an isolated or trusted workdir, never in the user's primary repo.

## Built-in Agents

Kilo ships specialized agents you switch between depending on the task. Pass `--agent <name>`:

```
terminal(command="kilo run --agent review 'Review this PR vs main for bugs and security issues'", workdir="~/project")
terminal(command="kilo run --agent plan 'Design the schema for a multi-tenant billing service'", workdir="~/project")
```

| Agent    | Use                                                                   |
| -------- | --------------------------------------------------------------------- |
| `code`   | Default. Implement and edit code from natural language                |
| `plan`   | Design architecture and write implementation plans before coding      |
| `ask`    | Answer questions about the codebase without touching files            |
| `debug`  | Troubleshoot and trace issues                                         |
| `review` | Review changes across performance, security, style, and test coverage |

## Interactive Sessions (Background)

For iterative work requiring multiple exchanges, run `kilo run -i` (direct interactive split-footer mode; requires a TTY, so use `pty=true`):

```
terminal(command="kilo run -i", workdir="~/project", background=true, pty=true)
# Returns session_id

# Send a prompt
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send follow-up input
process(action="submit", session_id="<id>", data="Now add error handling for token expiry")

# Exit cleanly — Ctrl+C
process(action="write", session_id="<id>", data="\x03")
# Or just kill the process
process(action="kill", session_id="<id>")
```

For a long bounded task, run `kilo run '...'` with `background=true, pty=true` and monitor the same way.

**Note:** bare `kilo` (no subcommand) opens the local Kilo Console in a browser — it is not a terminal session. For terminal orchestration use `kilo run` (one-shot) or `kilo run -i` (interactive).

Exit interactive sessions with Ctrl+C (`\x03`) or `process(action="kill")`. Do not rely on `/exit` — use the kill path.

### Resuming Sessions

After exiting, Kilo records a session ID. Resume with:

```
terminal(command="kilo run -c", workdir="~/project", background=true, pty=true)  # Continue last session
terminal(command="kilo run -s ses_abc123", workdir="~/project", background=true, pty=true)  # Specific session
```

Fork before continuing (new ID, keeps history):

```
terminal(command="kilo run --fork -s ses_abc123 'Try a different approach'", workdir="~/project")
```

Fetch a cloud session and continue locally:

```
terminal(command="kilo run --cloud-fork -s ses_abc123 'Continue this cloud session locally'", workdir="~/project")
```

## Common Flags

| Flag                             | Use                                                        |
| -------------------------------- | ---------------------------------------------------------- |
| `run 'prompt'`                   | One-shot execution and exit                                |
| `--auto`                         | Fully autonomous: auto-approve ALL permissions (CI/CD)     |
| `--continue` / `-c`              | Continue the last Kilo session                             |
| `--session <id>` / `-s`          | Continue a specific session                                |
| `--fork`                         | Fork the session before continuing (requires `-c` or `-s`) |
| `--cloud-fork`                   | Fetch a cloud session and continue locally (with `-s`)     |
| `--agent <name>`                 | Choose Kilo agent (code, plan, ask, debug, review)         |
| `--model provider/model` / `-m`  | Force specific model                                       |
| `--format json`                  | Machine-readable output/events                             |
| `--file <path>` / `-f`           | Attach file(s) to the message                              |
| `--thinking`                     | Show model thinking blocks                                 |
| `--variant <level>`              | Reasoning effort (high, max, minimal)                      |
| `--title <name>`                 | Name the session                                           |
| `--interactive` / `-i`           | Run in direct interactive split-footer mode (needs TTY)    |
| `--share`                        | Share the session                                          |
| `--dangerously-skip-permissions` | Auto-approve permission requests once                      |

## Procedure

1. Verify tool readiness:
   - `terminal(command="kilo --version")`
   - `terminal(command="kilo auth list")`
2. For bounded tasks, use `kilo run '...'` (no pty needed).
3. For CI/CD, use `kilo run --auto '...'`.
4. For iterative tasks, use `kilo run -i` with `background=true, pty=true`, or `kilo run '...'` with `background=true, pty=true` for a long bounded task.
5. Monitor long tasks with `process(action="poll"|"log")`.
6. If Kilo asks for input, respond via `process(action="submit", ...)`.
7. Exit with `process(action="write", data="\x03")` or `process(action="kill")`.
8. Summarize file changes, test results, and next steps back to user.

## PR Review Workflow

Kilo has a built-in PR command that fetches and checks out the PR branch, imports any session referenced in the PR body, then runs Kilo:

```
terminal(command="kilo pr 42", workdir="~/project", pty=true)
```

Or review in a temporary clone for isolation:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout <PR_NUMBER> && kilo run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')")
```

## Parallel Work Pattern

Use separate workdirs/worktrees to avoid collisions:

```
terminal(command="kilo run --auto 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="kilo run --auto 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")
```

## Session & Cost Management

List past sessions:

```
terminal(command="kilo session list")
```

Check token usage and costs:

```
terminal(command="kilo stats")
terminal(command="kilo stats --days 7 --models anthropic/claude-sonnet-4")
```

## Pitfalls

- `kilo run` (default, non-interactive) auto-rejects permission requests unless `--auto` or `--dangerously-skip-permissions` is set.
- `kilo run` one-shot does NOT need pty. `kilo run -i` (interactive) requires a TTY stdout — use `pty=true`.
- Bare `kilo` opens the web Kilo Console, not a terminal session. For terminal orchestration use `kilo run` / `kilo run -i`.
- PATH mismatch can select the wrong Kilo binary/model config.
- If Kilo appears stuck, inspect logs before killing:
  - `process(action="log", session_id="<id>")`
- Avoid sharing one working directory across parallel Kilo sessions.
- `--auto` is dangerous — only use it in trusted environments (it approves every action).
- `--dangerously-skip-permissions` removes the approval gate for one-shots — treat it like `--auto` and only use it in an isolated or trusted workdir, never the user's primary repo.

## Verification

Smoke test:

```
terminal(command="kilo run 'Respond with exactly: KILO_SMOKE_OK'")
```

Success criteria:

- Output includes `KILO_SMOKE_OK`
- Command exits without provider/model errors
- For code tasks: expected files changed and tests pass

## Rules

1. Prefer `kilo run` for one-shot automation — it's simpler and doesn't need pty.
2. Use `--auto` for CI/CD pipelines only — it auto-approves everything.
3. Use interactive mode (`-i`) only when iteration is needed.
4. Always scope Kilo sessions to a single repo/workdir.
5. For long tasks, provide progress updates from `process` logs.
6. Report concrete outcomes (files changed, tests, remaining risks).
7. Exit interactive sessions with Ctrl+C or kill — use the kill path, not `/exit`.
