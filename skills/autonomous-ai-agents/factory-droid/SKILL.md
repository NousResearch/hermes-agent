---
name: factory-droid
description: "Delegate coding to Factory's droid CLI for PRs and features."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Coding-Agent, Factory-AI, Droid, Autonomous, Refactoring, Code-Review]
    related_skills: [claude-code, codex, opencode]
---

# Factory Droid — Hermes Orchestration Guide

Delegate coding tasks to [Factory AI](https://factory.ai) via the `droid` CLI. `droid` is an autonomous AI software development agent with interactive mode, headless exec, specification mode, and multi-agent mission orchestration.

## When to Use

- User explicitly asks to use Factory / droid
- You want a coding agent with enterprise compliance (SOC 2 / ISO 27001/42001, HIPAA BAA available)
- The task benefits from spec-first planning (`--use-spec`) or multi-agent orchestration (`--mission`)
- You need a long-running autonomous task in an isolated worktree
- BYOK to any provider — use your own API keys for OpenAI, Anthropic, OpenRouter, Ollama

**Not sure if droid is the right call?** Compare against the `opencode` and `claude-code` skills for the delegation decision tree.

## Prerequisites

- **Install:** `curl -fsSL https://app.factory.ai/cli | sh`
- **Auth:** `droid` to log in interactively, or set up BYOK API keys at factory.ai/settings/api-keys
- **Verify:** `droid exec -o json --auto low "Respond with: OK"` returns `{"result":"OK",...}`
- **Git repository** for code tasks (recommended)

## One-Shot Tasks

Use `droid exec` for bounded, non-interactive tasks:

```
terminal(command="droid exec --auto medium \"Add retry logic to API calls and update tests\"", workdir="~/project", timeout=120)
```

**Structured JSON output:**

```
terminal(command="droid exec -o json --auto low \"List all functions in src/auth.py\"", workdir="~/project", timeout=60)
```

Returns `{"type":"result","subtype":"success","result":"...","session_id":"...","num_turns":...,"duration_ms":...,"usage":{...}}`.

**File-based prompt:**

```
terminal(command="droid exec -f .factory/review.md", workdir="~/project", timeout=120)
```

**Piped input for diff review:**

```
terminal(command="git diff HEAD~3 | droid exec \"Draft release notes for these changes\"", workdir="~/project", timeout=60)
```

## Autonomy Levels

`droid exec` uses tiered autonomy. Choose deliberately:

| Level | Intended for | Notable allowances |
|-------|-------------|-------------------|
| *(default)* | Read-only reconnaissance | File reads, git diffs, environment inspection |
| `--auto low` | Safe edits | Create/edit files, run formatters |
| `--auto medium` | Local development | Install deps, build/test, git commit |
| `--auto high` | CI/CD & orchestration | Git push, deploy scripts |
| `--skip-permissions-unsafe` | Isolated sandboxes only | Removes all guardrails (⚠️) |

**Examples:**

```
# Read-only analysis
terminal(command="droid exec \"Analyze the auth system\"", workdir="~/project")

# Low autonomy — safe edits
terminal(command="droid exec --auto low \"Add JSDoc to all functions in src/\"", workdir="~/project")

# Medium autonomy — full dev cycle
terminal(command="droid exec --auto medium \"Install deps, run tests, fix failures\"", workdir="~/project")

# High autonomy — deploy
terminal(command="droid exec --auto high \"Run tests, commit, push, deploy\"", workdir="~/project")
```

## Specification Mode

Plan before executing. `--use-spec` makes droid write a specification first, then implement:

```
terminal(command="droid exec --use-spec --auto high \"Add user profiles with avatar upload\"", workdir="~/project", timeout=300)
```

Use different models for planning vs execution:

```
terminal(command="droid exec --use-spec --spec-model claude-opus-4-7 --auto medium \"Design and implement payment flow\"", workdir="~/project", timeout=300)
```

## Multi-Agent Missions

Mission mode spawns worker and validator agents for large projects:

```
terminal(command="droid exec --mission -f mission.md", workdir="~/project", timeout=600)
```

Mission spec format (write to `mission.md`):

```markdown
# Mission: Refactor Auth

## Goal
Replace session-based auth with JWT-based auth.

## Scope
- src/auth/session.py → src/auth/jwt.py
- All route handlers importing from sessions
- Test files for auth

## Constraints
- Use PyJWT library
- Token expiry: 15 min access, 7 day refresh
```

With model tuning:

```
terminal(command="droid exec --mission -f mission.md --worker-model claude-sonnet-4-6 --validator-model claude-opus-4-7", workdir="~/project", timeout=600)
```

## Worktree Isolation

Run tasks in isolated git worktrees:

```
# Interactive
terminal(command="droid --worktree fix-auth-bug \"start debugging\"", workdir="~/project", timeout=300)

# Headless
terminal(command="droid exec --worktree refactor-tests --auto medium \"migrate tests\"", workdir="~/project", timeout=300)
```

Worktree lifecycle: interactive persists; `droid exec` auto-removes clean worktrees, preserves dirty ones (prints path). Git branch is never deleted.

## Session Continuation

```
# Resume last session
terminal(command="droid exec -s session-abc123 \"continue fixing auth\"", workdir="~/project")

# Fork into new session
terminal(command="droid exec --fork session-abc123 \"Try different approach\"", workdir="~/project")
```

## Model Selection

```
# Specific model
terminal(command="droid exec -m claude-sonnet-4-6 \"Refactor module\"", workdir="~/project")

# Different models for spec vs execution
terminal(command="droid exec --use-spec --spec-model claude-opus-4-7 --auto medium \"Design and implement\"", workdir="~/project")
```

Common model IDs: `claude-sonnet-4-6`, `claude-opus-4-7`, `claude-haiku-4`, `gpt-5`, `gpt-5-mini`, `deepseek-v4-flash`.

## Key Flags

| Flag | Effect |
|------|--------|
| `exec "prompt"` | Non-interactive execution |
| `-f, --file <path>` | Read prompt from file |
| `-m, --model <id>` | Select model by ID |
| `-o, --output-format <fmt>` | `text` (default), `json`, `stream-json` |
| `-s, --session-id <id>` | Continue an existing session |
| `--auto <level>` | Autonomy level: `low`, `medium`, `high` |
| `--use-spec` | Start in specification mode |
| `--spec-model <id>` | Model for spec planning |
| `--mission` | Multi-agent orchestration mode |
| `--worker-model <id>` / `--validator-model <id>` | Mission agent models |
| `-r, --reasoning-effort <level>` | `off`, `low`, `medium`, `high` |
| `--enabled-tools <ids>` / `--disabled-tools <ids>` | Tool control |
| `--skip-permissions-unsafe` | Remove guardrails (⚠️ sandbox only) |
| `-w, --worktree [name]` | Isolated git worktree |
| `--fork <id>` | Fork and resume a session |

## Cost & Spend Control

With BYOK, tokens come from your own API keys — no Factory cost. With managed models:

| Plan | Price |
|------|-------|
| BYOK | Free |
| Pro | $20/mo (10M+10M tokens) |
| Max | $200/mo (100M+100M tokens) |

Track usage: `droid exec -o json` returns `usage.input_tokens` / `usage.output_tokens` / `usage.cache_read_input_tokens` in the result.

## Pitfalls

1. **`droid exec` doesn't need pty** — only interactive `droid` is a TUI. Use non-pty terminal for exec.
2. **`droid auth status --text` is interactive-only** — verify auth with `droid exec` instead.
3. **`--use-spec` doubles token burn** — generates spec + code in one call.
4. **Mission mode is asynchronous** — very large missions may return before workers complete. Check via session resume.
5. **Dirty worktrees persist** — `droid exec` leaves dirty worktrees on disk. Clean up manually.
6. **Worktree branch naming** — `--worktree` derives `<current>-wt` from current branch. Explicit naming with `--worktree <name>` is more predictable.
7. **Shell escaping** — for complex prompts with special characters, use `-f prompt.md` instead of inline strings.

## Rules

1. **Prefer `droid exec`** for one-shot automation — simpler, no TUI dialog handling.
2. **Use `-o json`** for structured output parsing and cost tracking.
3. **Always set `workdir`** — keep droid focused on the right project.
4. **Set timeouts proportional to task** — 60s read-only, 120-300s dev, 600s+ missions.
5. **Use `--auto` levels deliberately** — never default to `--skip-permissions-unsafe`.
6. **Isolate missions in worktrees** — prevents file conflicts with parallel work.
7. **Report concrete outcomes** — files changed, test results, token cost, session ID.
