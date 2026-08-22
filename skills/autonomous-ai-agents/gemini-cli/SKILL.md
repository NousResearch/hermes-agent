---
name: gemini-cli
description: "Delegate coding, reviews, automation, and repository exploration to Google Gemini CLI / Gemini Code Assist CLI. Use when the user wants a Claude Code-like coding agent powered by Gemini, especially for large-context codebase analysis, headless scripting, Google Search-grounded tasks, MCP integrations, or parallel worktree agents."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Gemini, Google, Code-Assist, Code-Review, Automation, MCP, PTY]
    related_skills: [claude-code, codex, opencode, subagent-driven-development, requesting-code-review]
---

# Gemini CLI — Hermes Orchestration Guide

Delegate coding tasks to [Gemini CLI](https://github.com/google-gemini/gemini-cli), Google's open-source terminal coding agent. Gemini CLI is a Claude Code-like agent with interactive TUI mode, non-interactive headless mode, file operations, shell commands, web fetch/search, Google Search grounding, MCP support, GEMINI.md project context, sessions, checkpointing, policy controls, and experimental git worktrees.

Use this skill when the user says "Gemini CLI", "Gemini Code Assist", "Google coding agent", "类似 Claude Code 的 Gemini", or wants Gemini as one of several agents in a divide-and-conquer workflow.

## Current Known Baseline

As of this skill's creation, the npm package is:

```bash
npm view @google/gemini-cli version description bin
# version: 0.41.2
# bin: { gemini: 'bundle/gemini.js' }
```

The CLI command is `gemini`. Prefer verifying live help before relying on any flag because Gemini CLI evolves quickly:

```bash
gemini --version
gemini --help
gemini mcp --help
gemini skills --help
gemini extensions --help
gemini hooks --help
```

## Prerequisites

- **Node/npm:** Node.js installed; npm available.
- **Install:**
  ```bash
  npm install -g @google/gemini-cli
  # or no-install run:
  npx @google/gemini-cli
  ```
- **Version check:**
  ```bash
  gemini --version
  ```
- **Authentication:** choose one:
  1. **Sign in with Google / Gemini Code Assist** — best for individual users and Gemini Code Assist users; run `gemini` interactively and select Google sign-in. On remote PTYs, be ready to relay the OAuth URL/code. Some versions also accept `GOOGLE_GENAI_USE_GCA=true` to select the Gemini Code Assist auth path.
  2. **Gemini API key** — best for headless automation; set `GEMINI_API_KEY` from AI Studio.
  3. **Vertex AI** — enterprise/production; set `GOOGLE_GENAI_USE_VERTEXAI=true`, a project, region if needed, and Google credentials/API key.

### Authentication Matrix

| Scenario | Recommended auth | Notes |
|---|---|---|
| Individual Google account | Sign in with Google | Browser login; no project usually needed. |
| Google AI Pro/Ultra | Sign in with Google | Use the subscribed Google account. |
| Workspace/company/school account | Sign in with Google + `GOOGLE_CLOUD_PROJECT` | Org accounts commonly require a Cloud project. |
| Gemini Code Assist license | Sign in with Google / GCA + often `GOOGLE_CLOUD_PROJECT` | Set the licensed project when required; for non-interactive auth selection, some versions expect `GOOGLE_GENAI_USE_GCA=true`. |
| Headless scripts / cron / CI | `GEMINI_API_KEY` or Vertex AI | Avoid OAuth browser prompts. |
| Enterprise / GCP workloads | Vertex AI | Usually requires `GOOGLE_CLOUD_PROJECT`. |

### API Key Setup

```bash
# Get a key from https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="YOUR_KEY"
gemini -p "Say hello" --output-format json
```

### Vertex AI Setup

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
# Depending on environment, also configure ADC:
gcloud auth application-default login
# or service-account credentials:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

gemini -p "Say hello from Vertex AI" --output-format json
```

### Google Sign-In Setup

```bash
gemini
# Select: Sign in with Google
# Follow browser flow
```

For remote/headless gateway hosts, prefer API key or Vertex AI. Google sign-in requires a browser on a machine that can communicate with the terminal running Gemini CLI; unlike Claude Code's copy-code flow, Gemini's exact remote login UX may vary by version. If the CLI prints a URL/code, keep the process alive in a PTY/tmux session and relay the full URL/code to the user. See `references/remote-oauth-code-flow.md` for the observed remote OAuth/code flow, trust prompt, and auth error-code 41 handling.

## Hermes Orchestration Modes

Hermes interacts with Gemini CLI in three main ways.

### Mode 1: Headless Print Mode (`-p`) — Preferred for Most Tasks

Use headless mode for single tasks, automation, reviews, summaries, and CI-style work. It exits when done and is easiest to verify.

```bash
gemini -p "Explain the architecture of this codebase" --output-format json
```

Hermes pattern:

```python
terminal(
  command="gemini -p 'Review src/auth.ts for security issues. Return concrete findings.' --output-format json",
  workdir="/path/to/project",
  timeout=180,
)
```

Use headless mode when:

- You want one-shot code analysis, review, refactor request, or test writing.
- You need structured JSON output.
- You run from cron/CI/gateway without an interactive human.
- You are piping known context into Gemini.
- You want to avoid TUI dialogs.

Headless mode is triggered by `-p/--prompt` or by non-TTY execution. Output formats:

| Flag | Behavior |
|---|---|
| `--output-format text` | Plain final answer. |
| `--output-format json` | Single JSON object with `response`, `stats`, optional `error`. |
| `--output-format stream-json` | JSONL events: `init`, `message`, `tool_use`, `tool_result`, `error`, `result`. |

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | Success |
| `1` | General/API failure |
| `42` | Input/argument error |
| `53` | Turn limit exceeded |

### Mode 2: Interactive TUI via tmux — Multi-Turn Development

Interactive mode is useful for iterative development, slash commands, permission prompts, worktrees, session restore, and long multi-step work. Use tmux so Hermes can send input and capture output.

```bash
# Start session
terminal(command="tmux new-session -d -s gemini-work -x 140 -y 40")

# Launch Gemini in a project
terminal(command="tmux send-keys -t gemini-work 'cd /path/to/project && gemini' Enter")

# Let it start; if a trust/auth dialog appears, inspect before sending keys
terminal(command="sleep 5 && tmux capture-pane -t gemini-work -p -S -80")

# Send task
terminal(command="tmux send-keys -t gemini-work 'Refactor auth to use short-lived JWT access tokens and refresh-token rotation. Add tests.' Enter")

# Monitor progress
terminal(command="sleep 20 && tmux capture-pane -t gemini-work -p -S -100")

# Exit when done
terminal(command="tmux send-keys -t gemini-work '/quit' Enter")
```

Use interactive mode when:

- You need `/auth`, `/permissions`, `/model`, `/memory`, `/tools`, `/mcp`, `/resume`, `/restore`, `/plan`, or `/agents`.
- The task benefits from follow-up steering.
- You expect permission prompts and want to approve selectively.
- You want Gemini's interactive session memory and context display.

### Mode 3: Isolated Worktree Agents — Parallel Gemini Instances

Gemini CLI supports experimental git worktrees. Enable first:

```json
// ~/.gemini/settings.json or project .gemini/settings.json
{
  "experimental": {
    "worktrees": true
  }
}
```

Then:

```bash
gemini --worktree feature-search
# or
 gemini -w feature-search
```

For Hermes-run parallel agents, prefer explicit shell worktrees or Gemini's `-w` if verified. Worktrees prevent multiple agents from clobbering the same files.

```bash
# Agent A: implementation
tmux new-session -d -s gemini-impl -x 140 -y 40
 tmux send-keys -t gemini-impl 'cd /repo && gemini -w impl-auth' Enter

# Agent B: tests/review
tmux new-session -d -s gemini-test -x 140 -y 40
 tmux send-keys -t gemini-test 'cd /repo && gemini -w test-auth' Enter
```

After parallel exploration, Hermes must inspect diffs, run tests, and manually merge chosen changes. Do not assume sub-agent claims are correct.

## CLI Quick Reference

### Main Command

```bash
gemini [options] [query..]
```

Common options verified from `gemini --help`:

| Flag | Purpose |
|---|---|
| `-p, --prompt <text>` | Non-interactive/headless prompt. |
| `-i, --prompt-interactive <text>` | Execute initial prompt then stay interactive. |
| `-m, --model <model>` | Select model. |
| `-o, --output-format text|json|stream-json` | Output format. |
| `--raw-output` | Disable sanitization; only for trusted output. |
| `--accept-raw-output-risk` | Suppress raw-output warning. |
| `--skip-trust` | Trust current workspace for this session. |
| `-s, --sandbox` | Enable sandboxing. |
| `-y, --yolo` | Auto-accept all actions. Use cautiously. |
| `--approval-mode default|auto_edit|yolo|plan` | Permission behavior. |
| `--policy <paths>` | Additional policy files/dirs. |
| `--admin-policy <paths>` | Additional admin policy files/dirs. |
| `--allowed-mcp-server-names <names>` | Restrict MCP servers. |
| `--include-directories <dirs>` | Add workspace directories. |
| `-r, --resume <latest|index|id>` | Resume a previous session. |
| `--session-id <uuid>` | Start with explicit session UUID. |
| `--list-sessions` | List project sessions. |
| `--delete-session <index>` | Delete session. |
| `-w, --worktree [name]` | Start in a new git worktree. |
| `--acp` | Start in ACP mode. |
| `-e, --extensions <names>` | Select extensions. |
| `-l, --list-extensions` | List extensions. |

### Subcommands

```bash
gemini mcp        # manage MCP servers
gemini skills     # manage Gemini agent skills
gemini hooks      # manage hooks
gemini extensions # manage extensions
gemini gemma      # local Gemma routing
```

MCP commands:

```bash
gemini mcp add <name> <commandOrUrl> [args...]
gemini mcp remove <name>
gemini mcp list
gemini mcp enable <name>
gemini mcp disable <name>
```

Skills commands:

```bash
gemini skills list --all
gemini skills enable <name>
gemini skills disable <name> [--scope]
gemini skills install <source> [--scope] [--path]
gemini skills link <path>
gemini skills uninstall <name> [--scope]
```

Extensions commands:

```bash
gemini extensions install <source> [--auto-update] [--pre-release]
gemini extensions list
gemini extensions update [<name>] [--all]
gemini extensions disable [--scope] <name>
gemini extensions enable [--scope] <name>
gemini extensions link <path>
gemini extensions new <path> [template]
gemini extensions validate <path>
gemini extensions config [name] [setting]
```

Hooks:

```bash
gemini hooks migrate   # migrate hooks from Claude Code to Gemini CLI
```

## Interactive Slash Commands

Use these inside the Gemini TUI:

| Command | Purpose |
|---|---|
| `/help` or `/?` | Help. |
| `/about` | Version/system info. |
| `/auth` | Change authentication method. |
| `/model` | Change model. |
| `/permissions` | View/update permissions. |
| `/policies` | Inspect policies. |
| `/tools` | Inspect tools. |
| `/mcp` | Manage MCP interactively. |
| `/skills` | Manage Gemini agent skills. |
| `/extensions` | Manage extensions. |
| `/hooks` | Manage hooks. |
| `/memory show|reload|add` | Inspect/reload/add GEMINI.md memory. |
| `/init` | Initialize project context. |
| `/plan` | Plan mode. |
| `/agents list|reload|enable|disable|config` | Manage local/remote subagents. |
| `/resume` or `/chat` | Resume/list/save/share session checkpoints. |
| `/restore` | Restore automatic file-change checkpoint. |
| `/rewind` | Rewind session/code state. |
| `/stats` | Usage statistics. |
| `/theme`, `/settings`, `/terminal-setup`, `/vim` | UI/preferences. |
| `/quit` or `/exit` | Exit in many versions, but if the TUI input does not submit slash commands from a PTY automation session, use `Ctrl+C` once or kill/close the tracked process. Prefer headless `-p` for Hermes automation to avoid this issue. |

Prompt prefixes:

| Prefix | Meaning |
|---|---|
| `@path` | Reference files/directories/resources. |
| `!command` | Shell passthrough. |
| `/command` | Slash command. |

## GEMINI.md Project Context

Gemini CLI uses `GEMINI.md` files for persistent context, similar to Claude Code's `CLAUDE.md`.

Load order:

1. Global: `~/.gemini/GEMINI.md`
2. Workspace/project `GEMINI.md` files in configured directories and parents
3. Just-in-time `GEMINI.md` files discovered near files Gemini touches

Example project `GEMINI.md`:

```markdown
# Project: My Service

## Architecture
- FastAPI backend, SQLAlchemy ORM, PostgreSQL, Redis.
- Frontend is React + Vite.

## Commands
- `make test` runs all tests.
- `make lint` runs ruff, mypy, eslint.
- `make dev` starts local services.

## Rules
- Add tests for every behavior change.
- Do not modify `.env` or secrets.
- Prefer small, reviewable commits.
```

Manage memory interactively:

```text
/memory show
/memory reload
/memory add Always run make test before finalizing backend changes.
```

Customize context filenames:

```json
{
  "context": {
    "fileName": ["AGENTS.md", "GEMINI.md", "CONTEXT.md"]
  }
}
```

## Configuration Files

Settings precedence, low to high:

1. Built-in defaults
2. System defaults file
3. User settings file
4. Project settings file
5. System settings override
6. Environment variables / `.env`
7. CLI arguments

Important locations:

| Scope | Location |
|---|---|
| User settings | `~/.gemini/settings.json` |
| Project settings | `.gemini/settings.json` |
| Linux system defaults | `/etc/gemini-cli/system-defaults.json` |
| Linux system overrides | `/etc/gemini-cli/settings.json` |
| Project files | `.gemini/` |
| History/checkpoints | `~/.gemini/tmp/<project_hash>/`, `~/.gemini/history/<project_hash>` |

Minimal user settings example:

```json
{
  "general": {
    "preferredEditor": "code"
  },
  "experimental": {
    "worktrees": true
  },
  "general": {
    "checkpointing": {
      "enabled": true
    }
  }
}
```

Avoid duplicate top-level keys when writing real JSON; merge objects properly:

```json
{
  "general": {
    "preferredEditor": "code",
    "checkpointing": {
      "enabled": true
    }
  },
  "experimental": {
    "worktrees": true
  }
}
```

## Permissions, Policy Engine, and Safety

Gemini CLI has approval modes:

| Mode | Behavior |
|---|---|
| `default` | Prompt for approval. |
| `auto_edit` | Auto-approve edit tools; ask for broader actions. |
| `plan` | Read-only planning mode. |
| `yolo` / `-y` | Auto-approve all tools. Dangerous; use only in disposable sandboxes. |

Use policy files for fine-grained control. User policy example:

```bash
mkdir -p ~/.gemini/policies
cat > ~/.gemini/policies/safe-shell.toml <<'EOF'
[[rule]]
toolName = "run_shell_command"
commandPrefix = "rm -rf"
decision = "deny"
priority = 1000

[[rule]]
toolName = "run_shell_command"
commandPrefix = "git push --force"
decision = "deny"
priority = 1000

[[rule]]
toolName = "run_shell_command"
commandPrefix = "git"
decision = "ask_user"
priority = 100
EOF
```

Decisions:

- `allow`: execute automatically.
- `ask_user`: prompt user; in non-interactive mode this is treated as deny.
- `deny`: block. Global deny may remove tools from model awareness.

Important pitfall: project-level workspace policies may be unreliable in some versions; prefer user/admin policy files for security-critical rules.

## Sandboxing

Enable sandboxing for untrusted code or high-risk commands.

```bash
# Flag
gemini -s -p "analyze this repo and run tests"

# Environment
export GEMINI_SANDBOX=true
gemini -p "run test suite"

# Docker sandbox
export GEMINI_SANDBOX=docker
gemini -p "build the project"
```

Settings example:

```json
{
  "tools": {
    "sandbox": "docker"
  }
}
```

Docker/Podman sandbox mounts the current workspace at the same absolute path inside the container. Custom image:

```json
{
  "tools": {
    "sandbox": {
      "command": "docker",
      "image": "my-gemini-sandbox:latest"
    }
  }
}
```

Or set:

```bash
export GEMINI_SANDBOX_IMAGE="my-gemini-sandbox:latest"
```

## Checkpointing and Restore

Gemini can automatically checkpoint before file modifications. Enable in settings:

```json
{
  "general": {
    "checkpointing": {
      "enabled": true
    }
  }
}
```

Restoration is via interactive `/restore`. Checkpoints store a shadow Git snapshot and conversation/tool state under `~/.gemini/history/<project_hash>` and `~/.gemini/tmp/<project_hash>/checkpoints`.

Use this before risky refactors or broad edits. It does not replace normal Git discipline; still inspect `git diff` and commit intentionally.

## MCP Integration

Add MCP servers for GitHub, databases, browsers, cloud tools, etc.

```bash
# Add a stdio MCP server
gemini mcp add github npx @modelcontextprotocol/server-github

# List/enable/disable
gemini mcp list
gemini mcp disable github
gemini mcp enable github
```

Limit MCP exposure for a run:

```bash
gemini --allowed-mcp-server-names github -p "Summarize my open PRs"
```

When running with MCP in headless mode, prefer a policy that denies destructive actions unless explicitly needed.

## One-Shot Recipes

See also `references/headless-plan-review.md` for a verified Hermes pattern for scanning a repository plan with Gemini in non-interactive mode, including trusted-folder handling and model-routing verification from JSON stats.

### Install / Verify

```bash
node --version
npm --version
command -v gemini || npm install -g @google/gemini-cli
gemini --version
gemini --help | sed -n '1,120p'
```

### Headless Codebase Summary

```bash
gemini -p "Analyze this repository. Summarize architecture, entrypoints, test commands, and risks." --output-format json
```

Hermes should parse `response`, then verify claims by reading files or running commands.

### Security Review of Current Diff

```bash
git diff --stat
git diff | gemini -p "Review this diff for security bugs, correctness issues, missing tests, and risky migrations. Return prioritized findings." --output-format json
```

### Ask Gemini to Implement a Bounded Change

```bash
gemini -p "Implement the smallest safe change to add request timeout handling in src/api. Add or update tests. Do not change public APIs unless necessary." --approval-mode auto_edit --output-format json
```

After completion, Hermes must run:

```bash
git diff --stat
git diff
# project-specific tests, e.g.
npm test
# or
pytest
```

### Plan-Only Pass

```bash
gemini -p "Plan how to migrate auth from sessions to JWT. Do not edit files. Identify files, tests, risks, and rollback plan." --approval-mode plan --output-format json
```

### Stream Long Task

```bash
gemini -p "Run the test suite, diagnose failures, and propose fixes without editing files." --output-format stream-json
```

Hermes can capture JSONL events and watch `tool_use`, `tool_result`, and final `result`.

### Interactive Development Session

```bash
tmux new-session -d -s gemini-dev -x 140 -y 40
tmux send-keys -t gemini-dev 'cd /path/to/repo && gemini --skip-trust' Enter
sleep 5 && tmux capture-pane -t gemini-dev -p -S -80
tmux send-keys -t gemini-dev 'Implement feature X. Keep changes minimal and add tests.' Enter
sleep 30 && tmux capture-pane -t gemini-dev -p -S -120
```

### Parallel Divide-and-Conquer

Use Gemini + Claude + Hermes subagents, or multiple Gemini worktrees.

1. Planner: `gemini -p "Plan implementation..." --approval-mode plan`.
2. Implementer: Gemini or Claude in a worktree.
3. Reviewer: separate Gemini headless review of diff.
4. Test agent: Hermes subagent runs tests and diagnoses failures.
5. Integrator: Hermes merges final changes and verifies.

## Hermes Verification Rules

Never trust a Gemini agent's self-report for side effects. Before telling the user a coding task is done:

1. Inspect changed files:
   ```bash
   git status --short
   git diff --stat
   git diff
   ```
2. Run project-specific tests/lints/builds.
3. If Gemini created a worktree, inspect that worktree directly.
4. Confirm no secrets or unrelated files were changed.
5. Summarize exact changed files and verification commands.

## Common Pitfalls

1. **Assuming OAuth works on a server.** Remote gateway hosts often cannot complete browser login. Prefer `GEMINI_API_KEY` or Vertex AI for headless/server use. If the user chooses Google/Gemini Code Assist sign-in, keep the PTY alive, handle the workspace trust prompt first, relay the printed OAuth URL, and submit the returned authorization code into the same waiting process. A headless JSON error with code `41` means auth is not configured; choose `GEMINI_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_GENAI_USE_GCA`, or a settings.json auth method rather than treating it as an install failure.

2. **Using `-y/--yolo` on a real repo.** YOLO auto-accepts everything. Use only in disposable branches/worktrees/sandboxes, and still inspect diffs.

3. **Forgetting that `ask_user` policies deny in headless mode.** If a headless run needs a tool, policy must allow it; otherwise the tool may be unavailable.

4. **Mixing multiple agents in one worktree.** Use separate git worktrees for parallel coding agents to avoid clobbered changes.

5. **Expecting project policy files to be authoritative.** Some Gemini versions warn workspace policy tier is unreliable. Use user/admin policies for security-critical denies.

6. **Relying on stale CLI flags.** Gemini CLI changes quickly. Always run `gemini --help` and subcommand help on the installed version before constructing automation.

7. **Letting context files get huge or contradictory.** Keep `GEMINI.md` concise and project-specific. Use imports for modular context.

8. **Not setting `GOOGLE_CLOUD_PROJECT` for org accounts.** Workspace/Gemini Code Assist accounts often need a Cloud project.

9. **Assuming checkpointing is on.** It is disabled by default; enable via settings, not a removed `--checkpointing` flag.

10. **Confusing Gemini CLI skills with Hermes skills.** `gemini skills ...` manages Gemini's own agent skills under Gemini's ecosystem; Hermes `skill_view`/`skill_manage` manages Hermes skills. They are separate.

11. **Not using `workdir`.** Always set Hermes `terminal(workdir=...)` or `cd` explicitly so Gemini sees the intended repo and GEMINI.md files. Prefer placing active working git repositories in a normal project directory (for example, `$HOME/<repo>`), not inside hidden agent/tool directories such as `.hermes/` or `.codex/`, unless the user explicitly requests that location.

12. **Headless runs in untrusted repositories.** In non-interactive mode Gemini may refuse with a trusted-folder error or override `--approval-mode plan` to `default`. For automated read-only scans, add `--skip-trust` (or set `GEMINI_CLI_TRUST_WORKSPACE=true`) after verifying the intended `workdir`.

13. **Raw output risk.** `--raw-output` can allow unsafe terminal sequences. Use only for trusted contexts and include `--accept-raw-output-risk` only when necessary.

## Recommended Prompt Templates

### Implementation Prompt

```text
You are working in this repository. Implement: <feature>.
Constraints:
- Make the smallest safe change.
- Preserve public APIs unless explicitly required.
- Add or update tests.
- Do not edit secrets, lockfiles, generated files, or unrelated formatting unless necessary.
- After edits, run the relevant tests and report exact commands/results.
Return a concise summary of files changed, behavior changed, and remaining risks.
```

### Review Prompt

```text
Review the current diff for correctness, security, performance, maintainability, and missing tests.
Prioritize findings by severity.
For each finding include: file/path, line or function, why it matters, and a concrete fix.
Do not praise. If no blocking issues, say so and list non-blocking suggestions separately.
```

### Debugging Prompt

```text
Diagnose this failing command: <command>.
First inspect the error and relevant code paths. Identify root cause before editing.
If editing is needed, make the smallest fix and add a regression test.
Run the failing command again and report before/after results.
```

### Plan Prompt

```text
Create an implementation plan for <task>.
Do not edit files.
Include: affected files, data/model/API changes, tests, risks, rollback plan, and a sequence of small commits.
Call out unknowns that need user decisions.
```

## Verification Checklist

- [ ] `gemini --version` works.
- [ ] Authentication method chosen and verified with a small `gemini -p` call.
- [ ] For headless use, `GEMINI_API_KEY` or Vertex AI is configured; no browser prompt required.
- [ ] Project `GEMINI.md` exists or absence is intentional.
- [ ] Approval mode/policies match the risk level.
- [ ] Sandboxing/worktree used for risky or parallel edits.
- [ ] `git diff` inspected after any Gemini edits.
- [ ] Tests/lints/builds run and results captured.
- [ ] tmux sessions/worktrees cleaned up or clearly reported for continuation.
