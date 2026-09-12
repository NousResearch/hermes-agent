# Claude Code CLI Reference — Fact Sheet

Curated values and gotchas for the claude-code skill. Deliberately small: this holds the facts that drift or bite, not an exhaustive inventory. For anything not here, run `claude --help` / `claude -p --help` (offline, always version-correct) or read the official docs:

- CLI reference: https://code.claude.com/docs/en/cli-reference
- Permission modes: https://code.claude.com/docs/en/permission-modes
- Permissions: https://code.claude.com/docs/en/permissions
- Auto mode config: https://code.claude.com/docs/en/auto-mode-config
- Headless (`-p`): https://code.claude.com/docs/en/headless
- Hooks: https://code.claude.com/docs/en/hooks
- MCP: https://code.claude.com/docs/en/mcp
- Commands & skills: https://code.claude.com/docs/en/commands

Last verified against Claude Code v2.1.246. `claude --help` does not list every flag — absence there proves nothing.

## Read-Only Enforcement (the core gotcha)

| Flag | What it actually does |
|------|-----------------------|
| `--tools <list>` | Sets the available **built-in** tools (`""` = none, `"default"` = all). Removes unlisted built-ins entirely — it does not affect MCP tools. |
| `--allowedTools <list>` | Auto-approves the listed tools. Unlisted tools stay available and just need approval — it does **not** restrict. |
| `--disallowedTools <list>` | Blocks the listed tools regardless of other rules; a bare name also removes the tool from context (like a bare `deny` rule). |
| `--safe-mode` | Disables CLAUDE.md, skills, plugins, hooks, MCP servers, commands, agents. Auth and built-in tools still work. |
| `--setting-sources <sources>` | Loads settings from only the named sources (`user`, `project`, `local`). `--setting-sources user` stops Claude reading the repo's `.claude/settings.json` (env block, permission rules) and `.mcp.json` — the strongest isolation for an untrusted repo. |
| `--bare` | Drops OAuth, hooks, plugins, project MCP, and CLAUDE.md auto-discovery — fastest startup, needs `ANTHROPIC_API_KEY`. Still reads the project's `env` block. Slated to become the default for `-p` in a future release. |

In `-p` mode (or any non-TTY stdout) the trust dialog is skipped and there is no user to approve prompts, so every `--allowedTools` admit is auto-approved. A "do not edit" line in the prompt is a request, not enforcement. For a truly read-only review: `--safe-mode --setting-sources user --tools "Read,Grep,Glob"`.

## Tool Name Syntax

```
Read                    # all file reading
Edit                    # edit existing files
Write                   # create new files
Bash                    # all shell commands
Bash(git *)             # any git command — NOT read-only (checkout/reset/clean/stash write)
Bash(git diff *)        # diffs — still not read-only: accepts --output=<file>
Bash(git log *)         # history — accepts --output=<file>
Bash(git show *)        # commit/object contents — accepts --output=<file>
Bash(npm run lint:*)    # wildcard patterns
WebSearch / WebFetch
mcp__<server>__<tool>   # a specific MCP tool
```

Never grant `Bash` for a review at all: `git diff`, `git log`, and `git show` all accept `--output=<file>`, so even the "read-only" git families can write. Pipe the diff in instead (`git diff ... | claude -p ...`).

## Permission Modes (`--permission-mode`)

`default` (canonical; the CLI labels it `manual`, an accepted alias), `acceptEdits`, `auto`, `bypassPermissions`, `dontAsk`, `plan`.

What each mode runs without asking: `default` = reads only · `acceptEdits` = reads, file edits, common filesystem commands · `plan` = reads, plus classifier-approved commands · `auto` = routine actions auto-approved, riskier ones classifier-reviewed (reads and ordinary in-scope edits skip the classifier) · `dontAsk` = only pre-approved tools, auto-denies the rest · `bypassPermissions` = everything without prompts (deny rules and hard safety checks still apply).

**Auto mode is the built-in default for interactive terminal/VS Code sessions on Pro, Max, and Team plans** (v2.1.228+ macOS/Linux/WSL, v2.1.233+ Windows). `-p` and the Agent SDK have a built-in starting mode of `default` on every plan — a `--permission-mode` flag or applicable settings can still select another. The classifier blocks irreversible, destructive, or external actions; `deny` and `ask` rules fire before it, and broad allow rules such as `Bash(python:*)` are suspended. Pushes to any branch (including the default) and PR creation are allowed by default — add `permissions.ask` on `Bash(git push *)` and `Bash(gh pr create *)` for a human checkpoint.

Plan mode routes shell commands through the classifier by default (`useAutoModeDuringPlan` is on) when auto mode is available — a review convenience, not an OS security boundary. The docs' CI lockdown pattern is `--permission-mode dontAsk --allowedTools "Bash(npm test)" "Read"`.

## Model & Effort

- Model aliases: `fable`, `opus`, `sonnet`, `haiku` (or a full name like `claude-fable-5`). `--fallback-model haiku` auto-falls-back when the default is overloaded.
- Effort levels: `low`, `medium`, `high`, `xhigh`, `max`, `ultracode` (xhigh + ultracode on; v2.1.203+).

## JSON Result & Gating

```json
{
  "type": "result",
  "subtype": "success",
  "result": "The analysis text...",
  "session_id": "75e2167f-...",
  "num_turns": 3,
  "total_cost_usd": 0.0787,
  "duration_ms": 10276,
  "usage": { "input_tokens": 5, "output_tokens": 603 },
  "modelUsage": { "claude-fable-5": { "costUSD": 0.078, "contextWindow": 200000 } }
}
```

Success is `subtype == "success"`. Error subtypes — `error_max_turns`, `error_max_budget_usd`, `error_during_execution`, `error_max_structured_output_retries` — have no `result` field. With `--json-schema`, the validated object lands in `structured_output`.

`--max-turns` counts tool-use turns (one response plus all the tool calls it issues), not file reads. `--max-budget-usd` is a soft cap — a run can overshoot by one turn before it stops.

## Settings & Permissions Precedence

Managed (enterprise) settings > CLI flags > local project (`.claude/settings.local.json`, gitignored) > project (`.claude/settings.json`, git-tracked) > user (`~/.claude/settings.json`).

Deny rules win across all scopes: a `deny` blocks a call even when a matching `allow` also exists. A bare `deny: ["Bash"]` removes the tool from context entirely.

## CLAUDE.md / Memory

Hierarchy: `~/.claude/CLAUDE.md` (global) → `./CLAUDE.md` (project) → `./CLAUDE.local.md` (personal, at the project root). Modular form: `.claude/rules/*.md` (project) and `~/.claude/rules/*.md` (user). The 25KB / 200-line figure is the startup-load budget for `MEMORY.md`, not a total cap — topic files remain available on demand.

## Hooks

Configure in `.claude/settings.json` (project) or `~/.claude/settings.json` (global). Command hooks read tool input as JSON on **stdin** and return a decision on **stdout**. The `matcher` field names a tool (`"Bash"`, `"*"`); the optional `if` field narrows with permission-rule syntax. The reliably-set environment variable is `CLAUDE_PROJECT_DIR`.

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "if": "Bash(rm *)",
        "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-rm.sh"
      }]
    }]
  }
}
```

The hook script denies by printing a decision on stdout:

```bash
jq -n '{ hookSpecificOutput: { hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "Destructive command blocked" } }'
```

`exit 2` blocks regardless of any JSON on stdout; the reason shown to Claude is your stderr text (or the JSON denial reason if you emitted one). The full event list is 30+ (`PreToolUse`, `PostToolUse`, `SessionStart`, `PreCompact`, ...) and changes across releases — see the hooks docs rather than enumerating it here.

## MCP Scopes

| Scope | Storage | Visibility |
|-------|---------|------------|
| `local` (default) | `~/.claude.json`, keyed by project | This project, personal |
| `project` | `.mcp.json` at project root | This project, team-shared (git-tracked) |
| `user` | `~/.claude.json` | All projects, personal |

`claude mcp add` writes `local` unless you pass `--scope project` or `--scope user`.

## Subagent Precedence

Managed (enterprise) settings > `--agents` CLI flag (session) > `.claude/agents/` (project) > `~/.claude/agents/` (user) > plugins. Invoke manually with `@agent-<name>` (e.g. `@agent-security-reviewer`); `/agents` prints guidance for managing them.
