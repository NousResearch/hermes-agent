---
name: claude-code
description: "Delegate coding tasks and pull-request reviews."
version: 3.0.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, Code-Review, Refactoring, PTY, Automation]
    related_skills: [codex, hermes-agent, opencode]
---

# Claude Code Skill

Drives Anthropic's [Claude Code](https://code.claude.com/docs/en/cli-reference) CLI through the Hermes `terminal` tool in two modes: non-interactive print mode (`claude -p`) for one-shot and scripted tasks, and interactive PTY sessions (`terminal(pty=true, background=true)` + `process`) for multi-turn work. It does not manage Claude authentication or billing — use `claude auth` for that. The gotchas and drift-prone values (permission modes, model aliases, tool syntax, result schema) live in `references/cli-reference.md`; for the exhaustive flag and subcommand inventory, run `claude --help` or read the official docs.

## When to Use

- **One-shot coding** — fix a bug, add a feature, refactor → print mode.
- **CI, scripting, automation** → print mode.
- **PR review**, especially read-only or verdict-gated → print mode with `--tools` or `--safe-mode`.
- **Structured extraction** against a JSON schema → print mode with `--json-schema` (result lands in the `structured_output` field).
- **Multi-turn interactive work**, slash commands, or human-in-the-loop decisions → `terminal(pty=true)` + `process`.

## Prerequisites

- Install: `npm install -g @anthropic-ai/claude-code`, or `claude install [stable|latest|version]`.
- Auth: run `claude` once to log in (OAuth for Pro/Max), or set `ANTHROPIC_API_KEY`. Console billing: `claude auth login --console`. Enterprise SSO: `claude auth login --sso`.
- Version: `claude --version` must be v2.x+.
- Verify: `claude doctor` (health) and `claude auth status --text` (login).
- Print mode works on all platforms. Native `terminal(pty=true)` also works everywhere; the `tmux` fallback for driving the TUI is POSIX-only.

## How to Run

Print mode is the default for most tasks — no PTY, no dialogs, returns once and exits:

```
terminal(command="claude -p 'Add error handling to all API calls in src/' --allowedTools 'Read,Edit' --max-turns 10", workdir="/path/to/project", timeout=120)
```

Always set `workdir` to the repo root. `-p` (or any non-TTY stdout) skips the interactive dialogs, so whatever `--allowedTools` admits is auto-approved — read the Pitfalls before trusting a "review only" instruction.

Interactive mode runs the TUI under a native PTY and drives it with the `process` tool:

```
terminal(command="claude", pty=true, background=true, workdir="/path/to/repo")    # returns a session id
process(action="submit", session_id="<id>", data="Refactor the auth module")      # send the task
process(action="poll", session_id="<id>")                                          # read output
process(action="submit", session_id="<id>", data="/exit")                          # quit cleanly (write "\x03" to cancel)
```

On POSIX systems you can substitute tmux for the same flow (`tmux new-session`, `send-keys`, `capture-pane`); use it when you need fine key control for the confirmation dialogs or multiple panes.

## Quick Reference

| Task | Command |
|------|---------|
| One-shot task | `claude -p "..." --allowedTools 'Read,Edit' --max-turns 10` |
| Read-only review (trusted repo) | `claude -p "review..." --tools "Read,Grep,Glob" --max-turns 25` |
| Read-only review (untrusted fork) | `claude -p "..." --safe-mode --setting-sources user --tools "Read,Grep,Glob" --max-turns 25` |
| Gated review | `git diff main...HEAD \| claude -p "..." --tools "Read,Grep,Glob" --output-format json --max-turns 25` |
| Structured output | `claude -p "..." --output-format json --json-schema '<schema>'` → `structured_output` |
| Continue last session | `claude -p "..." --continue` |
| Resume a session | `claude -p "..." --resume <id> --fork-session` |
| CI / bare mode | `claude --bare -p "..." --allowedTools 'Read,Bash'` |
| Interactive | `claude` (under `terminal(pty=true)`) |
| Session to JSON | `--output-format json` → `subtype`, `result`, `session_id`, `num_turns`, `total_cost_usd` |

Gotchas and drift-prone values (permission modes, model aliases, tool-name syntax, JSON result schema): `references/cli-reference.md`. Full flag inventory: `claude --help` and the official docs.

## Procedure

### One-Shot Task (Print Mode)

```
terminal(command="claude -p 'Refactor the auth module to use JWT tokens' --allowedTools 'Read,Edit' --max-turns 20", workdir="/path/to/repo", timeout=180)
```

A turn is one model response plus all the tool calls it issues — a single turn can read several files. Size `--max-turns` to the number of decision rounds, not the file count, and use `--output-format json` to read `num_turns` and `total_cost_usd` and confirm the run didn't die at the cap.

### Read-Only PR Review (Enforced)

`--allowedTools "Read,Bash"` does not make a review read-only. `Edit`/`Write` are excluded, but `Bash` is a write backdoor: in `-p` mode there is no user to approve prompts, so every allowed tool is auto-approved — `echo > file`, `sed -i`, and `git checkout` all run. A "do not edit" line in the prompt is a request, not enforcement.

Enforce read-only, strongest first:

1. `--safe-mode --setting-sources user --tools "Read,Grep,Glob"` — for untrusted forks. `--tools` removes the unlisted **built-in** tools (the cleanest read-only lever; a bare-name `--disallowedTools "Bash,Edit,Write"` also removes them); `--safe-mode` disables project hooks, MCP servers, skills, plugins, and CLAUDE.md; and `--setting-sources user` stops Claude reading the repo's `.claude/settings.json` (env block, permission rules) and `.mcp.json` at all. Pipe the diff in:
   ```
   git diff origin/main...HEAD | claude -p "$(cat /tmp/review-prompt.txt)" --safe-mode --setting-sources user --tools "Read,Grep,Glob" --output-format json --max-turns 25 < /dev/null
   ```
2. `--tools "Read,Grep,Glob"` — same built-in removal, for repos you already trust. It restricts built-ins only: configured hooks and MCP tools can still write, so for a hard boundary use the `--safe-mode` recipe above.
3. `--permission-mode plan` — blocks edits while keeping the toolset for exploration. Caveat: when auto mode is available (the default on Pro/Max/Team interactive sessions), `useAutoModeDuringPlan` is on by default, so the classifier approves shell commands during planning rather than prompting you; plan-mode blocks are also disabled when bypass permissions is available in the session — a review convenience, not an OS security boundary.
4. `--allowedTools "Read,Grep,Glob"` — auto-approves the listed tools but does **not** remove anything else; unlisted tools remain available (so this is weaker than `--tools`). If you need git context, pipe it in (`git diff origin/main...HEAD | claude -p ...`) rather than granting `Bash`: even `git diff`, `git log`, and `git show` accept `--output=<file>`, so any `Bash(git *)` grant is a write path.

For isolation rather than read-only (changes land in a throwaway worktree, not your checkout), use `claude -w <name>`.

### Verdict-Gated Review (JSON)

For an APPROVE / REQUEST CHANGES gate, use `--output-format json`, not `text`. Text cannot tell you whether the run hit `--max-turns`; JSON carries `subtype` (`success` | `error_max_turns` | `error_max_budget_usd` | `error_during_execution` | `error_max_structured_output_retries`), `num_turns`, and `total_cost_usd`. An error result has no `result` field.

```
set -o pipefail
claude -p "$(cat /tmp/review-prompt.txt)" --tools "Read,Grep,Glob" --output-format json --max-turns 25 < /dev/null \
  | jq -e -r 'if .subtype == "success" then .result else error("GATE FAILED: \(.subtype)") end'
```

`set -o pipefail` makes the pipeline fail if `claude` exits nonzero; `jq -e` + `error(...)` makes `jq` exit nonzero on a non-success subtype, so the gate fails closed either way. Budget 20-30 turns for a line-level adversarial review. `< /dev/null` stops the `"$(cat ...)"` prompt form from blocking on a TTY stdin in cron/CI. POSIX only — the `pipefail` + `jq` gate assumes a POSIX shell; on Windows wrap it in a PowerShell equivalent.

### Interactive Multi-Turn (PTY + dialogs)

Claude Code may present confirmation dialogs on first launch. With `terminal(pty=true)` drive them through `process`; with tmux use `send-keys`:

Interactive terminal/VS Code sessions on Pro/Max/Team plans now start in **auto mode** (v2.1.228+ on macOS/Linux/WSL, v2.1.233+ on Windows): the classifier auto-approves routine actions instead of prompting. Press `Shift+Tab` to switch to Manual (`default`) to restore prompts for edits, shell, and network actions.

1. Workspace trust (first visit): default "Yes, I trust this folder" — just submit an empty line / `Enter`.
2. Bypass-permissions warning (only with `--dangerously-skip-permissions`): default "No, exit" — send `Down` then `Enter`.

```
terminal(command="tmux new-session -d -s claude-work -x 140 -y 40")
terminal(command="tmux send-keys -t claude-work 'cd /path/to/project && claude' Enter")
terminal(command="tmux send-keys -t claude-work Enter")                                       # trust dialog
terminal(command="tmux send-keys -t claude-work Down && sleep 0.3 && tmux send-keys -t claude-work Enter")  # permissions dialog
terminal(command="tmux send-keys -t claude-work 'Refactor the auth module' Enter")            # the task
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -60")                    # monitor
terminal(command="tmux send-keys -t claude-work '/exit' Enter")                               # end when done
```

Wait ~3-5s for the welcome screen before sending the task — input fired before the TUI renders gets eaten. The trust dialog appears once per directory; the permissions dialog recurs every launch with `--dangerously-skip-permissions`.

### Session Continuation

```
terminal(command="claude -p 'Continue and add connection pooling' --resume <id> --fork-session --max-turns 5", workdir="/path/to/repo")
terminal(command="claude -p 'What did you do last time?' --continue --max-turns 1", workdir="/path/to/repo")
```

`--continue` resumes the most recent session in the current directory; `--resume <id>` picks a specific one; `--fork-session` keeps the history under a new ID.

### Parallel Instances

Run independent tasks in separate PTY sessions (`terminal(pty=true, background=true)` per task) or tmux sessions (`tmux new-session -d -s task1 ...`), and poll each with `process(action="poll")` or `tmux capture-pane -t <session> -p -S -5`. Clean up tmux with `tmux kill-session -t <session>` when done.

## Pitfalls

1. **`--allowedTools "Read,Bash"` is not read-only.** `Bash` is a write backdoor in `-p` mode. Use `--tools "Read,Grep,Glob"` (or add `--safe-mode` for untrusted repos) — never a bare `Bash(git *)`.
2. **`--max-turns` is a hard stop, not a soft budget.** Hitting the cap ends the run with `subtype: error_max_turns` and no `result` field; in text mode that can masquerade as a finished answer. Check the subtype. Budget 20-30 turns for multi-file reviews.
3. **Non-interactive mode silently trusts the folder and runs its hooks.** The trust dialog is skipped via `-p` *and* whenever stdout is not a TTY (piped or redirected). For an untrusted fork, a hostile `.claude/settings.json` hook or `.mcp.json` server runs during your "review". Isolate with `--safe-mode --setting-sources user` (disables hooks/MCP/plugins/CLAUDE.md and stops the repo's settings files being read at all) or `--bare` (faster startup; drops OAuth so needs `ANTHROPIC_API_KEY`, but still reads the project's `env` block).
4. **Scripted `claude -p "$(cat file)"` can wait on stdin.** When stdin is a TTY the process may block for EOF. Add `< /dev/null` or pipe the input in cron/CI.
5. **`--dangerously-skip-permissions` dialog defaults to "No, exit".** Send `Down` then `Enter` to accept; print mode skips it entirely.
6. **`--max-budget-usd` is a soft cap.** The run can overshoot by one turn before it stops; it's not a hard ceiling.
7. **`--bare` skips OAuth** — it needs `ANTHROPIC_API_KEY` or an `apiKeyHelper` in settings.
8. **`--json-schema` needs enough `--max-turns`.** Claude must read files before it can emit the structured output, which takes several turns.
9. **On systems without a `python` symlink**, Claude's bash commands fail once on `python` then self-correct to `python3`.
10. **Context-fill thresholds are heuristic.** `/context` shows the grid; `~70%` degradation and `~85%` hallucination figures are community rules of thumb, not documented hard limits. Compact proactively with `/compact`.
11. **Background sessions persist.** Clean up tmux (`tmux kill-session -t <name>`) or `claude stop` for background agents. `claude rm <id>` deletes the session **and its worktree**, so prefer `claude stop` unless you've confirmed the worktree is disposable.
12. **Don't kill slow sessions** — a quiet Claude may be mid-multi-step-work. Check `capture-pane` / `process(action="poll")` before assuming a hang.
13. **`Write` is a separate tool from `Edit`.** Neither `--allowedTools "Read,Edit"` nor `"Read,Edit,Bash"` permits file *creation*: `Edit` only modifies existing files and `Bash` does not grant `Write`, so a task that creates a new file (new module, new test file) gets `permission_denials` on `Write` and the run stalls. When the task creates files, add `Write` (`--allowedTools "Read,Edit,Write"`). Grant `Bash` only when the task must run commands/tests — it is a write backdoor, never a default (see #1). A "read-only review" still uses `--tools "Read,Grep,Glob"`, never `Write`/`Edit`.

## Verification

A print-mode run succeeded only if JSON reports `subtype: "success"`. Check:

- `subtype == "success"` — any `error_*` subtype is a failed run (no `result` field).
- `result` actually answers the question (not a terse "done").
- `total_cost_usd` is in the expected range.

For interactive sessions, poll `process(action="poll")` (or `tmux capture-pane -t <session> -p -S -10`): a `❯` prompt means Claude is waiting on you (done or asking); `●` lines mean it's still working tools.
