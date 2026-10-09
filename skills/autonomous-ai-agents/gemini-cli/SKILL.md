---
name: gemini-cli
description: Delegate coding and repository reviews to Google Gemini.
version: 1.1.0
author: tomorrow (@tianxinzh), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Coding-Agent, Gemini, Google, Code-Review, Automation, MCP, PTY]
    category: autonomous-ai-agents
    related_skills: [claude-code, codex, opencode, subagent-driven-development, requesting-code-review]
---

# Gemini CLI Skill

Use Google's [Gemini CLI](https://github.com/google-gemini/gemini-cli) through
Hermes `terminal` for bounded coding, analysis, and review tasks. Hermes retains
responsibility for approvals, inspecting changes, and verifying results; this
skill does not configure Gemini as Hermes's own inference provider.

## When to Use

- The user requests Gemini CLI, Gemini Code Assist, or Google's coding agent.
- A task needs a Gemini codebase analysis, implementation, or second review.
- Independent coding tasks can run in separate worktrees with clear ownership.

For a small file edit, prefer Hermes `read_file`, `search_files`, and `patch`
directly unless the user specifically wants Gemini.

## Prerequisites

- Linux or macOS with a POSIX shell. These workflows do not cover native Windows.
- Hermes `terminal` access to the intended project; use `read_file` for its
  instructions and `GEMINI.md`. Interactive orchestration additionally uses tmux.
- Node.js/npm compatible with the installed Gemini CLI release and the `gemini`
  executable. If installation is needed and authorized, use the official package:
  `npm install -g @google/gemini-cli` through `terminal`.
- Existing cached Google authentication, or an approved Gemini API-key/Vertex AI
  setup. Keep credentials out of prompts, transcripts, and committed files.
  See [setup and controls](references/setup-and-controls.md) and
  [remote Google sign-in](references/remote-oauth-code-flow.md).

## How to Run

Set `terminal(workdir=...)` explicitly. First inspect the installed interface:

```python
terminal(command="gemini --version && gemini --help", workdir="/path/to/project")
```

Prefer a bounded headless request when no interactive approvals are needed:

```python
terminal(
    command="gemini -p 'Explain this repository and its test commands. Do not edit files or implement changes.' --approval-mode plan --output-format json",
    workdir="/path/to/project",
    timeout=180,
)
```

Check the command's exit status and JSON `error` before reading `response`.
When available, `stats.models` describes actual model routing; do not infer the
model from a requested alias alone. For long runs, use Hermes's supported
background/process controls and keep track of the running session.

**Plan Mode is not a read-only security boundary.** In current headless Gemini,
plan transitions can be auto-approved and exiting Plan Mode switches to YOLO.
An analysis-only prompt is guidance, not enforcement. If edits must be prevented,
use verified deny policies or a filesystem sandbox with the project mounted
read-only. A worktree isolates edits but does not prevent them.

See [headless plan review](references/headless-plan-review.md) for pre/post
checks and trusted-folder handling.

## Quick Reference

Verify flags and subcommands with the installed `--help` before relying on them.

| Interface | Use |
|---|---|
| `-p, --prompt` | One-shot headless task; exits when complete. |
| `--output-format json` | Response, statistics, and optional error envelope. |
| `--output-format stream-json` | JSONL events for progress-aware consumers. |
| `-i, --prompt-interactive` | Initial prompt followed by an interactive session. |
| `--approval-mode plan` | Plan-first analysis, subject to the headless warning above. |
| `--approval-mode default` | Interactive tool approvals. |
| `--sandbox` | Enable the configured sandbox; verify its actual restrictions. |
| `--resume` | Continue a prior session; verify the selected project/session. |
| `gemini mcp --help` | Inspect MCP integration commands before configuring access. |

Use `gemini --help`, then subcommand help, rather than copying a fixed catalogue
of model names, extension commands, or experimental flags.

## Procedure

1. **Establish scope.** Read project instructions with `read_file`; locate relevant
   files with `search_files`. Use `terminal` to inspect `git status --short` and
   `git diff` before starting. Record pre-existing tracked and untracked changes.
2. **Choose execution mode.** Use headless mode for bounded analysis or an already
   authorized task whose tools have explicit policies. Use interactive mode when
   permissions, authentication, or follow-up steering need human decisions.
3. **Give a concrete task.** Name the relevant files, intended result, boundaries,
   and test commands. For implementation, request the smallest change and tests;
   forbid unrelated edits. Never transmit secrets or unrelated private files.
4. **Monitor actual output.** Inspect errors and permission prompts before acting.
   A denied headless tool is not a reason to enable YOLO or broaden permissions.
5. **Verify independently.** Inspect the resulting diff and run the project's
   relevant checks. Preserve the user's earlier work. Report the exact files,
   commands, results, and remaining blockers.

### Interactive development

Use a unique tmux session, with the working directory set when creating it.
Run these commands through `terminal`; inspect each captured pane before sending
more input. Replace the example session name if it is already in use.

```bash
tmux new-session -d -s gemini-task -c /path/to/project -x 140 -y 40
tmux send-keys -t gemini-task -l 'gemini --approval-mode default'
tmux send-keys -t gemini-task Enter
tmux capture-pane -t gemini-task -p -S -80
```

Handle any trust/auth/permission prompt according to the user's authorization.
Once the task input is visible, submit the bounded task with literal key input:

```bash
tmux send-keys -t gemini-task -l 'Implement the agreed timeout fix and its regression test. Preserve unrelated changes.'
tmux send-keys -t gemini-task Enter
tmux capture-pane -t gemini-task -p -S -120
```

A pane snapshot is not completion proof. Continue observing while the task runs.
After verifying completion, send `/quit` to this Gemini session; close only an
idle tmux session created for this task. Leave active sessions clearly identified
if the user wants to continue them. Do not capture or share credential entry.

### Parallel work

Create a separate git worktree and branch for each editing agent, then point its
`terminal(workdir=...)` or tmux session there. Give agents non-overlapping tasks
and a shared interface contract. Manual worktrees avoid depending on Gemini's
experimental `--worktree` feature; check installed help before using that feature.

Review each worktree's diff and tests before integrating chosen changes. Do not
force-remove worktrees, discard changes, push, or merge without authorization.

## Pitfalls

- **Trust is a separate decision.** `--skip-trust` trusts the current workspace
  for the session and can load project configuration. Use it only after checking
  the exact directory and obtaining the necessary trust authorization, never as
  an automatic retry for an untrusted-folder error.
- **Headless approval is not interactive.** A tool requiring confirmation may be
  denied. Select interactive mode or a narrowly approved policy; do not bypass it.
- **Authentication failure is not an installation failure.** Inspect the reported
  error and use the chosen auth flow; do not print environment secrets to debug it.
- **Settings evolve.** Confirm available models, policies, checkpointing, MCP,
  extensions, and slash commands against installed help and official docs in
  [setup and controls](references/setup-and-controls.md).
- **Agent output is evidence to check.** Verify claimed edits, model routing, test
  results, and worktree paths yourself. Avoid raw terminal-output mode.
- **Skill systems differ.** Gemini's `gemini skills` manages Gemini skills;
  Hermes `skill_view` and `skill_manage` manage Hermes skills.

## Verification

- Confirm the executable version, intended repository, and chosen authentication.
- Check exit status and structured errors; separate model output from tool results.
- Compare pre/post `git status --short` and `git diff`, including untracked files.
- Read modified files with `read_file`; check for secrets and unrelated changes.
- Run relevant tests/lints/builds through `terminal`; distinguish passed, failed,
  and not-run checks. Do not claim completion from Gemini's self-report alone.
- Inspect every editing worktree, and report any session/worktree left running.
- If publication was authorized, verify the remote commit and its checks before
  reporting it pushed. Merging and deployment need their own authorization.
