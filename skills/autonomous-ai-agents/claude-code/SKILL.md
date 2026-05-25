---
name: claude-code
description: "Delegate coding to Claude Code CLI (features, PRs)."
version: 2.2.0
author: Hermes Agent + Teknium
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, Code-Review, Refactoring, PTY, Automation]
    related_skills: [codex, hermes-agent, opencode]
---
# Claude Code — Hermes Orchestration Guide

Use this skill when delegating coding or repository-review work to Anthropic Claude Code from Hermes. Prefer Claude Code for substantive code changes, PR reviews, refactors, and long-running implementation slices where a separate coding agent can work with repo context.

This main file is intentionally compact. Load detailed references only when needed:

- `references/cli-print-mode.md` — subcommands, print mode, JSON/streaming, full flags, env, cost/performance.
- `references/interactive-pty.md` — tmux/PTY operation, startup dialogs, slash commands, shortcuts, monitoring.
- `references/review-and-parallelism.md` — PR review, worktrees, parallel Claude instances, custom subagents.
- `references/project-configuration.md` — settings, permissions, CLAUDE.md, hooks, MCP, project context.
- `references/pitfalls-and-rules.md` — gotchas and Hermes-specific operating rules.

## Prerequisites

```bash
claude --version
claude --help
```

Before using Claude Code in a repo:

1. Verify repo identity and branch.
2. Read the local project instructions (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`, etc.).
3. Define a narrow task with expected files, tests, and stop conditions.
4. Avoid handing secrets or broad destructive authority to the subprocess.

## Preferred Mode: Print Mode

Use non-interactive print mode for most one-shot implementation, review, and analysis tasks:

```bash
claude -p "<task>"
```

Good for:

- focused code review,
- small/medium feature slices,
- test-fix loops,
- repository analysis,
- producing a patch plan or implementation summary.

In Hermes, usually run it in a foreground `terminal()` call for short jobs, or as a tracked background process for long jobs.

## Interactive Mode: Only When Needed

Use tmux/PTY only for multi-turn sessions, long exploration, or when Claude Code's interactive features are necessary:

```bash
tmux new-session -d -s claude-run 'claude --dangerously-skip-permissions'
tmux capture-pane -pt claude-run -S -120
```

Interactive mode has startup dialogs and permission prompts. Load `references/interactive-pty.md` before using it.

## Safe Hermes Pattern

1. **Prepare task prompt** with scope, files, constraints, checks, and output shape.
2. **Run Claude Code** in the intended repo/worktree.
3. **Monitor output** without trusting completion claims blindly.
4. **Inspect diff yourself** with `git status -sb` and `git diff`.
5. **Run tests/checks yourself** or require verifiable output.
6. **Commit only explicit paths** using the safe-commit workflow if Alexander asked for commit/push.
7. **Report compactly**: scope, files, checks, risks, next step.

## Prompt Skeleton

```text
You are working in <repo path> on branch <branch>.

Task:
<one narrow objective>

Constraints:
- Do not introduce unrelated refactors.
- Preserve existing architecture and project instructions.
- Touch only files necessary for this slice.
- If blocked, stop and report exactly what is missing.

Verification:
- Run/describe: <tests/build/lint>
- Return: summary, changed files, checks, risks.
```

## PR Review Quick Pattern

```bash
claude -p "Review PR <number> for correctness, tests, security, and scope. Classify findings as blocking/non-blocking and cite files/lines. Do not make changes."
```

For deep PR review, load `references/review-and-parallelism.md`.

## Common Pitfalls

1. **Interactive dialogs.** PTY startup may require accepting workspace trust or permission prompts.
2. **Overbroad prompts.** Claude Code may refactor too much unless scope is explicit.
3. **Trusting self-reports.** Always inspect diffs and run checks outside Claude's narrative.
4. **Wrong repo/worktree.** Verify path, branch, and remote before running.
5. **Permission bypass.** `--dangerously-skip-permissions` is only acceptable in controlled local repos with clear scope.
6. **Token-heavy reference loading.** Load detailed references only when a specific mode needs them.

## Verification Checklist

- [ ] Repo path, branch, and task scope verified.
- [ ] Correct mode chosen: print mode by default, PTY only when necessary.
- [ ] Claude output reviewed against actual filesystem diff.
- [ ] Relevant tests/checks run or explicitly documented as not run.
- [ ] No unrelated files staged or committed.
- [ ] Final handoff includes changed files, checks, risks, and next step.
