---
name: simplicio-orient
description: Terminal-first execution — answer facts with shell commands, never with the LLM. Run builds, tests, and shell operations with compressed output to save context.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [terminal, execution, shell, token-economy]
    related_skills: [simplicio-tasks]
---

# Simplicio Orient — Terminal-First Execution

> Answer facts with the shell, never with the LLM. Keep raw output out of context.

## Trigger

Any time you need a fact about the filesystem, git history, build output, test results, or system state. Run it in shell first; only bring the derived summary into context.

## Principles

1. **Shell first, LLM never** — Need to know if a file exists? `ls`. What's in the diff? `git diff`. Did tests pass? `pytest`. Never ask the LLM to guess filesystem state.
2. **Summarize, don't dump** — After running a command, summarize the output in 1-2 lines rather than dumping the full output into context. Only include relevant excerpts.
3. **Compress on the way out** — Use `tee` to cache results. Use `grep -c` instead of full output. Use `tail -5` instead of `cat`.
