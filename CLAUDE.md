# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, complete the checklist below and leave the repository in a clearly reviewable state. Do not push unless the user has explicitly authorized that push for this session.

**WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Prepare remote sync only when authorized**:
   ```bash
   git fetch
   git status
   # Only after explicit user authorization/review:
   # git pull --rebase
   # git push
   ```
5. **Clean up** - Clear temporary stashes and local-only artifacts when safe
6. **Verify** - Summarize changed files, test results, and whether anything remains uncommitted or unpushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Do not run `git push` without explicit user authorization for this session
- Do not auto-retry pull/push failures; stop, report the conflict/error, and ask for review
- If pushing is authorized, review `git status` and the outgoing diff before pushing
<!-- END BEADS INTEGRATION -->


## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

_Add your project-specific conventions here_
