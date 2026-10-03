# Product Release Ops Pattern

## Sources
- GitHub: merged PRs, commits, releases, changed files, tags, CI/deploy state — implementation evidence
- Linear: issues, cycles, projects, priorities, acceptance criteria, status — product intent
- Obsidian wiki: feature behavior, architecture, decisions, release history — durable synthesis

## Rules
- Keep internal changelogs separate from customer-facing release notes
- Public release notes must be rewritten into safe user-facing language and pass claim safety
- Promote durable shipped behavior into wiki/features/, wiki/architecture/, or wiki/decisions/ and update _index.md

## Current Setup
- gh authenticated as percusser
- Main repo: percusser/built-lms
- Linear team key: BUI