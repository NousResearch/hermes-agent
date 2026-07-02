---
name: study-engineering
description: Guide engineering and skill learning with StudyOS.
platforms: [linux, macos, windows]
---

# StudyOS Engineering Domain Pack

Use with `domain_pack="engineering.v1"`: codebase-driven learning, engineering
skills, and lightweight concept vaults. Before long reasoning, call
`study_prompt_context` with the active intent and `domain_pack="engineering.v1"`.
Treat fragments as turn-local context only; never mutate system prompts
mid-conversation.

## Workspace Types

- `engineering-repo`: source repo is the learning surface; keep experiments and
  reading notes close to code.
- `skill-vault`: vault stores long-lived concepts, references, records.
- `hybrid`: repo for exploration, lightweight vault for reusable concepts.

## Lightweight Rule

Do not copy exam-vault behavior into engineering learning. Avoid daily
dashboards, heavy Anki export, and full mistake systems unless requested.

Create a durable concept note only when it blocks code understanding, appears
across paths/repos, will be reused in implementation, or repeats as confusion.

Every concept note needs a source anchor: file path, symbol, paper, benchmark,
or command.

## Hybrid Pattern

For an AI infra repo plus a separate AI infra vault:
- keep source exploration in the repo,
- keep reusable concepts and terminology in the vault,
- keep schedules lightweight,
- use `prompt_summary.md` to record project boundaries and current learning
  objective.

## Planning Heuristics

- Plan around concrete skills and artifacts, not broad motivation.
- Prefer shallow scouting before deep exploration.
- Deep-dive only when the concept affects current implementation, benchmark,
  architecture reading, or repeated confusion.
- Keep notes small enough to maintain during real engineering work.
