---
name: study-os
description: Route StudyOS learning workflows.
platforms: [linux, macos, windows]
---

# StudyOS Router

Use this router for StudyOS learning help from an Obsidian vault. StudyOS is
opt-in and stores artifacts only under the vault's `.StudyOS/` directory.

Before long project-specific reasoning, call `study_prompt_context` with the
matching intent and active project. Use returned fragments as turn-local
context only. Never mutate system prompts mid-conversation.

## Intent Map

| Intent | Route |
| --- | --- |
| `planning` | `study-plan`: projects, curriculum, schedules. |
| `schedule_adjustment` | `study-plan`: validated schedule revisions. |
| `organizing` | `study-organize`: 整理 single problems into notes. |
| `reviewing` | `study-review`: 艾宾浩斯 spaced repetition and 复习. |
| `assessment` | `study-assessment`: weekly reports and mock exams. |
| `error_analysis` | `study-assessment`: 错题 clusters and remediation. |
| `kaoyan.v1` | `study-kaoyan`: 考研 domain guidance. |

## Core Tools

- `study_project`: initialize, select, inspect, or update project summaries.
- `study_schedule`: template, validate, save, list, or read schedules.
- `study_prompt_context`: capped base, intent, domain, and project fragments.
- Existing StudyOS tools handle search, 整理, 复习, weekly reports, curriculum,
  Anki candidates, memory sync, and concept graphs.

## Routing Rules

- If no project exists for 考研 planning, call `study_project(action="init")`.
- If the user gives a problem and says 整理, use `study-organize`.
- If the user says 复习 or a daily review job fires, use `study-review`.
- If the user asks for weekly, mock, exam, or 错题 analysis, use
  `study-assessment`.
- If the user asks for textbook/course/curriculum planning, use `study-plan`.

## Safety

- Desktop calendars render only persisted `study_schedule.v1` JSON artifacts.
- Do not parse chat prose into desktop events.
- Do not add cron jobs, Anki sync, or schedule editing UI in this slice.
