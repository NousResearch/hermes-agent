---
name: study-os
description: Route StudyOS learning workflows.
platforms: [linux, macos, windows]
---

# StudyOS Router

Use this router for StudyOS help from an Obsidian vault. StudyOS writes only
under `.StudyOS/`.

Before project-specific reasoning, call `study_prompt_context` with the matching
intent and project. Treat fragments as turn-local context only; never mutate system prompts mid-conversation.

## Intent Map

| Intent | Route |
| --- | --- |
| `planning`, `schedule_adjustment` | `study-plan` |
| `organizing` | `study-organize` |
| `reviewing` | `study-review` |
| `teaching` | `study-teach` |
| `assessment`, `error_analysis` | `study-assessment` |
| `kaoyan.v1` / `engineering.v1` | domain pack |

## Workspace Types

Classify the active project first:

- `exam-vault`: exams, problem sets, 错题, review.
- `engineering-repo`: source repo as learning surface.
- `skill-vault`: concepts, references, records.
- `hybrid`: repo exploration plus lightweight vault.

## Routing Rules

- 考研 init: `study_project(action="init", domain_pack="kaoyan.v1")`.
- Engineering init explicitly: `domain_pack="engineering.v1"`,
  `workspace_type="hybrid"`.
- If the user gives a problem and says 整理, use `study-organize`.
- Learn concept: `study-teach`; `study-lesson` only for explicit visualization
  or structural/temporal/stateful concepts.
- Grill decision: `study-grill`; never routine planning, 整理, 复习, or 错题.
- If the user says 复习 or a daily review job fires, use `study-review`.
- If the user asks for weekly, mock, exam, or 错题 analysis, use `study-assessment`.
- If the user asks for textbook/course/curriculum planning, use `study-plan`.

## Safety

- Desktop calendars render only persisted `study_schedule.v1` JSON artifacts.
- Do not parse chat prose into desktop events or mutate schedules in UI.
