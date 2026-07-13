---
name: study-os
description: Route StudyOS learning workflows.
platforms: [linux, macos, windows]
---

# StudyOS Router

StudyOS keeps durable project evidence under `.StudyOS/`. Use
`study_activity(resource, action, data)` for project state and records; use
`study_coach` only for conclusions grounded in recorded attempts.

## Shared Flow

1. Check `study_activity(resource="project", action="status")`. Initialize a
   project only when the user asks to set one up; do not create a project for a
   one-off explanation.
2. Load turn-local context with `prompt_context.load` and the matching intent.
   Never mutate system prompts.
3. Read existing notes, schedules, records, or attempts before proposing a
   change. State unknowns rather than inventing learner history.
4. Perform one focused learning action. Persist only requested or completed
   outcomes, then report what was read, written, and left unchanged.

## Route

| Request | Intent and skill |
| --- | --- |
| Plan or reschedule | `planning` / `schedule_adjustment`: `study-plan` |
| Organize a problem or note | `organizing`: `study-organize` |
| Recall, drill, or spaced review | `reviewing`: `study-review` |
| Teach or practice a concept | `teaching`: `study-teach` |
| Exam, weekly, or error diagnosis | `assessment` / `error_analysis`: `study-assessment` |

Use `study-kaoyan` for `kaoyan.v1` and `study-engineering` for
`engineering.v1`. Use `study-lesson` only for a requested or genuinely visual
concept; use `study-grill` only for a strategic decision.

## Evidence Rules

Record an `attempt` only after a learner response or other concrete evidence.
Do not infer mastery from fluent chat, review counts, or plans. Candidate
pattern changes are not applied automatically. Persist an accepted conclusion
as a `learning_record` (LearningRecord), a strategic choice as a `decision`
(LearningDecisionRecord), and a completed schedule only after it validates. Desktop calendars read saved
`study_schedule.v1` artifacts.
