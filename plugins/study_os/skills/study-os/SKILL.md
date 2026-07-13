---
name: study-os
description: Route StudyOS learning workflows.
platforms: [linux, macos, windows]
---

# StudyOS Router

Keep durable project evidence under `.StudyOS/`. Use `study_activity` for state
and records; use `study_coach` only for evidence-grounded conclusions.

## Shared Flow

1. Check `study_activity(resource="project", action="status")`; initialize only
   when asked, never for a one-off explanation.
2. Load `prompt_context.load` for the intent. Never mutate system prompts.
3. Read relevant records before changing them; state unknown history.
4. Perform one focused action and persist only requested, completed outcomes.

## Route

| Request | Intent and skill |
| --- | --- |
| Plan or reschedule | `planning` / `schedule_adjustment`: `study-plan` |
| Organize a problem or note | `organizing`: `study-organize` |
| Recall, drill, or spaced review | `reviewing`: `study-review` |
| Teach or practice a concept | `teaching`: `study-teach` |
| Exam, weekly, or error diagnosis | `assessment` / `error_analysis`: `study-assessment` |

Route `kaoyan.v1`, `engineering.v1`, and `research.v1` to `study-kaoyan`,
`study-engineering`, and `study-research`. Reserve `study-lesson` for visual
needs and `study-grill` for strategic decisions.

## Evidence Rules

Record an `attempt` only after a learner response or concrete observation. Do
not infer mastery from fluent chat, review counts, or plans; never auto-apply a
candidate pattern. Use LearningRecord for demonstrated progress and
LearningDecisionRecord for accepted strategy. Save only validated schedules.

For a focused loop, call `study_coach.start` with a contract, perform its
ActivitySpec, submit evaluated evidence through `study_coach.advance`, then
`snapshot` or `finish`. Active state is turn-local user context, never system
prompt content.
