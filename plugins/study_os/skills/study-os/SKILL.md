---
name: study-os
description: Route StudyOS learning workflows.
platforms: [linux, macos, windows]
---

# StudyOS Router

Set the Vault, enable the `study` toolset, then open a new chat. Use
`study_activity` for `.StudyOS` state and `study_coach` for conclusions.

Never prewrite a supported resource with terminal, Python, or file tools. Its
save action is canonical; `schedule.save` is registration. Follow
`study-plan`'s exact payload shape.

## Shared Flow

1. Check `project.status`; initialize only when asked.
2. Load `prompt_context.load` for the intent. Never mutate system prompts.
3. Read relevant records before changes; state unknown history.
4. Persist only requested, completed outcomes.

## Route

| Request | Intent and skill |
| --- | --- |
| Plan or reschedule | `planning` / `schedule_adjustment`: `study-plan` |
| What next or proactive proposals | `planning`: `study-plan` |
| Organize a problem or note | `organizing`: `study-organize` |
| Recall, drill, or spaced review | `reviewing`: `study-review` |
| Teach or practice a concept | `teaching`: `study-teach` |
| Exam, weekly, or error diagnosis | `assessment` / `error_analysis`: `study-assessment` |

Route Domain Packs to their matching skill; use `study-lesson` for visuals and
`study-grill` for strategic decisions.

## Evidence Rules

Record an `attempt` only after a learner response or observation. Never infer
mastery from chat, counts, or plans, nor auto-apply proposals. Cron may save a
proposal but cannot decide it or save a Schedule. Use LearningRecord for
demonstrated progress and LearningDecisionRecord for accepted strategy.

For a focused loop: `study_coach.start`, perform its ActivitySpec, `advance`
with evaluated evidence, then `snapshot` or `finish`. Active state is
turn-local user context, never system prompt content.
