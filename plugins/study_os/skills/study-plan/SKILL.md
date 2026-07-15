---
name: study-plan
description: Plan StudyOS projects, interventions, and schedules.
platforms: [linux, macos, windows]
---

# StudyOS Planning

For setup or schedule changes, first call `study_activity` for `project.status`,
then `prompt_context.load` with intent `planning` or `schedule_adjustment`.
Never mutate system prompts.

## Schedule Model

- Long-term roadmaps belong in `phases`; their date-only ranges may span days
  or months. `phase.goal` is the summary, `phase.goals` optional detail, and
  `phase.effort_minutes` optional aggregate workload. Effort need not equal the
  wall-clock range.
- `events` are exact, timezone-aware study sessions; `events` may be empty
  until daily scheduling is requested. Never duplicate a phase as a multi-day
  event or put aggregate workload in `event.duration_minutes`.
- Each event must be inside the Schedule range, last at most 720 minutes, and
  satisfy `duration_minutes == end - start`.

## Workflow

1. Read the active project, curricula, and target Schedule before replacement.
2. Classify it as `exam-vault`, `engineering-repo`, `skill-vault`, or `hybrid`;
   reserve `kaoyan.v1` for 考研.
3. Map observable objectives, prerequisites, source anchors, time, and a
   checkpoint. Never invent dates, scores, or availability. Curriculum is the
   source of truth; keep topic names stable.
4. If saving was not requested, return a compact draft. If authorized, call
   `curriculum.create`, then pass the same complete `study_schedule.v1` to:
   - `study_activity(resource="schedule", action="validate", project_id="...", data={...})`
   - `study_activity(resource="schedule", action="save", project_id="...", data={...})`

`data` is the Schedule itself: never use `data.schedule`, `data.data`, or a
prewritten JSON file. `schedule.save` is registration; it writes the canonical
file discovered on the panel's next refresh. Report its returned path and do
not claim a change before save succeeds.

## Proposals and Cron

Use project-scope `study_coach.prioritize` and `propose_plan`. List pending
proposals before `plan_proposal.save`. Only an explicit learner decision permits
accept/reject; acceptance does not mutate a Schedule. To apply one, read the
target, add `source_plan_proposal_id`, validate, then save.

After cadence and delivery are chosen, cron may propose and save a new proposal
but must never decide it or save a Schedule. Do not create a Schedule merely to
answer a planning question.
