---
name: study-plan
description: Plan StudyOS projects and schedules.
platforms: [linux, macos, windows]
---

# StudyOS Planning

Use for setup, curriculum, schedules, and schedule changes. Call
`study_activity(resource="project", action="status")`, then call
`study_activity(resource="prompt_context", action="load", data={"intent":"planning"})`
or `"schedule_adjustment"`; never mutate system prompts.

## Plan Before Persisting

1. Read the active project, existing curricula, and relevant schedules. For a
   schedule change, read the target schedule before proposing replacements.
2. Identify the project shape: `exam-vault`, `engineering-repo`, `skill-vault`,
   or `hybrid`. Use `kaoyan.v1` only for 考研; use `engineering.v1` for
   codebase- or artifact-driven learning.
3. Turn the request into observable objectives, prerequisites, source anchors,
   a realistic time budget, and a review/checkpoint. Do not fill unknown dates,
   scores, or availability with invented facts.
4. Present a compact proposed curriculum or schedule when the user has not yet
   asked to save it. When saving is requested, call `study_activity` with
   `curriculum.create`, then `schedule.validate`, then `schedule.save` using the
   same `study_schedule.v1` object.
5. Report the artifact paths and conflicts/assumptions. Never claim a desktop
   calendar changed until `schedule.save` succeeds.

## Constraints

Curriculum is the source of truth for what to learn; keep topic names stable
because schedules reference them. Map 考点 or engineering skills to source
material, representative practice, and prerequisites. Schedule events need a
timezone, must be inside the range, and require
`duration_minutes == end - start`. Do not create a schedule merely to answer a
planning question.
