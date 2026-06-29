---
name: study-plan
description: Plan StudyOS projects and schedules.
platforms: [linux, macos, windows]
---

# StudyOS Planning

Use this skill for project setup, curriculum planning, learning schedules, and
schedule_adjustment. Before long project-specific reasoning, call
`study_prompt_context(intent="planning")` or
`study_prompt_context(intent="schedule_adjustment")`. Treat fragments as
turn-local context only; never mutate system prompts mid-conversation.

## Workflow

1. Confirm or initialize the active project with `study_project`.
2. For 考研 work, use the `study-kaoyan` domain pack and the default
   `kaoyan-2027` manifest unless the user provides a different project.
3. Build or inspect standardized curriculum JSON with `study_create_curriculum`
   and `study_list_curricula`.
4. Generate schedules as `study_schedule.v1` JSON only.
5. Validate with `study_schedule(action="validate")` before saving.
6. Save with `study_schedule(action="save")`; desktop reads persisted JSON.

## Curriculum

Curriculum files are the source of truth for what to learn. Generate them from
textbook and exercise book structure, not from inconsistent review notes.

Required curriculum thinking:
- Extract 考点 from definitions, theorems, formulas, and problem variants.
- Map each 考点 to representative problems.
- Record prerequisites so `study_concept_graph` can order learning.
- Keep `curriculum` topic names stable because schedules can reference them.

## Schedule Contract

Schedules must be valid `study_schedule.v1` artifacts. Events must include
timezone offsets, fall inside the schedule range, and have
`duration_minutes == end - start`. Unknown fields are allowed for future use,
but desktop rendering ignores them.

Use read-only calendar assumptions: no drag/drop, no edit controls, and no
natural-language parsing in the desktop UI.
