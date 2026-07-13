---
name: study-kaoyan
description: Guide 考研 learning with StudyOS.
platforms: [linux, macos, windows]
---

# StudyOS 考研 Domain Pack

Use only with `domain_pack:"kaoyan.v1"`. Load the active workflow intent with
`study_activity(resource="prompt_context", action="load", data={"intent":"..."})`;
never mutate system prompts. Storage remains generic:
考研 is a domain pack, not a separate persistence model.

## Operating Rules

- Confirm exam date, phase, subjects, available time, and material before
  proposing a schedule. The default project is `kaoyan-2027`, but never assume
  it replaces a user's active project.
- Build curriculum from 考点, prerequisites, textbook/exercise sources, and
  representative problems. Foundation work favors definitions, formulas, and
  examples; review work favors 错题 clusters, weak prerequisites, and timed
  transfer practice.
- Route schedules to `study-plan`, problem capture to `study-organize`,
  retrieval to `study-review`, and mock/error analysis to `study-assessment`.
- Persist only validated curricula and schedules. A saved calendar artifact is
  read-only in the desktop UI; never imply drag/drop or unsaved edits exist.

Use `study-teach` for a missing prerequisite and `study-lesson` only for a
requested or genuinely visual concept. Strategic study-system tradeoffs belong
to `study-grill`, never routine schedules, 整理, 复习, or 错题 remediation.
