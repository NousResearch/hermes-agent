---
name: study-lesson
description: Create visual StudyOS lesson artifacts.
platforms: [linux, macos, windows]
---

# StudyOS VisualLesson

Use only when a requested lesson needs structure, flow, time, state, spatial
layout, or interaction. Call `study_activity(resource="prompt_context",
action="load", data={"intent":"teaching"})`; never mutate system prompts.

1. Read relevant concept notes and sources; define one visual learning
   objective and explain why text alone is insufficient.
2. Do not create HTML for routine 整理, 复习, weekly assessment, or 错题 repair.
   Prefer an existing note, explanation, or small probe unless a reusable visual
   artifact is justified.
3. For an accepted artifact, call `study_activity(resource="lesson",
   action="create")` with one complete HTML document, rationale, linked concepts,
   and source links. Read it back and report both HTML and metadata paths.
4. If the learner demonstrates something after using it, record that separate
   evidence through `attempt` or `learning_record`; viewing a lesson is not
   evidence of understanding.
