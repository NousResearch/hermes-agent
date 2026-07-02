---
name: study-lesson
description: Create visual StudyOS lesson artifacts.
platforms: [linux, macos, windows]
---

# StudyOS VisualLesson Workflow

Use this optional workflow when a learning topic needs a diagram, timeline,
state machine, spatial layout, or lightweight interaction. Treat loaded context
as turn-local; never mutate system prompts mid-conversation.

Do not create HTML by default. Use a VisualLesson only when at least one is
true:
- the concept depends on structure, flow, time, or state,
- the user asks to draw or visualize it,
- the artifact will be reviewed more than once,
- the page can become a reusable teaching reference.

For `exam-vault` projects, never create VisualLesson artifacts during routine
整理, 复习, weekly assessment, or 错题 remediation. Use them only for explicit
visualization requests or genuinely visual/structural concepts.

Before creating:
1. Call `study_prompt_context(intent="teaching")`.
2. Read relevant Box notes or concept graph if they exist.
3. Keep the lesson focused on one small objective.

Create with `study_lesson(action="create")`. Store complete HTML only under
`.StudyOS/projects/<project_id>/lessons/`. Link concepts and sources in
metadata. Do not duplicate Box; link to Box concepts instead.

If the user demonstrates understanding after using the lesson, record that with
`study_learning_record(action="create")`.
