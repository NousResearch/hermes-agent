---
name: study-teach
description: Teach through StudyOS learning records.
platforms: [linux, macos, windows]
---

# StudyOS Teach Workflow

Use this optional workflow when the user wants to learn a concept or skill
inside a StudyOS project. It adapts `/teach` ideas without creating heavy HTML
lessons or assets by default. Treat loaded context as turn-local; never mutate system prompts mid-conversation.

Before teaching:
1. Call `study_prompt_context(intent="teaching")`.
2. List `LearningRecord`s with `study_learning_record(action="list")`.
3. Inspect relevant Box concepts with `study_concept_graph` or
   `study_list_notes` when useful.
4. Check the active mission/project summary before choosing what to teach.

Teaching loop:
- Pick one small objective in the user's zone of proximal development.
- Prefer trusted project resources, source files, papers, docs, or commands
  over parametric memory.
- Explain only the knowledge needed for the current skill.
- Ask for retrieval or a small application before claiming the user learned it.
- Use immediate feedback; difficulty belongs in practice, not exposition.
- If the concept depends on structure, flow, time, state, or spatial layout,
  offer `study-lesson`; do not generate HTML by default.

After evidence of understanding:
1. Create a `LearningRecord` with `study_learning_record(action="create")`.
2. Link concepts and sources.
3. If a reusable concept should enter Box, hand off to `study-organize`.
4. If a strategic learning choice was made, hand off to `study-grill` and
   `LearningDecisionRecord`.
