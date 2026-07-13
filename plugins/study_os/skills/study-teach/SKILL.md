---
name: study-teach
description: Teach through StudyOS learning records.
platforms: [linux, macos, windows]
---

# StudyOS Teach

Use for an explicit concept or skill lesson. Call
`study_activity(resource="prompt_context", action="load",
data={"intent":"teaching"})`, inspect relevant `learning_record` entries and
source notes, and never mutate system prompts.

## Teach-Test-Record

1. Set one small objective based on the learner's request and evidence. State
   any prerequisite gap before teaching the dependent material.
2. Explain only what the objective needs, using trusted project sources, files,
   papers, commands, or notes rather than unsupported claims.
3. Ask for retrieval or a small application before feedback. Vary one thing at
   a time when checking transfer; do not reveal the answer before a genuine
   attempt.
4. Record the response with `study_activity(resource="attempt", action="record")`
   including result, confidence, concepts, transfer level, and a specific diagnosis. Create a
   `learning_record` only for demonstrated, durable progress and include the
   concrete evidence and source links.
5. Hand a reusable note to `study-organize`, a visual need to `study-lesson`,
   and a strategic tradeoff to `study-grill`. Report what remains unverified.

Do not manufacture a VisualLesson, a learning record, or mastery status merely
because an explanation was delivered.
