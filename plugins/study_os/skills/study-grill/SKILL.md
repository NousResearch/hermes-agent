---
name: study-grill
description: Bridge grilling sessions into StudyOS decisions.
platforms: [linux, macos, windows]
---

# StudyOS Grill Workflow

Use this optional workflow when the user asks to grill, stress-test, or sharpen
a learning plan inside a StudyOS project. This skill composes `/grilling`; it
does not replace it. Treat loaded context as turn-local; never mutate system prompts mid-conversation.

Do not use this workflow for routine planning, problem 整理, daily 复习, weekly
assessment, or 错题 remediation. In `exam-vault` projects, use it only for
strategic study decisions such as scope boundaries, priority tradeoffs, or
whether to change the learning system itself.

Before the interview:
1. Call `study_prompt_context(intent="planning")` for the active project.
2. List existing LearningDecisionRecords with `study_decision(action="list")`.
3. Inspect relevant concepts with `study_concept_graph` or `study_list_notes`
   when the question depends on existing Box knowledge.

During the interview, follow `/grilling`: ask one question at a time, provide
your recommended answer, and explore code or notes instead of asking when the
answer is discoverable.

After a stable decision:
1. Create a `LearningDecisionRecord` with `study_decision(action="create")`.
2. Link relevant Box concepts via `linked_concepts`.
3. Link source files, notes, repos, commands, or papers via `linked_sources`.
4. Do not write Box notes directly unless the user also asks to organize the
   resulting concept; hand that off to `study-organize`.

Boundary: StudyOS stores project state and decisions. `/grilling` owns the
interview protocol. Box stores reusable knowledge objects.
