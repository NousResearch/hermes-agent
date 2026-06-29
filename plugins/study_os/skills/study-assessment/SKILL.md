---
name: study-assessment
description: Analyze StudyOS exams and mistakes.
platforms: [linux, macos, windows]
---

# StudyOS Assessment

Use this skill for assessment, mock exams, weekly reports, 错题 analysis, and
error_analysis. Before long project-specific reasoning, call
`study_prompt_context(intent="assessment")` or
`study_prompt_context(intent="error_analysis")`. Treat fragments as turn-local
context only; never mutate system prompts mid-conversation.

## Weekly Workflow

1. Generate a report with `study_generate_weekly_report`.
2. Inspect cause x concept clusters for repeated patterns.
3. For repeated or deep confusion, call `study_concept_graph(concept="...")`.
4. Check prerequisite review levels and use recommended review order.
5. Create targeted review tasks with `study_create_review_task`.
6. Sync memory with `study_sync_memory` when memory is available.

## Mock and Exam Analysis

- Classify each mistake by cause, concept, pattern, severity, and next action.
- Log 错题 through `study_log_error`; do not keep exam conclusions only in chat.
- Prefer prerequisite repair over isolated answer explanations.
- Separate careless mistakes from concept gaps and method gaps.

## Output

Return a short diagnosis, the highest-impact remediation sequence, and 3-5
concrete next review tasks. Tie recommendations to StudyOS notes, concepts,
curriculum items, or schedule events when available.
