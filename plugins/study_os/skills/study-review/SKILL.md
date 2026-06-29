---
name: study-review
description: Run StudyOS spaced repetition reviews.
platforms: [linux, macos, windows]
---

# StudyOS Review

Use this skill for 复习, daily review, and 艾宾浩斯 spaced repetition. Before
long project-specific reasoning, call `study_prompt_context(intent="reviewing")`.
Treat fragments as turn-local context only; never mutate system prompts
mid-conversation.

## Daily 艾宾浩斯 Workflow

1. Load review preferences with `study_read_note(note=".StudyOS/study_profile.md",
   include_body=true)` when available.
2. Find due examples with `study_due_reviews()`.
3. For each due example, read the problem using `study_read_note`.
4. Ask the user for their solution before judging.
5. Be strict: missed conditions, concept confusion, or invalid reasoning fail.
6. Record the result with `study_record_review`.
7. On failure, also log a 错题 with `study_log_error`.
8. Summarize reviewed count, pass/fail count, weak concepts, and next actions.

## New Material

For first-time learning, use `study_learning_queue`, update concept
`learning_state` with `study_update_concept_state`, and log sessions with
`study_log_session`.

## Memory

After daily or weekly review, call `study_sync_memory` and pass returned
entries to the memory tool when that tool is available. This preserves weak
concepts, due counts, and last sync time across sessions.
