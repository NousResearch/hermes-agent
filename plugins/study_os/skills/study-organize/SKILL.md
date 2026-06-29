---
name: study-organize
description: Organize problems into StudyOS notes.
platforms: [linux, macos, windows]
---

# StudyOS Organize

Use this skill when the user gives a problem and asks to 整理, analyze, study,
or make notes. Before long project-specific reasoning, call
`study_prompt_context(intent="organizing")`. Treat fragments as turn-local
context only; never mutate system prompts mid-conversation.

## Single-Problem 整理 Workflow

Trigger words include 整理, 分析一下这道题, 研究一下, 帮我看看这道题,
做笔记, and 总结一下.

1. Explore before writing. Call `study_concept_graph()` to inspect existing
   concepts and isolated nodes.
2. Extract candidate concepts and reusable problem-type signals from the
   problem statement.
3. Search concepts with `study_list_notes(layer="concept", search_body=true,
   normalize=true)`.
4. Search problem patterns with `study_list_notes(layer="pattern",
   search_body=true, normalize=true)`.
5. For each candidate concept, call `study_concept_graph(concept="...")` to
   inspect prerequisites, dependents, exercised examples, and review levels.
6. Read likely matches with `study_read_note(include_body=true)`.
7. Use `study_extract_concepts` on related notes to avoid duplicate or isolated
   notes.

## Write Decisions

- Create or update `/Box/` concept notes only when the concept is missing or
  incomplete.
- Create `/Box/题型/题型：...` notes only when the problem has a reusable trigger
  signal and solution routine.
- Do not create `/examples/` files unless the user explicitly asks to add the
  problem to the example library.
- Do not rename existing notes just for style.

## User Summary

Report concepts, problem patterns, prerequisite/dependent relationships, and
key 易错点. Make clear what was found, created, updated, or deliberately skipped.
