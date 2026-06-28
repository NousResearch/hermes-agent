---
name: study-os
description: Use Hermes Study OS tools to read Obsidian learning notes and maintain daily review data.
platforms: [linux, macos, windows]
---

# Study OS

Use this skill when the user wants learning review, mistake analysis, Anki candidates, concept extraction, or a second-pass study plan from an Obsidian vault.

The Study OS tools are registered with backend-safe underscore names:

- `study_list_notes` means `study.list_notes`
- `study_read_note` means `study.read_note`
- `study_extract_concepts` means `study.extract_concepts`
- `study_log_error` means `study.log_error`
- `study_create_review_task` means `study.create_review_task`
- `study_generate_weekly_report` means `study.generate_weekly_report`
- `study_export_anki_candidates` means `study.export_anki_candidates`

## Vault Path

Prefer an explicit `vault_path` when the user gives one. Otherwise rely on the existing Obsidian convention:

1. `OBSIDIAN_VAULT_PATH`
2. `~/Documents/Obsidian Vault`

Do not create a database. The source of truth is Obsidian Markdown.

## Write Policy

Study OS generated data always goes under the vault root:

- `.StudyOS/errors/YYYY-MM.md`
- `.StudyOS/review_tasks.md`
- `.StudyOS/reports/YYYY-Www.md`
- `.StudyOS/anki_candidates/YYYY-MM-DD.md`

Do not edit source notes unless the user explicitly asks for source-note edits. Use Study OS tools for logs, plans, reports, and candidate exports.

## Daily Review Workflow

1. Use `study_list_notes` or `study_extract_concepts` to gather recent or targeted notes.
2. Use `study_log_error` for each meaningful mistake. Capture source note, concepts, patterns, cause, severity, and next action.
3. Use `study_create_review_task` for second-pass tasks with due date and priority.
4. Use `study_export_anki_candidates` only for high-value prompts: recognition signals, first moves, common traps, key distinctions, and durable transformations.

## Weekly Review Workflow

1. Use `study_generate_weekly_report` for the current week or requested date range.
2. Read the report and identify repeated causes, weak concepts, overdue tasks, and good Anki candidates.
3. Propose the next week as a small set of review tasks, not a broad study wish list.

## Output Style

Keep feedback in Simplified Chinese unless the user asks otherwise. Tie recommendations to concrete notes, concepts, mistakes, and tasks.
