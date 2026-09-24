# Life Ops Brief Helper Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Add a small local-first script that turns simple YAML/JSON life-ops items into a Joe-style Traditional Chinese morning brief, with exact `[SILENT]` support when nothing needs attention.

**Architecture:** A standalone script under `scripts/` parses a user-owned backlog file without touching external systems. It classifies overdue/due-soon/scheduled items from dates or recurring intervals, formats compact Traditional Chinese output, and supports deterministic tests with `--today`.

**Tech Stack:** Python stdlib + optional PyYAML already used in project tests. Focused pytest coverage via `scripts/run_tests.sh`.

---

### Task 1: Add failing tests for overdue/due-soon/SILENT behavior

**Files:**
- Create: `tests/scripts/test_life_ops_brief.py`
- Create: `scripts/life_ops_brief.py`

**Steps:**
1. Write tests that import the script module and exercise:
   - overdue date items are reported
   - recurring tasks compute next due date from `last_done` + `every_days`
   - no relevant items returns exactly `[SILENT]`
2. Run the focused tests and confirm failure because the script does not exist or behavior is missing.

### Task 2: Implement minimal parser/classifier/formatter

**Files:**
- Modify: `scripts/life_ops_brief.py`

**Steps:**
1. Implement `load_items`, `build_brief`, and CLI `main` with `--input`, `--today`, `--soon-days`.
2. Support JSON and YAML based on file extension.
3. Return exact `[SILENT]` when no due/soon/overdue items exist.
4. Run focused tests and confirm pass.

### Task 3: Add smoke/CLI coverage and verify

**Files:**
- Modify: `tests/scripts/test_life_ops_brief.py`

**Steps:**
1. Add CLI smoke test for JSON input.
2. Run focused tests through `scripts/run_tests.sh`.
3. Commit, push branch, open PR.
