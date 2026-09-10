# Personal Reminder Brief Helper Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a small local-first script that turns a YAML/JSON reminder list into a Joe-style morning brief with due/soon/overdue grouping and exact `[SILENT]` support.

**Architecture:** Keep the helper standalone under `scripts/` so it can be used by cron script hooks without adding new runtime services or private data access. Parse YAML/JSON reminders, classify them by date against an explicit `--today`, and render Traditional Chinese markdown.

**Tech Stack:** Python standard library + PyYAML (already a core dependency), pytest via `scripts/run_tests.sh`.

---

### Task 1: Add tests for reminder parsing and due classification

**Objective:** Define the desired behavior before implementation.

**Files:**
- Create: `tests/scripts/test_personal_reminder_brief.py`
- Create: `scripts/personal_reminder_brief.py`

**Steps:**
1. Write tests that load the script via `importlib.util.spec_from_file_location` and register it in `sys.modules` before executing.
2. Cover YAML/JSON parsing into reminders.
3. Cover overdue, due today, and soon classification.
4. Cover completed/future-only reminders returning exact `[SILENT]`.
5. Run focused test and verify RED: `scripts/run_tests.sh tests/scripts/test_personal_reminder_brief.py -q` should fail because the script is missing.

### Task 2: Implement minimal reminder brief helper

**Objective:** Make the tests pass with a reversible standalone script.

**Files:**
- Create: `scripts/personal_reminder_brief.py`

**Steps:**
1. Add a `Reminder` dataclass with title, due date, optional cadence, area, action, source, completed flag, and notes.
2. Add `load_reminders(path)`, `classify_reminders(reminders, today, soon_days)`, and `render_brief(...)` functions.
3. Add a CLI with `--input`, `--today`, `--soon-days`, and `--silent-if-empty`.
4. Run focused tests and smoke-check sample YAML output plus empty `[SILENT]` output.

### Task 3: Verify and prepare PR

**Objective:** Leave a clean, reviewable PR.

**Files:**
- Modify: generated files only from Tasks 1-2.

**Steps:**
1. Run `scripts/run_tests.sh tests/scripts/test_personal_reminder_brief.py -q`.
2. Run `python scripts/personal_reminder_brief.py --help`.
3. Commit with `feat: add personal reminder brief helper`.
4. Push to Joe fork and open a PR against `NousResearch/hermes-agent:main`.
