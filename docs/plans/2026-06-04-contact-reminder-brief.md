# Contact Reminder Brief Helper Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Add a small local-first helper that turns a JSON/YAML contact occasion file into a concise Traditional Chinese relationship-reminder brief, with exact `[SILENT]` output when nothing needs attention.

**Architecture:** A standalone script in `scripts/contact_reminder_brief.py` exposes pure functions for loading contacts, computing upcoming annual/one-off occasions, and rendering text. The CLI accepts `--input`, `--today`, `--window-days`, and `--format json|text` so cron jobs can run it deterministically.

**Tech Stack:** Python standard library plus optional PyYAML if installed; pytest tests under `tests/scripts/`.

---

### Task 1: Specify upcoming occasion behavior with failing tests

**Objective:** Capture the core behavior before implementation.

**Files:**
- Create: `tests/scripts/test_contact_reminder_brief.py`
- Create later: `scripts/contact_reminder_brief.py`

**Steps:**
1. Write tests that import the script with `importlib.util.spec_from_file_location()`.
2. Cover annual birthdays/anniversaries within the window.
3. Cover exact `[SILENT]` when no item is actionable.
4. Cover Feb 29 annual dates in non-leap years by observing them on Feb 28.
5. Run focused tests and confirm they fail because the script does not exist yet.

### Task 2: Implement minimal pure functions and renderer

**Objective:** Pass the focused tests with a small, dependency-light script.

**Files:**
- Create: `scripts/contact_reminder_brief.py`

**Steps:**
1. Implement JSON/YAML loading.
2. Implement `build_brief(records, today, window_days)` returning structured items.
3. Implement leap-day normalization for annual dates.
4. Implement Traditional Chinese text rendering with TL;DR and action bullets.
5. Run focused tests and confirm pass.

### Task 3: Add CLI smoke path and verify

**Objective:** Make the helper useful from cron/shell.

**Files:**
- Modify: `scripts/contact_reminder_brief.py`
- Modify: `tests/scripts/test_contact_reminder_brief.py`

**Steps:**
1. Add `main()` and argparse flags.
2. Test text output and `[SILENT]` behavior through pure helper or subprocess-safe CLI entrypoint.
3. Run `scripts/run_tests.sh tests/scripts/test_contact_reminder_brief.py`.
4. Run a manual sample invocation with a temp JSON file.
