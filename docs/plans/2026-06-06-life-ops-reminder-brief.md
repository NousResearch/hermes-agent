# Life Ops Reminder Brief Implementation Plan

> **For Hermes:** Use test-driven-development skill while implementing this small helper.

**Goal:** Add a local-first script that turns a YAML/JSON life-ops task list into a deterministic morning reminder brief, returning exact `[SILENT]` when nothing is due.

**Architecture:** A standalone `scripts/life_ops_reminder_brief.py` module with pure parsing/evaluation/formatting helpers plus a tiny CLI. Tests cover date math, recurring tasks, one-off tasks, stale plan-path caveats, and `[SILENT]` behavior.

**Tech Stack:** Python stdlib + PyYAML already in project dependencies; pytest via existing test wrapper.

---

### Task 1: Specify expected brief behavior in tests

**Objective:** Lock down user-facing behavior before implementation.

**Files:**
- Create: `tests/scripts/test_life_ops_reminder_brief.py`

**Steps:**
1. Write failing tests for one-off overdue/due/soon grouping and exact `[SILENT]` output.
2. Write failing tests for recurring `last_done` + `every_days` task date math.
3. Run the focused test file and verify failures are because the script does not exist yet.

### Task 2: Implement the helper minimally

**Objective:** Add a small deterministic CLI and pure helpers to satisfy the tests.

**Files:**
- Create: `scripts/life_ops_reminder_brief.py`

**Steps:**
1. Implement loading YAML/JSON payloads with an `items` list.
2. Normalize one-off and recurring tasks into evaluated reminders.
3. Format Traditional Chinese-friendly concise output with grouped Overdue / Due today / Soon sections.
4. Emit exact `[SILENT]` when no task is inside the reporting window.
5. Run focused tests and fix until green.

### Task 3: Verify and package for review

**Objective:** Keep the PR small, reviewed, and reversible.

**Files:**
- Modify only the spec, test, and script above.

**Steps:**
1. Run focused test wrapper: `scripts/run_tests.sh tests/scripts/test_life_ops_reminder_brief.py`.
2. Run a CLI smoke check against a temporary YAML fixture.
3. Commit on branch `joe/nightly-life-ops-reminder-brief`.
4. Push to Joe fork remote and open a PR against `NousResearch/hermes-agent:main` if auth allows.
