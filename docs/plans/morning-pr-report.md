# Morning PR Report Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Add a small local script that turns a Hermes Agent worktree into a concise Traditional Chinese morning PR report for Joe.

**Architecture:** Keep this as a reversible, standalone developer utility under `scripts/` with pure formatting/parsing helpers covered by tests. The script shells out to `git`/optional `gh` only at the CLI boundary and can be used by future nightly cron jobs without changing Hermes runtime behavior.

**Tech Stack:** Python standard library, pytest via `scripts/run_tests.sh`.

---

### Task 1: Specify report behavior with tests

**Objective:** Lock down report formatting, git status parsing, and silent-empty behavior before implementation.

**Files:**
- Create: `tests/scripts/test_morning_pr_report.py`
- Create: `scripts/morning_pr_report.py`

**Steps:**
1. Add tests for parsing `git status --porcelain=v1` lines into stable changed-file entries.
2. Add tests for Traditional Chinese report output including branch, PR URL, files, verification, blockers, and Joe-focused why/impact.
3. Add a test that `--silent-if-empty` emits exactly `[SILENT]` when there are no commits, no changed files, and no PR URL.
4. Run the focused test and verify RED.

### Task 2: Implement minimal script

**Objective:** Make the failing tests pass with a small standalone script.

**Files:**
- Modify: `scripts/morning_pr_report.py`

**Steps:**
1. Add `ChangedFile` and `GitSnapshot` dataclasses.
2. Implement `parse_porcelain_status`, `should_silence`, and `format_report`.
3. Add CLI collection helpers around `git` and optional `gh pr view`.
4. Run focused tests and verify GREEN.

### Task 3: Verify and prepare PR

**Objective:** Validate the utility, commit, push, and open a PR.

**Steps:**
1. Run `scripts/run_tests.sh tests/scripts/test_morning_pr_report.py`.
2. Smoke-run `python scripts/morning_pr_report.py --title ... --why ... --verify ...`.
3. Commit with `feat: add morning PR report helper`.
4. Push branch and create a PR against `main`.
