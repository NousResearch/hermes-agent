# Cron Job Tags Implementation Plan

> **For Hermes:** Implement directly with strict TDD in this nightly cron run.

**Goal:** Add lightweight tags to cron job records so Joe can group proactive jobs by domain (health, leads, side-projects, reminders) without overloading names or prompts.

**Architecture:** Store an optional `tags: list[str]` field on cron job dicts. Normalize tags at create/update/read boundaries so legacy jobs get `[]`, duplicates collapse case-insensitively, and malformed hand-edited records cannot crash list/get callers.

**Tech Stack:** Python cron storage (`cron/jobs.py`) and pytest coverage (`tests/cron/test_jobs.py`).

---

### Task 1: Add failing tag-normalization tests

**Objective:** Lock in expected behavior before implementation.

**Files:**
- Modify: `tests/cron/test_jobs.py`

**Steps:**
1. Add tests that `create_job(..., tags=[...])` trims whitespace, drops blanks, and de-duplicates case-insensitively while preserving first spelling/order.
2. Add tests that `list_jobs()` normalizes legacy missing/null/scalar tag fields to a safe list.
3. Run focused tests and verify RED failure.

### Task 2: Implement tag normalization

**Objective:** Make cron jobs persist and read safe normalized tag lists.

**Files:**
- Modify: `cron/jobs.py`

**Steps:**
1. Add `_normalize_tags(tags)` helper.
2. Apply it in `_normalize_job_record()`.
3. Add `tags` parameter to `create_job()` and persist normalized tags.
4. Add update handling so `update_job(job_id, {"tags": ...})` normalizes before save.
5. Run focused tests and verify GREEN.

### Task 3: Smoke and PR

**Objective:** Verify no focused regressions, commit, push branch, open PR.

**Commands:**
- `scripts/run_tests.sh tests/cron/test_jobs.py -q`
- `git diff --check`
