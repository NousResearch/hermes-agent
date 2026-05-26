# Cron Show Command Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement this plan task-by-task.

**Goal:** Add a safe `hermes cron show <job>` command and matching `cronjob(action="show")` tool action so users/agents can inspect one scheduled job without scanning the full cron list.

**Architecture:** Reuse the existing `resolve_job_ref()` name-or-ID resolver. Keep storage unchanged; expose a formatted full job payload from the tool and a readable detail view from the CLI. Do not alter scheduler behavior.

**Tech Stack:** Python argparse CLI (`hermes_cli/main.py`, `hermes_cli/cron.py`), cron tool wrapper (`tools/cronjob_tools.py`), pytest via `scripts/run_tests.sh`.

---

### Task 1: Add failing CLI behavior test

**Objective:** Prove `cron show` resolves a job and prints full details including prompt.

**Files:**
- Modify: `tests/hermes_cli/test_cron.py`

**Steps:**
1. Add `test_show_prints_job_details` using `create_job(...)` and `cron_command(Namespace(cron_command="show", job_id=job["id"]))`.
2. Assert stdout contains `Job:`, job ID, name, schedule, prompt, and delivery target.
3. Run `scripts/run_tests.sh tests/hermes_cli/test_cron.py::TestCronCommandLifecycle::test_show_prints_job_details -q`.
4. Expected RED: fails because `show` is unknown.

### Task 2: Add failing tool behavior test

**Objective:** Prove `cronjob(action="show")` returns a single detailed job payload.

**Files:**
- Modify: `tests/tools/test_cronjob_tools.py`

**Steps:**
1. Add `test_show_returns_full_job_details` after create/list tests.
2. Create a job with a prompt over 100 chars.
3. Call `cronjob(action="show", job_id=created["job_id"])`.
4. Assert success, `job.job_id`, `job.prompt` full text, and `job.prompt_preview` truncated.
5. Run focused test; expected RED because action is unknown.

### Task 3: Implement minimal show action

**Objective:** Make both tests pass with the least production change.

**Files:**
- Modify: `tools/cronjob_tools.py`
- Modify: `hermes_cli/cron.py`
- Modify: `hermes_cli/main.py`

**Steps:**
1. Add optional full prompt to `_format_job(..., include_prompt=False)`.
2. Add `if normalized == "show"` after job resolution in `cronjob()` returning `{success: True, job: _format_job(job, include_prompt=True)}`.
3. Add `cron_show()` in `hermes_cli/cron.py`, using `_cron_api(action="show", job_id=...)` and printing human-readable fields.
4. Dispatch `show` in `cron_command()`.
5. Register argparse subparser `cron show job_id`.
6. Run focused tests until green.

### Task 4: Verify

**Objective:** Confirm no regressions in cron-related surfaces.

**Commands:**
- `scripts/run_tests.sh tests/hermes_cli/test_cron.py tests/tools/test_cronjob_tools.py -q`
- Optional smoke: `python -m hermes_cli.main cron --help` if import path permits.

### Task 5: Commit and PR

**Objective:** Leave Joe a reviewable branch/PR.

**Commands:**
- `git status --short`
- `git add .hermes/plans/2026-05-27-cron-show-command.md hermes_cli/cron.py hermes_cli/main.py tools/cronjob_tools.py tests/hermes_cli/test_cron.py tests/tools/test_cronjob_tools.py`
- `git commit -m "feat: add cron show command"`
- `git push -u joe HEAD`
- `gh pr create --repo NousResearch/hermes-agent --head joe102084:joe/nightly-cron-show --base main --title "feat: add cron show command" --body-file -`
