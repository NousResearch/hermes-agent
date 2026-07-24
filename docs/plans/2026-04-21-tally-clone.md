# Hermes Tally Clone Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Create a working Tally clone in Hermes that sends Andrew a daily billing report, clones the old Production Hub Tally summary, and separately tracks Kimi direct vs Kimi via OpenRouter using Hermes native session data.

**Architecture:** Use a lightweight Python reporter script to combine two sources: old Production Hub billing endpoints for the legacy 12-provider budget view, plus Hermes `state.db` for native provider/base-URL split tracking. Schedule the reporter via Hermes cron to deliver the report to this Telegram chat each morning.

**Tech Stack:** Python stdlib, SQLite, local HTTP requests to localhost:5053, Hermes cron jobs, Hermes env/config.

---

### Task 1: Add a reusable Tally clone script

**Objective:** Create a Python script that builds a full Tally-style JSON context from Production Hub billing APIs plus Hermes native state.db usage.

**Files:**
- Create: `scripts/tally_clone_report.py`

**Step 1:** Read `~/.hermes/state.db` and aggregate yesterday + MTD by `billing_provider`, `billing_base_url`, and `model`.

**Step 2:** Call:
- `http://localhost:5053/api/billing/budget-status`
- `http://localhost:5053/api/billing/subscriptions`
- per-provider routes needed for freshness/health checks

**Step 3:** Emit JSON with:
- `generated_at`
- `legacy_budget_status`
- `legacy_subscriptions`
- `endpoint_health`
- `hermes_native_split`
- `comparison_notes`

**Step 4:** Verify with:
- `python3 scripts/tally_clone_report.py`

### Task 2: Install the live cron script

**Objective:** Put the working script in `~/.hermes/scripts/` so cron can run it directly.

**Files:**
- Create: `~/.hermes/scripts/tally_clone_report.py`

**Step 1:** Copy the script from the repo to the Hermes script directory.

**Step 2:** Verify with:
- `python3 ~/.hermes/scripts/tally_clone_report.py`

### Task 3: Schedule the daily Tally clone

**Objective:** Create a Hermes cron job that runs every morning and sends the formatted report to Andrew in this Telegram chat.

**Files:**
- No repo file required; cron job stored in Hermes cron registry.

**Step 1:** Create a daily cron using the script for context.

**Step 2:** Prompt should format:
- old Tally-style headline
- yesterday + MTD + projected EOM
- provider alerts
- subscriptions total
- Hermes native Kimi comparison block:
  - Moonshot direct
  - OpenRouter Kimi
  - sessions, tokens, costs

**Step 3:** Verify by manually running the cron once.

### Task 4: Validate and ship

**Objective:** Ensure the script runs, cron is scheduled, and the code is committed to a PR branch.

**Files:**
- Modify: git branch in repo

**Step 1:** Run syntax check:
- `python3 -m py_compile scripts/tally_clone_report.py`

**Step 2:** Commit:
- `git add docs/plans/2026-04-21-tally-clone.md scripts/tally_clone_report.py`
- `git commit -m "feat: add tally clone billing reporter"`

**Step 3:** Push and open PR.
