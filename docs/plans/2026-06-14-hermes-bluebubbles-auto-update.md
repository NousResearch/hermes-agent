# Hermes BlueBubbles Auto-Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the local running Hermes checkout automatically updated to upstream `main` while preserving the BlueBubbles/iMessage fixes.

**Architecture:** Maintain a dedicated fork branch, `fix/bluebubbles-canonical-chat-id`, that rebases onto official `NousResearch/hermes-agent/main`. A local updater script validates the rebased branch in a staging worktree, pushes the branch to the user's fork, deploys it into the runtime checkout at `~/.hermes/hermes-agent`, and restarts the Hermes gateway only after validation succeeds. Failures stop before deployment whenever possible; restart failures roll the runtime checkout back to the previous commit.

**Tech Stack:** Python stdlib, Git worktrees, pytest, uv, Hermes gateway CLI, Codex cron automation.

---

### Task 1: Add updater unit tests

**Files:**
- Create: `tests/scripts/test_auto_update_bluebubbles_fix.py`

**Steps:**
1. Test remote resolution for the current local layout: official remote plus user fork remote.
2. Test that untracked local config/skills do not count as blocking tracked changes.
3. Test that dependency sync only runs for environment files such as `pyproject.toml` and `uv.lock`.
4. Test that BlueBubbles validation runs only `py_compile` and `tests/gateway/test_bluebubbles.py`.

### Task 2: Implement the updater script

**Files:**
- Create: `scripts/auto_update_bluebubbles_fix.py`

**Steps:**
1. Detect official and fork remotes, adding the fork remote when absent.
2. Fetch both remotes.
3. Create a staging worktree from `fix/bluebubbles-canonical-chat-id`.
4. Rebase the staging branch onto official `main`.
5. Sync dependencies into the runtime venv when dependency files changed or pytest is missing.
6. Run BlueBubbles validation.
7. Push the rebased branch to the fork.
8. Deploy the validated branch to `~/.hermes/hermes-agent`.
9. Restart Hermes gateway and report status.

### Task 3: Initial branch migration

**Files:**
- Modify: `gateway/platforms/bluebubbles.py`
- Modify: `tests/gateway/test_bluebubbles.py`
- Add updater files from Tasks 1-2.

**Steps:**
1. Add the user's fork remote to the runtime checkout.
2. Create `fix/bluebubbles-canonical-chat-id` from current upstream `main`.
3. Reapply the two BlueBubbles fixes from the earlier fork branch.
4. Commit the updater and tests.
5. Push the branch to the fork.

### Task 4: Install recurring automation

**Steps:**
1. Create a Codex cron automation that runs the updater from `~/.hermes/hermes-agent`.
2. Keep the automation active.
3. Log every run to `~/.hermes/logs/bluebubbles-auto-update.log`.

### Task 5: Verify deployment

**Steps:**
1. Run updater dry-run or equivalent validation.
2. Run BlueBubbles tests.
3. Verify `hermes --version` points at `~/.hermes/hermes-agent`.
4. Verify `hermes gateway status` no longer reports a stale service definition after restart.
