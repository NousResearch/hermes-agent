# PR 98470 Passive Worker Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an exact-main, authorship-preserving, two-file replacement for the passive worker-contract claim hidden inside cumulative PR #98470.

**Architecture:** Start from verified upstream `main` commit `1cf36398135f4848a1d04b2167ffb564b7881d35` and cherry-pick the original passive-contract commit `e0b848fed2f0bb446237c066eaa991e454096a43`. Treat the production-file allowlist as fail-closed: any conflict or additional production path stops integration instead of widening scope.

**Tech Stack:** Python 3.13, pytest, Ruff, compileall, Git.

**Spec:** `docs/superpowers/specs/2026-08-31-pr-98470-passive-worker-contract-design.md`

## Global Constraints

- Preserve the original commit author and commit message.
- Production changes are limited to `agent/worker_contract.py` and `tests/agent/test_worker_contract.py`.
- Do not include lifecycle integration, model or provider changes, Qwen changes, routing, concurrency, egress, Kanban, PR-feedback, desktop, cron, gateway, packaging, or Python pinning.
- Do not push, create or close a PR, force-push, delete a branch, or merge.
- Focused checks are diagnostic evidence, not a full-suite or hosted exact-head CI receipt.

---

### Task 1: Integrate the Passive Worker Contract

**Files:**
- Create: `agent/worker_contract.py`
- Create: `tests/agent/test_worker_contract.py`
- Verify: `docs/superpowers/specs/2026-08-31-pr-98470-passive-worker-contract-design.md`

**Interfaces:**
- Consumes: original commit `e0b848fed2f0bb446237c066eaa991e454096a43` and upstream base `1cf36398135f4848a1d04b2167ffb564b7881d35`.
- Produces: the public classes and functions defined by `agent.worker_contract`, with their contract tests preserved byte-for-byte unless current-main compatibility requires a reviewed two-file-only adjustment.

- [ ] **Step 1: Verify the exact clean base and source provenance**

Run:

```bash
git status --short
git rev-parse refs/remotes/origin/main
git merge-base HEAD refs/remotes/origin/main
git show -s --format='%H%n%an%n%ae%n%s' e0b848fed2f0bb446237c066eaa991e454096a43
git diff-tree --no-commit-id --name-only -r e0b848fed2f0bb446237c066eaa991e454096a43
```

Expected: clean tracked state; branch ancestry contains base `1cf36398135f4848a1d04b2167ffb564b7881d35`; source author is preserved; source paths are exactly the two allowlisted files.

- [ ] **Step 2: Apply the original commit without rewriting authorship**

Run:

```bash
git cherry-pick e0b848fed2f0bb446237c066eaa991e454096a43
```

Expected: a clean cherry-pick that creates only the two allowlisted production paths. On any conflict, abort with `git cherry-pick --abort` and stop for reassessment.

- [ ] **Step 3: Verify the branch diff is narrowly scoped**

Run:

```bash
git diff --name-status 1cf36398135f4848a1d04b2167ffb564b7881d35..HEAD
git diff --stat 1cf36398135f4848a1d04b2167ffb564b7881d35..HEAD
git diff --check 1cf36398135f4848a1d04b2167ffb564b7881d35..HEAD
```

Expected: the two contract files plus the approved spec and plan documents; no excluded subsystem paths; no whitespace errors.

- [ ] **Step 4: Run the complete contract test module**

Run:

```bash
/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m pytest -q tests/agent/test_worker_contract.py
```

Expected: all collected contract tests pass with zero failures.

- [ ] **Step 5: Run focused lint and compilation**

Run:

```bash
/Users/mikedemott/.hermes/hermes-agent/venv/bin/ruff check agent/worker_contract.py tests/agent/test_worker_contract.py
/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m compileall -q agent/worker_contract.py tests/agent/test_worker_contract.py
```

Expected: both commands exit zero.

- [ ] **Step 6: Run adjacent agent compatibility tests**

Run:

```bash
/Users/mikedemott/.hermes/hermes-agent/venv/bin/python -m pytest -q tests/agent/test_subagent_lifecycle.py
```

Expected: all collected adjacent lifecycle tests pass with zero failures; the passive module remains unwired and introduces no lifecycle regression.

- [ ] **Step 7: Verify final provenance and cleanliness**

Run:

```bash
git show -s --format='%H%n%P%n%an%n%ae%n%s' HEAD
git status --short
git log --oneline --decorate -3
```

Expected: the integration commit retains Mike DeMott and the original noreply identity; the worktree is clean; the branch contains only the design commit, plan commit, and passive-contract commit above upstream main.
