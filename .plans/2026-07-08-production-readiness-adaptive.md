# Hermes Agent Production Readiness Implementation Plan

> **For Hermes:** Use adaptive orchestration with targeted subagents/reviews for implementation and verification. Follow TDD where production behavior changes are introduced.

**Goal:** Make the current Hermes Agent changes production-ready by adding missing regression coverage, hardening progress/status formatting, documenting phased rollout, and verifying with focused quality gates.

**Architecture:** This branch preserves the existing work in progress and completes it in vertical slices: gateway progress observability, subagent model-routing ergonomics, TTS language selection, and release-quality verification. The changes stay small and production-safe: deterministic helpers, no external network dependency in tests, no credentials or deployment side effects.

**Tech Stack:** Python 3.11+, pytest/pytest-xdist, ruff PLW1514, Hermes gateway/agent/delegation/TTS modules, GitHub branch workflow.

---

## Phase 0: Branch and Baseline Discovery

### Task 0.1: Preserve current work on a feature branch

**Objective:** Ensure all implementation happens on a dedicated Git branch without overwriting existing uncommitted work.

**Files:**
- Modify: none
- Verify: Git state only

**Steps:**
1. Run `git status --short --branch`.
2. Create branch `prod/adaptive-production-ready-20260708` from the current working tree.
3. Verify with `git branch --show-current`.

**Acceptance Criteria:**
- Current branch is `prod/adaptive-production-ready-20260708`.
- Pre-existing uncommitted changes remain intact.

---

## Phase 1: Planning and Production Readiness Scope

### Task 1.1: Save this implementation plan

**Objective:** Provide a durable, reviewable plan before implementation.

**Files:**
- Create: `.plans/2026-07-08-production-readiness-adaptive.md`

**Verification:**
- Run `test -f .plans/2026-07-08-production-readiness-adaptive.md`.
- Read the file and confirm it contains phases, acceptance criteria, and verification commands.

---

## Phase 2: Gateway Adaptive Progress Cards

### Task 2.1: Add deterministic helper coverage for gateway progress cards

**Objective:** Ensure long-running Telegram/gateway progress reports are structured workstream cards, show bounded percentages, include active subagents, and never silently regress to generic pings.

**Files:**
- Modify/Create: `tests/gateway/test_gateway_workstream_progress.py`
- Production code already touched: `gateway/run.py`

**Step 1: Write failing tests**
- Import `_progress_bar`, `_activity_percent`, `_format_duration`, and `_format_gateway_workstream_progress` from `gateway.run`.
- Test percentage clamping: negative → `0`, huge → `100`; active non-terminal summaries cap at `95` when budget exceeds max.
- Test duration formatting for seconds/minutes/hours.
- Test a workstream card contains `## Workstream`, `TaskFlow`, `Subagents / workers`, main agent, child subagent model/id, and a visible `█/░` bar.

**Step 2: Run RED**
- `python -m pytest tests/gateway/test_gateway_workstream_progress.py -q`
- Expected initially: fail if coverage file does not exist or formatting is missing.

**Step 3: Implement only the helper/test gaps**
- Keep helper output Telegram-friendly and deterministic.
- Avoid timestamps in assertions except checking an `Updated:` prefix.

**Step 4: Run GREEN**
- `python -m pytest tests/gateway/test_gateway_workstream_progress.py -q`
- Expected: pass.

---

## Phase 3: Subagent Model Routing Hardening

### Task 3.1: Cover top-level and per-task model override propagation

**Objective:** Make the new `delegate_task(model=...)` and per-task `tasks[].model` routing production-safe and test-protected.

**Files:**
- Modify: `tests/tools/test_delegate.py`
- Production code already touched: `tools/delegate_tool.py`

**Step 1: Write failing tests**
- Assert `DELEGATE_TASK_SCHEMA` exposes top-level `model` and nested task `model` properties.
- Patch `_build_child_agent` and `_run_single_child` to confirm:
  - single-task top-level `model` reaches `_build_child_agent(model=...)`.
  - batch per-task models override the top-level model independently.

**Step 2: Run RED**
- `python -m pytest tests/tools/test_delegate.py -q -k 'model or schema_valid'`

**Step 3: Implement/fix propagation if needed**
- Preserve credential-config fallback: `task.model` → top-level `model` → configured delegation model → parent model.
- Do not expose `max_iterations` to model schemas.

**Step 4: Run GREEN**
- `python -m pytest tests/tools/test_delegate.py -q -k 'model or schema_valid'`

---

## Phase 4: TTS Language Voice Selection

### Task 4.1: Complete Edge TTS language voice coverage and fallback behavior

**Objective:** Ensure Arabic/English auto voice selection is deterministic, configurable, and safe when config is invalid.

**Files:**
- Modify: `tests/tools/test_tts_speed.py`
- Production code already touched: `tools/tts_tool.py`

**Step 1: Write failing tests**
- Existing tests cover Arabic and English selection.
- Add tests for invalid `voices_by_language` falling back to configured default voice.
- Add tests for blank language-specific voice falling back to configured default voice.

**Step 2: Run RED/GREEN**
- `python -m pytest tests/tools/test_tts_speed.py -q -k 'EdgeTtsSpeed or language_voice'`

**Acceptance Criteria:**
- Edge TTS voice selection remains deterministic and does not require network/audio generation.

---

## Phase 5: Production Verification and Review

### Task 5.1: Run targeted test suite

**Objective:** Verify the touched surfaces without requiring the full repository test suite.

**Commands:**
- `python -m pytest tests/gateway/test_gateway_workstream_progress.py tests/tools/test_delegate.py tests/tools/test_tts_speed.py tests/gateway/test_gateway_inactivity_timeout.py -q`

**Acceptance Criteria:**
- All targeted tests pass.

### Task 5.2: Run lint/type import checks for touched files

**Objective:** Catch syntax and encoding regressions.

**Commands:**
- `python -m py_compile gateway/run.py run_agent.py tools/delegate_tool.py tools/tts_tool.py hermes_cli/commands.py`
- `python -m ruff check gateway/run.py run_agent.py tools/delegate_tool.py tools/tts_tool.py hermes_cli/commands.py tests/gateway/test_gateway_workstream_progress.py tests/tools/test_delegate.py tests/tools/test_tts_speed.py`

**Acceptance Criteria:**
- Commands pass, or any pre-existing environment/tool limitation is documented.

### Task 5.3: Independent code review gate

**Objective:** Use a fresh reviewer context to review the final diff for security, correctness, tests, and scope.

**Inputs:**
- `git diff --stat`
- `git diff` for touched files

**Acceptance Criteria:**
- Reviewer verdict has no blocking security or logic issues.
- Parent agent verifies any concrete issues are fixed.

---

## Phase 6: GitHub Branch Preparation

### Task 6.1: Commit and optionally push the branch

**Objective:** Prepare the branch for GitHub without merging to main.

**Commands:**
- `git status --short --branch`
- `git add .plans/2026-07-08-production-readiness-adaptive.md gateway/run.py run_agent.py tools/delegate_tool.py tools/tts_tool.py tests/tools/test_tts_speed.py tests/tools/test_delegate.py tests/gateway/test_gateway_workstream_progress.py hermes_cli/commands.py`
- `git commit -m "test: harden adaptive production-readiness changes"`
- If policy allows pushing after verification: `git push -u origin prod/adaptive-production-ready-20260708`

**Acceptance Criteria:**
- Branch exists locally.
- Commit contains the plan and verified production-readiness changes.
- Push is performed only after verification and with clear reporting; merge is not performed without explicit approval.
