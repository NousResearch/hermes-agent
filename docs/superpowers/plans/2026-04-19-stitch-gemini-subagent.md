# Stitch Gemini Subagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `delegate_task` callers and skills such as `google-stitch` to route specific subagents to a different provider/model than the global `delegation` default.

**Architecture:** Extend the delegation schema and runtime so top-level and per-task overrides can specify `provider` and `model`, with provider resolution reusing the existing runtime-provider path. Then document the new routing pattern in the `google-stitch` skill so prototyping subagents can target Gemini 3+ while coding subagents continue using Codex.

**Tech Stack:** Python, pytest, Hermes skills markdown

---

### Task 1: Add Failing Tests For Per-Task Delegation Overrides

**Files:**
- Modify: `tests/tools/test_delegate.py`

- [ ] Add tests that prove `delegate_task(tasks=[...])` can pass distinct `provider` and `model` overrides per task.
- [ ] Add tests that prove top-level `provider` / `model` overrides work without changing `config.yaml`.
- [ ] Run targeted delegate tests and verify the new assertions fail before implementation.

### Task 2: Implement Provider/Model Overrides In Delegate Tool

**Files:**
- Modify: `tools/delegate_tool.py`
- Test: `tests/tools/test_delegate.py`

- [ ] Extend the tool schema to accept top-level and per-task `provider` / `model`.
- [ ] Resolve effective child credentials with priority `task override > top-level override > delegation config > parent inherit`.
- [ ] Keep existing behavior unchanged when no override is supplied.
- [ ] Re-run targeted delegate tests until they pass.

### Task 3: Document Google Stitch Routing

**Files:**
- Modify: `skills/software-development/google-stitch/SKILL.md`

- [ ] Update the skill to instruct Hermes to dispatch prototyping/frontend subagents with Gemini 3+ overrides.
- [ ] Preserve current Stitch CLI usage and timeout guidance.
- [ ] Re-read the skill for consistency with the new `delegate_task` arguments.

### Task 4: Verify, Commit, And Prepare PR

**Files:**
- Modify: any touched files above

- [ ] Run focused verification commands for delegation tests.
- [ ] Review `git diff` for only intended changes.
- [ ] Commit on the feature branch with a focused message.
- [ ] Push the branch to `fork` and open a PR against the fork-ready upstream flow.
