# Managed-Agent Policy Guardrails Implementation Plan

> **For Hermes:** Use Multica/Codex formal development for implementation; keep this as the repo-local ledger plan.

Status: in_progress

**Goal:** Move Managed-Agent/Kanban worker policy from passive metadata into an explicit, visible, testable policy contract that workers and reviewers can follow without overclaiming OS sandbox enforcement.

**Architecture:** Keep enforcement honest and incremental: local workers still do not have OS/container sandboxing, but every spawned worker receives a structured policy contract in env/context, CLI output surfaces the contract, and tests prove read-only/test-only/code-edit/sandbox-strict semantics are communicated consistently. This phase is policy-contract guardrails, not destructive-command interception.

**Tech Stack:** Python, Hermes Kanban DB/dispatcher, pytest.

---

## Non-goals

- Do not implement Docker/VM/OS sandboxing in this phase.
- Do not mutate user config or install external runtimes.
- Do not block all shell commands at OS level yet; that requires a later tool/runtime enforcement layer.
- Do not store secrets, `.env`, tokens, DB dumps, raw videos, model weights, or private credentials in `.hermes`.

---

## Acceptance Criteria

1. Worker policy descriptors include a structured contract with:
   - allowed operations;
   - forbidden operations;
   - enforcement level (`contract` for local worker);
   - whether edits/destructive commands are allowed;
   - explicit note that local `os_sandbox=false` unless a future backend proves otherwise.
2. Dispatcher injects this contract as a bounded JSON env var, e.g. `HERMES_KANBAN_POLICY_CONTRACT`.
3. Worker context includes a clear `## Worker policy contract` section with policy-specific instructions.
4. `read_only` and `test_only` policies clearly instruct the worker not to edit files; `code_edit` allows workspace-scoped edits; `sandbox_strict` requires strongest available isolation but admits local contract-only status.
5. CLI show/json surfaces enough policy/capability detail for operators/reviewers.
6. Tests cover descriptor shape, worker context text, spawn env injection, CLI visibility, and no OS sandbox overclaim.
7. Existing phase-1 tests remain green.

---

## File Ownership / Expected Changes

Single implementation child should own all touched files to avoid overlap.

- Modify: `hermes_cli/kanban_db.py`
  - Add `worker_policy_contract(policy, capabilities=None)` helper or equivalent.
  - Extend `WORKER_POLICY_DESCRIPTORS` with `allowed_operations`, `forbidden_operations`, `enforcement_level`, and clear notes.
  - Include `policy_contract` in run metadata where useful.
  - Inject `HERMES_KANBAN_POLICY_CONTRACT` in `_default_spawn`.
  - Add a `## Worker policy contract` section to `build_worker_context`.
- Modify: `hermes_cli/kanban.py`
  - Surface policy contract/capability summary in `kanban show` output and JSON where low-risk.
- Modify: `tests/hermes_cli/test_kanban_db.py`
  - Add tests for contract descriptors, context text, env injection, and no sandbox overclaim.
- Modify: `tests/hermes_cli/test_kanban_cli.py`
  - Add/update CLI show coverage if needed.
- Update: `.hermes/tasks/2026-05-12-managed-agent-policy-guardrails.md`
  - Record Multica IDs, status changes, verification evidence.

---

## Task 1: Policy contract helper and descriptor expansion

**Objective:** Make policy semantics structured and reusable.

**Files:**
- Modify: `hermes_cli/kanban_db.py`
- Test: `tests/hermes_cli/test_kanban_db.py`

**Steps:**
1. Write/adjust tests proving `worker_policy_descriptor("read_only")` or new helper returns structured fields:
   - `allows_edits=False`
   - `allows_destructive_commands=False`
   - `enforcement_level="contract"`
   - non-empty `allowed_operations` and `forbidden_operations`
2. Implement minimal descriptor/helper updates.
3. Run: `python -m pytest -q tests/hermes_cli/test_kanban_db.py -o 'addopts='`.

---

## Task 2: Dispatch env + run metadata contract

**Objective:** Ensure spawned workers receive policy contract as machine-readable env and dispatcher metadata records it.

**Files:**
- Modify: `hermes_cli/kanban_db.py`
- Test: `tests/hermes_cli/test_kanban_db.py`

**Steps:**
1. Add failing test around `_default_spawn` test stub/env capture for `HERMES_KANBAN_POLICY_CONTRACT`.
2. Add failing test that dispatcher metadata includes `policy_contract` or equivalent bounded structure.
3. Implement env injection and metadata inclusion.
4. Run focused tests.

---

## Task 3: Worker context guardrail section

**Objective:** Put policy instructions into the prompt workers actually read.

**Files:**
- Modify: `hermes_cli/kanban_db.py`
- Test: `tests/hermes_cli/test_kanban_db.py`

**Steps:**
1. Add test that `build_worker_context` for `read_only` includes:
   - `## Worker policy contract`
   - `Do not edit files` or equivalent;
   - `os_sandbox=false` / no OS sandbox overclaim.
2. Add test that `code_edit` includes workspace-scoped edit permission.
3. Implement context rendering.
4. Run focused tests.

---

## Task 4: CLI operator visibility

**Objective:** Make policy guardrails visible in `hermes kanban show` for humans.

**Files:**
- Modify: `hermes_cli/kanban.py`
- Test: `tests/hermes_cli/test_kanban_cli.py`

**Steps:**
1. Add/update CLI show test proving policy contract/capability summary appears.
2. Keep output concise; avoid dumping huge JSON.
3. Run CLI tests.

---

## Verification Commands

Run from repo root with venv active:

```bash
source venv/bin/activate
python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py -o 'addopts='
python -m pytest -q tests/test_hermes_memory_provider.py tests/agent/test_auxiliary_temperature_retry.py tests/agent/test_prompt_builder.py -o 'addopts='
git diff --check
```

If feasible before final push:

```bash
python -m pytest -q tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_cli.py tests/tools/test_kanban_tools.py -o 'addopts='
```

---

## Execution Discipline

- Think before coding: state assumptions in issue comments/ledger when policy semantics are ambiguous.
- Simplicity first: implement the minimum code needed for the acceptance criteria; no speculative sandbox runtime.
- Surgical changes: touch only the files listed above unless a test requires a small adjacent update.
- Goal-driven execution: add regression/probe tests first, then loop until focused tests pass.
- Do not claim local OS/container sandbox enforcement. Use `os_sandbox=false` unless real backend detection exists.
