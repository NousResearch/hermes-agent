# MASTER EXECUTION PLAN: Multi-Provider Memory for Hermes Agent

**Date:** 2026-04-28
**Methodology:** Subagent-Driven Development (skill v1.1.0)
**Constraint:** NO CAP — all 8 providers simultaneously for benchmarking
**Working Directory:** `/home/d/Desktop/agenda/hermes-agent` (fork of NousResearch/hermes-agent)
**Branch:** `feat/multi-provider-memory`

---

## Execution Model

This plan uses **batch execution with verification gates** and **two-stage review** (spec compliance then code quality) per task. Each batch has 3 parallel executor slots. Verification gates run between batches. Fresh subagents for every task — no context reuse.

```
Batch 1 (3 executors) → Gate 1 Verifier →
Batch 2 (3 executors) → Gate 2 Verifier →
Batch 3 (3 executors) → Gate 3 Verifier →
Final Integration Verifier (fresh, never reused)
```

**Strict agent separation:**
- Executors implement tasks
- Spec reviewers verify against plan spec
- Quality reviewers verify code quality
- Gate verifiers run batch verification commands
- Final verifier reviews entire implementation
- NEVER reuse an executor as a reviewer
- NEVER reuse a gate verifier as the final verifier

---

## PHASE -1: Fork & Branch Setup

**Purpose:** Create a clean fork of NousResearch/hermes-agent, set up dev environment, create feature branch.
**When:** Before anything else.
**Estimated time:** 10 minutes.

### Step 1: Fork the repository

```bash
# Fork and clone in one shot
gh repo fork NousResearch/hermes-agent --clone --remote-name origin

# Or if fork already exists:
gh repo clone <YOUR_USER>/hermes-agent
cd hermes-agent
git remote add upstream https://github.com/NousResearch/hermes-agent.git
```

### Step 2: Verify remotes

```bash
git remote -v
# Expected:
# origin    https://github.com/<YOUR_USER>/hermes-agent.git (fetch)
# origin    https://github.com/<YOUR_USER>/hermes-agent.git (push)
# upstream  https://github.com/NousResearch/hermes-agent.git (fetch)
# upstream  https://github.com/NousResearch/hermes-agent.git (push)
```

### Step 3: Sync with upstream

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Step 4: Create feature branch

```bash
git checkout -b feat/multi-provider-memory
git push -u origin feat/multi-provider-memory
```

### Step 5: Set up dev environment

```bash
# Install in editable mode (uv sync makes hermes command use fork code)
uv sync

# Verify installation
hermes --version

# Run tests to verify clean state
pytest tests/ -o 'addopts=' -q
```

### Step 6: Verify working directory

```bash
pwd
# Expected: /home/d/Desktop/agenda/hermes-agent (or wherever you cloned)

git status
# Expected: On branch feat/multi-provider-memory, nothing to commit

git log --oneline -5
# Expected: latest upstream commits
```

### Step 7: Copy plan files to fork

```bash
# Copy the master plan and PLAN.md into the fork for reference
cp /home/d/Desktop/agenda/multimemslot/PLAN.md /home/d/Desktop/agenda/hermes-agent/
cp /home/d/Desktop/agenda/multimemslot/MASTER-PLAN.md /home/d/Desktop/agenda/hermes-agent/
```

**Deliverable:** Clean fork at `/home/d/Desktop/agenda/hermes-agent` on branch `feat/multi-provider-memory`, dev environment working, tests passing.

---

## PHASE 0: Pre-Flight Audit

**Purpose:** Verify the plan against the live fork before any code is written.
**When:** After forking, before any implementation.

### Audit Verifier (1 subagent)

**Goal:** Audit the implementation plan against the live fork codebase

**Context:**
```
You are a plan auditor. You did NOT write this plan.

Read the plan at: /home/d/Desktop/agenda/multimemslot/PLAN.md
Read the fork at: /home/d/Desktop/agenda/hermes-agent

Verify:
- [ ] All file paths in the plan exist in the fork
- [ ] Line numbers referenced are approximately correct
- [ ] Code snippets to find-match actually exist in the files
- [ ] No structural changes since plan was written (check git log upstream)
- [ ] Config keys referenced exist in hermes_cli/config.py
- [ ] Test files referenced exist or can be created
- [ ] No conflicting PRs merged since plan was written
- [ ] The fork is clean (git status shows nothing modified)

Report: PASS with confidence level, or FAIL with specific discrepancies.
```

**Toolsets:** `['terminal', 'file']`
**Output:** `/home/d/Desktop/agenda/multimemslot/audit-report.md`

**If audit fails:** Patch the plan before proceeding. Re-audit if changes are substantial.

---

## PHASE 1: Gateway Lifecycle Fixes

**Purpose:** Fix 6 pre-existing bugs that become critical with multi-provider.
**Dependencies:** None (can start immediately after audit).
**Estimated time:** 3-4 hours.

### Batch 1.1: Independent Bug Fixes (3 parallel executors)

All three tasks are INDEPENDENT — no shared files, no dependencies.

#### Task 1.1.1: Fix #7192 — on_pre_compress() return value

**Executor:**
```
GOAL: Fix bug #7192 — capture on_pre_compress() return value and pass to compressor

CONTEXT:
- File: run_agent.py (~line 6081)
- Find: self._memory_manager.on_pre_compress(messages)  (bare statement)
- Replace: memory_context = self._memory_manager.on_pre_compress(messages) or ""
- File: agent/context_compressor.py
- Update compress() signature to accept memory_context: str = ""
- Pass memory_context to _generate_summary()
- Inject into summary prompt if non-empty

FOLLOW TDD:
1. Write failing test in tests/agent/test_context_compressor.py
2. Run: pytest tests/agent/test_context_compressor.py -v (verify FAIL)
3. Write minimal implementation
4. Run: pytest tests/agent/test_context_compressor.py -v (verify PASS)
5. Run: pytest tests/ -q (verify no regressions)
6. Commit: git add -A && git commit -m "fix(agent): capture on_pre_compress return value (#7192)"

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
Verify cwd with pwd before writing.
```

**Spec Reviewer:**
```
CHECK:
- [ ] on_pre_compress return value is captured (not bare statement)
- [ ] memory_context passed to compressor
- [ ] compressor signature updated
- [ ] Empty string handled gracefully
- [ ] No other call sites of on_pre_compress affected
- [ ] Test covers non-empty memory_context case
OUTPUT: PASS or list of gaps.
```

**Quality Reviewer:**
```
CHECK:
- [ ] Follows project conventions
- [ ] Proper error handling
- [ ] Clear variable names
- [ ] Adequate test coverage
- [ ] No side effects on existing compression
OUTPUT: APPROVED or REQUEST_CHANGES.
```

---

#### Task 1.1.2: Fix #7193 — on_turn_start() never wired

**Executor:**
```
GOAL: Fix bug #7193 — wire on_turn_start() into run_agent.py

CONTEXT:
- File: run_agent.py
- Find: self._user_turn_count += 1
- Add immediately after:
  try:
      last_user_msg = ""
      if self._messages and self._messages[-1].get("role") == "user":
          content = self._messages[-1].get("content", "")
          last_user_msg = content if isinstance(content, str) else str(content)[:500]
      self._memory_manager.on_turn_start(self._user_turn_count, last_user_msg)
  except Exception as exc:
      logger.debug("on_turn_start hook failed: %s", exc)

FOLLOW TDD:
1. Write test in tests/agent/test_memory_hooks.py
2. Verify on_turn_start is called after turn count increment
3. Commit: git commit -m "fix(agent): wire on_turn_start hook (#7193)"

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify hook is called, turn count is correct, error handling present.
**Quality Reviewer:** Verify conventions, test coverage, no side effects.

---

#### Task 1.1.3: Fix #15118 — dual routing in sequential dispatch

**Executor:**
```
GOAL: Fix bug #15118 — add MemoryManager check in _execute_tool_calls_sequential

CONTEXT:
- File: run_agent.py — _execute_tool_calls_sequential method
- Find the loop that calls handle_function_call(name, args, ...)
- Add MemoryManager check BEFORE handle_function_call:
  if self._memory_manager and self._memory_manager.has_tool(name):
      result = self._memory_manager.handle_tool_call(name, args)
  else:
      result = handle_function_call(name, args, ...)

- This mirrors the existing logic in _invoke_tool (concurrent path)

FOLLOW TDD:
1. Write test that verifies memory tool dispatch in sequential mode
2. Commit: git commit -m "fix(agent): route memory tools in sequential dispatch (#15118)"

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify routing logic matches _invoke_tool, memory tools work in sequential mode.
**Quality Reviewer:** Verify no regression in non-memory tool dispatch.

---

### Gate 1 Verifier (1 fresh subagent)

```
GOAL: Run Batch 1.1 verification gate

You are a verifier. You did NOT implement these tasks.
Check the filesystem state objectively and report PASS/FAIL.

Run these commands and compare against expected output:

1. cd /home/d/Desktop/agenda/hermes-agent && pwd
   Expected: /home/d/Desktop/agenda/hermes-agent

2. git log --oneline -5
   Expected: 3 new commits with messages:
   - fix(agent): capture on_pre_compress return value (#7192)
   - fix(agent): wire on_turn_start hook (#7193)
   - fix(agent): route memory tools in sequential dispatch (#15118)

3. pytest tests/agent/test_context_compressor.py -q
   Expected: all passed

4. pytest tests/agent/test_memory_hooks.py -q
   Expected: all passed

5. pytest tests/ -q
   Expected: no regressions

Report exactly which gates passed or failed.
```

---

### Batch 1.2: Gateway Fixes (3 parallel executors)

These tasks are INDEPENDENT of each other but depend on Batch 1.1 being committed.

#### Task 1.2.1: Fix #7358 — os.environ → contextvars

**Executor:**
```
GOAL: Fix bug #7358 — replace run_in_executor with asyncio.to_thread for contextvars propagation

CONTEXT:
- File: gateway/run.py
- Find all loop.run_in_executor(None, ...) calls in agent execution paths
- Replace with: await asyncio.to_thread(...)
- In gateway/session_context.py: remove os.environ fallback for gateway contexts
- Add clear_session_vars(tokens) in finally blocks in _handle_message()

COMMIT: fix(gateway): propagate contextvars to worker threads (#7358)

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify all run_in_executor calls replaced, finally blocks added, os.environ fallbacks removed.
**Quality Reviewer:** Verify thread safety, no regressions in gateway message handling.

---

#### Task 1.2.2: Fix #16155 — pass memory_manager into AIAgent

**Executor:**
```
GOAL: Fix bug #16155 — reuse MemoryManager across gateway agent recreations

CONTEXT:
- File: run_agent.py — AIAgent.__init__()
- Add optional parameter: memory_manager: Optional[MemoryManager] = None
- If provided, use it instead of creating new MemoryManager
- File: gateway/run.py — when building AIAgent for gateway sessions
- Pass the session's MemoryManager instance

This transitively fixes bug #9973 (prefetch cache lost).

COMMIT: fix(agent): reuse memory manager across gateway agent recreations (#16155)

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify memory_manager parameter added, gateway passes it, prefetch cache survives.
**Quality Reviewer:** Verify no regression in CLI mode (where memory_manager is None).

---

#### Task 1.2.3: Fix #11205 — call on_session_end in gateway

**Executor:**
```
GOAL: Fix bug #11205 — call on_session_end in gateway flush path

CONTEXT:
- File: gateway/run.py — _flush_memories_for_session() method
- BEFORE the flush agent spawn block, add:
  live_agent = self._cached_agents.get(session_key)
  if live_agent and hasattr(live_agent, '_memory_manager') and live_agent._memory_manager:
      try:
          history = session_store.get_messages(session_key)
          live_agent._memory_manager.on_session_end(history)
      except Exception as exc:
          logger.warning("on_session_end hook failed: %s", exc)

COMMIT: fix(gateway): call on_session_end on live agent before flush (#11205)

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify on_session_end is called before flush agent spawn, error handling present.
**Quality Reviewer:** Verify no regression in existing flush behavior.

---

### Gate 2 Verifier (1 fresh subagent)

```
GOAL: Run Batch 1.2 verification gate

1. cd /home/d/Desktop/agenda/hermes-agent && pwd
   Expected: /home/d/Desktop/agenda/hermes-agent

2. git log --oneline -6
   Expected: 6 commits total (3 from Batch 1.1 + 3 from Batch 1.2)

3. grep -r "asyncio.to_thread" gateway/run.py | wc -l
   Expected: > 0 (contextvars propagation in place)

4. grep "memory_manager" run_agent.py | grep "Optional" | head -3
   Expected: shows memory_manager parameter in AIAgent.__init__

5. grep "on_session_end" gateway/run.py
   Expected: shows call in _flush_memories_for_session

6. pytest tests/ -q
   Expected: no regressions

Report exactly which gates passed or failed.
```

---

## PHASE 2: Core Changes

**Purpose:** Remove the guard, update config, wire the plugin loader.
**Dependencies:** Phase 1 committed (gateway fixes in place).
**Estimated time:** 2-3 hours.

### Batch 2.1: Config + Plugin Loader (3 parallel executors)

All three tasks are INDEPENDENT.

#### Task 2.1.1: Add providers config key

**Executor:**
```
GOAL: Add memory.providers list key to hermes_cli/config.py

CONTEXT:
- File: hermes_cli/config.py
- Find memory config defaults dict
- Add: "providers": [] alongside existing "provider": ""
- Add validation: ensure providers is always a list of strings
- Handle edge cases: string input (wrap in list), non-list (reset to []), None (treat as [])

COMMIT: feat(config): add memory.providers list key for multi-provider support

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify key added, validation handles edge cases, backward compatible.
**Quality Reviewer:** Verify config loading doesn't break with old format.

---

#### Task 2.1.2: Add get_active_memory_providers()

**Executor:**
```
GOAL: Add get_active_memory_providers() to plugins/memory/__init__.py

CONTEXT:
- File: plugins/memory/__init__.py
- Add new function that reads config and returns list of provider names
- Supports both old format (memory.provider: "honcho") and new (memory.providers: ["honcho", "mem0"])
- New format takes precedence when non-empty
- Keep existing load_memory_provider(name) unchanged
- Keep existing _get_active_memory_provider() for backward compat

COMMIT: feat(plugins): add get_active_memory_providers() for multi-provider loading

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify function exists, returns list, handles both config formats.
**Quality Reviewer:** Verify existing single-provider code path unaffected.

---

#### Task 2.1.3: Update discover_plugin_cli_commands()

**Executor:**
```
GOAL: Update CLI command discovery for multi-provider

CONTEXT:
- File: plugins/memory/__init__.py — discover_plugin_cli_commands()
- Currently returns CLI commands for ONE active provider
- Change to iterate over get_active_memory_providers() and collect all
- Return list of command dicts from all active providers

COMMIT: feat(plugins): discover CLI commands from all active memory providers

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify all active providers' CLI commands are discovered.
**Quality Reviewer:** Verify no command name collisions, backward compatible.

---

### Gate 3 Verifier (1 fresh subagent)

```
GOAL: Run Batch 2.1 verification gate

1. cd /home/d/Desktop/agenda/hermes-agent && pwd

2. python -c "from hermes_cli.config import get_config; c = get_config(); print(c.get('memory', {}).get('providers', 'MISSING'))"
   Expected: [] (empty list, key exists)

3. python -c "from plugins.memory import get_active_memory_providers; print(get_active_memory_providers())"
   Expected: [] (empty list, function works)

4. git log --oneline -9
   Expected: 9 commits total

5. pytest tests/ -q
   Expected: no regressions
```

---

### Batch 2.2: Memory Manager Changes (3 parallel executors)

These tasks TOUCH THE SAME FILE but different methods — must be sequential, not parallel.

#### Task 2.2.1: Remove guard and _has_external

**Executor:**
```
GOAL: Remove single-external-provider guard in MemoryManager.add_provider()

CONTEXT:
- File: agent/memory_manager.py
- Change 1: In __init__(), remove self._has_external: bool = False
- Change 2: In add_provider():
  - Remove the if self._has_external: ... return block
  - Add duplicate-name check instead
  - Collect schemas BEFORE state mutation (fixes #9948)
  - Add summary log with ext_count, total_tools
  - Add tool budget warning at 20+
  - Add namespace prefix warning

COMMIT: feat(agent): remove single-external-provider guard for multi-provider support

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify guard removed, duplicate check added, schemas collected first, logging added.
**Quality Reviewer:** Verify error handling, no half-registration, backward compatible.

---

#### Task 2.2.2: Add remove_provider() method

**Executor:**
```
GOAL: Add remove_provider(name) method to MemoryManager

CONTEXT:
- File: agent/memory_manager.py
- New method: remove_provider(name: str) -> bool
- Cannot remove builtin
- Removes from _providers list
- Removes tool entries from _tool_to_provider
- Calls provider.shutdown()
- Returns True if removed, False if not found

COMMIT: feat(agent): add remove_provider() for runtime provider deregistration

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify method exists, removes correctly, handles edge cases.
**Quality Reviewer:** Verify thread safety, no resource leaks.

---

#### Task 2.2.3: Wire providers in run_agent.py

**Executor:**
```
GOAL: Replace single provider load with multi-provider loop in run_agent.py

CONTEXT:
- File: run_agent.py — memory wiring section
- Replace single _get_active_memory_provider() + load_memory_provider() with:
  from plugins.memory import get_active_memory_providers, load_memory_provider
  for provider_name in get_active_memory_providers():
      try:
          plugin_provider = load_memory_provider(provider_name)
          if plugin_provider and plugin_provider.is_available():
              self._memory_manager.add_provider(plugin_provider)
          elif plugin_provider:
              logger.info("Memory provider '%s' loaded but not available.", provider_name)
          else:
              logger.warning("Memory provider '%s' not found.", provider_name)
      except Exception as exc:
          logger.warning("Failed to load memory provider '%s': %s", provider_name, exc)

COMMIT: feat(agent): load all configured memory providers in agent init

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify loop iterates all providers, error handling per provider, backward compatible.
**Quality Reviewer:** Verify no regression with single provider config.

---

### Gate 4 Verifier (1 fresh subagent)

```
GOAL: Run Batch 2.2 verification gate

1. cd /home/d/Desktop/agenda/hermes-agent && pwd

2. python -c "
from agent.memory_manager import MemoryManager
mm = MemoryManager()
# Should not have _has_external attribute
print('has _has_external:', hasattr(mm, '_has_external'))
print('has remove_provider:', hasattr(mm, 'remove_provider'))
"
Expected: has _has_external: False, has remove_provider: True

3. git log --oneline -12
   Expected: 12 commits total

4. pytest tests/ -q
   Expected: no regressions

5. grep "get_active_memory_providers" run_agent.py
   Expected: shows the loop importing and calling the function
```

---

## PHASE 3: Tool System Polish

**Purpose:** Namespace enforcement, holographic rename, toolset integration.
**Dependencies:** Phase 2 committed (multi-provider loading works).
**Estimated time:** 2-3 hours.

### Batch 3.1: Tool System Changes (3 parallel executors)

#### Task 3.1.1: Holographic tool rename with aliases

**Executor:**
```
GOAL: Rename Holographic tools from fact_* to holographic_* with backward-compat aliases

CONTEXT:
- File: plugins/memory/holographic/__init__.py
- Rename fact_store → holographic_store in tool schema
- Rename fact_feedback → holographic_feedback in tool schema
- Add alias mapping in handle_tool_call:
  _TOOL_ALIASES = {"fact_store": "holographic_store", "fact_feedback": "holographic_feedback"}
- Log deprecation warning when alias is used

COMMIT: refactor(plugins): rename holographic tools with provider prefix and aliases

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify renamed, aliases work, deprecation warning logged.
**Quality Reviewer:** Verify no break for existing users of fact_store/fact_feedback.

---

#### Task 3.1.2: Add toolset filtering for memory tools

**Executor:**
```
GOAL: Add toolsets_enabled parameter to get_all_tool_schemas()

CONTEXT:
- File: agent/memory_manager.py — get_all_tool_schemas()
- Add optional parameter: toolsets_enabled: Optional[Set[str]] = None
- If provided and "memory" not in toolsets_enabled, return empty list
- Default behavior unchanged (None = no filtering)
- Update call site in model_tools.py to pass toolsets_enabled

COMMIT: feat(agent): add toolset filtering for memory provider tools

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify filtering works, default unchanged, call site updated.
**Quality Reviewer:** Verify no regression when memory toolset is enabled.

---

#### Task 3.1.3: Update tests for multi-provider

**Executor:**
```
GOAL: Update test suite for multi-provider behavior

CONTEXT:
- File: tests/agent/test_memory_provider.py (update existing)
- Add test: test_accepts_multiple_external_providers()
- Add test: test_rejects_duplicate_provider_name()
- Add test: test_remove_provider()
- Add test: test_tool_budget_warning_at_20_plus()
- Add test: test_namespace_prefix_warning()
- Add test: test_schema_first_mutation_rollback()
- Remove or update any tests that assert single-provider behavior

COMMIT: test(agent): add multi-provider memory tests

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

**Spec Reviewer:** Verify all new tests exist, old tests updated, coverage adequate.
**Quality Reviewer:** Verify tests are clean, no flaky assertions.

---

### Gate 5 Verifier (1 fresh subagent)

```
GOAL: Run Batch 3.1 verification gate

1. cd /home/d/Desktop/agenda/hermes-agent && pwd

2. python -c "
from plugins.memory.holographic import HolographicProvider
p = HolographicProvider()
names = [s['name'] for s in p.get_tool_schemas()]
print('holographic tools:', names)
print('has alias support:', hasattr(p, '_TOOL_ALIASES'))
"
Expected: holographic tools include holographic_store/holographic_feedback, has alias support: True

3. pytest tests/agent/test_memory_provider.py -q
   Expected: all passed

4. pytest tests/ -q
   Expected: no regressions

5. git log --oneline -15
   Expected: 15 commits total
```

---

## PHASE 4: Integration & Dogfooding

**Purpose:** Verify everything works together, run benchmarks, prepare for PR.
**Dependencies:** Phases 1-3 committed.
**Estimated time:** 1-2 hours.

### Batch 4.1: Integration Testing (1 executor + 2 verifiers)

#### Task 4.1.1: Full integration test

**Executor:**
```
GOAL: Run full integration test with all 8 providers configured

CONTEXT:
- Set config: memory.providers: ["honcho", "mem0", "hindsight", "byterover", "holographic", "openviking", "retaindb", "supermemory"]
- Start hermes from fork
- Verify all 8 providers register
- Verify tool count (should be ~37 memory tools)
- Verify tool budget warning appears
- Run: hermes chat -q "test multi-provider memory"
- Verify no errors

COMMIT: test: full integration test with all 8 memory providers

WORKING DIRECTORY: /home/d/Desktop/agenda/hermes-agent
```

---

#### Integration Verifier 1: Single-provider regression

```
GOAL: Verify single-provider behavior is unchanged

CONTEXT:
- Set config: memory.provider: "honcho" (old format, single provider)
- Start hermes from fork
- Verify only honcho is active
- Verify honcho tools work
- Verify no warnings about multi-provider
- Run full test suite: pytest tests/ -q

Report: PASS or FAIL with specific regressions.
```

---

#### Integration Verifier 2: Multi-provider benchmark

```
GOAL: Collect benchmark data for multi-provider operation

CONTEXT:
- Set config with all 8 providers
- Measure: startup time, prefetch latency, tool count, context usage
- Run: hermes chat -q "remember that I prefer Python over JavaScript"
- Verify all providers receive the memory
- Run: hermes chat -q "what programming language do I prefer?"
- Verify all providers can recall
- Collect and report metrics

Report: Benchmark data table.
```

---

### Final Integration Verifier (1 fresh subagent — NEVER reused from earlier)

```
GOAL: Final review of entire implementation before PR

You are the final integration verifier. You did NOT implement ANY of these tasks.
Review the complete implementation objectively.

CHECKLIST:
- [ ] All 15+ commits are clean and conventional
- [ ] No files modified that shouldn't be
- [ ] git diff --stat shows only expected changes
- [ ] Full test suite passes: pytest tests/ -q
- [ ] Single-provider regression passes
- [ ] Multi-provider with all 8 providers works
- [ ] Config migration works (old format still works)
- [ ] Gateway mode works with multi-provider
- [ ] Holographic aliases work
- [ ] remove_provider() works
- [ ] Tool budget warning appears at 20+
- [ ] No security issues (prompt injection, env var leaks)
- [ ] No performance regression with single provider
- [ ] Commit messages follow convention
- [ ] No stray workspace changes (package.json, bun.lock, etc.)

OUTPUT: APPROVED (ready for PR) or REQUEST_CHANGES (list issues).
```

---

## Parallelizability Matrix

```
PHASE -1: Fork & Branch Setup
  └─ [manual] — 10 min, no subagents

PHASE 0: Pre-Flight Audit
  └─ [1 verifier] — serial

PHASE 1: Gateway Fixes
  ├─ Batch 1.1: [Task 1.1.1 ∥ Task 1.1.2 ∥ Task 1.1.3] → [Gate 1 Verifier]
  └─ Batch 1.2: [Task 1.2.1 ∥ Task 1.2.2 ∥ Task 1.2.3] → [Gate 2 Verifier]

PHASE 2: Core Changes
  ├─ Batch 2.1: [Task 2.1.1 ∥ Task 2.1.2 ∥ Task 2.1.3] → [Gate 3 Verifier]
  └─ Batch 2.2: [Task 2.2.1 → Task 2.2.2 → Task 2.2.3] → [Gate 4 Verifier]
      (sequential — same file)

PHASE 3: Tool System
  └─ Batch 3.1: [Task 3.1.1 ∥ Task 3.1.2 ∥ Task 3.1.3] → [Gate 5 Verifier]

PHASE 4: Integration
  └─ [1 executor + 2 verifiers] → [Final Integration Verifier]

TOTAL: 15 tasks, 5 gate verifiers, 1 final verifier, 1 audit verifier + 1 fork setup (manual)
PARALLEL SLOTS: 3 max concurrent subagents
ESTIMATED WALL TIME: 8-12 hours (including fork setup and dogfooding)
```

---

## Scope Expansion Protocol

If implementation reveals issues NOT in the plan:

1. **Don't stop the plan** — dispatch fixers in parallel, not sequentially
2. **Mark as D (discovered)** — distinguish from P (planned) tasks
3. **Test full suite after each batch** — pre-existing failures surface here
4. **Investigate before patching** — architectural issues need research, not quick fixes
5. **Report to user** — "N planned tasks done, M discovered failures fixed, K issues investigated"

---

## Context Compaction Recovery

If context compacts during execution:

1. **STOP** — do not dispatch new subagents
2. **Re-read:** PLAN.md, all workflow docs, git status
3. **Re-load skills:** subagent-driven-development, writing-plans
4. **Verify filesystem:** `git log --oneline`, `git diff --stat`
5. **Confirm with user** before resuming

---

## Commit Sequence (15 commits)

```
Phase 1 — Gateway Fixes:
  1. fix(agent): capture on_pre_compress return value (#7192)
  2. fix(agent): wire on_turn_start hook (#7193)
  3. fix(agent): route memory tools in sequential dispatch (#15118)
  4. fix(gateway): propagate contextvars to worker threads (#7358)
  5. fix(agent): reuse memory manager across gateway agent recreations (#16155)
  6. fix(gateway): call on_session_end on live agent before flush (#11205)

Phase 2 — Core Changes:
  7. feat(config): add memory.providers list key for multi-provider support
  8. feat(plugins): add get_active_memory_providers() for multi-provider loading
  9. feat(plugins): discover CLI commands from all active memory providers
  10. feat(agent): remove single-external-provider guard for multi-provider support
  11. feat(agent): add remove_provider() for runtime provider deregistration
  12. feat(agent): load all configured memory providers in agent init

Phase 3 — Tool System:
  13. refactor(plugins): rename holographic tools with provider prefix and aliases
  14. feat(agent): add toolset filtering for memory provider tools
  15. test(agent): add multi-provider memory tests
```
