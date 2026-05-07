# Audit Report: Multi-Provider Memory Implementation Plan

**Auditor:** Automated plan auditor
**Date:** 2026-04-28
**Plan:** /home/d/Desktop/agenda/hermes-agent/PLAN.md
**Codebase:** /home/d/Desktop/agenda/hermes-agent (branch: feat/multi-provider-memory)

---

## VERDICT: CONDITIONAL PASS — Core plan is sound, but 6 bug claims are factually wrong

The plan's central thesis is CORRECT: the `_has_external` single-provider guard is
the blocker, and removing it is a one-line change that unlocks multi-provider.
However, 6 of the 8 "pre-existing bugs" listed in Workstream 2 are either already
fixed or were never broken. The implementation work is significantly smaller than
the plan estimates (~100 lines, not ~250).

---

## Check Results

### 1. Directory Verification
**PASS** — `pwd` returns `/home/d/Desktop/agenda/hermes-agent`

### 2. File Path Verification
**PASS** — All 7 files exist:
- agent/memory_manager.py ✅
- agent/memory_provider.py ✅
- plugins/memory/__init__.py ✅
- hermes_cli/config.py ✅
- run_agent.py ✅
- gateway/run.py ✅
- plugins/memory/holographic/__init__.py ✅

### 3. Guard Verification in memory_manager.py
**PASS** — Guard exists exactly as described:
- `_has_external` at lines 202, 216, 228 ✅
- "Only one external memory provider" warning at line 222 ✅
- Guard logic: `if self._has_external:` → reject with warning → return ✅

### 4. Config memory.provider Key
**PASS** — `"provider": ""` at hermes_cli/config.py line 830 ✅
- Memory section spans lines 821-831
- Comment at line 829 says "Only ONE external provider is allowed at a time"
- No `"providers": []` key yet (correctly — plan intends to ADD it)

### 5. All 8 Plugins
**PASS** — plugins/memory/ contains exactly:
byterover, hindsight, holographic, honcho, mem0, openviking, retaindb, supermemory ✅

### 6. Holographic Uses fact_* Prefix
**PASS** — `fact_store` (line 38) and `fact_feedback` (line 76) confirmed ✅
- 22 total `fact_` references in the file
- Plan correctly flags this as a naming convention violation

### 7. Git Status
**PARTIAL PASS**
- Branch: `feat/multi-provider-memory` ✅
- Upstream: `https://github.com/NousResearch/hermes-agent.git` ✅
- ⚠️ `uv.lock` is modified (unstaged) — not fully clean
- ⚠️ Branch is 1 commit ahead of origin (unpushed)

### 8. Upstream Remote
**PASS** — upstream → https://github.com/NousResearch/hermes-agent.git ✅

---

## Critical Discrepancies: Bug Claims vs. Actual Code

### Bug #7193 — "on_turn_start never wired" — WRONG
**Plan claim:** on_turn_start is never called, needs wiring in run_agent.py
**Actual code:** ALREADY WIRED at run_agent.py:10122
```python
self._memory_manager.on_turn_start(self._user_turn_count, _turn_msg)
```
This is inside a `if self._memory_manager:` guard with try/except. Fully functional.

### Bug #15118 — "Dual routing (handle_function_call instead of memory_manager)" — WRONG
**Plan claim:** Tool calls go through `handle_function_call` bypassing memory_manager
**Actual code:** Memory tools ARE already routed through memory_manager:
- Parallel path: run_agent.py:8841-8842 (`self._memory_manager.handle_tool_call`)
- Sequential path: run_agent.py:9455 (`self._memory_manager.handle_tool_call`)
Both paths check `self._memory_manager.has_tool(function_name)` first.

### Bug #7358 — "os.environ race conditions" — MOSTLY WRONG
**Plan claim:** Gateway uses `run_in_executor` without contextvars
**Actual code:** Gateway ALREADY uses `copy_context()` + `ctx.run` pattern:
```python
# gateway/run.py:8769-8773
async def _run_in_executor_with_context(self, func, *args):
    """Run blocking work in the thread pool while preserving session contextvars."""
    loop = asyncio.get_running_loop()
    ctx = copy_context()
    return await loop.run_in_executor(None, ctx.run, func, *args)
```
`copy_context` is imported from `contextvars` at line 29. Session vars are set
via `_set_session_env()` (line 4599) and cleared via `_clear_session_env()` (line 8764).

### Bug #11205 — "on_session_end never called in gateway" — WRONG
**Plan claim:** Gateway never calls on_session_end, references `_flush_memories_for_session()`
**Actual code:** 
- `_flush_memories_for_session()` DOES NOT EXIST in the codebase
- Gateway DOES call `shutdown_memory_provider()` which calls `on_session_end()`:
  - In `_cleanup_agent_resources()` (gateway/run.py:2095-2110)
  - During gateway shutdown (gateway/run.py:3041-3056)
  - In run_agent.py `shutdown_memory_provider()` (line 4311-4346)

### Bug #16155 — "Provider re-init per message" — WRONG
**Plan claim:** Gateway creates a new AIAgent per message, losing provider state
**Actual code:** Gateway CACHES AIAgents via `_agent_cache` OrderedDict:
- Cache initialized at gateway/run.py:859
- `_AGENT_CACHE_MAX_SIZE = 128` (line 41)
- `_AGENT_CACHE_IDLE_TTL_SECS = 3600.0` (line 42)
- Comment at line 848-851 explicitly explains the caching rationale

### Bug #9973 — "Prefetch cache lost" — LIKELY WRONG
**Plan claim:** Prefetch results lost when agent is recreated
**Actual code:** Since agents are cached (see #16155 above), the prefetch cache
survives across turns within the same session.

---

## Discrepancy in Plan Section 1.6 (run_agent.py Wiring)

**Plan shows "BEFORE" code as:**
```python
provider_name = _get_active_memory_provider()
if provider_name:
    plugin_provider = load_memory_provider(provider_name)
```

**Actual code (run_agent.py:1635-1643):**
```python
_mem_provider_name = mem_config.get("provider", "") if mem_config else ""
if _mem_provider_name:
    from agent.memory_manager import MemoryManager as _MemoryManager
    from plugins.memory import load_memory_provider as _load_mem
    self._memory_manager = _MemoryManager()
    _mp = _load_mem(_mem_provider_name)
```

The actual code reads config directly rather than calling `_get_active_memory_provider()`.
The pattern is equivalent but the plan's "BEFORE" snippet doesn't match literally.

---

## What the Plan Gets RIGHT

1. **Core thesis:** The `_has_external` guard IS the single blocker for multi-provider ✅
2. **Guard location and logic:** Exactly as described (lines 202, 216, 228) ✅
3. **Config structure:** `memory.provider` key exists, `providers` list needs adding ✅
4. **Plugin inventory:** All 8 plugins present and accounted for ✅
5. **Holographic naming violation:** `fact_*` prefix confirmed ✅
6. **remove_provider() doesn't exist:** Confirmed — needs to be added ✅
7. **get_active_memory_providers() doesn't exist:** Confirmed — needs to be added ✅
8. **No budget warning in get_all_tool_schemas():** Confirmed — needs adding ✅
9. **on_pre_compress return value discarded:** Confirmed at run_agent.py:8611 ✅

---

## Revised Effort Estimate

| Workstream | Plan Estimate | Actual Estimate | Notes |
|------------|--------------|-----------------|-------|
| WS1: Core changes | ~100-115 lines | ~80 lines | Guard removal + config + loader + wiring |
| WS2: Gateway fixes | ~150 lines | ~15 lines | Only Bug #7192 (on_pre_compress) is real |
| WS3: Tool system | ~50 lines | ~40 lines | Remove_provider + budget warning |
| **Total** | **~250 lines** | **~135 lines** | Work is ~54% of estimate |

The only real bug is **#7192 (on_pre_compress return value discarded)**.
Task 1.8 needs the return value captured and passed to the compressor.
Everything else in Workstream 2 is already implemented.

---

## Recommendation

1. **Proceed with Workstream 1** (tasks 1.1–1.8) — core changes are real and needed
2. **Skip Workstream 2 tasks 2.1–2.3** — bugs #7358, #16155, #9973, #11205 are already fixed
3. **Only do Bug #7192 fix** (capture on_pre_compress return value) — this is the one real bug
4. **Proceed with Workstream 3** (tasks 3.1–3.4) — tool system polish is valid
5. **Fix git state** — commit or stash `uv.lock` changes, push the 1 unpushed commit
6. **Update plan** — remove incorrect bug claims to avoid confusion during implementation

---

## Confidence Level

**92%** — High confidence. All checks were performed against live code via
grep/read operations. The one area of uncertainty is whether the gateway's
agent caching is truly complete (i.e., whether there are code paths that
DO create fresh agents per message that I didn't find). But the primary
caching mechanism is clearly present and documented.
