# Multi-Provider Memory — Final Polish Plan (VERIFIED)

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fix all remaining issues found by 3 holistic verifiers before PR creation.

**Branch:** feat/multi-provider-memory at /home/d/Desktop/agenda/hermes-agent

**Principle:** No commits until verifiers return only compliments.

---

## CRITICAL (2 items)

### Task 1: Fix `_get_current_memory_provider()` in plugins_cmd.py

**Problem:** Reads only `memory.provider` (legacy string). Ignores `memory.providers` list.
When multi-provider is configured, returns empty string → plugins UI shows "built-in".

**File:** `hermes_cli/plugins_cmd.py:761-768`  ✅ LINE NUMBERS CONFIRMED

**Current code:**
```python
def _get_current_memory_provider() -> str:
    """Return the current memory.provider from config (empty = built-in)."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("memory", {}).get("provider", "") or ""
    except Exception:
        return ""
```

**Fix:** Use `memory.providers` list first, fall back to legacy. Return comma-joined string or first provider.

```python
def _get_current_memory_provider() -> str:
    """Return the current memory provider(s) from config (empty = built-in)."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        mem = config.get("memory", {})
        providers = mem.get("providers", [])
        if providers:
            active = [p for p in providers if p]
            return active[0] if len(active) == 1 else ", ".join(active)
        return mem.get("provider", "") or ""
    except Exception:
        return ""
```

**NOTE — caller compatibility:** `_configure_memory_provider()` at line 805 uses
the return value as a comparison key (`if name == current` at line 817). When
multiple providers are active, the comma-joined string won't match any single
name → radio picker won't pre-select. This is acceptable: the picker is
single-select anyway, and users would need to use `hermes memory setup` for
multi-provider management. The fix is still correct because it returns the
actual state rather than a misleading empty string.

---

### Task 2: Fix `_save_memory_provider()` in plugins_cmd.py

**Problem:** Writes ONLY `config["memory"]["provider"] = name`. Destroys multi-provider config.
Users who set up multiple providers via `hermes memory setup` lose all but one when using `hermes plugins`.

**File:** `hermes_cli/plugins_cmd.py:781-788`  ✅ LINE NUMBERS CONFIRMED

**Current code:**
```python
def _save_memory_provider(name: str) -> None:
    """Persist memory.provider to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "memory" not in config:
        config["memory"] = {}
    config["memory"]["provider"] = name
    save_config(config)
```

**Fix:** Write to BOTH legacy field (backwards compat) AND providers list.

```python
def _save_memory_provider(name: str) -> None:
    """Persist memory provider to config.yaml (both legacy and list format)."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "memory" not in config:
        config["memory"] = {}
    config["memory"]["provider"] = name
    if name:
        config["memory"]["providers"] = [name]
    else:
        config["memory"]["providers"] = []
    save_config(config)
```

**Note:** The `_configure_memory_provider()` function (line 801) is a radio picker (single-select). This is acceptable for now — the plugins UI is a single-select toggle. But the SAVE must preserve the list format.

---

## SHOULD-FIX (10 items)

### Task 3: Remove dead `_get_active_memory_provider()` function

**Problem:** Never called anywhere. Superseded by `get_active_memory_providers()`.

**File:** `plugins/memory/__init__.py:308-320`  ✅ LINE NUMBERS CONFIRMED

**Fix:** Delete lines 307-320 (blank line before + entire function):
```python
# DELETE:
def _get_active_memory_provider() -> Optional[str]:
    """Read the active memory provider name from config.yaml.

    Returns the provider name (e.g. ``"honcho"``) or None if no
    external provider is configured.  Lightweight — only reads config,
    no plugin loading.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("memory", {}).get("provider") or None
    except Exception:
        return None
```

**Verify:** grep for `_get_active_memory_provider` — only 3 hits (all in the definition itself). ✅ CONFIRMED NO CALLERS

---

### Task 4: Fix stale docstring in `memory_provider.py`

**Problem:** Lines 3-5 say "One external provider is active at a time...The MemoryManager enforces this limit." — contradicts lines 9-10 which already say "Multiple external providers can run simultaneously."

**File:** `agent/memory_provider.py:1-25`  ✅ LINE NUMBERS CONFIRMED

**Fix:** Replace lines 3-5:
```python
"""Abstract base class for pluggable memory providers.

Memory providers give the agent persistent recall across sessions.
Multiple external providers can run simultaneously alongside the
always-on built-in memory (MEMORY.md / USER.md) via the
``memory.providers`` list in config.yaml.
```

Also fix line 14:
```python
# Before: activated by memory.provider config.
# After:  activated by memory.providers list (or legacy memory.provider string) in config.
```

---

### Task 5: Fix stale docstring in `memory_manager.py`

**Problem:** Lines 1-2 say "at most ONE external plugin memory provider" — contradicts lines 8-10.
Lines 15-16 say "# Only ONE of these:" — false.

**File:** `agent/memory_manager.py:1-27`  ✅ LINE NUMBERS CONFIRMED

**Fix:** Replace lines 1-2:
```python
"""MemoryManager — orchestrates the built-in memory provider plus one or more
external plugin memory providers.
```

Replace lines 14-16 (usage example):
```python
    self._memory_manager.add_provider(BuiltinMemoryProvider(...))
    # One or more external providers:
    for provider_name in active_providers:
        self._memory_manager.add_provider(load_provider(provider_name))
```

---

### Task 6: Fix `_memory_provider()` in dump.py

**Problem:** Reads only `memory.provider` (legacy string). Multi-provider configs show "built-in" in dump.

**File:** `hermes_cli/dump.py:118-122`  ✅ LINE NUMBERS CONFIRMED

**Current code:**
```python
def _memory_provider(config: dict) -> str:
    """Return the active memory provider name."""
    mem = config.get("memory", {})
    provider = mem.get("provider", "")
    return provider if provider else "built-in"
```

**Fix:**
```python
def _memory_provider(config: dict) -> str:
    """Return the active memory provider name(s)."""
    mem = config.get("memory", {})
    providers = mem.get("providers", [])
    if providers:
        active = [p for p in providers if p]
        if active:
            return ", ".join(active)
    provider = mem.get("provider", "")
    return provider if provider else "built-in"
```

---

### Task 7: Fix honcho/cli.py to write to providers list

**Problem:** Line 545 writes only `hermes_config["memory"]["provider"] = "honcho"`.
If user has other providers in the list, they're preserved but honcho isn't added to list.

**File:** `plugins/memory/honcho/cli.py:541-550`  ✅ LINE NUMBERS CONFIRMED

**Fix:** Write to both fields:
```python
    # --- Auto-enable Honcho as memory provider in config.yaml ---
    try:
        from hermes_cli.config import load_config, save_config
        hermes_config = load_config()
        hermes_config.setdefault("memory", {})["provider"] = "honcho"
        # Also add to providers list if not present
        providers = hermes_config["memory"].get("providers", [])
        if "honcho" not in providers:
            providers.append("honcho")
            hermes_config["memory"]["providers"] = providers
        save_config(hermes_config)
        print("  Memory provider set to 'honcho' in config.yaml")
    except Exception as e:
        print(f"  Could not auto-enable in config.yaml: {e}")
        print("  Run: hermes config set memory.provider honcho")
```

---

### Task 8: Fix context_compressor.py singular reference

**Problem:** Line 861 says "the active memory provider" (singular). Should be plural.

**File:** `agent/context_compressor.py:861`  ✅ LINE NUMBERS CONFIRMED

**Fix:** Change "the active memory provider" to "active memory providers":
```
The following context was extracted by active memory providers before this compaction. Treat it as additional important information that MUST be preserved in the summary:
```

---

### Task 9: Fix test docstring in test_plugin_cli_registration.py

**Problem:** Line 117 says "memory.provider" but test actually mocks `get_active_memory_providers`.

**File:** `tests/hermes_cli/test_plugin_cli_registration.py:116-117`  ✅ LINE NUMBERS CONFIRMED

**Fix:**
```python
    def test_returns_nothing_when_no_active_provider(self, tmp_path, monkeypatch):
        """No commands when no memory providers are configured."""
```

---

### Task 10: Add direct unit test for `get_active_memory_providers()`

**Problem:** The actual config-reading logic has zero direct tests. All tests monkeypatch it.

**File:** CREATE `tests/plugins/memory/test_get_active_memory_providers.py`
(Corrected path: function lives in `plugins/memory/__init__.py`, so test goes in
`tests/plugins/memory/` which already exists with other memory plugin tests.)

**Tests needed:**
1. `memory.providers: ["honcho", "mem0"]` → `["honcho", "mem0"]`
2. `memory.providers: []` + `memory.provider: "honcho"` → `["honcho"]`
3. `memory.providers: ["honcho", ""]` → `["honcho"]` (falsy filter)
4. Both set → providers list wins
5. Neither set → `[]`
6. Exception in load_config → `[]`

---

### Task 11: Add test for `_get_configured_providers()` and `_set_configured_providers()`

**Problem:** Backbone CLI helpers have zero tests.

**File:** CREATE `tests/hermes_cli/test_memory_setup.py`  ✅ CONFIRMED DOES NOT EXIST

**Tests needed:**
1. `_get_configured_providers({"memory": {"providers": ["a", "b"]}})` → `["a", "b"]`
2. `_get_configured_providers({"memory": {"provider": "a"}})` → `["a"]` (legacy fallback)
3. `_get_configured_providers({})` → `[]`
4. `_set_configured_providers(config, ["a", "b"])` → config has both `providers: ["a", "b"]` and `provider: "a"` (legacy compat)

---

### Task 12: Add test for `discover_plugin_cli_commands()` with MULTIPLE active providers

**Problem:** All 4 tests use single provider. The iteration loop is never exercised with 2+.

**File:** `tests/hermes_cli/test_plugin_cli_registration.py`  ✅ EXISTS

**Test needed:**
```python
def test_discovers_multiple_active_plugins(self, tmp_path, monkeypatch):
    """Both active providers' CLI commands are discovered."""
    # Create two plugins with cli.py
    # Monkeypatch get_active_memory_providers to return ["plugin1", "plugin2"]
    # Verify both are in results
```

---

## ADDITIONAL TASKS (found during verification, missing from original plan)

### Task 13: Fix test_doctor.py to test providers list format

**Problem:** `TestDoctorMemoryProviderSection` (line 167-232) only uses legacy
`memory.provider` string in test configs (line 175). Should also test the
new `memory.providers` list format to match doctor.py which handles both.

**File:** `tests/hermes_cli/test_doctor.py:167-232`  ✅ LINE NUMBERS CONFIRMED

**Changes needed:**
1. Update class docstring line 168: "should respect memory.provider config"
   → "should respect memory.providers config"
2. Update `_make_hermes_home` to accept providers list parameter
3. Add test: `providers: ["honcho", "mem0"]` config → both checked by doctor
4. Add test: `providers: ["honcho"]` config (single item list) → works like legacy
5. Existing legacy-format tests should still pass (doctor.py handles fallback)

---

### Task 14: web_server.py config schema (DEFERRED — acknowledged)

**File:** `hermes_cli/web_server.py:292-296`
```python
"memory.provider": {
    "type": "select",
    "description": "Memory provider plugin",
    "options": ["builtin", "honcho"],
},
```

**Status:** Hardcoded single-select. Does not support `memory.providers` list.
Plan Task 13 already marks this as NICE-TO-HAVE / defer. Confirmed OK to defer.

---

## NICE-TO-HAVE (4 items)

### Task 15: PLAN.md / MASTER-PLAN.md in PR
**Decision:** Remove from branch before PR to keep diff clean.

### Task 16: Untracked files cleanup
**Files:** audit-report.md, final-verification.md, .hermes/plans/
**Decision:** Add to .gitignore or clean up.

### Task 17: Stale test docstring in test_memory_provider_init.py
**File:** `tests/run_agent/test_memory_provider_init.py:8`  ✅ CONFIRMED
Line 8: "Blank memory.provider should remain opt-out even if Honcho fallback looks configured."
**Fix:** Low priority — the test is correctly testing backward compatibility with
legacy format. Docstring is technically accurate for what it tests. Could add
"(legacy format)" for clarity but not required.

### Task 18: memory_context parameter naming
**Files:** `agent/context_engine.py:82`, `agent/context_compressor.py:711`
**Decision:** Defer — renaming is disruptive, current name is acceptable.

---

## REMOVED / VERIFIED NOT NEEDED

(none — all original tasks are still valid)

---

## Execution Order

1. Tasks 1-2 (CRITICAL — plugins_cmd.py)
2. Tasks 3-8 (SHOULD-FIX — dead code + stale docs + other files)
3. Tasks 9-13 (SHOULD-FIX — tests, including new test_doctor.py task)
4. Run full test suite
5. Verify with `hermes memory status` and `hermes doctor`
6. Final verifier pass — must return only compliments
7. Commit

---

## Verification Summary

| Task | Lines Verified | Code Correct | Callers Checked | Notes |
|------|---------------|-------------|----------------|-------|
| 1 | 761-768 ✅ | ✅ | Line 805/817 — radio picker won't pre-select with comma-joined | Acknowledged |
| 2 | 781-788 ✅ | ✅ | — | — |
| 3 | 308-320 ✅ | N/A (delete) | 0 callers ✅ | — |
| 4 | 3-5, 14 ✅ | ✅ | — | Lines 9-10 already updated, lines 3-5 contradict |
| 5 | 1-2, 15-16 ✅ | ✅ | — | Lines 8-10 already updated, lines 1-2 contradict |
| 6 | 118-122 ✅ | ✅ | — | — |
| 7 | 541-550 ✅ | ✅ | — | — |
| 8 | 861 ✅ | ✅ | — | — |
| 9 | 116-117 ✅ | ✅ | — | — |
| 10 | N/A (new) | ✅ | — | Path corrected: tests/plugins/memory/ |
| 11 | N/A (new) | ✅ | — | — |
| 12 | Exists ✅ | ✅ | — | — |
| 13 | 167-232 ✅ | ✅ | — | NEW: test_doctor.py providers format |
| 14 | 292-296 ✅ | N/A (defer) | — | — |
