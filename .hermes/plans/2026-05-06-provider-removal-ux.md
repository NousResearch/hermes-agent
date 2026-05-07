# Plan: Provider Removal UX

**Branch:** `feat/multi-provider-memory`  
**Date:** 2026-05-06  
**Working tree baseline:** clean except pyproject.toml + uv.lock

## Problem

The multi-provider memory PR lets users add providers to `memory.providers` list, but there is no way to remove a single provider. Only `hermes memory off` exists (nukes all). The interactive "Add another?" prompt is unnecessary friction. The `_save_memory_provider()` function has a bug where it always overwrites the legacy `memory.provider` key to the last-added name instead of the first.

## Elements

### E1: `_remove_memory_provider()` in `plugins_cmd.py`
**Scope:** `hermes_cli/plugins_cmd.py` — insert after `_save_memory_provider()` (line ~825)  
**Change:** New function that pops a name from `memory.providers`, updates legacy `memory.provider` to `existing[0]` or `""`. Returns bool.  
**Verification gate:** Unit test calling the function on a mock config dict.  
**Test file:** `tests/hermes_cli/test_plugins_cmd_providers.py` (new)

### E2: Fix `_save_memory_provider()` legacy key bug
**Scope:** `hermes_cli/plugins_cmd.py` line 823  
**Change:** `config["memory"]["provider"] = name` → `config["memory"]["provider"] = existing[0] if existing else ""`  
**Verification gate:** Test that adding providers [a, b] leaves legacy key as "a", not "b".  
**Test file:** same as E1

### E3: Remove "Add another?" prompt from `memory_setup.py`
**Scope:** `hermes_cli/memory_setup.py` — two locations:
  - Lines 353-360 in `cmd_setup_provider()`: delete the "Add another?" prompt + recursive call
  - Lines 462-470 in `cmd_setup()`: delete the prompt + recursive call  
**Change:** After saving config, just print status and return.  
**Verification gate:** Test that `cmd_setup` returns after one provider without prompting.  
**Test file:** `tests/hermes_cli/test_memory_setup.py` (update existing)

### E4: `hermes memory remove <provider>` subcommand
**Scope:** `hermes_cli/main.py` lines ~9778-9813  
**Change:** Add `remove` subparser with a required `provider` positional arg. In `cmd_memory()`, handle `sub == "remove"`: import `_remove_memory_provider`, call it, print result or error.  
**Verification gate:** `hermes memory remove fakeprovider` prints "not found". After adding a provider to config, `hermes memory remove <name>` removes it.  
**Test file:** `tests/hermes_cli/test_memory_setup.py` (new tests)

### E5: Web API — add `memory_providers` list field to PUT endpoint
**Scope:** `hermes_cli/web_server.py` lines 3883-3901  
**Change:** Add `memory_providers: Optional[List[str]] = None` to `_PluginProvidersPutBody`. In handler, if `memory_providers is not None`, call `_set_configured_providers()` directly (full list replacement). Also import `_remove_memory_provider` for a `DELETE` endpoint. Add `DELETE /api/dashboard/plugin-providers/{name}` endpoint.  
**Verification gate:** Test the endpoint with a mock config.  
**Test file:** `tests/hermes_cli/test_web_server_providers.py` (new)

### E6: Curses checklist in `_configure_memory_provider()`
**Scope:** `hermes_cli/plugins_cmd.py` lines 837-872  
**Change:** Replace single-select radio with multi-select checklist. Each provider gets a checkbox (checked = active). "built-in only" at the bottom unchecks all. ENTER confirms. Uses `curses_checklist` if available, otherwise falls back to numbered selection.  
**Verification gate:** Test that toggling a provider off removes it from the list, toggling on adds it.  
**Test file:** `tests/hermes_cli/test_plugins_cmd_providers.py` (new tests)

## Dependency graph

```
E1 (remove function) ──→ E4 (CLI subcommand) ──→ E3 (remove "add another?" prompt)
                  └──→ E5 (web API)
                  └──→ E6 (curses checklist)

E2 (legacy key fix) is independent of all others
```

- E1 must come first (E4, E5, E6 depend on it)
- E2 is independent
- E3 is independent but can share a test file with E4
- E4, E5, E6 are parallel after E1 is done

## Execution plan

**Batch 1 (parallel):** E1 + E2 + E3 — touch different concerns, minimal file overlap  
**Batch 2 (parallel):** E4 + E5 + E6 — all depend on E1 being done  
**Final audit:** read all modified files, run tests, verify no stale "add another?" prompts remain
