# Multi-Provider Memory — PR Readiness Report

**Date:** 2026-04-28
**Branch:** `feat/multi-provider-memory`
**Fork:** https://github.com/someaka/hermes-agent

---

## 1. What Was Done

### 1.1 Core Feature: Multi-Provider Memory (11 commits, committed)

Support for loading multiple memory providers simultaneously. Results merged
across all active providers at runtime.

| Commit | Type | Description |
|--------|------|-------------|
| `b903ab87` | docs | Multi-provider memory implementation plans |
| `fbac4844` | fix | Capture on_pre_compress return value (#7192) |
| `b9dd060f` | feat | `get_active_memory_providers()` for multi-provider loading |
| `3bb52f72` | feat | `memory.providers` list key in config |
| `926b460f` | feat | Load all configured providers in agent init |
| `cd27b3ce` | feat | Remove single-external-provider guard |
| `823e82c3` | feat | `remove_provider()` for runtime deregistration |
| `f8e2d091` | feat | Tool budget warning and namespace validation |
| `f525cba1` | feat | Toolset filtering for memory provider tools |
| `e44cb3cb` | refactor | Rename holographic tools with provider prefix + aliases |
| `e81e32f3` | test | 6 new multi-provider memory tests |

**Tests:** 68 passing (was 62, +6 new)

**Files modified (committed):**
- `hermes_cli/config.py` — `memory.providers` list key
- `plugins/memory/__init__.py` — `get_active_memory_providers()`
- `run_agent.py` — multi-provider init loop + on_pre_compress fix
- `agent/memory_manager.py` — guard removed + remove_provider + budget + namespace + toolset
- `agent/context_compressor.py` — memory_context parameter
- `agent/context_engine.py` — abstract interface update
- `plugins/memory/holographic/__init__.py` — renamed tools with aliases
- `tests/agent/test_memory_provider.py` — 6 new tests

### 1.2 CLI Multi-Provider Support (uncommitted)

| File | Changes |
|------|---------|
| `hermes_cli/memory_setup.py` | Setup wizard writes to `memory.providers` list, "add another?" loop, status shows all providers, `_get_configured_providers()` / `_set_configured_providers()` helpers |
| `hermes_cli/main.py` | Parser description updated, `memory off` clears both `provider` and `providers` |
| `plugins/memory/__init__.py` | `get_active_memory_providers()` bug fix (`get_config` → `load_config`), `discover_plugin_cli_commands()` iterates all active providers |

### 1.3 Environment Fix (uncommitted)

- Installed `numpy` in `.venv` — required by mnemosyne's `BeamMemory`. Without
  this, `is_available()` returns False and mnemosyne silently doesn't load.

### 1.4 Config Applied

`~/.hermes/config.yaml` now has:
```yaml
memory:
  provider: mnemosyne        # legacy single-string (backwards compat)
  providers:                 # new list format (takes precedence)
    - mnemosyne
    - holographic
```

---

## 2. Verified Working

Both providers load and expose their tools:

```
mnemosyne:   available=True, 7 tools
  mnemosyne_remember, mnemosyne_recall, mnemosyne_sleep,
  mnemosyne_stats, mnemosyne_invalidate,
  mnemosyne_triple_add, mnemosyne_triple_query

holographic: available=True, 2 tools
  holographic_store, holographic_feedback
```

---

## 3. Polish Required for PR

### 3.1 Stale "only one provider" references (5 files)

These docstrings/comments still say "only one provider at a time":

| File | Line | What to fix |
|------|------|-------------|
| `plugins/memory/__init__.py` | 12 | Module docstring: "Only ONE provider can be active at a time" |
| `agent/memory_manager.py` | 8 | Module docstring: "Only ONE external (non-builtin) provider is allowed at a time" |
| `agent/memory_provider.py` | 9 | Module docstring: "Only one external provider runs at a time" |
| `hermes_cli/config.py` | 829 | Comment: "Only ONE external provider is allowed at a time" |

### 3.2 `hermes doctor` only checks single provider

`hermes_cli/doctor.py` lines 1131-1202 read `memory.provider` (single string)
and only check one provider. Should iterate `memory.providers` list and check
each active provider.

### 3.3 Test file uses old function

`tests/hermes_cli/test_plugin_cli_registration.py` still monkeypatches
`_get_active_memory_provider` (the old single-provider function) at lines
102, 128, 146, 164. These tests need updating to use
`get_active_memory_providers` or the tests need to verify both old and new
code paths work.

### 3.4 Unused helper

`_curses_multiselect()` was added to `memory_setup.py` but `cmd_setup()`
uses `_curses_select()` (single-select) with an "add another?" loop instead.
The helper is ready for future use but is currently dead code. Either:
- Remove it (clean PR)
- Keep it (it's 13 lines, useful if someone wants multi-select later)

### 3.5 Commit the uncommitted changes

The 3 modified files + numpy install need to be committed as one or more
conventional commits before the PR. Suggested structure:

```
fix(plugins): use load_config in get_active_memory_providers
feat(cli): multi-provider memory setup wizard
fix(env): add numpy dependency for mnemosyne BeamMemory
```

### 3.6 Test coverage gap

The 6 existing multi-provider tests verify the agent-side loading. No tests
exist for:
- `cmd_setup()` writing to `memory.providers` list
- `cmd_status()` reading from providers list
- `_get_configured_providers()` / `_set_configured_providers()` helpers
- `discover_plugin_cli_commands()` iterating multiple providers
- `memory off` clearing both config keys

---

## 4. Dogfood Checklist

After restart, verify:

- [ ] Both providers show in `hermes memory status`
- [ ] Both tool sets available in agent session
- [ ] `holographic_store` add/search/probe works
- [ ] `mnemosyne_remember` / `mnemosyne_recall` works
- [ ] `hermes memory setup` → pick one → "Add another?" loop works
- [ ] `hermes memory off` clears both config keys
- [ ] `hermes doctor` reports on memory provider(s)
- [ ] Agent logs show both providers loaded
- [ ] Tool budget not exceeded with 9 memory tools active
- [ ] No regressions in existing memory behavior

---

## 5. PR Preparation

When dogfood is satisfied:

```bash
cd /home/d/Desktop/agenda/hermes-agent

# Stage and commit polish changes
git add hermes_cli/memory_setup.py hermes_cli/main.py plugins/memory/__init__.py
git commit -m "fix(plugins): use load_config, multi-provider CLI support"

# Run full test suite
.venv/bin/python -m pytest tests/agent/test_memory_provider.py -v

# Push
git push origin feat/multi-provider-memory

# Create PR
gh pr create \
  --title "feat(agent): support multiple simultaneous memory providers" \
  --body "..."
```
