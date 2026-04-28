# Multi-Provider Memory Support for Hermes Agent

**Date:** 2026-04-28
**Status:** Ready for review
**Estimated Effort:** ~250 lines across 7 files, 3-5 days focused work
**Constraint:** NO CAP on provider count — all 8 must work simultaneously for benchmarking

---

## Executive Summary

Hermes Agent's memory system currently allows only ONE external memory provider at a time, enforced by a single boolean guard in `MemoryManager.add_provider()`. The iteration plumbing (prefetch_all, sync_all, build_system_prompt, lifecycle hooks) was DESIGNED for multi-provider since Issue #3943 and already scales to N providers. The blocker is a one-line guard removal plus ~100 lines of defensive code.

This plan covers three workstreams:
1. **Core changes** — remove the guard, update config, wire the plugin loader
2. **Gateway fixes** — 6 pre-existing bugs that become critical with multiple providers
3. **Tool system** — namespace enforcement, deregistration, dual routing fix

## Current Architecture

```
┌─────────────────────────────────────────────────────┐
│                  SYSTEM PROMPT                       │
│  ┌───────────────────────────────────────────────┐  │
│  │  Built-in Memory (always active)              │  │
│  │  MEMORY.md (2200 chars) + USER.md (1375 chars)│  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  External Provider (0 or 1 active)            │  │
│  │  Selected via memory.provider config key      │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  Session Search (always available)             │  │
│  │  FTS5 over all past sessions in state.db       │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Goal state:** Multiple external providers active simultaneously.

## Available Providers (8 total, 36 tools)

| Provider | Tools | Prefix | Storage |
|----------|-------|--------|---------|
| Honcho | 5 | `honcho_*` | Cloud/self-hosted |
| Mem0 | 3 | `mem0_*` | Cloud |
| Hindsight | 3 | `hindsight_*` | Local/cloud |
| ByteRover | 3 | `brv_*` | Local/cloud |
| Holographic | 2 | `fact_*` ⚠️ | Local SQLite |
| OpenViking | 5 | `viking_*` | Self-hosted |
| RetainDB | 10 | `retaindb_*` | Cloud |
| Supermemory | 4 | `supermemory_*` | Cloud |

⚠️ Holographic violates the naming convention (`fact_*` instead of `holographic_*`).

All 36 tool names are UNIQUE — zero literal collisions across providers.

---

## Workstream 1: Core Changes

**Files:** `hermes_cli/config.py`, `plugins/memory/__init__.py`, `agent/memory_manager.py`, `run_agent.py`
**Lines:** ~100-115

### 1.1 — Add `providers` config key

**File:** `hermes_cli/config.py`

```python
# In the memory config defaults:
"memory": {
    "memory_enabled": True,
    "user_profile_enabled": True,
    "memory_char_limit": 2200,
    "user_char_limit": 1375,
    "provider": "",          # Legacy: single provider (still supported)
    "providers": [],         # NEW: list of providers (takes precedence)
}
```

Backward compatible: `provider: "honcho"` still works. `providers: ["honcho", "mem0"]` takes precedence when non-empty.

### 1.2 — Add `get_active_memory_providers()` to plugin loader

**File:** `plugins/memory/__init__.py`

```python
def get_active_memory_providers() -> List[str]:
    """Return list of active memory provider names from config.
    
    Supports both old format (memory.provider: "honcho") and
    new format (memory.providers: ["honcho", "mem0"]).
    """
    try:
        from hermes_cli.config import get_config
        config = get_config()
        memory_config = config.get("memory", {})
        
        # New list format takes precedence
        providers = memory_config.get("providers", [])
        if providers:
            return [p for p in providers if p]
        
        # Fall back to legacy single-string format
        single = memory_config.get("provider", "")
        return [single] if single else []
    except Exception:
        return []
```

Existing `load_memory_provider(name)` stays unchanged — the multi-provider loop happens in run_agent.py.

### 1.3 — Remove single-external-provider guard

**File:** `agent/memory_manager.py` — `add_provider()`

**BEFORE:**
```python
def add_provider(self, provider: MemoryProvider) -> None:
    """Only one external (non-builtin) provider is allowed — a second
    attempt is rejected with a warning."""
    is_builtin = provider.name == "builtin"
    if not is_builtin:
        if self._has_external:
            existing = next(
                (p.name for p in self._providers if p.name != "builtin"), "unknown"
            )
            logger.warning(
                "Rejected memory provider '%s' — external provider '%s' is "
                "already registered. Only one external memory provider is "
                "allowed at a time. Configure which one via memory.provider "
                "in config.yaml.",
                provider.name, existing,
            )
            return
        self._has_external = True
    # ... register provider and index its tools
```

**AFTER:**
```python
def add_provider(self, provider: MemoryProvider) -> None:
    """Register a memory provider. Multiple external providers are allowed."""
    is_builtin = provider.name == "builtin"
    if not is_builtin:
        # Check for duplicate provider name
        if any(p.name == provider.name for p in self._providers if p.name != "builtin"):
            logger.warning(
                "Memory provider '%s' is already registered; skipping duplicate.",
                provider.name,
            )
            return

    # Collect schemas BEFORE mutating state (fixes #9948)
    try:
        schemas = provider.get_tool_schemas()
    except Exception as exc:
        logger.error(
            "Memory provider '%s' failed during schema loading: %s — NOT registered",
            provider.name, exc,
        )
        return

    self._providers.append(provider)

    # Index tool names → provider
    for schema in schemas:
        tool_name = schema["name"]
        if tool_name in self._tool_to_provider:
            logger.warning(
                "Tool name '%s' already registered by '%s'; "
                "ignoring duplicate from '%s'",
                tool_name, self._tool_to_provider[tool_name].name, provider.name,
            )
            continue
        self._tool_to_provider[tool_name] = provider

    ext_count = sum(1 for p in self._providers if p.name != "builtin")
    total_tools = len(self._tool_to_provider)
    logger.info(
        "Memory provider '%s' registered (%d tools). "
        "Active: %d external provider(s), %d total tool(s).",
        provider.name, len(schemas), ext_count, total_tools,
    )
    if total_tools > 30:
        logger.warning(
            "High tool count (%d) may degrade model tool-calling accuracy.",
            total_tools,
        )
```

### 1.4 — Remove `_has_external` boolean

**File:** `agent/memory_manager.py` — `__init__()`

```python
# BEFORE:
def __init__(self) -> None:
    self._providers: List[MemoryProvider] = []
    self._tool_to_provider: Dict[str, MemoryProvider] = {}
    self._has_external: bool = False

# AFTER:
def __init__(self) -> None:
    self._providers: List[MemoryProvider] = []
    self._tool_to_provider: Dict[str, MemoryProvider] = {}
    # _has_external removed — multiple external providers now supported
```

`_has_external` is only used in `add_provider()` for the guard. After guard removal, it's dead code. No other file reads it.

### 1.5 — Add `remove_provider()` method

**File:** `agent/memory_manager.py`

```python
def remove_provider(self, name: str) -> bool:
    """Deregister a memory provider by name. Returns True if removed."""
    if name == "builtin":
        logger.warning("Cannot remove builtin memory provider")
        return False

    for i, p in enumerate(self._providers):
        if p.name == name:
            # Remove tool mappings
            tools_to_remove = [t for t, prov in self._tool_to_provider.items() if prov is p]
            for t in tools_to_remove:
                del self._tool_to_provider[t]

            # Shutdown the provider
            try:
                p.shutdown()
            except Exception as exc:
                logger.warning("Provider '%s' shutdown failed: %s", name, exc)

            self._providers.pop(i)
            logger.info("Memory provider '%s' removed (had %d tools)", name, len(tools_to_remove))
            return True

    logger.warning("Provider '%s' not found for removal", name)
    return False
```

### 1.6 — Loop over providers in run_agent.py

**File:** `run_agent.py` — memory wiring section

**BEFORE:**
```python
self._memory_manager = MemoryManager()
self._memory_manager.add_provider(BuiltinMemoryProvider(...))

provider_name = _get_active_memory_provider()
if provider_name:
    plugin_provider = load_memory_provider(provider_name)
    if plugin_provider and plugin_provider.is_available():
        self._memory_manager.add_provider(plugin_provider)
```

**AFTER:**
```python
self._memory_manager = MemoryManager()
self._memory_manager.add_provider(BuiltinMemoryProvider(...))

# Load all configured external providers
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
```

### 1.7 — Wire `on_turn_start()` hook (Bug #7193)

**File:** `run_agent.py` — after `self._user_turn_count += 1`

```python
self._user_turn_count += 1

# Notify memory providers of new turn
try:
    last_user_msg = ""
    if self._messages and self._messages[-1].get("role") == "user":
        content = self._messages[-1].get("content", "")
        last_user_msg = content if isinstance(content, str) else str(content)[:500]
    self._memory_manager.on_turn_start(self._user_turn_count, last_user_msg)
except Exception as exc:
    logger.debug("on_turn_start hook failed: %s", exc)
```

### 1.8 — Capture `on_pre_compress()` return value (Bug #7192)

**File:** `run_agent.py` — around line ~6081

```python
# BEFORE:
self._memory_manager.on_pre_compress(messages)

# AFTER:
memory_context = self._memory_manager.on_pre_compress(messages) or ""
```

Then pass `memory_context` to the compressor in `agent/context_compressor.py`:

```python
def compress(self, messages, current_tokens=None, memory_context: str = ""):
    summary = self._generate_summary(messages, memory_context=memory_context)
    # ...

def _generate_summary(self, messages, memory_context: str = ""):
    extra_context = ""
    if memory_context:
        extra_context = f"\n\nMemory provider context:\n{memory_context}\n"
    # Include in summary prompt
```

---

## Workstream 2: Gateway Lifecycle Fixes

**Files:** `gateway/run.py`, `run_agent.py`, `agent/memory_manager.py`, `agent/context_compressor.py`
**Lines:** ~150

Six pre-existing bugs that become critical with multiple providers:

| # | Issue | Severity | Multi-Provider Impact |
|---|-------|----------|----------------------|
| 7192 | on_pre_compress return discarded | MEDIUM | 8 insight sources all lost |
| 7193 | on_turn_start never wired | MEDIUM | 8 cadence logics all broken |
| 7358 | os.environ race conditions | HIGH | 8 sessions cross-contaminate |
| 16155 | Provider re-init per message | HIGH | 8-32s latency per message |
| 9973 | Prefetch cache lost | HIGH | 8 prefetch pipelines all broken |
| 11205 | on_session_end never called | CRITICAL | 8 providers lose final extraction |

### 2.1 — Fix #7358: os.environ → contextvars

**File:** `gateway/run.py`

Replace `run_in_executor` with `asyncio.to_thread` for contextvars propagation:

```python
# BEFORE:
loop.run_in_executor(None, run_sync)

# AFTER:
await asyncio.to_thread(run_sync)
```

Add `clear_session_vars()` in finally blocks:

```python
tokens = set_session_vars(platform=event.platform, chat_id=event.chat_id, ...)
try:
    # ... handle message ...
finally:
    clear_session_vars(tokens)
```

Remove `os.environ` fallbacks in `get_session_env()` for gateway contexts.

### 2.2 — Fix #16155: Pass memory_manager into AIAgent (Option B)

**File:** `gateway/run.py` + `run_agent.py`

```python
# In gateway/run.py, when building AIAgent:
agent = AIAgent(
    ...,
    memory_manager=session_memory_manager,  # reuse from session
)

# In run_agent.py AIAgent.__init__():
def __init__(self, ..., memory_manager: Optional[MemoryManager] = None):
    if memory_manager is not None:
        self._memory_manager = memory_manager
    else:
        self._memory_manager = MemoryManager()
        # ... existing provider registration
```

This transitively fixes Bug #9973 (prefetch cache lost) because the same MemoryManager survives agent recreation.

### 2.3 — Fix #11205: Call on_session_end in gateway

**File:** `gateway/run.py` — `_flush_memories_for_session()`

```python
# ADD THIS before the flush agent spawn block:
live_agent = self._cached_agents.get(session_key)
if live_agent and hasattr(live_agent, '_memory_manager') and live_agent._memory_manager:
    try:
        history = session_store.get_messages(session_key)
        live_agent._memory_manager.on_session_end(history)
    except Exception as exc:
        logger.warning("on_session_end hook failed for session %s: %s", session_key, exc)
```

---

## Workstream 3: Tool System Changes

**Files:** `agent/memory_manager.py`, `run_agent.py`, `plugins/memory/holographic/__init__.py`
**Lines:** ~50

### 3.1 — Fix dual routing (Bug #15118)

**File:** `run_agent.py` — `_execute_tool_calls_sequential`

```python
# BEFORE:
async def _execute_tool_calls_sequential(self, ...):
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = tool_call.function.arguments
        result = handle_function_call(name, args, ...)

# AFTER:
async def _execute_tool_calls_sequential(self, ...):
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = tool_call.function.arguments
        if self._memory_manager and self._memory_manager.has_tool(name):
            result = self._memory_manager.handle_tool_call(name, args)
        else:
            result = handle_function_call(name, args, ...)
```

### 3.2 — Holographic tool rename with aliases

**File:** `plugins/memory/holographic/__init__.py`

```python
# Rename in tool schemas:
# fact_store → holographic_store
# fact_feedback → holographic_feedback

# Add backward-compat aliases in handle_tool_call:
_TOOL_ALIASES = {
    "fact_store": "holographic_store",
    "fact_feedback": "holographic_feedback",
}

def handle_tool_call(self, tool_name, args, **kwargs):
    tool_name = self._TOOL_ALIASES.get(tool_name, tool_name)
    # ... existing dispatch logic
```

### 3.3 — Tool budget warning

**File:** `agent/memory_manager.py` — `get_all_tool_schemas()`

```python
# Add after schema collection:
TOOL_BUDGET_WARN_THRESHOLD = 20
total_memory_tools = len(self._tool_to_provider)
if total_memory_tools > TOOL_BUDGET_WARN_THRESHOLD:
    logger.warning(
        "Memory tool budget: %d tools registered (threshold: %d). "
        "May degrade tool-calling accuracy.",
        total_memory_tools, TOOL_BUDGET_WARN_THRESHOLD,
    )
```

### 3.4 — Namespace validation

**File:** `agent/memory_manager.py` — `add_provider()`

```python
# Add after tool indexing:
for schema in schemas:
    tool_name = schema.get("name", "")
    if provider.name != "builtin" and not tool_name.startswith(provider.name[:4]):
        logger.warning(
            "Provider '%s' tool '%s' does not follow naming convention "
            "'<provider>_<action>' (e.g., 'honcho_search').",
            provider.name, tool_name,
        )
```

---

## Config Migration

```yaml
# Old (still works):
memory:
  provider: "honcho"

# New (takes precedence when non-empty):
memory:
  providers: ["honcho", "mem0", "hindsight"]

# Enable all 8 for benchmarking:
memory:
  providers: ["honcho", "mem0", "hindsight", "byterover", "holographic", "openviking", "retaindb", "supermemory"]
```

## Implementation Order

```
1. Workstream 2, Tasks 2.1-2.3 (gateway fixes — must ship first)
2. Workstream 1, Tasks 1.1-1.2 (config + plugin loader)
3. Workstream 1, Tasks 1.3-1.5 (memory_manager changes)
4. Workstream 1, Tasks 1.6-1.8 (run_agent wiring + hook fixes)
5. Workstream 3, Tasks 3.1-3.4 (tool system polish)
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tool name collision → silent drop | LOW (current providers unique) | HIGH | First-wins with warning |
| Context overflow from combined prefetch | MEDIUM (8 × ~300 tokens) | HIGH | Add max_prefetch_chars budget |
| Tool schema bloat (66 tools) | CERTAIN | MEDIUM | Acceptable for benchmarking |
| Sequential latency (8 × init) | CERTAIN | MEDIUM | Acceptable for benchmarking |
| Gateway lifecycle bugs | CERTAIN without fixes | HIGH | Fix before shipping |
| Config migration confusion | LOW | LOW | Legacy format still works |

## What Does NOT Need Changes

- `agent/memory_provider.py` — ABC needs zero code changes (docstring update only)
- `tools/registry.py` — memory tools stay in MemoryManager, not registry
- `toolsets.py` — memory tools are dynamic, shouldn't be in toolset definitions
- All iteration methods (prefetch_all, sync_all, build_system_prompt, etc.) — already scale to N providers
