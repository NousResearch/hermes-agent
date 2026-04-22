# hermes_memory

Bundled Hermes **curated persistent memory**, **provider orchestration**, and **optional memory backends** you can reuse in other projects.

## Layout

| Path | Role |
|------|------|
| `memory_provider.py` | Abstract `MemoryProvider` base class |
| `memory_manager.py` | `MemoryManager`, prefetch fencing (`build_memory_context_block`, `sanitize_context`) |
| `builtin_memory_tool.py` | `MemoryStore`, `MEMORY.md` / `USER.md`, `memory_tool`, `MEMORY_SCHEMA` |
| `plugins/memory/` | Bundled backends (Honcho, Mem0, Hindsight, …) — discovered via `load_memory_provider()` |

Tool **registration** stays in repo `tools/memory_tool.py` (Hermes integrates with `tools.registry`). For reuse elsewhere, call `memory_tool(...)` or copy the schema from `MEMORY_SCHEMA`.

## Imports

```python
from hermes_memory import MemoryManager, MemoryStore, MemoryProvider
from hermes_memory.plugins.memory import load_memory_provider, discover_memory_providers
```

Legacy shim: `from plugins.memory import load_memory_provider` still works in this repo (`plugins/memory/__init__.py` re-exports).

## Sessions vs curated memory

Long-running **conversation logs** are separate (Hermes uses `hermes_state.SessionDB`). This package is **curated facts + optional vector/service recall**, not full transcript storage.
