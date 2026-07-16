# BETA-007 — Strategic and technical memory scopes

Beta uses `ScopedMemory` as a policy adapter over the existing `MemoryManager`; it creates no database or provider. Strategic memory contains Chief preferences, goals, decisions, priorities, operating rules, and team structure. Technical memory is tagged to one specialist and contains domain facts, environment details, recurring solutions, and tool limits.

Beta may read strategic memory or explicitly query a specialist scope. A specialist may read only its own technical scope and may write it only when its manifest grants `read_write`; specialists cannot write strategic memory.

When Hindsight is active, retain and recall use its existing tools with `beta:strategic` or `beta:specialist:<id>` tags. Recall supports per-call tags with `all_strict`, preventing cross-scope retrieval. Other MemoryManager providers receive the same scope as write metadata/query context.

## Validation

```bash
python -m pytest -q tests/agent/beta/test_memory_scope.py
```
