Title
feat(memory): add Supermemory memory provider

Summary
This PR adds Supermemory as a native Hermes memory provider.

It implements the upstream `MemoryProvider` interface rather than the general hook-based plugin system, so it works with the current memory-provider lifecycle and setup flow. The provider supports automatic recall, cleaned turn capture, session-end conversation ingest, and four explicit memory tools.

What is included
- `plugins/memory/supermemory/plugin.yaml`
- `plugins/memory/supermemory/README.md`
- `plugins/memory/supermemory/__init__.py`
- focused tests for provider behavior and failure modes

Behavior
When enabled, the provider can:
- prefetch relevant memory context before turns
- include profile facts on the first turn and on a configurable cadence
- store cleaned user-assistant turns after each completed response
- ingest the full session on session end via Supermemory conversations API
- expose explicit tools for memory store, search, forget, and profile access

Tools
- `supermemory_store`
- `supermemory_search`
- `supermemory_forget`
- `supermemory_profile`

Setup
Use the standard memory setup flow:

```bash
hermes memory setup
```

Select `supermemory`, then provide:
- `SUPERMEMORY_API_KEY` in `.env`
- optional non-secret config written to `$HERMES_HOME/supermemory.json`

Config surface
The provider currently supports:
- `container_tag`
- `auto_recall`
- `auto_capture`
- `max_recall_results`
- `profile_frequency`
- `capture_mode`
- `entity_context`
- `api_timeout`

Design notes
- Implemented as a native memory provider so it fits Hermes's pluggable memory system and `hermes memory setup` / `status` flow.
- The implementation preserves the useful behavior of a working local Supermemory integration, but removes local-only naming and packaging assumptions.
- Recall context is fenced and stripped before capture to avoid recursive memory pollution.
- `on_memory_write()` is intentionally conservative in v1 and mirrors `add` writes only. Supermemory is not a simple CRUD table, so pretending replace/remove are lossless would be dishonest.
- Session-end ingest is kept in the provider because Supermemory's conversation ingestion endpoint is a meaningful part of the backend's graph-building behavior.

Failure behavior
- `is_available()` performs no network calls
- missing API key or missing SDK leaves the provider unavailable without crashing Hermes
- recall, capture, and ingest failures degrade quietly and do not break the agent loop

Validation
Ran:

```bash
python3 -m pytest tests/plugins/memory/test_supermemory_provider.py tests/agent/test_memory_provider.py tests/agent/test_memory_plugin_e2e.py -q
```

Result:
- `76 passed`

Reviewer-facing summary
This PR adds Supermemory as a first-class Hermes memory provider in the same structural shape as the existing built-in providers. It is setup-compatible, failure-tolerant, and intentionally conservative where backend semantics differ from simple CRUD expectations.

The main reason for this PR is straightforward: Supermemory is a real external memory backend, and the right way to integrate it upstream is through the native `MemoryProvider` interface, not through a user-local hook plugin.

Points to review
- provider shape and config UX alignment with other memory providers
- recall formatting and profile cadence behavior
- whether the add-only `on_memory_write()` bridge is the right v1 scope
- test coverage for the provider lifecycle and failure paths
