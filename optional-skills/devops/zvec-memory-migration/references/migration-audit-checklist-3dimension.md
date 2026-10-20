# Migration Audit Checklist

Use this checklist to evaluate any system replacement (LanceDB → Zvec, or any
other DB/service swap) across **three validation dimensions**.

## Dimension A: Official Docs Coverage

Check the NEW system's official documentation against what the skill or plugin
claims to support. For each major feature area:

| Area | Official docs cover? | Skill covers? | Gap? |
|------|--------------------|---------------|------|
| Core CRUD (create/read/update/delete) | | | |
| Query types (vector/scalar/hybrid) | | | |
| Index types (HNSW/FLAT/IVF) | | | |
| Rerankers / fusion strategies | | | |
| Embedding backend options | | | |
| Configuration / init | | | |
| Schema/DLL operations | | | |
| Troubleshooting / error catalog | | | |

**Pass criteria**: ≤1 small gap (e.g. an edge-case section missing from the skill's
reference docs). ≥2 gaps → don't declare the old system replaceable yet.

## Dimension B: Predecessor Feature Parity

Map every user-facing feature of the OLD system to the NEW system. Use a
feature-by-feature comparison table:

| Feature | Old system | New system | Compatible? |
|---------|-----------|------------|-------------|
| API / tool schema names | | | 1:1 or different? |
| Parameter names and defaults | | | |
| Return format / error format | | | JSON matching? |
| Async hooks (sync_turn, on_session_end, etc.) | | | |
| Thread safety model | | | |
| Config keys and defaults | | | |
| Health check / availability | | | |
| Data schema / storage format | | | |

**Key insight:** Even if the new system is "better" overall, a single parameter
name change in a user-facing tool schema can break the agent's ability to call
it, because tools are auto-generated from schemas and the LLM memorizes the
parameter names.

**Pass criteria**: All user-facing schemas identical. Internal differences (faster
search, native vs. two-pass hybrid) are acceptable as upgrades. Any schema
change → must update the tool description to cue the LLM.

## Dimension C: Migration Readiness

This is the **ground-truth validation run** — not a code review, an actual
execution:

1. **Data migration** — run the migration script end-to-end
   - [ ] Rows copied match exactly (count before == count after)
   - [ ] No embedding re-generation needed
   - [ ] Schema field mapping verified (all fields present in target)
   - [ ] Migration is idempotent (can re-run safely)

2. **Smoke test** — instantiate the new plugin and test every tool schema:
   - [ ] `vec_memory_add` — store 3 items with different session_ids
   - [ ] `vec_memory_search hybrid` — query with vector semantics
   - [ ] `vec_memory_search vector` — same query, vector-only mode
   - [ ] `vec_memory_search keyword` — same query, FTS-only mode
   - [ ] `vec_memory_list` — filter by session_id, by time range
   - [ ] `vec_memory_delete` — delete by ID
   - [ ] `vec_memory_stats` — verify count, sessions, backend label

3. **Path alignment** — the migration script output path **must match** what the
   plugin expects on startup. Common mismatches:
   - Migration creates at `/zvec_memory/`, plugin reads `/zvec_memory/memories/`
   - Migration uses absolute path, plugin uses `$HERMES_HOME` expansion
   - Collection names differ between schema and plugin config

   **Fix**: Set migration script's output path to `{zvec_dir}/{collection_name}/`

4. **Config alignment** — the `config.yaml` block must contain every key the
   plugin's `_load_plugin_config()` reads:
   - [ ] `memory.provider` set to the plugin's name
   - [ ] `plugins.<name>.zvec_dir` (or equivalent) matches migration output
   - [ ] `$HERMES_HOME` left unexpanded in config (plugin expands at runtime)

5. **Rollback path** — verify the old system is undisturbed:
   - [ ] Old data directory still intact
   - [ ] Setting `memory.provider` back to old name + restart restores old
   - [ ] Both directories can coexist

## When to Run This Audit

Run this before every production migration that involves:
- Changing the backing store of a live Hermes memory system
- Replacing a plugin that has active data
- Upgrading to a new SDK that changes parameter names or return types

**Don't** run it for:
- Adding a new empty plugin (no data to migrate)
- Documentation-only changes
- Features behind a config flag that default off
