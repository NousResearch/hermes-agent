# Built-In Memory Walling Design

Date: 2026-05-30

## Status

Design-only. No production memory files are migrated, rewritten, pruned, or
reattached by this proposal. Existing `profile_global` behavior remains the
default until a separate implementation batch explicitly enables another mode.

## Current Behavior

Built-in memory uses `tools.memory_tool.MemoryStore` and resolves storage through
`get_hermes_home() / "memories"`.

- Agent memory is stored in `MEMORY.md`.
- User profile memory is stored in `USER.md`.
- Both files are profile-scoped, not session-scoped, chat-scoped, or
  Discord-thread-scoped.
- At agent startup, `MemoryStore.load_from_disk()` loads both files and freezes a
  prompt snapshot.
- Mid-session built-in memory tool writes update the same profile files on disk,
  but do not update the already-frozen prompt snapshot.

This means two intentionally separate Discord project threads using the same
Hermes profile receive the same built-in memory snapshot.

## Why Walling Is Needed

Discord project threads can represent separate client projects, editorial
pipelines, or operational domains. Profile-global built-in memory can leak
decisions across those project walls, for example:

- Project A's approval policy influencing Project B.
- A deferred integration decision in one thread becoming assumed in another.
- User/project-specific constraints appearing authoritative outside their
  intended thread.

External memory providers can avoid this by honoring `gateway_session_key`, but
built-in `MEMORY.md` / `USER.md` do not yet have a namespace model.

## Proposed Modes

### `profile_global`

Existing behavior. Memory paths stay:

- `<HERMES_HOME>/memories/MEMORY.md`
- `<HERMES_HOME>/memories/USER.md`

This must remain the default for backward compatibility.

### `discord_thread_scoped`

For Discord thread sources with a stable `gateway_session_key`, built-in memory
uses a thread namespace derived from that key. Profile-global memory is not
injected unless explicitly allowed.

Suggested path shape:

- `<HERMES_HOME>/memories/scopes/discord_thread/<safe_scope_id>/MEMORY.md`
- `<HERMES_HOME>/memories/scopes/discord_thread/<safe_scope_id>/USER.md`

`safe_scope_id` should be a deterministic filesystem-safe digest of
`gateway_session_key`, not raw Discord identifiers in arbitrary path segments.

### `project_namespace_scoped`

For platforms or workflows that define an explicit project namespace, built-in
memory uses that namespace instead of a platform-specific thread key.

Suggested path shape:

- `<HERMES_HOME>/memories/scopes/project/<safe_project_id>/MEMORY.md`
- `<HERMES_HOME>/memories/scopes/project/<safe_project_id>/USER.md`

This mode should require an explicit namespace from config or routing metadata.
It should not infer a project from display names.

### `walled_project`

Strict project wall. Built-in memory is excluded unless it is in the active
project namespace. Profile-global `MEMORY.md` / `USER.md` are not injected by
default.

This mode is appropriate for high-isolation project threads. It should be
opt-in per gateway session policy or per project namespace.

## Configuration Shape

Proposed config, defaulting to current behavior:

```yaml
memory:
  builtin_scope:
    mode: profile_global
    include_profile_global_in_walled_projects: false
    diagnostics: true
```

Optional future per-platform override:

```yaml
gateway:
  platforms:
    discord:
      builtin_memory_scope:
        thread_mode: profile_global
```

Do not read these as final names. They are placeholders for implementation
review; the important contract is default-off walling and no implicit migration.

## Non-Migration Default

Changing modes must not move or rewrite existing profile-global memory.

When a scoped mode is enabled:

- Empty scoped memory files may be created only when the memory tool writes.
- Existing profile-global files remain where they are.
- Profile-global files are not copied into scoped memory automatically.
- Operators can intentionally seed scoped memory through an explicit command or
  documented manual process in a separate migration design.

This avoids orphaning existing memory while preventing silent cross-project
injection.

## Operator Diagnostics

When a Discord thread enters a scoped or walled mode, log metadata-only
diagnostics:

- selected built-in memory mode
- platform
- chat_type
- chat_id or thread_id
- gateway_session_key hash, not raw sensitive content
- whether profile-global memory was included or excluded
- scoped memory file existence and entry counts

Never log memory entry contents.

If `profile_global` memory exists but is excluded by walling, the diagnostic
should say that memory exists outside the active wall without printing it.

## Path Safety

Scoped path resolution should:

- use `get_hermes_home()` as the root
- derive scope directory names from a stable digest
- never accept raw user-controlled path segments
- never write outside `<HERMES_HOME>/memories`
- keep `MEMORY.md` and `USER.md` filenames unchanged inside the selected scope
- preserve existing file locking and atomic replace behavior

## Rollout Plan

1. Add a pure `BuiltinMemoryScope` resolver with tests for each mode.
2. Add `MemoryStore(memory_scope=...)` or equivalent dependency injection while
   preserving the current constructor default.
3. Wire agent startup to pass scope metadata only when config opts in.
4. Add metadata-only diagnostics for scoped/walled sessions.
5. Add gateway-focused tests for Discord thread sessions using temp homes.
6. Document operator commands for inspecting scoped memory entry counts without
   printing contents.
7. Consider an explicit, separate seeding/migration command after runtime walling
   has soaked.

## Rollback Plan

Rollback is config-only if no migration is performed:

1. Set mode back to `profile_global`.
2. Restart the affected non-production gateway process during a maintenance
   window.
3. Leave scoped files in place for later inspection.
4. Do not merge scoped files back into profile-global memory automatically.

Because scoped files are separate and profile-global files are untouched, rollback
does not require state repair.

## Test Plan

Current behavior coverage:

- Same profile plus different Discord thread sources load the same built-in
  memory snapshot.
- Built-in memory writes go to profile-global files by default.

Future behavior coverage:

- `profile_global` preserves existing paths and prompt snapshot behavior.
- `discord_thread_scoped` derives distinct paths from distinct
  `gateway_session_key` values.
- `project_namespace_scoped` derives distinct paths from explicit namespaces.
- `walled_project` excludes profile-global memory unless explicitly allowed.
- Scoped writes create only scoped files in temp homes.
- Diagnostics include only metadata and counts, never memory contents.
- Invalid or missing scope metadata falls back according to explicit config,
  without inventing a project namespace.

Current spec tests:

- `tests/agent/test_memory_walling_scope.py`
  - documents current profile-global behavior
  - includes strict xfail tests for default-off future walling behavior

## Remaining Gaps

- No `BuiltinMemoryScope` resolver exists yet.
- No built-in memory config exists for scoped/walled modes.
- No operator command exists for scoped memory counts.
- No migration or seeding command is proposed in this batch.
