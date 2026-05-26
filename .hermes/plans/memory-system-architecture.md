# Hermes memory system architecture recon + plan

Date: 2026-05-22
Worktree: `/home/oscar/workspace/hermes-agent-memory`
Branch: `feat/personal-memory-architecture`
Scope: read-only recon + planning. No code changes made.

## Current implementation map

### Built-in always-on memory

- Implementation: `tools/memory_tool.py`
- Storage:
  - `$HERMES_HOME/MEMORY.md`
  - `$HERMES_HOME/USER.md`
- Entry delimiter: `\n§\n`
- Defaults from `hermes_cli/config.py`:
  - `memory.memory_enabled: true`
  - `memory.user_profile_enabled: true`
  - `memory.memory_char_limit: 2200`
  - `memory.user_char_limit: 1375`
- Writes:
  - `add`, `replace`, `remove`
  - substring replace/remove, with ambiguous-match protection
  - file locking + atomic temp-file rename
  - exact duplicate rejection
  - prompt-injection / exfil / invisible Unicode scan before add/replace
- Prompt behavior:
  - `MemoryStore.load_from_disk()` captures a frozen system-prompt snapshot
  - mid-session writes persist to disk but do not affect the active prompt until prompt rebuild/session restart/compression invalidation
  - `agent/system_prompt.py` injects memory/user blocks into the volatile prompt tier

### External memory provider layer

- Interface: `agent/memory_provider.py`
- Orchestrator: `agent/memory_manager.py`
- Discovery: `plugins/memory/__init__.py`
- Config selection: `memory.provider` in config; empty means built-in only
- Current local status: built-in only; no external provider enabled
- Installed provider plugins:
  - `byterover`
  - `hindsight`
  - `holographic`
  - `honcho`
  - `mem0`
  - `openviking`
  - `retaindb`
  - `supermemory`
- Constraint: only one external provider can be active at a time; built-in memory is additive and always separate
- Provider hooks available:
  - `system_prompt_block()`
  - `prefetch(query)`
  - `queue_prefetch(query)`
  - `sync_turn(user, assistant)`
  - `on_turn_start(turn_number, message, **kwargs)`
  - `on_session_end(messages)`
  - `on_session_switch(new_session_id, ...)`
  - `on_pre_compress(messages)`
  - `on_memory_write(action, target, content, metadata=None)`
  - `on_delegation(task, result, ...)`
  - provider-specific tools via `get_tool_schemas()` / `handle_tool_call()`

### Turn lifecycle

- Startup: `agent/agent_init.py`
  - initializes built-in `MemoryStore` unless `skip_memory`
  - loads selected external provider from `memory.provider`
  - passes platform/profile/session/chat identity into provider initialization
- Before model call: `agent/conversation_loop.py`
  - calls `memory_manager.on_turn_start(...)`
  - calls `memory_manager.prefetch_all(original_user_message)` once per turn
  - injects returned external memory context only into the current user API message
- Injection fence:
  - `build_memory_context_block()` wraps external prefetch in `<memory-context>`
  - system note says recalled memory is not new user input but is authoritative reference data
  - `StreamingContextScrubber` prevents memory-context leakage into streamed UI deltas
- After completed turn: `run_agent.py::_sync_external_memory_for_turn`
  - skips interrupted turns
  - calls `sync_all(original_user_message, final_response)`
  - calls `queue_prefetch_all(original_user_message)`
- Session/compression boundaries:
  - `commit_memory_session()` calls provider `on_session_end()` without shutting providers down
  - `shutdown_memory_provider()` calls `on_session_end()` then provider shutdown
  - context compression calls `on_pre_compress()` before dropping old context

### Built-in-to-external bridge

- Built-in memory tool writes are mirrored to external providers for `add` and `replace` only.
- Bridge locations:
  - `agent/tool_executor.py`
  - `agent/agent_runtime_helpers.py`
- Metadata includes execution provenance such as task/tool/session/platform context.
- `remove` is intentionally not bridged by current tests (`test_on_memory_write_remove_not_bridged`).

## Constraints / design pressure

1. Prompt budget is intentionally tiny.
   - `MEMORY.md`: 2200 chars
   - `USER.md`: 1375 chars
   - This forces kernel-style durable facts, not diary logs.

2. Prompt caching matters.
   - Built-in memory is snapshotted and system prompt is cached.
   - Mid-session mutable memory injection would hurt cache stability unless isolated outside the cached prompt.

3. External memory context already has the right place to be dynamic.
   - Per-turn recalled context is injected into the current user API message only.
   - It is fenced/scrubbed and does not mutate session history.

4. External providers are best-effort.
   - Exceptions are swallowed or logged.
   - A broken memory backend must not block responses.

5. One external provider only.
   - This avoids tool schema bloat and conflicting memory semantics.
   - Any architecture that wants multiple stores needs either a meta-provider or a new memory-manager policy.

6. Current docs still contain at least one stale/undesirable recommendation.
   - `website/docs/user-guide/features/memory.md` says completed work can be saved to memory.
   - Current system/user preference says stale task logs do not belong in core memory.

7. Current active profile is already nearly full.
   - Live prompt shows memory/user stores at ~98% / ~96%.
   - This validates the need for better routing, compaction, and non-prompt storage.

## Main architecture recommendation

Build a layered personal memory architecture that **keeps `USER.md` and `MEMORY.md` small** while adding real capacity through operational references, deeper memories, and selectively loaded notes. The problem is not merely bad routing; the current files are small, easy to overfill with low-value crap, and already full before the assistant/user relationship has much history. The replacement target is: core kernels stay tiny, while larger memory/preference growth lands in an Obsidian/knowledge-db-backed operational memory layer.

### Layer 0: Core prompt kernel

Keep current `MEMORY.md` / `USER.md` as the smallest always-on kernel.

Purpose:
- durable user preferences
- durable environment facts
- durable routing rules
- stable conventions that should influence every turn

Non-goals:
- task progress
- PR/issue/commit logs
- per-project scratch state
- contacts/entities
- long procedural workflows

Implementation direction:
- update docs/tool guidance to strongly discourage diary/task-log memory
- optionally add memory pressure nudges when usage crosses ~85-90%
- preserve frozen-snapshot/prompt-cache behavior
- treat these files as bootloader/kernel config, not the database
- keep only the preferences/facts that must influence nearly every turn

### Layer 1: Operational references + expanded memories

Use an Obsidian/knowledge-db-backed note layer for larger memories, preference history, operational references, and rich context that should be searchable/selectively loaded but not always injected.

Likely locations:
- `/home/oscar/workspace/_vault/Ops/` for operational references/runbooks
- `/home/oscar/workspace/_vault/memory/` for deeper personal memory/preference notes
- `(dropped — unused empty db)` for an optional search/index layer

Purpose:
- store more memories and preferences as the assistant/user relationship develops
- keep richer context human-readable and auditable
- support selective retrieval into turns without bloating the system prompt
- provide a place for "this matters, but not every turn" facts

Access path:
- search/load relevant notes on demand
- optionally add a thin `memory_ref`/`ops_ref` retrieval tool later
- do not inject the whole vault; this is not prompt stuffing with better furniture

### Layer 2: Entity/project files

Use structured files outside the core prompt for entities, contacts, projects, recurring systems.

Likely locations:
- `~/.hermes/profiles/<profile>/entities/*.json` or similar for profile-private entities
- `/home/oscar/workspace/_vault/` for human-readable long-lived notes
- project-local `.hermes/` files for project-specific state

Purpose:
- relationship graph
- project profiles
- service inventories
- operational runbooks
- non-global facts that should be fetched when relevant

Access path:
- explicit tools/skills/search, not always-on injection

### Layer 3: Procedural skills

Use skills for workflows and repeatable procedures.

Purpose:
- debugging playbooks
- deployment procedures
- project-specific recurring workflows
- lessons learned that are procedural, not facts

Implementation direction:
- keep authoring/maintenance through existing skill tooling
- after complex tasks, save workflow as skill instead of memory
- possible future consolidation with Obsidian is explicitly out of scope for the first pass

### Layer 4: Session search / transcript recall

Use session DB for ephemeral history and completed-work recall.

Purpose:
- "what did we do last time?"
- PR/issue/commit numbers
- task progress and stale artifacts
- conversation breadcrumbs

Implementation direction:
- avoid copying stale artifacts into core memory
- improve agent guidance to call `session_search` when user references prior work

### Layer 5: External semantic memory provider

Use one provider for scalable semantic recall.

Best candidates by current constraints:
- `holographic`: local, inspectable SQLite, no API dependency; good default for experimentation and privacy
- `honcho`: richer user modeling/cross-session dialectic; stronger but network/API/config heavier
- `openviking`/`hindsight`: worth evaluating if the desired shape is knowledge-base/tree plus retrieval

Implementation direction:
- start with a local provider if we want safe dogfooding without external cost/secrets
- keep provider context dynamic via existing `prefetch_all` path, not system prompt expansion
- measure latency/token impact before enabling broadly

## Proposed phases

### Phase 1: Documentation + guardrails + routing semantics

No behavioral risk.

- Update memory docs to match current routing policy:
  - global durable facts -> core memory
  - workflow -> skill
  - project/contact/entity -> structured entity/project file
  - temporary/task progress -> session search
- Add/adjust tests that forbid diary/task-progress language in memory tool docs/schema.
- Optional: add a compact `memory-routing.md` developer/user reference.
- Define the new three-way memory split explicitly:
  - core kernel: `USER.md` / `MEMORY.md`
  - expanded memories/preferences: Obsidian/knowledge DB notes
  - scoped state: contacts/projects/entities/kanban/session search

### Phase 2: Obsidian/knowledge-db operational memory layer

Primary capacity fix.

- Create/standardize vault folders for ops refs and expanded memory notes.
- Add a small retrieval convention before adding code: predictable paths, headings, tags, and search terms.
- Decide whether retrieval is initially skill/file-tool driven or exposed through a dedicated Hermes tool.
- Keep the core memory tool from dumping large facts into `MEMORY.md`; route expanded facts to notes.

### Phase 3: Core memory pressure UX

Small behavior changes; low risk.

- Improve full-memory errors with routing suggestions.
- Add a status command or nudge that names likely candidates for migration/consolidation.
- Keep writes manual/agent-driven; do not auto-delete core memory.

### Phase 4: Local structured entity/project store

Medium scope.

- Add a small profile-scoped entity/project store with JSON files and search/list/get/update tools.
- Make routing explicit: memory tool remains for global kernel; entity tool is for scoped durable records.
- Use `get_hermes_home()` and profile-safe paths.
- Do not inject entity store globally; retrieve on demand or via skills.

### Phase 5: External provider dogfood

Medium/high variance depending on provider.

- Enable `holographic` or another local provider in a test profile first.
- Verify:
  - init works
  - per-turn prefetch is fenced and not persisted into history
  - interrupted turns are not synced
  - session/compression boundaries call provider hooks
  - latency is acceptable
- Only then consider production assistant profile.

### Phase 6: Meta-provider / multi-store policy, only if needed

Higher-risk architecture.

- Current `MemoryManager` enforces one external provider.
- If multiple retrieval backends are needed, build a single umbrella provider that routes internally rather than weakening global manager constraints immediately.

## Test targets for any code work

Run targeted tests first:

```bash
python -m pytest tests/tools/test_memory_tool.py -q -o 'addopts='
python -m pytest tests/agent/test_memory_provider.py -q -o 'addopts='
python -m pytest tests/agent/test_streaming_context_scrubber.py -q -o 'addopts='
python -m pytest tests/run_agent/test_commit_memory_session_context_engine.py -q -o 'addopts='
python -m pytest tests/run_agent/test_memory_provider_init.py -q -o 'addopts='
```

Then area tests for any provider touched:

```bash
python -m pytest tests/plugins/memory -q -o 'addopts='
python -m pytest tests/honcho_plugin -q -o 'addopts='
```

Full suite before PR:

```bash
python -m pytest tests/ -q -o 'addopts='
```

## Open questions before code work

1. Should expanded memories/preferences live in plain Markdown first, or Markdown plus SQLite index immediately?
2. Should retrieval be initially skill/file-tool driven, or should we add a dedicated `memory_ref`/`ops_ref` tool?
3. What exact vault taxonomy should be canonical: `ops/`, `memory/`, `projects/`, `entities/`, or a different split?
4. Should memory pressure management stay advisory, or should there be an explicit assisted compaction/migration command?
5. Should `remove` built-in memory writes be mirrored to external providers, or is the current add/replace-only bridge intentional enough to keep?

## Current recommendation

Do Phase 1 and Phase 2 first. Phase 1 aligns the docs/tests with the intended routing policy. Phase 2 is the actual capacity fix: add an Obsidian/knowledge-db operational memory layer so Aster can grow richer preferences and memories without bloating the always-on prompt kernel. Entity/project/kanban stores remain separate and structured; skills consolidation is explicitly out of scope for this pass.
