# Hermes Memory Architecture (Phases 1–6)

This document is the authoritative design for Hermes' deterministic, inspectable,
Git-friendly memory system. Markdown is the source of truth; the SQLite index is a
derived cache only. Code lives under `hermes_cli/memory_api/`, `hermes_cli/memory_router/`,
and `hermes_cli/memory_index/`.

---

## 0. One primary question per layer (architectural invariant)

Every memory layer MUST answer exactly one primary question. Future features
belong to exactly one primary question whenever possible. This is the
controlling invariant for the whole taxonomy — it is what keeps the system from
silent sprawl.

| Layer | Primary question | Backing |
|-------|------------------|---------|
| L1 | Who am I? | Identity files (USER.md, IDENTITY.md, MEMORY.md) |
| L2 | What is the current state of this project? | `memory/projects/<slug>/STATUS.md` (present state only) |
| L3 | What happened? | Conversation archive (closed sessions) |
| L4 | Why did we decide this? | ADRs (`memory/adr/<project>/NNN-slug.md`) |
| L5 | Where can I find information? | Search / index over the above |
| L6 | How do I access memory? | The MemoryAPI facade + Router (this layer) |

Rationale: L2 must describe the *present*, not preserve the *past*. History
already has excellent homes (L3 archive, L4 ADRs, L5 search). L2 answers the
question every reopen asks: "Where were we, and what should we do next?" It
links to history by reference; it never copies it.

---

## 1. Implemented architecture (current state)

### 1.1 Facade → Router → Provider

```
caller
  └─ MemoryAPI  (hermes_cli.memory_api.facade.MemoryAPI)   ← THIN, no providers
       └─ MemoryRouter  (hermes_cli.memory_router.router.MemoryRouter)  ← routing only
            ├─ Capability registry (intent → handler)
            └─ Providers (concrete storage owners, instantiated ONCE)
```

- **MemoryAPI (facade)** is a translation layer only. It holds **no concrete
  provider instances** and **never imports a storage backend**. It speaks intents.
- **MemoryRouter** is the sole owner of routing AND of the single provider
  instance per capability. It exposes convenience methods
  (`project()`, `decision()`, `draft_decision()`, `accept_decision()`,
  `remember()`, `capability_status()`) that wrap `route(intent, method, **kw)`.
- **Providers** own storage behavior (markdown files, etc.) and are the only
  place filesystem paths are constructed.

### 1.2 Routing-all-operations invariant (IMPLEMENTED — Item 1)

All operations — reads AND writes — go through Router intents. The facade does
not directly invoke any concrete provider.

| Caller API | Router intent | method |
|------------|---------------|--------|
| `api.search / archive / recent` | SEARCH / ARCHIVE / RECENT | — |
| `api.context()` | (fans out via router) | — |
| `api.project(key)` | PROJECT_STATE | `get` |
| `api.set_project(state, by)` | PROJECT_STATE | `set` (the ONLY L2 write) |
| `api.propose_project(key, **kw)` | PROJECT_STATE | `propose` (writes NOTHING) |
| `api.decision(...)` | DECISION | `decision` / `by_project` / `recent` |
| `api.draft_decision(...)` | DECISION | `draft` |
| `api.accept_decision(id, by=)` | DECISION | `accept` |
| `api.remember(content, layer)` | REMEMBER | `remember` |

The facade's old provider registry (`_providers`, `register_provider`,
`_select_write_provider`, `_select_read_provider`) was **removed**. Adding or
swapping a backend now touches only the Router's registry — never the facade or
any caller. This resolves the earlier dual-instance coupling (F3): there is
exactly one `ProjectProvider` instance, owned by the Router.

### 1.3 Statelessness invariant (IMPLEMENTED — Item 2)

**Providers own storage access but hold NO authoritative mutable in-memory
state.** All truth lives on disk (e.g. STATUS.md). Two reads of the same project
always agree; nothing is cached or mutated in memory across calls. Mutating a
returned `ProjectState` cannot affect a subsequent read.

### 1.4 Slug collision detection (IMPLEMENTED — Item 3)

`ProjectProvider.set()` guards against two *distinct project keys* that resolve
to the same storage directory (slug). If a `STATUS.md` already exists at the
target slug path under a **different** `project` key, `set()` raises
`CapabilityError` — explicit failure, never ambiguous overwrite. Re-saving the
exact same key is idempotent (allowed).

### 1.5 Trust model (L4 ADRs, carried forward)

- `draft_decision()` creates a PROPOSED draft (non-authoritative).
- `accept_decision(id, approved_by=...)` is the ONLY authority transition; it
  requires a human `approved_by`.
- `decision()` (read) surfaces ACCEPTED ADRs only; proposed drafts are excluded
  at the provider read boundary.
- Hermes may draft; humans approve. No autonomous acceptance.

### 1.6 Trust model (L2 projects, authority option B)

- `STATUS.md` is the curated L2 truth, human-authored.
- `set_project()` is the only write path; it requires `updated_by` (human).
- `propose_project()` returns an in-memory draft and writes NOTHING.
- `last_verified` / `verified_by` are informational; never auto-populated.

### 1.7 Writes are never silent

An unsupported/unavailable write raises `CapabilityError`. No fabricated
"success" for an unpersisted write. `set()` verifies the file landed on disk
before returning.

---

## 2. Architectural invariants (DESIGN-ONLY — do not implement yet)

These are declared invariants for future evolution. They are NOT implemented in
the current code and must not be assumed present.

### 2.1 Declarative context participation

**STATUS: IMPLEMENTED — build spec in `docs/memory-future-invariants.md`.**

Each capability declares `contributes_to_context: bool` + a `context_category`
(string). `MemoryAPI.context()` iterates the Router registry and includes only
opted-in capabilities, dispatching each by `context_category` into the matching
`ContextBundle` slot. The four product-opted capabilities (reviewed decisions):

- `L1-identity`  → category "identity"
- `L3-archive`   → category "recent"
- `L4-adr`       → category "decision"
- `L2-project`   → category "project" (single active, same precedence rules)

THE GUARDRAIL (enforced by contract tests): default `contributes_to_context =
False`. A newly registered capability contributes NOTHING to context until a
human explicitly opts it in — future backends do NOT automatically participate
merely by existing. This is what stops "more backends" from silently changing
what Hermes loads by default. Verified by
`tests/plugins/memory/test_future_invariants.py`
(`test_context_ignores_unopted_capability`, `test_context_opted_out_when_flag_false`).

### 2.2 Provider self-registration

**STATUS: IMPLEMENTED — build spec in `docs/memory-future-invariants.md`.**

Providers register themselves via `@_registrar` modules under
`hermes_cli/memory_router/registrations/` (one module per default capability).
The Router's `__init__` imports ONLY that package boundary
(`load_default_capabilities`) and never a concrete provider class
(`AdrProvider` / `ProjectProvider` / `IndexCapability` / `IdentityCapability`),
so removing a capability is a one-line edit with zero other changes. The
Router's single-instance handles (`_adr_provider` / `_project_provider`) are
sourced from the registry (`registry.by_name(...).provider`), preserving
exactly ONE instance per capability. Verified by
`tests/plugins/memory/test_future_invariants.py`
(`test_router_has_no_concrete_provider_imports`,
`test_router_sources_single_provider_instances_from_registry`,
`test_removing_a_provider_drops_capability`).

### 2.3 Other standing invariants (carried)

- Markdown is source of truth; SQLite is a derived index (never authority).
- No LLM extraction, no embeddings, no Graphiti/Holographic in the core layers.
- No autonomous L2 writes; no automatic STATUS.md generation.
- `ARCHIVE` intent stays distinct from present-state reads.

---

## 3. Capability → intent map

| Intent | Layer | Capability | Available |
|--------|-------|------------|-----------|
| SEARCH | L5/L1 | L5-search | yes |
| ARCHIVE | L3 | L3-archive | yes |
| RECENT | L3 | L3-recent | yes |
| DECISION | L4 | L4-adr | yes (draft/accept/decision/by_project/recent) |
| PROJECT_STATE | L2 | L2-project | yes (get/set/propose) |
| REMEMBER | L1 | L1-remember | **no writer wired** (routing exists; raises CapabilityError) |
| (identity) | L1 | L1-identity | yes (pointer resolution only) |

---

## 4. Forward-review guidance (3 backends, thousands of projects, 2 years)

- **Routing-all-ops** (§1.2) is what keeps callers backend-agnostic. A new
  backend is added by registering a capability + handler in the Router; the
  facade and all callers are untouched.
- **Single provider instance** (§1.1) removes dual-path coupling and path
  duplication.
- **Slug collision** (§1.4) is cheap insurance at scale — prefer explicit
  failure over ambiguous resolution.
- **Statelessness** (§1.3) makes providers safely replaceable and cache-free.
- **Declarative context** (§2.1) is the guardrail that stops "more backends"
  from silently changing what Hermes loads by default.
- Resist speculative escape hatches (e.g. a generic `invoke()` /
  capability-discovery shim). They trade away the explicit, reviewable routing
  graph for magic.

---

## 5. Test status

- Full memory suite: **550 passed** (Phase 1–6 + routing/slug/statelessness
  contract tests).
- Contract tests cover: provider parse/round-trip, slug collision, statelessness,
  facade-routes-through-router, propose-writes-nothing, remember-raises-when-unwired,
  ADR draft/accept trust boundary, context no-fabrication.
