# Hermes Memory Subsystem — Overview

This is the friendly entry point. For the detailed phase design (§16 API, §17
ADR, §18 project) see [`memory-architecture.md`](./memory-architecture.md); for
the two invariants see [`INVARIANTS.md`](./INVARIANTS.md); for the archive
ownership rule see [`ARCHIVE_CONTRACT.md`](./ARCHIVE_CONTRACT.md).

## 1. Purpose & Project Goals

The memory subsystem is a **deterministic, inspectable, Git-friendly** memory
system. Markdown files are the source of truth; a SQLite index is a derived,
rebuildable cache only.

Goals (from Joe's directives):

- One stable interface (the `MemoryAPI` facade) between callers and concrete
  storage — callers never know storage details.
- Pluggable backends (L1 identity, L2 project, L3 archive, L4 ADR, L5 search,
  and future stores) behind one contract.
- No silent writes: an unavailable write raises a typed `CapabilityError`,
  never a fabricated "success".
- No LLM extraction / embeddings / Graphiti / Holographic in the core layers.

Explicit non-goals: no new storage invented by this subsystem beyond what the
phases define; no autonomous L2 writes; no automatic `STATUS.md` generation.

## 2. High-Level Architecture

```
caller
  └─ MemoryAPI   (hermes_cli.memory_api.facade)    ← THIN, no providers, speaks intents
       └─ MemoryRouter (hermes_cli.memory_router.router)  ← routing + single provider instance per capability
            ├─ Capability registry (intent → handler)
            └─ Providers (concrete storage owners, instantiated ONCE)
```

- **MemoryAPI (facade)** is a translation layer only. It holds **no concrete
  provider instances** and **never imports a storage backend**.
- **MemoryRouter** is the sole owner of routing AND of the single provider
  instance per capability. Providers own storage behavior (markdown files) and
  are the only place filesystem paths are constructed.
- **Providers** are stateless: all truth lives on disk; two reads always agree.

## 3. The Layers (L1–L6)

Every layer answers exactly one primary question — the controlling invariant
that prevents silent sprawl.

| Layer | Primary question | Backing | Status |
|-------|------------------|---------|--------|
| L1 | Who am I? | Identity files (USER.md, IDENTITY.md, MEMORY.md) | shipped |
| L2 | What is the current state of this project? | `memory/projects/<slug>/STATUS.md` (present only) | shipped |
| L3 | What happened? | Conversation archive (closed sessions) | shipped |
| L4 | Why did we decide this? | ADRs `memory/adr/<project>/NNN-slug.md` | shipped |
| L5 | Where can I find information? | Search / index over the above | shipped |
| L6 | How do I access memory? | The `MemoryAPI` facade + Router | shipped |

**L6 disambiguation.** In *this* subsystem, "Layer 6" is the access facade +
Router — **NOT a sixth content layer**. It answers *"How do I access memory?"*
Do **not** confuse it with the Second Brain project's own "Layer 6" (the Fix
Button, `~/Projects/second-brain/FIX_BUTTON_DESIGN.md`, `sb-sorter` profile).
Those are two unrelated systems that both reuse the "L6" label.

## 4. Two Architectural Invariants

### A. Provider self-registration
Providers register themselves via `@_registrar` modules under
`hermes_cli/memory_router/registrations/` (one module per capability). The
Router imports only that package boundary (`load_default_capabilities`) and
never a concrete provider class, so removing a capability is a one-line edit.
The Router's single-instance handles are sourced from the registry, preserving
exactly one instance per capability.

### B. Declarative context participation
Each capability declares `contributes_to_context: bool` + a `context_category`.
`MemoryAPI.context()` iterates the registry and includes only opted-in
capabilities. **Default `contributes_to_context = False`** — a newly registered
capability contributes nothing to context until a human explicitly opts it in.
This is the guardrail that stops "more backends" from silently changing what
Hermes loads by default.

## 5. Hard Constraints (carry into every doc)

- Writes never silent (typed `CapabilityError`, never a no-op "success").
- Markdown + raw = source of truth; SQLite = derived, rebuildable cache only.
- No LLM extraction / embeddings / Graphiti / Holographic in core.
- One provider instance per capability; stateless providers.
- Slug collision → explicit `CapabilityError`.
- Context participation is opt-in, default OFF.

## 6. Status

- Full memory suite: **562 passed** (Phase 1–6 + routing/slug/statelessness +
  invariant contract tests).
- Both architectural invariants (A, B) implemented 2026-07-09.
- Docs consolidated 2026-07-09 into `docs/memory/` (see `README.md`).
