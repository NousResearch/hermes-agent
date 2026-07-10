# Memory Subsystem — Future Invariants Build Spec

Prepared handoff for a dedicated session that will implement the two
**architectural invariants** declared (but NOT implemented) in
`docs/memory/docs/memory/memory-architecture.md` §2. This session builds them from scratch. The
current code is GREEN (550 memory tests pass); do not regress it.

---

## 0. Status before this build

- Facade is already backend-agnostic (no concrete provider imports — verified).
- Router owns the single provider instance per capability (F3 resolved).
- NOT yet done: provider self-registration, declarative context participation.
- `HERMES_SESSION.md` / `HERMES_PROJECTS.md` are NOT present in this workspace;
  the session-start reads are dormant. This doc IS the handoff.

---

## 1. Invariant A — Provider self-registration

### Current shape (what to replace)
`hermes_cli/memory_router/router.py`, `MemoryRouter.__init__` (~lines 80–183),
imports concrete classes and wires them by hand:

- `from hermes_cli.memory_api.adr import AdrProvider` → registers `L4-adr`
- `from hermes_cli.memory_api.project import ProjectProvider` → registers `L2-project`
- `IdentityCapability()` → registers `L1-identity`
- `L1-remember` registered as an unavailable stub
- `L3-archive` / `L3-recent` / `L5-search` registered via their providers

The Router currently holds concrete imports → violates the "Router free of
concrete imports" end-state.

### Target shape
- Each provider module self-registers on import via a decorator, e.g.
  `@register_capability(name="L2-project", intents=[Intent.PROJECT_STATE], available=True)`
  that calls `MemoryRouter.registry.register(...)`.
- `MemoryRouter.__init__` imports ONLY the registry + intent enum, then imports
  the provider modules (side-effect registration) — or a small
  `_load_default_capabilities()` does the imports. No `from ... import AdrProvider`
  type references in the routing logic.
- `self._project_provider` / `self._adr_provider` single-instance handles: keep
  them, but obtain the instance from the registry (`registry.get("L2-project")`)
  rather than storing a separately-constructed object, so there is still exactly
  ONE instance.

### Acceptance criteria
- `MemoryRouter` source contains no import of `AdrProvider` / `ProjectProvider` /
  `IdentityCapability` at module/class level (only inside the loader, if needed).
- Removing a provider module's import drops that capability with zero other edits.
- All 550 existing tests still pass; add a test asserting the Router registry is
  populated by importing provider modules (not by hand-wiring).

---

## 2. Invariant B — Declarative context participation

### Current shape (what to replace)
`hermes_cli/memory_api/facade.py`, `MemoryAPI.context()` (lines 66–137) hardcodes
the fan-out:
1. identity  → `self._router.search(query, scope="L1-identity")`
2. recent    → `self._router.recent(...)`
3. decision  → `self.decision()` (accepted ADRs)
4. project   → single active project via `self.project(resolved)`

This is intentional product-level composition (per architecture §2.1). The build
must preserve EXACTLY this behavior while making participation declarative.

### Target shape
- Extend `Capability` (`hermes_cli/memory_router/registry.py`) with:
  `contributes_to_context: bool = False` and `context_category: str = ""`.
- `CapabilityRegistry.register(...)` gains `contributes_to_context` /
  `context_category` params (default `False` / `""`).
- In `context()`, iterate `self._router.registry.list()`; for each capability
  where `contributes_to_context` is True, dispatch its context contribution by
  `context_category` (identity / recent / decision / project) into the matching
  `ContextBundle` slot.
- Explicitly opt IN the three/four current capabilities:
  - `L1-identity`  → category "identity"
  - `L3-recent`    → category "recent"
  - `L4-adr`       → category "decision"
  - `L2-project`   → category "project" (single active, same precedence rules)

### THE GUARDRAIL (do not skip)
> Future backends must NOT automatically participate in context merely because
> they exist. Opting in is a deliberate, reviewed product decision.

Therefore: **default `contributes_to_context=False`**. A newly registered
capability contributes NOTHING to context until a human explicitly flips it on.
This is the whole point of the invariant — it prevents "more backends" from
silently changing what Hermes loads by default. Add a test that registers a
fake available capability and asserts `context()` does NOT include it.

### Acceptance criteria
- `context()` produces byte-equivalent `ContextBundle` content to today's
  hardcoded version for the four opted-in capabilities (diff the two on a
  fixture project; assert equality of bundle fields).
- A capability registered with `contributes_to_context=False` is invisible to
  `context()` (test with a stub).
- The single-active-project rule and "no fabrication when absent" still hold.

---

## 3. Build order (suggested)

1. Invariant A (self-registration) first — it's mechanical and de-risks B.
2. Invariant B (declarative context) second — depends on the registry shape
   from A. Keep the explicit opt-in list reviewed and small.
3. Update `docs/memory/docs/memory/memory-architecture.md` §2.1 / §2.2 → "IMPLEMENTED", move the
   guardrail description into the code/contract tests.

---

## 4. Verification plan (run at the end)

- `.venv/bin/python -m pytest tests/plugins/memory/ -q` → expect 550 + new
  contract tests, 0 failures.
- Import sanity: `MemoryRouter` source has no concrete provider imports at
  routing level; `Capability` has `contributes_to_context`.
- CLI smoke: `HERMES_HOME=/tmp/x python -m hermes_cli.main memory project set
  "<key>" --title t --status active --by joe --narrative n` then `... project
  show "<key>"` → still works through the new routing.

---

## 5. Hard constraints carried from prior phases

- Writes NEVER silent (CapabilityError on unavailable).
- Markdown = source of truth; SQLite = derived index only.
- No LLM extraction / embeddings / Graphiti / Holographic in core layers.
- L2 human-curated (authority B); L4 Hermes-drafts / human-accepts.
- Single provider instance per capability (no path duplication).
- Stateless providers (no mutable in-memory authoritative state).
- Slug collision → explicit CapabilityError, never ambiguous overwrite.
