# Hermes Memory Subsystem — Documentation

This directory is the consolidated, **version-controlled** home for the Hermes
memory subsystem design docs. The repo (`hermes-agent`) is the source of truth.
`~/.hermes` is the **runtime home** and is NOT a git repository — design docs
do not belong there as authoritative copies.

## Why this tree exists

The memory design docs had drifted into multiple copies with **conflicting
content**:

- `~/.hermes/memory-architecture.md` — 47 KB, 904 lines. The detailed Phase
  4/5/6 design journal (§16 API, §17 ADR, §18 project). The **source code
  cites this copy** (`facade.py` → "§16.4", `adr.py` → "§17",
  `project.py` → "§18.1/§18.11", `indexer.py` → "§7/§14").
- `hermes-agent/docs/memory-architecture.md` — 9.7 KB, 208 lines. A condensed
  arch summary (layer table L1–L6 + §0–§2 + IMPLEMENTED markers). Did NOT
  contain §16/§17/§18.
- Two skills contradicted each other about which copy was "authoritative."

Same filename, two genuinely different documents — that was the root of the
confusion. This tree resolves it (see Consolidation Status).

## Document Map (current)

| File                    | Purpose                                                          | Status  |
|-------------------------|------------------------------------------------------------------|---------|
| `README.md`             | This file — structure + navigation                               | current |
| `OVERVIEW.md`           | General architecture, project goals, per-layer (L1–L6) explainer | current |
| `memory-architecture.md`| Detailed phase design (§16 API, §17 ADR, §18 project) — cited by code | current (consolidated from the live 47 KB doc) |
| `INVARIANTS.md`         | The two architectural invariants (self-registration, declarative context) | current (from `docs/memory-future-invariants.md`) |
| `ARCHIVE_CONTRACT.md`   | Archive ownership / lifecycle contract                           | current (from `docs/memory-archive-contract.md`) |

## Layer model (one line each)

- **L1** Identity / hot memory
- **L2** Project state (human-curated, authority B)
- **L3** Archive (SQLite derived index — not an authority)
- **L4** ADR decisions (Hermes drafts `proposed`, human accepts)
- **L5** Session lifecycle
- **L6** Access facade — "How do I access memory?"

## L6 disambiguation (important)

"Layer 6" in **this** subsystem = the `MemoryAPI` facade + `MemoryRouter` — the
answer to *"How do I access memory?"* It is **NOT a sixth content layer**.

The Second Brain project has its own, unrelated "Layer 6 — Fix Button"
(`~/Projects/second-brain`). Do not confuse the two.

## Consolidation status (DONE — 2026-07-09)

1. Canonical detailed doc = the **47 KB live copy** (code cites it). Moved to
   `docs/memory/memory-architecture.md`.
2. The condensed 9.7 KB copy's unique bits (layer table, §2 status, L6 note)
   were ported into `OVERVIEW.md` + this README; the old copy was deleted.
3. `INVARIANTS.md` and `ARCHIVE_CONTRACT.md` moved in from `docs/`.
4. The runtime (`~/.hermes/memory-architecture.md`) and old-repo
   (`docs/memory-architecture.md`, `docs/memory-future-invariants.md`,
   `docs/memory-archive-contract.md`) duplicates were deleted.
5. All 11 source-file citations and the contradictory skill references were
   repointed at `docs/memory/...`. A post-move grep confirmed zero stale
   references to the old paths.

## Hard constraints (carry into every doc)

- Writes never silent (typed `CapabilityError`, never a no-op "success").
- Markdown + raw = source of truth; SQLite = derived, rebuildable cache only.
- No LLM extraction / embeddings / Graphiti / Holographic in core.
- One provider instance per capability; stateless providers.
- Slug collision → explicit `CapabilityError`.
- Context participation is **opt-in, default OFF** (a new capability contributes
  nothing until a human opts it in).
