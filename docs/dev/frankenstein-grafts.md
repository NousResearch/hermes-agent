# Frankenstein grafts — Spearhead provenance

**Status:** durable provenance note for the May 2026 Spearhead/Gond Frankenstein-graft wave.

This document records which ideas from external agent/memory/orchestration scout reports were adopted, spiked, monitored, or explicitly skipped for Hermes/Spearhead. It is intentionally a provenance and decision ledger, not a blanket import plan. Shiny repo != free organ transplant; we are not building a prompt-shaped junk drawer with a pulse.

## Source chain

Primary Kanban evidence:

- `t_301854d9` — broad Gond runtime graft continuation; superseded by narrower finish card after iteration-budget exhaustion.
- `t_6b0caa94` — final narrow patch: tripwire public-message fix, regression assertions, and scout-report path listing.
- `t_6affd5cd` — independent dirty-diff review of the Frankenstein runtime graft scope.
- `t_3e95cbea` — focused Gond Phase 2A card that integrated the Claude scout reports into this provenance decision matrix.
- `t_1fddc40d` — reconciliation card making the integration durable after the original `/tmp` worktree/report paths were no longer present.

Original Claude scout report paths recorded by `t_3e95cbea`:

- `/tmp/hermes-claude-scouts/reports/memory_indexing.md`
- `/tmp/hermes-claude-scouts/reports/mnemo_provenance.md`
- `/tmp/hermes-claude-scouts/reports/hyperagents_eval.md`
- `/tmp/hermes-claude-scouts/reports/agency_roles.md`
- `/tmp/hermes-claude-scouts/reports/gateway_hardening.md`

At reconciliation time, the `/tmp/hermes-claude-scouts/` and `/tmp/hermes-gond-frankenstein-runtime/` paths had already been cleaned up, so this durable note is reconstructed from Kanban task metadata and session handoff evidence rather than pretending the transient files were still readable. The decision matrix below mirrors the completed `t_3e95cbea` metadata.

## Phase 1 runtime graft already applied/reviewed

The parent handoffs record a local Frankenstein runtime graft covering:

- gateway outbound guard / send policy / tripwire wiring;
- provider-resilience and safety handling;
- fresh-final stream behavior;
- targeted regression tests;
- documentation of scope, safety behavior, and risks.

Independent review `t_6affd5cd` found the dirty file set matched the expected 12-file runtime graft scope and found no destructive added-line patterns. Follow-up `t_6b0caa94` removed sentinel-name exposure from public tripwire block text and added tests ensuring canary/sentinel names are not exposed in public sent content or send-message tool JSON.

## Phase 2A decision matrix

| Source / pattern | Decision | Hermes/Spearhead adaptation | Non-goals / guardrails |
| --- | --- | --- | --- |
| Gateway hardening scout | **Implement now** | Keep call-site enforced outbound guard, send policy, stream-safety, and tripwire suppression tests. | No dead helper-only hardening; every safety path needs call-site wiring and regression coverage. |
| CocoIndex-style indexing patterns | **Implement now, selectively** | Use the idea of explicit indexing/provenance pipelines where it fits Hermes-native retrieval and corpus workflows. | Do not import an external indexing stack just to feel fancy. Keep dependency surface small unless a later spike proves need. |
| MemPalace memory/retrieval patterns | **Spike** | Explore hybrid BM25/vector retrieval, neighbor expansion, and citation/line provenance as pattern candidates. | Do not adopt ChromaDB/full provider stack or AAAK compression into Hermes core without a separate approval gate. |
| Mnemo/Cortex provenance | **Spike** | Investigate durable provenance objects, source references, validity windows, and evidence trails for memory/knowledge updates. | No unverifiable knowledge writes; no inferred provenance when source text was not actually read. |
| agency-agents role/checklist contracts | **Implement now, selectively** | Convert useful handoff, evidence, and minimal-change checklists into Hermes skills/contracts instead of importing personas wholesale. | No bulk persona prompt import; no roleplay bloat; keep instructions operational and testable. |
| HyperAgents / DGM eval gates | **Spike** | Use eval gates and mutation/promotion discipline as inspiration for safe skill/tool evolution and code changes. | No unattended DGM promotion, no AGPL-tainted imports into Hermes core, no self-modifying production behavior without explicit approval. |

## Explicit skip list

Do not implement these as part of the Frankenstein graft without a new, explicit approval/review path:

- MemPalace AAAK compression.
- MemPalace ChromaDB/full-provider adoption in Hermes core.
- Bulk agency-agents persona imports.
- Unsigned or ungated Developer Passport enforcement.
- Unattended DGM promotion.
- AGPL darwinian-evolver imports into Hermes core.
- Gateway hardening that exists only as dead helper code without call-site tests.

## Evidence requirements for future grafts

A future Frankenstein graft is acceptable only when it has:

1. exact source/provenance link or local report path;
2. decision state: `implement-now`, `spike`, `monitor`, or `skip`;
3. Hermes-native adaptation plan, not a wholesale framework transplant;
4. explicit safety/licensing/dependency guardrails;
5. targeted tests or a written reason tests do not apply;
6. a Kanban handoff containing changed files, commands run, and remaining risks.

If any of those are missing, the graft is not done. It is just a monster under a bedsheet, and we are not paying for its GitHub Copilot subscription.
