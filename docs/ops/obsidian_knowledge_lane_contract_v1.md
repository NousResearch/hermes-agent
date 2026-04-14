# Obsidian Knowledge Lane Contract v1 (`XK-201`)

Date: 2026-03-28  
Status: active (canonical for XK lane foundation)  
Owner: Architect  
Scope: Obsidian knowledge canonicalization lane (`XK-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XK` is the canonical Obsidian knowledge lane for governed vault-origin knowledge inside the expanded roadmap.

This contract canonicalizes three minimum surfaces required by `XK-201`:
1. lane scope and authority boundaries,
2. deterministic source-classification rules,
3. governance checklist obligations before promotion claims.

`XK-201` does **not** implement materialization mappings (`XK-202`) or freshness/retrieval SLO runtime gates (`XK-203`).

---

## 2) Canonical inputs and bounded outputs

### Canonical inputs
- `ops/obsidian/README.md`
- `docs/ops/obsidian_portable_setup.md`
- `docs/ops/shared_memory_fabric_lifecycle_contract_v1.md`
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XK-201..XK-203`)

### Canonical outputs for `XK-201`
- this lane contract (`docs/ops/obsidian_knowledge_lane_contract_v1.md`),
- source-classification artifact (`state/continuity/latest/xk_201_obsidian_source_classification_rules_2026-03-28.json`),
- governance checklist artifact (`state/continuity/latest/xk_201_obsidian_lane_governance_checklist_2026-03-28.json`),
- closeout evidence report (`reports/xk_201_obsidian_knowledge_lane_contract_v1_closeout_2026-03-28.md`).

Deferred to follow-on slices:
- `XK-202`: deterministic Obsidian→shared-memory mapping + conflict semantics.
- `XK-203`: freshness SLO + retrieval abstain quality gates in continuity/operator surfaces.

---

## 3) Lane boundary and non-goals

### In-boundary for `XK-201`
- Canonical lane scope for governed Obsidian sources.
- Deterministic source-class taxonomy and class-level handling intent.
- Ownership and mutation boundary between Obsidian lane artifacts, shared-memory promotion, and archive/quarantine destinations.

### Out-of-boundary for `XK-201`
- No new runtime schedulers/watchdogs.
- No direct promotion runtime rewiring.
- No retrieval threshold tuning or SLO enforcement runtime.

---

## 4) Authority and ownership model

| Domain | Canonical authority | Mutation rights | Verification obligation |
|---|---|---|---|
| Lane doctrine/boundary contract | Architect control plane | main session only | source-of-truth map parity + queue evidence refs |
| Source-class taxonomy | XK lane owner (governed) | ticketed updates only; append-only class versioning | classification artifact schema + path containment |
| Obsidian extraction tooling (`ops/obsidian/*`) | XK worker lane | implementation updates allowed inside lane scope | deterministic output + no vault write-back by exporter |
| Shared-memory canonical authority | B1 shared-memory governance | only via promotion/materialization contracts | conformity with `shared_memory_fabric_lifecycle_contract_v1` |
| Archive/quarantine boundaries | XR/XG governance | rule updates via canonical governance docs only | out-of-core scope filter parity |

Authority constraints:
1. Obsidian lane cannot directly declare canonical shared-memory truth without promotion/materialization contract compliance.
2. Obsidian exporter output is memory-bound only; vault mutation is out-of-scope.
3. Source classes marked restricted/excluded cannot enter promotion candidates.

---

## 5) Source classification contract (minimum)

`XK-201` defines four source classes:

1. `canonical_governed`
   - governed notes with required frontmatter (`type,id,status,trust_level,created,updated`),
   - eligible for deterministic export and downstream promotion consideration.

2. `operational_reference`
   - lane-adjacent operational notes that may inform synthesis,
   - export allowed with lower promotion priority and explicit traceability requirements.

3. `incident_or_decision_log`
   - incident/decision streams allowed for evidence and context,
   - promotion requires explicit conflict/demotion compatibility checks in `XK-202`.

4. `restricted_or_excluded`
   - secrets, blocked roots, or excluded patterns (`.obsidian`, execution/artifact churn paths, secret-file denylist),
   - never eligible for promotion/materialization.

Normative include roots are inherited from `ops/obsidian/README.md` and `docs/ops/obsidian_portable_setup.md`.
If include roots are configured but missing, processing fails closed.

---

## 6) Promotion boundary contract (XK↔B1↔archive)

Boundary rule set:
1. `XK` may produce deterministic export/index/retrieval evidence only.
2. Promotion to canonical shared-memory objects requires `XK-202` contract compliance and B1 lifecycle semantics.
3. Archive/quarantine destinations remain governed by out-of-core scope filters; `XK` cannot silently reclassify archived material as canonical.
4. Any ambiguity between Obsidian content and existing canonical memory must resolve through conflict-set workflow, never silent overwrite.

---

## 7) Required governance checklist (minimum)

Before claiming `XK-201` done, all must be true:
1. lane contract is canonicalized and referenced in source-of-truth map,
2. source-class rules are published as machine-readable artifact,
3. ownership matrix and boundary constraints are explicit,
4. no claims made for `XK-202`/`XK-203` runtime behavior,
5. queue-layer state updated with evidence refs and deterministic status reason.

---

## 8) Validation entrypoints for this slice

- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`
- queue-layer parity check (slice state/evidence refs) in `state/continuity/latest/true_expanded_roadmap_queue_layer.json`

---

## 9) Closeout condition for `XK-201`

`XK-201` is complete only when:
1. Obsidian lane foundation contract is canonical,
2. source classes and governance checklist are published as bounded artifacts,
3. source-of-truth map registers these contract surfaces,
4. queue slice `XK-201` is `DONE` with evidence refs,
5. downstream dependency posture is updated without over-claiming `XK-202`/`XK-203` implementation.
