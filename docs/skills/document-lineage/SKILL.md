# Document Lineage Map

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Priority:** 3 of 7

## Purpose

Define which documents are authoritative for which facts, establish a dependency graph between document layers, and provide lookup rules so any agent encountering a factual claim can determine its authority level.

---

## Document Layers (Dependency Order)

```
Layer 0: Production Code + Config
  (selector_engine.py, ranker_v2_pairwise.py, decision_engine.py,
   production_data/ranker_v2_model.json, production_data/decision_rulesets/manifest.json,
   production_data/manager_registry.json)
     |
Layer 1: GitHub Model Documentation
  (model_documentation_root.md, docs/MODEL_DOCUMENTATION.md,
   governance/agent_governance.md, governance/STATUS.md,
   governance/AGENT_ROUTING_POLICY.md, governance/HASH_ROTATIONS.md)
     |
Layer 2: Town Skills (Content Library + skills://)
  (14 Hermes-relevant skills documented in May 18 inventory)
     |
Layer 3: Operational Artifacts
  (CI Diagnostic Report, CI Fix Checklist, Full Sweep Audit Report,
   Ranker Research Prep Pack, Build Specs, Code Review Report,
   Doc Review Log, 13F Manager Directories)
     |
Layer 4: External-Facing Documents (.docx exports)
  (WakeRobin_DEM_Executive_Overview, WakeRobin_Model_Documentation_v1.7.0)
```

**Authority flows downward.** Layer 0 is ground truth. Each downstream layer should be derivable from the layer above it. When layers conflict, the upstream layer wins.

---

## Per-Fact Authority Table

| Fact | Authoritative Source | Layer | Known Stale Copies |
| --- | --- | --- | --- |
| Selector weights | `selector_engine.py` + `decision_rulesets/manifest.json` | L0 | .docx (L4): still shows 65/35 split |
| Ranker weights | `ranker_v2_model.json` provenance block | L0 | Model docs (L1): say "2 features" vs runtime 6 |
| Active ruleset ID | `manifest.json` | L0 | .docx footer (L4): shows v1.13.0 |
| Agent count | `governance/agent_governance.md` | L1 | Exec Overview (L4): 26; compliance memo (L3): 17 |
| Signal names | Production code variable names | L0 | .docx (L4): sponsorship/momentum; GitHub (L1): mixed |
| Clinical score denominator | `clinical-scoring` skill (L2) post-H1 fix | L2 | GitHub (L1): shows pre-H1 value 117 |
| inst_delta_z status | `selector_engine.py` (selector) + `ranker_v2_pairwise.py` (ranker) | L0 | selector-ranker skill (L2): was wrong pre-H3 fix |
| Tier numbering | No single authority (W4) | -- | 3 incompatible systems across L1 governance docs |
| IC measurement scope | `run_rank_ic_backtest.py` code (L0) | L0 | All pre-Spec-100 IC claims cited wrong field |
| Cron schedule | `crontab -l` on production host | L0 | screener-ops skill (L2): says 5:30 PM; W5 notes inconsistency |

---

## Sync-State Protocol (Conceptual)

When a fact changes at Layer 0:

1. **Immediately:** Update the Layer 1 GitHub doc that references it (same PR or follow-up PR within 48h)
2. **Within 1 week:** Update the Layer 2 skill that references it
3. **At next .docx release:** Update Layer 4 external documents
4. **Layer 3 artifacts:** These are point-in-time and expected to go stale. They should carry an explicit `as_of_date` and never be cited as current without verification.

**Current gap:** No mechanism enforces this cascade. It is entirely manual and has broken repeatedly (C1, C2, C6).

---

## Tier Numbering Cross-Reference (Resolving W4)

| Document | System | Tiers | Scope |
| --- | --- | --- | --- |
| `governance/AGENT_ROUTING_POLICY.md` | Agent routing | Tiers 0-4 (5 tiers) | Which agents handle which task types |
| `governance/agent_governance.md` | Agent authority | Tiers 0-3 (4 tiers) | Authority levels (observe_only through mutate_config) |
| `ccft-aware-routing-policy` (ai-projects) | CCFT routing | 7 tiers | Model routing by task complexity |

These are three different taxonomies applied to three different dimensions. They are NOT versions of the same system.

---

## Open Questions

1. Should tier numbering be unified into a single namespace, or is the current separation (routing / authority / model-routing) correct and just needs cross-references?
2. Should the sync-state protocol be enforced by a post-merge CI check, or is manual cascade sufficient?
3. Should Layer 4 (.docx) exports be auto-generated from Layer 1 markdown, or do they serve a different audience requiring manual curation?