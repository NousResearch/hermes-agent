# True Expanded Roadmap Out-of-Core Scope Filter Rules (v1)

Date: 2026-03-28  
Status: active (canonical for XR-003 scope quarantine)  
Owner: Architect  
Slice: `XR-003 out_of_core_archive_quarantine_index`

## Purpose
Define deterministic, fail-closed path filtering so legacy/downstream materials cannot contaminate core execution queues, canonical lane mapping, or promotion decisions.

## Canonical authority
This rule set is authoritative for out-of-core quarantine decisions in the true expanded roadmap queue.

Primary references:
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (XR queue truth)
- `state/continuity/latest/true_expanded_out_of_core_archive_index_2026-03-28.json` (indexed quarantine catalog)
- `state/continuity/latest/true_expanded_out_of_core_path_classification_sample_2026-03-28.json` (deterministic sample classifications)

## Deterministic filter contract

Precedence order (top to bottom):
1. **Explicit allowlist** (in-scope core/expanded execution artifacts)
2. **Explicit quarantine denylist** (legacy/downstream clusters)
3. **Unknown path fallback = quarantine** (fail-closed)

No “soft include” is allowed for paths matched by quarantine rules.

## Rule set

| Rule ID | Match type | Pattern(s) | Classification | Queue/Promotion action |
|---|---|---|---|---|
| `ALLOW-CORE-CANON` | prefix | `docs/ops/`, `ops/openclaw/`, `scripts/`, `tests/`, `state/continuity/latest/`, `reports/openclaw_*`, `reports/core_roadmap_*`, `reports/a1_*`, `reports/a2_*`, `reports/a3_*`, `reports/a4_*`, `reports/a5_*`, `reports/a6_*`, `reports/b1_*`, `reports/b2_*`, `reports/b3_*`, `reports/b4_*`, `reports/b5_*`, `reports/b6_*`, `reports/c2_*`, `reports/true_expanded_roadmap_*`, `reports/repo_archive_upgrade_asset_mining_true_expanded_roadmap_2026-03-28.md` | `IN_SCOPE_CORE_OR_EXPANDED` | Eligible for normal queue/promote flow |
| `QUAR-WALLETDB-SRC` | prefix | `src/walletdb/` | `OUT_OF_CORE_QUARANTINED` | Exclude from queue candidate generation and canonical lane-map references unless explicitly activated by dedicated extraction slice (e.g. `XR-008`). XR-008 currently landed as contract/evidence only and remains fail-closed for runtime promotion until namespace adaptation safety is proven. |
| `QUAR-WALLETDB-LEGACY` | glob | `scripts/legacy/walletdb_*` | `OUT_OF_CORE_QUARANTINED` | Exclude from queue candidate generation; reference-only pattern mining |
| `QUAR-PERSONAL-HEALTH` | glob | `reports/personal_health_*`, `reports/future_*health*`, `reports/openclaw_personal_health_system_lane_brief_2026-03-20.md` | `OUT_OF_CORE_QUARANTINED` | Exclude from required-core execution queue and source-of-truth map updates |
| `QUAR-YQ-TERMINAL-LEGACY` | glob | `reports/yq_terminal_*` | `OUT_OF_CORE_QUARANTINED` | Exclude from core/expanded queue decisions; historical/reference-only |
| `FAIL-CLOSED-UNKNOWN` | fallback | any path not matching explicit allowlist | `OUT_OF_CORE_QUARANTINED_UNTIL_REVIEW` | Treat as quarantined until explicit rule is added |

## Execution-safe operating requirements

1. **No queue contamination**
   - Quarantined paths MUST NOT be used to mark slices READY/DONE in core or true-expanded queue layers.

2. **No canonical-doc contamination**
   - Quarantined paths MUST NOT be promoted into canonical lane-map references without an explicit activation slice and evidence pack.

3. **Fail-closed unknown handling**
   - Any path without deterministic classification is quarantined by default.

4. **Activation escape hatch (explicit only)**
   - Quarantined material may re-enter execution scope only via a dedicated activation/extraction slice with bounded objective, evidence, and verifier output.

## Minimal operator check

Before promoting or queueing any artifact, evaluate:
1. path classification rule id,
2. classification result,
3. whether action is allow / quarantine / requires activation slice.

If uncertain, apply `FAIL-CLOSED-UNKNOWN` and stop promotion.
