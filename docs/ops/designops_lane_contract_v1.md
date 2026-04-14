# DesignOps Lane Contract v1 (`XD-101`)

Date: 2026-03-28  
Status: active (canonical for XD lane foundation)  
Owner: Architect  
Scope: Design-system governance for expanded downstream lanes (`XD-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XD` is the canonical DesignOps lane for machine-readable design-system operations.

It governs:
- design tokens,
- component specs,
- interaction contracts,
- and gate obligations (`G1..G6`) required before design-surface promotion.

This contract establishes lane authority boundaries and promotion/release coupling. It does **not** implement gate runtime integration (that is `XD-103`).

---

## 2) Canonical inputs and bounded outputs

### Canonical inputs
- `ops/openclaw/architecture/ui_design_edd.v1.yaml`
- `ops/openclaw/architecture/schemas/design_component_spec_frontmatter.schema.json`
- `ops/openclaw/architecture/trading_terminal_design_language.v1.yaml`
- `ops/openclaw/architecture/design_token_registry.foundation.v1.json`

### Canonical outputs for `XD-101`
- lane doctrine + authority contract (this file),
- ownership matrix artifact,
- gate dependency reference artifact,
- queue-layer slice closeout evidence.

Deferred to follow-on slices:
- `XD-102` schema/template pack versioning,
- `XD-103` runtime gate wiring into continuity/release ladder.

---

## 3) Lane boundary and non-goals

### In-boundary
- Canonical design token taxonomy and baseline token registry foundation.
- Component spec contract conformance via existing schema/tooling.
- Gate dependency declarations for later runtime/release integration.

### Out-of-boundary for `XD-101`
- New runtime producers or watchdog loops.
- Cross-domain release ladder extension (`XG-802`).
- Full token/component/interaction schema pack expansion (`XD-102`).

---

## 4) Authority and ownership model

DesignOps authority is split to prevent ambiguous mutation rights:

| Domain | Canonical authority | Mutation rights | Verification obligation |
|---|---|---|---|
| Lane doctrine / boundary rules | Architect control plane | main session only | doctrine link parity in source-of-truth map |
| Token registry foundation | DesignOps lane owner (bounded) | ticketed updates, append-only version entries | schema/lint + reference integrity |
| Component spec frontmatter schema usage | XD workers | no schema bypass; spec-only updates allowed | `validate_component_spec.sh` PASS |
| Gate dependency references (`G1..G6`) | XD + C2 governance coupling | reference updates only in doctrine/evidence until `XD-103` | dependency artifact parity check |

Authority constraints:
1. No undeclared token introduction in component specs.
2. No component promotion claim without declared gate requirements.
3. No bypass of schema/a11y gate declarations.

---

## 5) Promotion policy

Design artifacts are promoted only when all three conditions hold:

1. **Contract validity**: frontmatter/spec artifact validates against canonical contract expectations.
2. **Gate declaration completeness**: artifact declares required gates `G1_SCHEMA..G6_ALIGNMENT`.
3. **Traceability**: evidence references are published into lane closeout packet and queue-layer refs.

Fail-close policy:
- missing gate declaration, schema mismatch, or orphan token reference => promotion denied.

---

## 6) Release coupling contract (dependency reference level for `XD-101`)

`XD-101` establishes declared coupling only (runtime wiring deferred):

- Design gate stack (`G1..G6`) is normative for all design promotions.
- Release-path coupling target: `C2` release evidence ladder extension (`XG-802`) + runtime gate integration (`XD-103`).
- Continuity projection target: `state/continuity/latest/true_expanded_roadmap_queue_layer.json` for `XD-*` lifecycle truth.

Until `XD-103` lands, gate results are contractual obligations, not live-produced runtime artifacts.

---

## 7) Validation entrypoints for this slice

- `bash ops/openclaw/architecture/validate_component_spec.sh --json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 8) Closeout criteria for `XD-101`

`XD-101` is complete only when:
1. this lane contract is present and canonical,
2. ownership matrix + gate dependency refs are published as machine-readable artifacts,
3. source-of-truth map references this contract,
4. queue-layer slice `XD-101` is transitioned to `DONE` with evidence refs.
