# Doctrine Object Contract v1

Date: 2026-03-20  
Status: active (bounded contract; docs/schema only)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Scope: doctrine/principle objects used by philosophy/judgment substrate and life-philosophy integration lanes

## Canonical source set
1. `/home/yeqiuqiu/.openclaw/media/inbound/Philosophy_Judgment_Substrate_Design---46349a98-6bf4-4020-84ec-aad5f9a68a8e.pdf`
2. `/home/yeqiuqiu/.openclaw/media/inbound/Life-Philosophy_Integration_Doctrine---068c2b63-5ef5-4035-be7f-2cfb99cf514b.pdf`
3. `/home/yeqiuqiu/.openclaw/media/inbound/Unified_Knowledge_Integration_Blueprint---7e6d4059-b476-4e20-b9da-6708d373ea20.pdf`
4. `/home/yeqiuqiu/clawd-architect/obsvault_yq_terminal/07_Execution/Reports/2026-03-20_PDF_Batch_05_Analysis.md`
5. `/home/yeqiuqiu/clawd-architect/obsvault_yq_terminal/07_Execution/Reports/2026-03-20_PDF_Batch_06_Analysis.md`

---

## 1) Purpose and boundary
The doctrine object is the minimum durable unit for storing a principle/rule in a way that is:
- provenance-backed,
- conflict-aware,
- lane-aware,
- and executable by governance/promotion tooling.

This contract is intentionally bounded to object structure and validation semantics. It does **not** define runtime retrieval/routing code.

---

## 2) Doctrine object (normative fields)
A valid object MUST include the following categories.

### 2.1 Core principle payload
- `doctrine_id`: stable unique id (`doc_*`).
- `version`: contract/object version (v1 uses `1.0`).
- `title`: short human label.
- `principle_text`: normative text (the doctrine itself).
- `status`: `draft | active | superseded | archived`.
- `doctrine_type`: `principle | policy | rubric | heuristic | anti_pattern`.

### 2.2 Provenance and source traceability
- `source_refs[]` (min 1), each with:
  - `source_id`
  - `uri_or_path`
  - `source_type`
  - `locator` (page/chapter/section range)
  - `citation_text`
- Optional `evidence_hash` should be attached when available.

Invariant: doctrine without at least one concrete source reference is invalid for promotion.

### 2.3 Confidence and uncertainty
- `confidence.score` in `[0,1]`
- `confidence.evidence_quality`: `low | medium | high`
- `confidence.last_calibrated_at`
- `confidence.uncertainty_notes`

Invariant: low-confidence doctrine (`score < 0.5`) is allowed for draft/review, but must not auto-promote to authoritative retrieval contexts without explicit approval.

### 2.4 Domain/lane targeting
- `domain_tags[]` (min 1): semantic domains (`philosophy`, `life`, `trading`, etc.).
- `lane_scope[]` (min 1): where doctrine may be applied.

Invariant: doctrine objects are lane-scoped by default; cross-lane use requires explicit promotion/bridge path (no raw leakage).

### 2.5 Contradiction and conflict model
- `contradictions[]`: links to doctrine ids with:
  - `relation`: `contradicts | tensions_with | reframes`
  - `severity`: `low | medium | high`
  - `resolution_state`: `unresolved | in_review | resolved | accepted_tradeoff`
  - `resolution_note` (optional)

Invariant: unresolved high-severity contradiction blocks silent promotion.

### 2.6 Examples and anti-patterns
- `examples[]`: positive, context-bounded application examples.
- `anti_patterns[]`: failure modes / misuse patterns + mitigations.

Invariant: examples and anti-pattern arrays must exist (may be empty at draft stage, should be filled before broad rollout).

### 2.7 Precedence and optionality semantics
- `precedence`:
  - `level`: `constitutional | default | situational | advisory`
  - `rank`: integer ordering inside level
  - `overridable` + `override_requires_review`
- `optionality`:
  - `mode`: `mandatory | default_on | contextual | optional`
  - `activation_conditions[]`
  - `disable_conditions[]`

Invariant: precedence and optionality must both be explicit to avoid silent doctrine clashes.

### 2.8 Governance + review lifecycle
- `governance`: owner, promotion state, approver metadata, decision refs.
- `review`: cadence + `next_review_at` + optional `last_review_at`.
- `created_at`, `updated_at`, optional supersession links.

Invariant: active doctrine requires scheduled review (`next_review_at`) to remain freshness-auditable.

---

## 3) Conflict-aware application rules (v1)
1. When multiple doctrines apply, sort by `precedence.level` then `precedence.rank`.
2. If a lower-precedence doctrine is chosen over a higher one, emit explicit override rationale and decision ref.
3. If contradictions are unresolved and severity is high, route to review queue instead of auto-application.
4. If confidence is low and scenario is high-stakes, enforce verify-then-escalate behavior.

---

## 4) Artifacts in this slice
- Schema: `docs/ops/schemas/doctrine_object.schema.json`
- Template: `docs/ops/templates/doctrine_object.template.json`
- Lint/pre-promotion gate: `scripts/doctrine_object_lint.py`

These are the canonical v1 machine-readable artifacts for validation and authoring.

---

## 5) Out of scope for v1
- Full contradiction-graph service implementation.
- Retrieval-policy engine wiring.
- Promotion queue runtime automation.

Those should land in follow-on slices after contract adoption.
