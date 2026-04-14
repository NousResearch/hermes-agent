# Markdown Conversion Quality Gate v1

Date: 2026-03-20  
Status: active (bounded v1)

## Purpose
Provide deterministic, fail-closed validation for source→markdown conversion artifacts before they enter production knowledge ingestion.

This gate is intentionally narrow:
1. validate packet contract,
2. verify artifact path/hash integrity,
3. enforce structural markdown quality minimums,
4. detect obvious conversion noise/corruption,
5. enforce reference-coverage floor against source text projection.

## Artifacts
- Contract: `docs/ops/markdown_conversion_quality_gate_v1.md`
- Packet schema: `docs/ops/schemas/markdown_conversion_gate_packet.schema.json`
- Packet template: `docs/ops/templates/markdown_conversion_gate_packet.template.json`
- Runtime: `scripts/markdown_conversion_quality_gate.py`
- Default decision log: `state/continuity/knowledge_ingestion/markdown_conversion_gate_decisions.jsonl`

## Gate order (deterministic)
1. `schema`
2. `artifact_integrity`
3. `markdown_structure`
4. `markdown_noise`
5. `reference_coverage`

If any gate fails, downstream gates are marked `skipped`.

## Fail-closed rules
- Missing `jsonschema` validator or unreadable schema blocks.
- Any path/hash mismatch blocks.
- Non-markdown source conversions require `source_text_artifact` for coverage checks.
- Unbalanced fenced blocks, low structural density, high control-char ratio, repeated-line inflation, or low alphabetic density block.
- Decision output is deterministic JSON (`clawd.markdown_conversion_gate.decision.v1`).

## Command
- Direct:
  - `python3 scripts/markdown_conversion_quality_gate.py --packet <packet.json> --json`
- Continuity dispatcher:
  - `bash ops/openclaw/continuity.sh markdown-gate --packet <packet.json> --json`

## Slice 24 contract extension (high-volume heuristic hardening)
Canonical bad-conversion corpus artifacts (used for deterministic heuristic-eval closeout):
- Contract schema: `docs/ops/schemas/b3_bad_conversion_golden_corpus.schema.json`
- Contract template: `docs/ops/templates/b3_bad_conversion_golden_corpus.template.json`
- Canonical fixture: `tests/fixtures/b3/bad_conversion_golden_corpus_v1.json`

Corpus rules:
- Cases must be deterministic and reproducible from inline source/markdown text.
- Each case pins expected `block_gate` + `block_reason` for fail-closed behavior.
- Minimum corpus coverage: structure failure, noise failure, and reference-coverage failure.

Verification entrypoint:
- `tests/test_markdown_conversion_quality_gate.py::test_markdown_gate_bad_conversion_golden_corpus_contract`

## Out of scope (v1)
- OCR reconstruction quality ranking.
- Semantic correctness scoring by model.
- Auto-repair or auto-rewrite of markdown artifacts.
