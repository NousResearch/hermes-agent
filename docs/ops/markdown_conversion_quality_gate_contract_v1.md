# Markdown Conversion Quality Gate Contract v1

Status: active (wave-6 slice-1 contract)

## Purpose

Define a deterministic, fail-closed quality gate for long-form source conversions before downstream ingestion/promotion.

This gate is intentionally bounded to **conversion-quality validation**:
- file-set integrity,
- source-map/chunk alignment,
- source-faithful structure checks,
- markdown safety/profile checks.

Out of scope in v1:
- full ingestion lifecycle,
- classification/promotion workflow,
- converter/parser redesign.

## Required conversion artifact set (v1)

A candidate packet must declare and resolve:
- `book.md` (canonical assembled markdown),
- `chunks/` directory with chunk markdown files,
- `source_map.jsonl` mapping each chunk to source locator metadata,
- source file pointer + sha256.

## Gate order (fail-closed)

1. **schema**
   - candidate shape must pass `docs/ops/schemas/markdown_conversion_candidate.schema.json`.

2. **file_set**
   - package paths resolve under package root,
   - `book.md`, `chunks/`, `source_map.jsonl` all exist,
   - chunk count inside declared bounds,
   - source path/hash check passes.

3. **source_map**
   - JSONL rows parse as objects,
   - each row has `chunk_id`, `chunk_path`, `source_locator`,
   - no duplicate chunk ids/paths,
   - all mapped chunk paths resolve in `chunks/`,
   - (if enabled) all chunk files are covered by source map.

4. **structure**
   - `book.md` has headings,
   - each chunk has headings,
   - (if enabled) first chunk heading aligns to book heading sequence in map order.

5. **markdown_profile**
   - reject control chars,
   - reject unsafe raw html tags (`script`/`iframe`/`object`/`embed`) when enabled,
   - reject `javascript:` links when enabled,
   - enforce line-length ceiling,
   - enforce heading hierarchy discipline,
   - optional frontmatter presence/required keys.

## Canonical fail reasons

- `schema_invalid`
- `gate_unavailable`
- `fileset_unready`
- `source_map_invalid`
- `structure_unfaithful`
- `markdown_profile_violation`

Any fail blocks the candidate. Later gates are marked `skipped`.

## Decision output contract

Runner emits machine-readable decision JSON:
- `decision`: `PASS` or `BLOCK`
- `final_state`: `CONVERSION_QUALITY_VERIFIED` or `BLOCKED`
- `block_gate` + `block_reason` when blocked
- per-gate status rows (`pass` / `fail` / `skipped`)
- append-only decision record metadata

## Runner

- Script: `scripts/markdown_conversion_gate_runner.py`
- Exit codes:
  - `0` = pass
  - `2` = block/fail-closed
