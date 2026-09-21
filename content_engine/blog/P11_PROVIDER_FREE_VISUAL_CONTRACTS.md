# P11 — Provider-free visual contracts

P11 now records exactly what an article image was planned to look like **before** any image generator is called. It does not change, activate, configure or test an image provider. P10 owns the later local-ComfyUI runtime decision.

## What P11 does

1. Loads the private canonical reference root at `/home/kensei/content-references`.
2. Verifies every selected reference remains physically inside that root and matches its SHA-256 record.
3. Builds one deterministic visual plan per article: hero/section layouts, shared style/palette/motif, and role-bound reviewed reference IDs.
4. Persists `visual-plan.json` and `asset-manifest.json` before the existing legacy generator is reached.
5. Fails closed: the current 26 core references are `review-required`, so none is approved for real reference-image generation.

## What P11 does not do

- It does not call, select, configure or replace Codex, OpenAI, FAL, ComfyUI, or any cloud provider.
- It does not generate a real image as P11 proof.
- It does not publish, schedule, queue, or approve an image.
- It does not merge the frozen legacy P11 branch.
- It does not promote a reference from `review-required` to `permitted`.

## Canonical reference data

`manifest.jsonl` contains 262 hash-bound records:

| Class | Records | Status |
|---|---:|---|
| `sahil_curated` | 121 | review required |
| `baoyu_derived` | 124 | review required |
| `derived` | 17 | blocked pending parent linkage |

`core-pack.jsonl` selects 26 visually reviewed candidates. A core record must match the baseline record’s ID, path, hash and provenance, and may only narrow its allowed visual roles.

## Contract modules

### `blog/reference_catalog.py`

- Enforces schema-v2 provenance fields.
- Rejects absolute paths, `..` traversal and symlinks resolving outside the root.
- Validates file hashes and core-to-baseline identity.
- Exposes reviewed core records for planning.
- Allows generation eligibility only for individually promoted `permitted` core records.

### `blog/visual_plan.py`

- Requires an explicit `ReferenceCatalog`.
- Binds each selected core record to `layout`, `style`, `composition`, `palette` or `subject`.
- Preserves reference hash and provenance in the plan.
- Requires one hero, shared article family, and distinct layouts for multi-asset articles.
- Persists deterministic JSON only through an explicit file path.

### `blog/asset_manifest.py`

- Records a `planned` state before generation, with an unbound provider/model and no output digest.
- Supports generated/QA evidence later: provider/model, requested/actual dimensions, output hash, OCR policy/result, visual-QA/rejection reasons and review status.
- Binds the visual-plan digest, exact prompt plus digest, and every reference ID/hash/provenance/role.
- Rejects credential fields and unsafe output paths.
- Persists deterministic JSON only through an explicit file path.

## Existing caller seam

`blog_illustrator.illustrate()` now performs this order:

```text
art brief → reviewed-core visual plan → planned provenance manifest → unchanged legacy generator
```

If the plan or manifest cannot be created, it stops before generation. This is an implementation contract, not a live visual-quality or provider-runtime proof.

## Tests

- Catalogue unit tests cover schema-v2 preservation, hash checking, physical symlink containment and fail-closed eligibility.
- Visual-plan unit tests cover role-bound core references, provenance, distinct layouts, deterministic persistence and invalid assignments.
- Asset-manifest unit tests cover planned/generated lifecycles, prompt/hash binding, reference provenance, credential rejection and deterministic persistence.
- The blog-illustrator integration test mocks the legacy generator and proves the plan and planned manifest exist before that boundary is reached. No test invokes a provider.

P11 is not complete or mergeable until these tests, independent review, and the wider P11 acceptance gates all pass.
