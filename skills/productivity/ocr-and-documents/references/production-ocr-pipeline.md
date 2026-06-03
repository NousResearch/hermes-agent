# Production OCR Pipeline Pattern

Use this when processing large local document batches (hundreds of PDFs/scans) where ad-hoc `pdftotext`/`tesseract` commands are not enough.

## Durable lesson

For Spearhead-scale local OCR, implement/operate a batch pipeline with explicit safety and quality gates:

1. **Text-PDF short-circuit**: run `pdftotext -layout` first and skip OCR when selectable text is sufficient.
2. **Rendering**: render image-only PDFs via Poppler (`pdftoppm`) at ~300 DPI for final-ish OCR, lower only for quick triage.
3. **Sandbox/timeout**: run subprocesses with no shell, per-command timeout, restrictive umask, and POSIX resource limits where available.
4. **Tesseract stability**: cap OpenMP (`OMP_THREAD_LIMIT=1`, `OMP_NUM_THREADS=1`) inside sandboxed batch runs; otherwise Tesseract can fail with `libgomp: Thread creation failed` under process limits.
5. **Retry matrix**: try PSM modes such as `3`, `6`, `11`; stop once confidence passes the gate.
6. **Metrics**: derive `word_count`, `char_count`, mean/median confidence, and low-confidence word ratio from TSV.
7. **Review routing**: mark `needs_review` when mean confidence is low, low-confidence ratio is high, word count is zero, or all OCR attempts fail.
8. **Layout outputs**: persist text, TSV, hOCR, ALTO XML, and PAGE XML when downstream citation/coordinates/layout reconstruction may matter.
9. **Manifests**: write per-document `manifest.json` and batch `batch-manifest.json`; include enough metadata to resume/review without re-running OCR.
10. **Artifacts**: keep verification output and `SHA256SUMS` under `~/spearhead-execution/...` if worker workspaces may be garbage-collected.

## Current bundled helper

The bundled helper script lives at:

`skills/productivity/ocr-and-documents/scripts/production_ocr_pipeline.py`

Example:

```bash
python skills/productivity/ocr-and-documents/scripts/production_ocr_pipeline.py \
  /path/to/docs-or-one-file \
  --output-dir ~/spearhead-execution/ocr-batch-YYYYMMDD \
  --lang ces+eng \
  --dpi 300 \
  --timeout 120

python skills/productivity/ocr-and-documents/scripts/production_ocr_pipeline.py --check
```

Run its tests before trusting changes:

```bash
pytest -q tests/skills/test_production_ocr_pipeline.py
```

## Approval boundary

Do not call this “production complete” for hostile external uploads unless there is also container/bubblewrap-style isolation and a reviewed I/O/network design. The helper is a strong local baseline, not a full untrusted-upload service boundary.

Do not run representative Spearhead real-scan benchmarks over private/business documents without an explicit approval gate for corpus path, allowed outputs, retention, and redaction policy.
