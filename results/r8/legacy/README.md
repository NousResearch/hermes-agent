# Legacy historical tests — NOT current contract

`test_holographic_cognitive.HISTORICAL.py`
sha256: `74b840a100200e1da4db5175263bdf3ff0c60ab46abd6288223744a27d357898`

- Byte-identical recovery from `stash@{1}` (moved here from
  `tests/plugins/memory/`; hash verified before/after the move).
- 50 tests: 34 pass against the current tree, 16 fail with `AttributeError`
  only — they call a pre-R1 API (`MemoryStore.get_fact`/`lineage`,
  `FactRetriever.cognitive_search`, `HolographicMemoryProvider.run_maintenance`
  /`llm_calls`) intentionally replaced by the R-line design.
- HISTORICAL evidence. Do NOT move back into the active suite. Do NOT edit to
  make it pass. Per-test classification: `../reports/test_migration_matrix.md`.
- The `.HISTORICAL.py` suffix keeps pytest from collecting it.
