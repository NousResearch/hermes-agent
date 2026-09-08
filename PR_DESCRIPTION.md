# Pull Request: Tool Schema Minification, Deterministic Prefix Caching, and High-SNR Output Pruning

## Summary

This PR addresses all critical code review items and introduces production-grade latency, token efficiency, and prompt caching optimizations across the Hermes toolchain runtime:

1. **Deterministic Tool Schema Minifier (`tools/schema_minifier.py`)**:
   - Losslessly strips redundant JSON Schema metadata (`$schema`, empty `enum`/`allOf`/`anyOf`/`required` arrays) while preserving `additionalProperties: false` required for strict Structured Outputs.
   - Recursively normalizes and sorts internal dictionary keys deterministically for 100% stable prefix serialization and KV-cache reuse across inference backends.
   - Preserves original tool list ordering to avoid behavioral attention shifts across model architectures.
   - Includes robust recursion depth guards (`_depth > 30`) and circular reference detection.

2. **High-SNR Tool Output Reducer (`tools/tool_output_reducer.py`)**:
   - Intelligently parses oversized tool stdout/stderr (e.g., from `terminal`).
   - Identifies and prioritizes critical diagnostic blocks (Python tracebacks, Rust panics, Go panics, Pytest/Cargo failure summaries, critical exception lines) within a 55% signal budget.
   - Fixes slice edge cases (`tail_chars == 0`), adds ReDoS size guards before running regex on >200k char outputs, and preserves JSON payloads without disruptive marker injection.

3. **Byte/Char Consistency**:
   - Maintains byte-based truncation in `code_execution_tool.py` and character-based high-SNR reduction in `terminal_tool_result.py`.

---

## Test Plan

- [x] Unit test suite in `tests/tools/test_tool_perf_optimizations.py`:
  - `TestSchemaMinifierAndPrefixCache`: Pruning metadata, preserving `additionalProperties: false`, maintaining tool ordering, cycle/recursion safety, and deterministic serialization hashes.
  - `TestHighSNROutputReducer`: Unchanged short outputs, zero-tail slice edge case, JSON payload safety, ReDoS large input handling, multi-line traceback extraction, Rust panic extraction, and budget allocations.
- [x] Verified through CI test runner:
  ```bash
  scripts/run_tests.sh tests/tools/test_tool_perf_optimizations.py
  # Summary: 1 files, 10 tests passed, 0 failed (100% complete) in 2.5s
  ```
