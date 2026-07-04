# feat(memory): Float16 Semantic Memory Provider

## Summary

New memory provider that stores float16-compressed embeddings instead of
FTS5 keyword search, achieving **2x smaller storage with zero retrieval
quality loss** compared to full-precision embeddings, and **2.2x better
recall** than the current FTS5 default.

## Benchmark Results (50 conversation pairs, 150 queries)

| Provider | Recall@5 | MRR | Semantic Gap@5 | Storage/turn | Query |
|---|---|---|---|---|---|
| FTS5 (current) | 0.42 | 0.20 | 0.00 | 1.6KB | 0.1ms |
| FP32 Embeddings | 0.94 | 0.44 | 0.86 | 1.5KB | 7.1ms |
| **Float16 (this PR)** | **0.94** | **0.44** | **0.86** | **0.8KB** | **7.5ms** |

**Semantic Gap Recall@5** measures retrieval when query vocabulary has
zero overlap with stored text — FTS5 gets 0.00, our provider gets 0.86.

The float16 compression preserves ranking at Kendall tau 0.99 vs FP32
(verified experimentally). Q4 quantization was also evaluated but
degrades ranking at scale (tau ~0.3 at N=50), so float16 is the default.

## Architecture

```
User turn → MemoryManager.sync_turn()
                ↓
         KVMemoryProvider
         ├── capture.py   → sentence-transformers (80MB, CPU, 5ms)
         ├── storage.py   → SQLite with WAL mode
         ├── retrieval.py → cosine similarity + temporal decay
         └── quantize.py  → Q4 optional (aggressive mode)
```

Pure plugin — zero changes to core Hermes. Only dependency is
`sentence-transformers` (pip install).

## Files changed

```
plugins/memory/kv_memory/        9 source files  (~1900 lines)
plugins/memory/kv_memory/tests/  4 test files   (~680 lines, 60 tests)
benchmarks/memory/               3 files        (~690 lines)
validate_kv_retrieval.py         1 research script (~1055 lines)
```

## How to use

```bash
# Install dependency
pip install sentence-transformers

# Configure
hermes memory setup kv-memory
# Or manually add to config.yaml:
#   memory:
#     provider: kv-memory
#   plugins:
#     kv-memory:
#       embedding_backend: auto
#       top_k: 5
```

## Related

- Closes #44075 (semantic session search)
- Related to #44093 (sqlite-vec hybrid search)
- Related to #29902 (memory provider as search backend)
