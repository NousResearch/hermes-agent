# External-Content FTS Snapshot Benchmark — 2026-07-17

## Scope

This benchmark exercised the external-content FTS migration on a checksum-verified disposable copy of an archived Hermes production snapshot. It did **not** modify the live database, live checkout, gateway unit, or deployed code.

> **Code-version note:** The measured run preceded the independent-review remediation that made startup trigger repair mode-aware, prevented no-trigram repeat rebuilds, randomized staging names, persisted the layout marker, and moved schema discovery behind `BEGIN IMMEDIATE`. The external table definitions, canonical search document, rebuild commands, final triggers, and integrity checks measured here are unchanged. The exact post-review diff passes the repository's 524-test relevant state/repair/session-search gate, but the multi-GiB benchmark itself was not repeated.

- Source snapshot: `/home/ubuntu/.hermes/state-snapshots/20260614-122344-pre-update/state.db`
- Disposable copy: `/tmp/hermes-fts-benchmark-inline-v2.db`
- Source/copy SHA-256: `9fc2ca0a45a3cc23101ede2fbc220333ceda22dd38a46d78bfe4cccea479a9fc`
- Benchmark report: `/tmp/hermes-fts-benchmark-report-v2.json`
- SQLite used by the benchmark venv: `3.50.4`
- Separate compatibility spike passed on the VM system SQLite `3.37.2`.

## Dataset

- Sessions: `5,927`
- Messages: `187,949`
- Initial FTS mode: inline/contentful for base and trigram indexes
- Initial file size: `3,222,536,192` bytes

The harness sampled:

- 50 standard/tool-oriented FTS queries from the real vocabulary
- 14 real CJK/trigram queries from message content
- Public `SessionDB.search_messages()` projections
- Direct rowid/snippet results for both FTS tables

## Migration and integrity

- Transactional migration: `861.896 s` (`14.365 min`)
- Final FTS mode: external
- External-content `integrity-check rank=1`: passed for both tables
- Final compact clone `PRAGMA quick_check`: `ok` (`65.486 s`)
- Base rowid/snippet mismatches: `0`
- Trigram rowid/snippet mismatches: `0`
- Public API result mismatches: `0`
- Post-VACUUM base mismatches: `0`
- Post-VACUUM trigram mismatches: `0`

## Storage

| Metric | Before | After migration, before VACUUM | After VACUUM |
|---|---:|---:|---:|
| Main file | 3,222,536,192 B | 4,527,972,352 B | 2,093,027,328 B |
| Freelist | 0 B | 2,431,594,496 B | 0 B |
| Base FTS content shadow | 557,199,360 B | 0 B | 0 B |
| Trigram FTS content shadow | 557,199,360 B | 0 B | 0 B |
| Base FTS total | 717,815,808 B | 160,968,704 B | 160,841,728 B |
| Trigram FTS total | 1,712,324,608 B | 1,141,075,968 B | 1,140,678,656 B |

Physical file savings after VACUUM:

- `1,129,508,864` bytes
- `1.051937 GiB`
- `35.0503%`

The migration temporarily increased the main file by `1,305,436,160` bytes while staged indexes coexisted with old indexes. After the swap, dropped pages became `2,431,594,496` bytes of reusable freelist space. `VACUUM` took `115.493 s` (`1.925 min`) and returned that space to the filesystem.

## Warm sampled query timing

| Workload | Before median | After median | Before p95 | After p95 |
|---|---:|---:|---:|---:|
| Base FTS, 50 queries | 55.7700 ms | 19.8077 ms | 88.0409 ms | 26.1953 ms |
| Trigram FTS, 14 queries | 1.3623 ms | 0.5842 ms | 46.9513 ms | 10.5274 ms |

These latency changes combine external-content retrieval with a freshly rebuilt and defragmented FTS index and warm-cache effects. They are useful end-to-end observations, not proof that external content alone causes the speedup.

## Operational conclusions

1. The migration preserves sampled search IDs, snippets, public API results, and CJK/trigram behavior.
2. Actual reclaimed size on this snapshot is 35.05%; the current 8.69 GB live database still needs its own verified clone benchmark before claiming the earlier ~2.84 GiB projection as measured fact.
3. The operation requires a maintenance window. The dual-index rebuild held one write transaction for 14.4 minutes and produced substantial staged/WAL I/O.
4. Production execution should require the gateway and other writers to be stopped, explicit free-space preflight, an online backup, post-migration FTS integrity checks, and a separate approval for `VACUUM`.
5. Migration and `VACUUM` must remain separate operations so search/integrity can be validated before the compact rewrite.
