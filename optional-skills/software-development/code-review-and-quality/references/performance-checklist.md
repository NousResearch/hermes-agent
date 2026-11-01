# Performance Review Checklist

Use this checklist alongside the `code-review-and-quality` skill when reviewing a change for performance regressions. Confirm each item before a performance-sensitive change merges.

## Queries & Data Access
- [ ] No N+1 query patterns (batch or join instead of per-row lookups)
- [ ] Queries are bounded — `LIMIT` / pagination on every list endpoint and result set
- [ ] No `SELECT *` where projection suffices
- [ ] Indexes cover the changed hot-path queries

## Loops & Hot Paths
- [ ] No unbounded loops or recursion on untrusted/large inputs
- [ ] No large object/array allocation inside hot loops
- [ ] Work that can be done once is hoisted out of loops
- [ ] Streaming/chunking used for large inputs instead of loading all into memory

## Concurrency & I/O
- [ ] Blocking/synchronous calls that should be async are made async
- [ ] No accidental serial execution where parallelism is safe
- [ ] External calls have timeouts and bounded retries (no infinite retry)

## UI / Rendering
- [ ] No unnecessary re-renders (memoization, stable keys, avoided inline objects in deps)
- [ ] List/table virtualization for large datasets
- [ ] Assets are lazy-loaded / code-split where appropriate

## Caching
- [ ] Repeated expensive computation is cached with a defined invalidation policy
- [ ] Cache keys cannot collide across tenants/users
- [ ] No caching of data that must stay strongly consistent

## Measurement
- [ ] Behavior is justified by a benchmark or profiler result, not a guess
- [ ] Change does not regress p95/p99 latency or throughput on the relevant path

If a hot-path regression is found, mark it **Required** and quantify the impact (e.g. "~50ms per item") before merging.
