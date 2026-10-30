# Integration and measurement contract

Before writing data, record the inspected application revision and preparation/read/delete entry points. Document ownership fields, idempotency keys, batch bounds, absolute entity counts, relationship constraints, rejection semantics, and how to reconcile ambiguous writes. A known disposable local fixture may use its own SQLite seeder; never generalize this into direct SQL against an unknown target.

Use stable logical indices and field-specific deterministic values. Cover repeated names, equal sort values, Unicode, and a skewed relationship distribution. Keep the same logical prefix at every stage and separate random run identity from reproducible content. A manifest can identify ranges and namespaces rather than listing thousands of records, provided cleanup/reconciliation can derive exact ownership.

Separate preparation, correctness traversal, request sampling, and UI completion. Record UTC timestamps, platform/runtime versions, app commit, configuration, stage confirmation and actual counts. The first measured request is not necessarily a cold database read: counts, seeding and correctness checks may already warm caches. State exactly what was controlled.

At concurrency one, use one warmup followed by at least ten samples, fixed query values/page sizes/viewport, and the same client implementation. Count timeout and error samples even if no response body is available. Keep first/warmup/warmed populations separate. Compare medians only with matching operations; a ratio without an SLA is an observation, not a performance failure or diagnosis.

An independent reference should filter/sort generated records without using the target SQL or response as its expected answer. Check complete page walks, final/empty pages, stable tie-breaks, exact row content and complete filtered exports. On a partial batch or timeout, query persisted ownership/counts, checkpoint the reconciled prefix and either resume idempotently or stop. Do not mark a stage complete until counts and relations match.

On interruption, persist completed samples, unfinished coverage and the reason; then attempt bounded scoped cleanup and verify sentinels. A cleanup failure is not success: retain a manifest identifying residual records. Do not erase evidence or modify recorded measurements to match expectations.
