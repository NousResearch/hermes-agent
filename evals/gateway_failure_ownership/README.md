# Gateway failure-writer ownership

Offline integration: `_handle_message` → session preparation and lease → production
`_run_agent` / TurnRunner → real AIAgent → loopback HTTP/SSE → on-disk SQLite →
gateway persistence/delivery → exception fallback.

```sh
.venv/bin/python evals/gateway_failure_ownership/probe.py "$PWD" /tmp/ownership.json
scripts/run_tests.sh -j 1 tests/gateway/test_failure_writer_ownership.py
```

The first argument selects the production checkout, so the same fixture can A/B
another checkout without editing it. The receipt records its temporary HERMES_HOME.
Inherited credentials are cleared, tools are disabled, and socket connections are
restricted to loopback. No live messaging account or paid provider is contacted.

## Fault boundaries and controls

- Real HTTP 400 from the loopback peer; successful replies use HTTP/SSE.
- Controlled AIAgent constructor failure, or voice-policy failure after persistence.
  The handler and exception writer remain production code.
- Chat A succeeds with ID `100`; chat B resumes that session and fails construction
  with a different input and the same ID. Both inputs must remain, with unmodified
  platform IDs for quote/reply lookup.
- A separate Python process commits an unrelated, nonobserved input during a keyless
  constructor failure. Both writers' inputs must survive; the gateway lease is not
  treated as universal writer authority.
- Separately accepted identical keyed/keyless inputs, identical timestamps,
  same-delivery pre-agent retry, provider failure/recovery, and healthy follow-up.
- A normal history read fails closed before agent construction. There is no longer
  a raw baseline to read or an extra keyless admission query.
- 5,000 archived 4-KiB rows remain in SQLite. The complete keyless failure handler
  must stay below 8 MiB traced peak allocation, with zero raw transcript scans.
- Durable owner markers are unique across accepted inputs and absent from provider
  wire messages. The foreign writer supplies its own independent marker.

The second invariant uses real SQLite archives and a compression tree containing
an earlier `ws_orphan_reap` sibling and a live successor. It checks both published
reroutes and map-free restart routing; root/middle ancestors and successor writes
can establish ownership, while reaped, undone, observed, foreign-marker, and
unmarked rows cannot. These are storage-boundary checks, not provider-driven
compaction.

## Receipts

The expanded full-handler matrix has 20 checkpoints (19 deliveries plus marker
propagation). Base `869228cab4a8276d3b4c78da9d9939670c47bd0f`: **7/20**; reviewed
intermediate `653cd72ef63b013798748774a2509f453b566a77`: **1/20**; fixed: **20/20**.
All fault boundaries were reached. Base and fixed runs made 10 loopback requests
and recorded zero external connection attempts. Counts are cumulative exact-row
sequence checks, not independent defect counts.

The reviewed intermediate lost chat B's input (1 row instead of 2), also lost the
keyless input beside the independent writer, and allocated 28,034,300 bytes in
the archived-history handler. The fixed run retained all 18 expected active user
rows (including the independent writer) and used 107,679 bytes in that handler.
The memory measurement excludes fixture seeding and receipt serialization.

Exactly two invariant tests are retained. Eleven targeted files passed 63 tests;
broader directory suites and CI are not claimed. No historical rows are rewritten.
Existing rows without a marker cannot establish delivery ownership, so historical
redelivery is not deduplicated by this mechanism. This is exception-path input
arbitration, not global exactly-once delivery or content deduplication.
