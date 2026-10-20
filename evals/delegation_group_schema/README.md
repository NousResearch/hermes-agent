# Delegation grouping schema receipts

`probe.py TREE OUTPUT [--independent]` assembles real eager CLI tool definitions
in a fresh, credential-free temporary home with networking forbidden. Run it
in separate interpreters against base and fix with the same arguments.
Requires `tiktoken` (`o200k_base`); counts use compact OpenAI function JSON.
This measures schema footprint, not model quality or billed savings.

Base: `b2aa855b626ff8688eb34b95c60ee8b6a4af3679`.

| Policy | Base delegate tokens | Fixed delegate tokens | Change |
| --- | ---: | ---: | ---: |
| Default off | 927 | 826 | -101 |
| Explicit on | 927 | 913 | -14 |

Only `delegate_task` changed in either same-config comparison. Total eager
CLI schemas were 7586 → 7485 (off), 7586 → 7572 (on). The field remains
available when enabled; its parameter schema is unchanged. Repeated assembly
is byte-stable. The two invariants also check static/previous schema immutability,
legacy task normalization, and grouped/ungrouped delivery partitioning.
The default-off exposure assertion failed on base before implementation.

## Real-model N=1 child smoke

Pinned to configured Nous `google/gemini-3.7-flash`; no provider fallback.
A separate temporary home held only copied Nous authentication and minimal
explicit configuration (no personal skills/memory). One forced parent tool call
using the real default-off registry schema produced a one-task array without
`group`. The actual delegation handler then built and ran one real AIAgent child,
which completed in one API call with `GROUP_GATE_OK`. Parent tool bytes were
unchanged across child execution. This is a schema-acceptance/execution smoke,
not a comparative quality evaluation or proof of asynchronous delivery timing.

Before inference, the live catalog was read and an approximately $0.016 budget
estimated for two calls (20k input + 1024 output allowance).

| Inference | Input | Output | Reasoning (included in output) | API-reported `usage.cost` |
| --- | ---: | ---: | ---: | ---: |
| Parent schema call | 796 | 104 | 75 | $0.000987 |
| Child | 708 | 42 | 37 | $0.0006885 |
| Total | 1504 | 146 | 112 | $0.0016755 |

Parent request ID: `gen-1788889571-fBLF8eStu3UAevfGtvgD`.
The raw provider usage exposes the same values as `upstream_inference_cost`;
these are API-reported costs, not an independently reconciled Portal invoice.
The child result's local estimator instead reported `$0.000551` with
`cost_status: estimated`; do not substitute that estimate for the raw receipt.
Local full receipt: `/tmp/delegation-group-live-receipt.json`.
