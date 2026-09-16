# Task 6–12 verification

- Binary transfers are bounded at 10 MiB and use explicit base64 source/output, UTF-8 path handling, content type, overwrite-by-Graph-item semantics, and traversal rejection.
- Collection requests pass `$top` before serialization; search requests pass escaped filter criteria.
- To Do and Planner are separate internal capabilities and use separate Graph builder roots.
- Unsupported app/delegated combinations are represented by support status and blocked before client creation.
- Preflight and normalized results redact credentials, authorization material, headers, nested additional data, cycles, and oversized values.
- The only secret fallback in the handler is Hermes `agent.secret_scope`; test contexts may inject a value without persistence.
- `msgraph-sdk>=1.62.0,<2` is declared in both the plugin manifest and optional extra because generated builders were tested against 1.62.0.
