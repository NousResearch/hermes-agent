# BACKLOG

## Stage 1 baseline - accepted with documented failures (2026-04-29)

Pinned upstream: 258449c468166f404ae28f97f6e4157ff72d0893
Test result: 36 failed, 17695 passed, 56 skipped
Failure clusters: WhatsApp identifier normalization, Slack DM progress
callbacks, TUI gateway protocol, ACP file reading, cache concurrency,
MCP structured content. None impact operator's intended use case
(coding agent invocation, skill execution, /escalate).

Walk-back to find green commit attempted; abandoned per operator
direction. Failures accepted as preexisting upstream baseline.
Future test runs check for NEW failures only.

Commit-prefix judgment call: the operator-provided shell snippet used
`docs(backlog)`, but the mandatory overlay prefix scheme does not include
that scope. Used `docs(claude)` instead to preserve the scheme.
