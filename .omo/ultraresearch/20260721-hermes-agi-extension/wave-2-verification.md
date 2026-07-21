# Wave 2 — Implementation Verification

Implemented schema v5 with a nullable `completion_contract_digest` migration.
The digest uses versioned canonical JSON and ordered subgoals, while raw
criteria remain outside the receipt. The CLI renders a 12-character digest
prefix only for new receipts.

Foreground terminal commands stale older evidence before normal result
classification; a recognized later verification records fresh evidence.

Focused suite result: 147 passed, 0 failed. `compileall`, `ruff check`, and
`git diff --check` also passed.
