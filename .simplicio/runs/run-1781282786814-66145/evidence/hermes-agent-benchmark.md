# Hermes Agent Background Validation Benchmark

Run: `run-1781282786814-66145`

## Criteria

- Time to first functional result: compare Simplicio `foreground-functional-gates.json` against the first usable Hermes functional check.
- Total time to final merge-ready proof: compare Simplicio `background-validation.json` elapsed time and final gate with the Hermes end state.
- Evidence clarity: both systems must expose unit/background status, functional status, and final gate status.
- Lock handling: compare visible Cargo target wait time and overlapping Cargo job avoidance.
- Operator experience: confirm the terminal remains usable while unit tests run.

Simplicio evidence is local-first and deterministic; the Hermes side should be measured in a matching external run before publishing numeric claims.
