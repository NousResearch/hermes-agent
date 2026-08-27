# Controller-Owned Benchmark Verification

Use this after agent-driven implementation or remediation that claims a benchmark is green.

## Verify the executable, not merely the source tree

A benchmark launcher may resolve an installed/release binary before the freshly built debug binary. This can make a fixed source tree appear broken—or make stale behavior appear current.

1. Inspect the launcher’s binary-resolution order.
2. Build the exact binary under test.
3. Pin it explicitly through the launcher’s supported environment variable or absolute path.
4. Record the resolved binary path in the receipt.
5. Re-run from the controller session.

Example pattern:

```bash
cargo build --manifest-path path/to/server/Cargo.toml
SERVER_BIN=/absolute/path/to/target/debug/server \
  python benchmark.py --launch-local
```

Do not accept an agent’s “benchmark passed” summary when the controller run used a different executable.

## Resolve receipt disagreement before completion

If an agent reports 9/9 but the controller observes 8/9:

- Treat the controller result as authoritative.
- Re-run once with the same fixed corpus and exact binary.
- Instrument the failing scenario with observed IDs/content/timestamps, not just a boolean.
- Determine whether the defect is implementation, fixture timing, stale process/binary, or benchmark logic.
- Keep the claim boundary red until the discrepancy is explained and a deterministic rerun passes.

Never average contradictory receipts or quote the better result.

## Temporal benchmark determinism

For bitemporal tests:

- Use explicit fixture timestamps rather than `now() + offset` when possible.
- Include valid-time and recorded-time cutoffs in the receipt.
- Remove/recreate the database directory for every run.
- Assert both inclusion and exclusion for current and historical views.
- Emit observed historical contents when a contradiction-preservation assertion fails.

## Broad plan discipline

When a user says “finish everything,” agent time-boxes are batch boundaries, not permission to stop after a handoff. Continue through deferred P0/P1 correctness items when tools and time remain. Final output must separate:

- shipped and controller-verified;
- implemented but not controller-verified;
- still failing;
- honestly not tested because the public surface does not expose the capability.
