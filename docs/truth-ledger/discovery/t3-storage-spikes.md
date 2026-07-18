# T3 storage/concurrency spikes (throwaway)

Date: 2026-07-17
Task: t_682fbfbc
Workspace: /Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2

## Command

```bash
python - <<'PY'
# spike script (atomic replace, flock append, sqlite UNIQUE concurrency, partial-tail scan)
PY
```

## Output summary

- Temp root: `/var/folders/s1/br7_h42s1vjchrf5_n71plzw0000gp/T/t3_spike_e2y4grxo`
- `atomic_replace_cycles=100`, final file always parseable JSON line with trailing newline.
- `locked_append_lines=1600`, `locked_append_unique_pairs=1600` (8 threads x 200 writes).
- `sqlite_attempts=1000` inserting the same idempotency key concurrently resulted in `sqlite_same_key_rows=1`.
- Partial-tail scan over malformed JSONL (`{"a":1}\n{"b":2}\n{"c":3`) recovered `valid_prefix=2`, quarantined suffix `1` line.

## Interpretation

These spikes support the T3 design choices:

1. Temp-file + `fsync` + `os.replace` is suitable for atomic spool record creation.
2. File lock serialization can protect JSONL append on POSIX.
3. SQLite UNIQUE constraints enforce logical idempotency under concurrent retries.
4. Prefix-only recovery can preserve immutable history while isolating malformed tail fragments.
