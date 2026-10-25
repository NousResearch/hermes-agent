# Verified offline session maintenance (operator scripts)

These Linux/user-systemd scripts replace the unsafe pattern `trap 'hermes gateway start' EXIT`.
They are **not** automatic startup tasks or a new agent tool. Use them only from a separate
operator shell in an explicitly approved offline window, after review and with enough disk
space. Never run them from an agent whose own conversation is in the target store.

## Contract

- Resolve the target through `HERMES_HOME`, never a hardcoded default profile. Select the exact
  user-systemd unit with `--service`; an active unit must actually hold that database. Other
  holders refuse the operation. Keep all other clients/schedulers stopped throughout the window.
- Recovery supports a **standalone launch home only**. Active multiplexed/served-profile topologies
  are explicitly rejected before stop; supporting them is a separate workflow, not inferred here.
  Supply the independently approved adapter roster with repeated `--expect-platform`, or explicitly
  declare `--no-platforms`. Missing adapters or stale per-platform PID/start-time evidence fail readiness.
- Stop only a previously active unit. Unknown/failed/transitional unit states refuse. A previously
  inactive unit remains inactive. No `--force`, automatic rollback, retry, or restart on failure.
- Run holder preflight before writable connections, and repeat it before/after maintenance phases.
  A shared lock adjacent to the canonical database serializes both entry points through restart/readiness;
  it is **not** a global barrier preventing unrelated
  clients from starting. Process visibility and maintaining the offline window remain prerequisites.
- Preserve and test a compressed cold backup; use standalone snapshots for comparison. Live WAL
  sidecars are never deleted. `immutable=1` is used only for the cold standalone snapshots, not live
  or actively written databases.
- Require an independently approved UID budget before starting. Every session row, physical
  message ID and non-UID message field is protected, including reasoning/metadata and storage types.
  Every existing non-NULL UID remains on its original row. Only NULL-to-new lowercase 32-hex
  identities are allowed, with global uniqueness and explicit range/count ceilings. Existing duplicate
  UID memberships, unrelated metadata/tables, runtime epoch and non-FTS schema stay protected.
- A budget is a **verification allowance**, not an instruction to migrate that many rows. The
  normal writable-open migration may exceed it; that blocks verification/restart and requires
  operator review, never an automatic rollback or a relaxed retry.
- Require current FTS table/view/trigger definitions, cleared migration markers, unchanged nonempty
  representative search results and SQLite `quick_check=['ok']`. Exit code zero alone is insufficient.
- Persist and fsync a completed offline receipt before restart. The supervisor independently reads
  the new run's receipt and approved budget. After restart it requires current, matching gateway
  PID/state/session-store readiness and connected configured platforms. Service activation alone
  is not application readiness. Maintenance verification and recovery failure are reported separately.

## Usage (requires explicit operational approval)

From the checkout root, use the checkout's PM-prepared Python. Review a JSON budget, for example
**synthetic values, not a recommendation for any existing store**:

```json
{"before_cursor": 0, "max_id": 2000, "max_changes": 2000}
```

Optional `expected_after_cursor` / `expected_changes` enforce exact expectations in addition to
ceilings. Choose two distinct FTS queries known to match existing user/assistant history. Do not
create test conversations. Search terms and evidence may be sensitive: run files are private and
must not be committed to a repository.

```sh
# Read-only layout/holder inspection; no service effects.
HERMES_HOME=/absolute/profile python scripts/session_maintenance/offline_rebuild.py --check

# Offline execution + state-aware recovery, ONLY after explicit approval.
HERMES_HOME=/absolute/profile python scripts/session_maintenance/service_runner.py \
  --execute-offline --service hermes-gateway.service \
  --uid-budget /absolute/reviewed-budget.json --search 'alpha' --search 'beta' --expect-platform telegram
```

`$HERMES_HOME/maintenance/run-<id>/` retains a recovery journal and a unique offline run containing
compressed backup, before/after snapshots, optimizer output and receipt. Do not remove evidence on
failure. A blocked receipt is never silently converted into success: add a separate provenance-aware
supplement if a later independent review explains an intentional difference. The scripts do not
restore a stopped CLI or infer which disabled jobs should run.

The worker can run without service supervision using `--execute-offline --uid-budget ... --search ...`
only when all holders have already been stopped. It never starts/stops a service itself.

## Tests and provenance

The verification/recovery tests derive from independently exercised synthetic 41-test and 37-test
review suites (which overlap). Host-specific paths, transcripts and incident records are not included.
Additional tests cover preflight ordering, late holders, invalid/missing budgets, errors after exit zero,
receipt postconditions, invalid return types, unknown service state and stale readiness. The real CLI
integration uses a tiny temporary home, zstd and the actual optimizer; no real service is controlled.

```sh
scripts/run_tests.sh tests/scripts/session_maintenance/ tests/hermes_cli/test_sessions_maintenance_safety.py
```

References that define the constraints:
- [Session storage recovery](https://hermes-agent.nousresearch.com/docs/user-guide/session-storage-recovery)
- [Session storage and profile isolation](https://hermes-agent.nousresearch.com/docs/developer-guide/session-storage)
- [Sessions and retention](https://hermes-agent.nousresearch.com/docs/user-guide/sessions)
- [Contribution and testing guidance](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)

No incident-specific historical `quick_check` claim should be inferred from a passing synthetic test.
