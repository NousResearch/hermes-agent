# staging/ — home-local ops files for A1b (deploy after review)

Gap **(a)** of A1 (the code-skew import oracle) lives at its real repo path
(`gateway/code_skew.py` + `tests/test_code_skew_import_oracle.py`) and ships in
this PR normally.

Gap **(b)** — the partial-restart ledger + re-drive — edits files that are
**not part of this product repo**. They are home-local fleet-ops tooling that
lives in the live `~/.hermes/` tree:

| staged path | deploys to (live) |
|---|---|
| `staging/fleet/fleet-gateway-restart.sh` | `~/.hermes/fleet/fleet-gateway-restart.sh` |
| `staging/scripts/runtime-parity-check.py` | `~/.hermes/scripts/runtime-parity-check.py` |
| `staging/tests/test_pending_restart_ledger.sh` | `~/.hermes/fleet/` (or scripts/tests) |
| `staging/tests/test_pending_restart_redrive.py` | `~/.hermes/scripts/` |

Per `git-worktree-isolation` §4b (multi-tree write-surface isolation), these
are staged **inside the worktree** and were **never written to the live tree**.
The orchestrator/human deploys `staging/` → live after review (the diffs here
are against the live files as of this branch's base).

## What (b) adds

1. **Pending-restart ledger (write side)** in `fleet-gateway-restart.sh`:
   `fgr_ledger_add` / `fgr_ledger_clear` (atomic tmp-rename, fail-closed) at
   `~/.hermes/state/pending-gateway-restart.json`. A `BUSY-SKIPPED` label is
   appended (`{target_sha, since_epoch, first_skipped, last_skipped, attempts}`);
   a label that VERIFIES is cleared. The ledger is the durable memory the
   terminal busy-skip lacked.

2. **Re-drive (act side)** in `runtime-parity-check.py::_redrive_pending_restarts`
   (folded onto the existing parity tick — no new launchd unit, R4 sprawl-safe):
   for each pending gateway that is **now idle** (`active_agents==0`, INV-1),
   restart+verify via the shared `fleet-gateway-restart.sh --only <label>`
   (reuse, not a new restarter) and clear its ledger entry. Still-busy labels
   stay pending (never amputated). Routing is edge-triggered: nothing pending
   → silent; ≥1 brought current → single #logs; a label stuck past the
   age+attempt threshold → #alerts once.

## Verify

```
bash staging/tests/test_pending_restart_ledger.sh          # ledger write/clear
/usr/bin/python3 staging/tests/test_pending_restart_redrive.py   # re-drive routing
```

Both green as of this commit. The existing home-local
`~/.hermes/scripts/test_runtime_parity_check.py` also passes unchanged against
the staged module (the re-drive no-ops on an empty ledger).
