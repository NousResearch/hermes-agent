# Independent corrected attempt lifecycle AR — 2026-10-08

NO FINDINGS

Reviewed exact HEAD `cddf1809b31bc67647afad97baa9380055ebccde`, corrected parent `0e85d306fc2b38db79408373308f039942c2fc4a`, full slice base `8a33891bdd58c3e0795ebfb277bcdde1003e91ea`.

## Prior findings resolved

Executed both exact Python reproduction blocks from the V1 report against current HEAD, without altering source/test files. F1 now raises `AuxiliaryExplicitCancellation`; F2 records no transport close before releasing the still-running worker. New regressions additionally cover acquire/consume owner-first expiry with both raw cancellation Event and frozen `_AuxiliaryCancellationDecision`; clearing the original Event does not erase the frozen cancellation winner. They assert no destructive cleanup on cancellation.

Finalizer now belongs to the sync request adapter, retained by the bound method dispatched through `asyncio.to_thread`. Releasing/cancelling the outer async wrapper cannot release the active adapter. The async regression proves no premature close, eventual cleanup after worker completion, and close on a thread other than the event-loop thread. The finalizer callback captures only the leaf transport and does not form a retention cycle back to its adapter owner. Sync cleanup remains covered by the existing release/GC regression.

The FD fixture only patches `hermes_cli._early_recovery.restore_interrupted_pull` to avoid unrelated installed/linked-worktree recovery during import. It does not mock `_shutdown_socket`, `force_close_tcp_sockets`, client close, or the adapter watchdog. Original socket-shutdown event/thread assertions and owner-only FD close assertions remain intact and pass independently in the per-file runner.

## Verification

Ran `python scripts/run_tests_parallel.py --file-retries 0 -j 4 --files tests/agent/test_auxiliary_scope17_attempt_lifecycle.py:tests/agent/test_auxiliary_explicit_cancellation.py:tests/agent/test_codex_aux_no_progress_timeout.py:tests/agent/test_codex_aux_timeout_fd_ownership.py -q` with the specified Hermes Python: **37 passed, 0 failed**, four fresh per-file subprocesses, no retries (12.9s). Breakdown: 13 attempt-lifecycle, 13 explicit-cancellation, 8 no-progress, 3 FD-ownership tests.

Reviewed full slice against base: exactly three owned changed tracked files. Codex wrappers bypass shared cache insertion, preserve other provider caching, and reacquire fresh clients for same-provider timeout retry in both sync and async paths. Existing no-progress retry, hard-ceiling/stall fallback and concurrent cancellation isolation cases pass. No stale outer guard added; acquisition and consumption use the one adapter-local guard. No additional leak/regression found within this reviewed slice.

`git diff --check 8a33891bdd58 HEAD` passes. Source: **8397 lines**, full slice 29 insertions / 32 deletions. Attempt tests: 293 inserted lines. FD fixture: 6 inserted lines.

SHA-256:

- `agent/auxiliary_client.py`: `9bad5c52272836a4fa20f04e182acb2a39b8325a26e06f22f76ce883a0a16f77`
- `tests/agent/test_auxiliary_scope17_attempt_lifecycle.py`: `5c654c31e2ef528f0bbbc432de6896900ec0e19d1b6b904c870584138f21ce10`
- `tests/agent/test_codex_aux_timeout_fd_ownership.py`: `4a3d7b3d026ab34c070456db3797d1ce9fc1f7e3eac654f136dff661aff9731d`

No source/test edits, provider calls, network, credentials, push, merge or deployment. Per-file runner refreshed ignored `test_durations.json` as standard runner bookkeeping. Prior report and other task artifacts preserved.

Full `python scripts/check --commit HEAD`: **11 checks, ok**. Health evaluated all three owned files against `8a33891bdd58`: **0 blocking, 0 advisory** (19.5s). Final reviewed HEAD remained `cddf1809b31bc67647afad97baa9380055ebccde`.
