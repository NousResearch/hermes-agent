# Pass 2 progress — local correction checkpoint

Actor: actual Hermes CLI worker, not BUREAU/CLIENT. Only C:/Users/sibag/hermes-phase3-authority-history edited. No staging, commits, push, merge, installs, live boards/profiles/config, activation, sends, worker launches, real PID signals or automatic-recovery implementation.

Read fully: PASS2, original BRIEF, RESULT, SPEC-REVIEW, repository AGENTS.md, design input (read only), isolated launcher, testing wrapper, original history implementation/tests and verification helper. Search tool failed twice on Windows path translation; targeted stdlib AST/file discovery used instead.

## Implemented and observed

- P0 credential/final invariant/writer tests: pass2-red-p0 = 8 failed, pass2-green-p0 = 8 passed. Added unchanged-final-state misleading observation cases: pass2-red-final-observation = 2 failed / 8 passed; pass2-green-final-observation = 50 passed including original history suite.
- Persistent enrolled task/run writer triggers plus connection-local UDF gate around audited write_txn. Pre/post task/run/credential validation; exact capture credential binding and last per-run observation, including unchanged final rows. This is not administrator-proof.
- Strict shared capture/read validator, duplicate-key/nonfinite JSON refusal, exact binding versions/run IDs/start types, task/run/owner relations, retained binding lookup, full-stream validation even before requested cursor, unique source IDs, exact positive sequence adjacency, INSERT OR REPLACE immutability guards including additional connections. Original observed source created_at retained as observed_at.
- pass2-red-strict = 25 failed / 1 passed. Sequence test strengthened to the review's MAX=COUNT counterexample: pass2-red-sequence = 1 failed. pass2-green-strict-attempt = 76 passed across strict/P0/original history.
- Schema SQL validation and unused retained owner/run binding validation: pass2-red-schema-binding = 2 failed / 27 passed; pass2-green-schema-binding = 29 passed.
- Real schedule/unblock (unowned, synthetic terminal, active) and prospective scheduled enrollment; enrolled active-delete refusal preserves ordinary-board deletion. Terminal delete journal faults roll back both APIs. Board-removal filesystem failure reservation tested. pass2-red-lifecycle = 5 failed / 4 passed; pass2-green-lifecycle = 49 passed including original history.
- Bounded real method matrix: claim extension, reclaim deferral, runtime timeout, stale detection, crash, rate limit, protocol violation, each success and journal-fault rollback; typed dependency wait and block-loop/unblock. Only process boundaries mocked in this new matrix; no workers/signals. pass2-lifecycle-matrix = 4 failed / 20 passed (fixture mistake: negative claim TTL did not expire a claim); corrected fixture ages rows explicitly inside audited event transaction. pass2-lifecycle-matrix-fixed = 24 passed.
- pass2-adjacent-boards = 56 passed; pass2-adjacent-init-txn = 13 passed; pass2-dashboard-adjacent = 7 passed (explicit HTTP completion/block/schedule/drag/drop/reopen/running-refusal/delete). These preceded latest schema validation edit and must be rerun finally.
- pass2-history-first = 1 failed / 39 passed: prior corruption fixture hit new writer refusal; now explicitly drops/restores writer guard to simulate administrator corruption. No assertions weakened. Original active-delete expectation amended to authorized refusal, with unchanged history/run assertion.

## Still blocked / continuation

Acceptance INCOMPLETE; independent re-review required. Need final source-bound manifest/compile/diff/installed 3-hash comparison and all final safe suites. Preserve all original evidence; use pass2 variants, not verify.py unchanged (it overwrites old files).

Remaining scrutiny: retained source provenance beyond source-ID uniqueness (including synthetic run/task provenance); enforceable insertion ownership for history/bindings; full task coverage/first history_bound consistency; copied identity support. A real supported identity-copy path was found in hermes_cli/profiles.py: create_profile(clone_all=True) copytree (1059 onward), named profile export (1932 onward), import_profile staged extraction then shutil.move (2023 onward). These have been read but NOT edited/tested. They must refuse enrolled identity reuse before activation; do not claim there is no supported clone/import path. Also audit backup/restore/snapshot modules before accepting continuity. Arbitrary filesystem copies and rollback remain unsupported/UNKNOWN. No external fencing/new-incarnation/recovery workflow authorized.

Need Python API/coverage/instance/limitations documentation. Original candidate v1 without new observed_at/guards now refuses reopen; no silent migration/backfill is provided. Must document candidate-only incompatibility, not change installed boards.
