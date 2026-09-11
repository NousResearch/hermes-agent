# Phase 3 authority history progress

Actor: Hermes CLI worker, not BUREAU/CLIENT. Local checkout only; no staging/commit/push/merge/activation; no live state or other worktree edits.
Base 09109fec98016ffd7fef8622223073d296c02fa4; branch feat/phase3-authority-history. Initial status only supplied brief untracked.
Complete brief, repo AGENTS.md, read-only design and relevant test instructions read.

Implemented candidate slice: owned additive journal migration; permanent board enrollment/incarnation; prospective immutable task and actor bindings; private token digest to durable run binding; token-free versioned event snapshots inserted in _append_event transaction; renewal event; deletion tombstone; strict cursor reader; immutable SQL guards; validation before write transaction and legacy migration; enrolled-board removal refusal.
NOT complete or independently reviewed. No lossless/full lifecycle acceptance claimed.

Real evidence:
- red-enrollment.log: 1 failed (missing API).
- green-enrollment.log: 1 failed (test incorrectly assumed create_task returns Task); fixed test to actual string API.
- green-enrollment-fixed.log: 1 passed.
- red-bindings-retention.log: 11 failed, 5 passed.
- green-bindings-attempt.log: 1 failed, 15 passed (test supplied nonexistent cleanup kwarg); removed kwarg after reading actual complete_task implementation.
- green-bindings.log: 16 passed.
- red-integrity-process.log: 4 failed, 21 passed.
- green-integrity-process.log: 25 passed. Real two-process exactly-one-claim and abrupt os._exit(23) before/after COMMIT verified. No real task workers/LLMs spawned.
- adjacent-boards.log: 2 failed, 54 passed. Existing test held SQLite handle open across directory removal on Windows; changed that test to connect_closing, rerun pending.
- adjacent-txn.log: 6 failed, 2 passed. Boundary fake returned None for newly required schema SELECT; now models empty unenrolled schema with []; actual validator remains enabled. Rerun pending.

Isolation launcher .phase3-evidence/run_isolated.py scrubs all non-allowlisted env before child import, preserves Windows essentials, redirects HOME/USERPROFILE/HERMES_HOME/APPDATA/LOCALAPPDATA/TEMP/TMP/TMPDIR under .phase3-evidence/state, verifies cached tempfile reset and exact candidate path. Existing installed Python 3.11.4 used read-only; pytest available; no installs. run_tests.sh inspected but unsafe unchanged because env -i drops Windows essentials and local TEMP. Per-file isolated invocations used instead. Source baseline hashes preserved in installed-before.json.

Exact continuation:
1. Rerun adjacent-boards and adjacent-txn with launcher; run other audited adjacent DB lifecycle suites (avoid real spawn or git commit tests).
2. Complete RED/GREEN lifecycle matrix: review claims, manual/stale reclaim, worker renewal/extensions/defer, block/unblock/typed dependency, archive/delete active and archived, spawn failure, crash/rate-limit/protocol violation/runtime cap, manual status writers and migration repair. Inspect other modules that write SQL directly. Do not launch real workers or signal existing PIDs.
3. Known unclosed design gates: board removal enrollment race (guard currently checks then closes before filesystem operation); capture depends on emitted events and has no guard against coherent unjournaled SQL transitions; strict JSON duplicate keys/type/integrity not comprehensively tested; historical tombstone currently active hard-delete has no run snapshot after task deletion; live actor binding validation on prior run closures should be hardened; no explicit clone/restore/new-incarnation workflow (must remain fail-closed / out of scope).
4. Add full structural/sequence/binding validation tests, pagination, unknown versions and mutation rollback across all enrolled paths. Do not backfill missing history.
5. Normalize mixed newlines introduced by patch only in touched files if necessary; syntax and git diff --check; rerun final focused tests after all edits.
6. Save PHASE3-HISTORY-RESULT.md with complete manifest, exact evidence, blockers, final source hashes, continuation and review pending.

Old execution-tree exclusion remains unresolved and outside this amendment. Journal is not automatic recovery or authority activation. Search tool failed on Windows translated paths; read_file and AST inspection used instead. One combined read-only source inspection command returned exit -1 and did not provide usable evidence.

## Final checkpoint (supersedes earlier pending steps)

PHASE3-HISTORY-RESULT.md is saved with complete manifest, exact commands, final results and explicit acceptance gaps. Read that file first for continuation; this earlier progress trail is retained, not silently rewritten as complete.

Further completed RED/GREEN work:
- lifecycle-matrix.log: 36 passed.
- red-unemitted.log: 2 failed, 36 passed; green-unemitted.log: 38 passed. Added transaction NET-state audit that refuses missing task/run journal records before COMMIT.
- red-removal-race.log: 1 failed, 38 passed; final-focused.log: 39 passed. Board removal now commits an immutable enrollment-refusal marker before filesystem action. Enrolled removal still refuses before changes.
- red-dashboard.log and red-dashboard-child.log: each 1 failed, 39 passed; final-dashboard-focused.log: 40 passed. Both direct dashboard status and dependent-child demotion use the owned writer now.
- final-boards.log: 56 passed; final-init.log: 5 passed; final-txn.log: 8 passed; final-plugin-smoke.log: 2 passed. Final selected runs report no failures or skips/xfails.
- verification.log: 10 Python files compile without pyc, diff check passes, index empty, base HEAD unchanged, three installed baseline hashes unchanged.

Acceptance remains INCOMPLETE, independent parent-Hermes review pending. Highest-priority continuation: cover remaining lifecycle paths; harden malformed record/binding reader tests; close/prove fail-closed boundaries for writes bypassing write_txn; improve active-delete terminal run disposition. No CLI activation, clone/restore workflow or old-worker exclusion is claimed. No staging/commit/push/merge/activation occurred. No profiles, memory, skills, live boards or other worktrees were edited.
