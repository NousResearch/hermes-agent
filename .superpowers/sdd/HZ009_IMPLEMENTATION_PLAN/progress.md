# SDD ledger — plan: /Users/talesauxagr/.openclaw/workspace/agents-workspaces/tales-aux/shared/4_outputs/hermes-zeca-v0215-regression-recovery-20261009/HZ009_IMPLEMENTATION_PLAN.md

- Spec: `/Users/talesauxagr/.openclaw/workspace/agents-workspaces/tales-aux/shared/4_outputs/hermes-zeca-v0215-regression-recovery-20261009/SPEC.md`
- Branch: `fix/hermes-zeca-v0215-regressions-20261009`
- Starting HEAD: `8658f8eb29`
- Baseline: `tests/agent/test_compression_rotation_state.py` → `40 passed` using the parent checkout dev venv and a hermetic `/private/tmp` home.
- Graph routing: CodeGraph and CRG are unavailable in this session; repository-native inspection is the recorded degradation.

## Pre-flight interface scan

| Tasks | Producer → consumer | Finding |
|---|---|---|
| 1 → 2 | `SessionCompressionMixin.apply_compressor_route_reset` → `CompressorRouteTicket.commit` | Clean: Task 2 explicitly consumes Task 1's atomic primitive and preserves the legacy-store fallback only for direct/non-transactional calls. |
| 2 → 4 | `prepare_route_update` / `CompressorRouteTicket` → token-budget coordinator | Clean: Task 4 consumes the ticket and requires policy-on fail-closed behavior for third-party engines without it. |
| 3 → 4 | reclaim/swap tickets → coordinator sequencing/effects | Clean: Task 3 owns credential compensation; Task 4 only sequences owners and publishes success after commits. |
| 2 ↔ 3 | compressor and credential tickets share the coordinator boundary | Clean: ownership is disjoint; reverse-order abort is assigned to Task 4. |
| 1 | tests vs implementation | Clean: success and trigger-abort tests exercise one SQLite write/transaction and unrelated JSON preservation. |
| 2 | tests vs implementation | Clean: prepare/abort, exact-once commit, same-route guards, stale tickets, ABI, and legacy stores are mutually consistent. |
| 3 | tests vs implementation | Clean: route veto, resource ownership, CAS/concurrency, single-use refresh, and success ordering match the stated owner contract. |
| 4 | tests vs implementation | Clean: explicit effect sink, reverse abort, request ephemerals, policy-off parity, full regression and static checks match the spec. |

- Ruling: use `HERMES_PYTHON=/Volumes/MauxOs/Hermes/hermes-agent/.venv/bin/python` with hermetic temp/home variables for tests because the isolated worktree intentionally has no local venv and the canonical parallel runner is sandbox-blocked when spawning; direct pytest preserves the same code checkout and avoids changing dependencies. Cost if wrong: minor runner-environment drift, mitigated by the final explicit command and full static checks.

Task 1: complete (commits 8658f8e..334e811, review clean; controller verification: `42 passed`; reviewer Critical/Important/Minor: `0/0/0`).

Task 2: fix round 1/5 (1 addressed, 1 open — equal-valued durable ABA remained; commits 2afde74..edeb240).
Task 2: fix round 2/5 (0 addressed, 1 open — transaction predecessor was captured before the atomic reset; commits edeb240..8529c54).
Task 2: Ruling: extend `apply_compressor_route_reset` with optional internal `advance_route_revision` and `return_route_predecessor` keywords while preserving the original default call/return contract — cross-instance ABA and the predecessor race cannot be closed from an owner-local generation or value CAS; both marker and predecessor must participate in the same SQLite transaction — cost if wrong: internal callers could depend on a narrower inspected signature, mitigated by default-compatibility tests and unchanged `ContextCompressor.update_model` ABI.
Task 2: fix round 3/5 (1 addressed, 0 open — transaction-local predecessor closes the remaining race; commits 8529c54..13b4807).
Task 2: complete (commits 334e811..13b4807, review clean; controller verification: `54 passed` on route-ticket + Task 1 files; canonical adjacency reported `213 passed`; reviewer final verdict Approved).

Task 3: minor (deferred): `prepare_reclaim` calls `_is_sole_credential()` and therefore scans pool rows read-only instead of literally inspecting only the target row; no unrelated mutation/probe/prune occurs, but literal target-only metadata ownership remains for final triage.
Task 3: fix round 1/5 (4 addressed, 2 open — supported admin mutation paths bypassed epoch; scratch refresh discarded terminal verdicts; commits e8e9574..e6efcee).
Task 3: Ruling: add `agent/credential_pool_admin.py` to the Task 3 fix scope — a shared ABA-safe mutation epoch cannot be truthful while supported public admin reset/remove/move/add paths bypass it — cost if wrong: broader regression surface in credential-pool administration, mitigated by public-admin ABA tests and expanded 256-test regression.
Task 3: fix round 2/5 (2 addressed, 0 open — public-admin epoch and typed terminal outcomes approved; commits e6efcee..da571b1).
Task 3: complete (commits 13b4807..da571b1, review clean; controller verification: canonical `47 passed`; expanded `256 passed` and lifecycle adjacency `79 passed` reported; one deferred Minor).

Task 4: fix round 1/5 (2 addressed, 2 open — staged restore failure still commit-able; selection abort could erase interleaved selection; commits 5ee52ef..968f07b).
Task 4: fix round 2/5 (1 addressed, 1 open — pending refresh after owned selection advanced epoch and prevented compensation; commits 968f07b..fe838c5).
Task 4: fix round 3/5 (1 addressed, 0 open — refresh now precedes transaction-local selection baseline; commits fe838c5..66357f7).
Task 4: complete (commits da571b1..66357f7, review clean; controller verification: focal `98 passed`; explicit final selection `926 passed, 4 skipped`; Ruff, 2,085 compatibility pointers and diff-check clean).
Task 3: complete (commit e8e9574a62; post-commit canonical adjacency `37 passed`; expanded pool/credential regression `246 passed`; Ruff, py_compile, compatibility pointers and diff checks clean; self-review Critical/Important/Minor: `0/0/0`).
Task 3: fix round 1/5 complete (5 Important addressed, 0 Important open; commit e8e9574a62..e6efcee540; canonical adjacency `44 passed`; expanded credential/pool regression `253 passed`; additional lifecycle/OAuth/header adjacency `79 passed`; Ruff, py_compile, compatibility pointers and diff checks clean; one target-only Minor remains deferred).
Task 3: fix round 2/5 complete (2 Important addressed, 0 Important open; commit e6efcee540..da571b14b5; public-admin ABA plus manual-DEAD and singleton-removal contracts RED→GREEN; canonical adjacency `47 passed`; expanded credential/pool regression `256 passed`; lifecycle/OAuth/header adjacency `79 passed`; focused reclaim/admin/terminal `33 passed`; Ruff, py_compile, compatibility pointers and diff checks clean; one target-only Minor remains deferred).

HZ-009 final review fix pass: complete (implementation commit `73d18348631d5ba495e59e74613c909c501be00b`; 3 Important addressed, 0 Important open). Compressor owner-local commit/abort serialization, pool-before-exact-auth reclaim ordering, and borrowed single-use authoritative-source serialization were each driven RED→GREEN with deterministic thread contracts. Final focal: `32 passed`; expanded focal: `164 passed`; exact 21-file selection after self-review: `930 passed, 4 skipped`; Ruff, py_compile, compatibility pointers, and diff checks clean. One pre-existing target-only read-only Minor remains accepted/deferred. Full evidence: `final-fix-report.md`.
