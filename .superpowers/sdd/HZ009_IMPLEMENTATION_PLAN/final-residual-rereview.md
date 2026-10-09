# HZ-009.1 final residual re-review

Scope: static, independent review of `7a31e1100a..b2ef1de73c`, limited to the two Important residual findings accepted in `final-fix-rereview.md`, plus inspection of the delta for any new Critical/Important breakage. I did not rerun a suite. The implementer's recorded focal and regression results were evaluated as supporting evidence, while the verdict below rests on the code, lock hierarchy, persistence behavior, and test contracts in the review package.

## Finding Verdicts

### Important 1 — borrowed reclaim versus normal refresh / Anthropic cross-profile lock cycles: ADDRESSED

The delta introduces one component-owned single-use refresh boundary in `agent/credential_pool.py:1501-1559`. Ordinary refresh enters it before any auth/source operation (`agent/credential_pool.py:1561-1578`), giving the relevant paths the hierarchy `local pool -> active profile -> exact owner/root -> external source`. Reentrant pool publication and persistence remain inside the outer `RLock` (`agent/credential_pool.py:1120-1151`, `agent/credential_pool.py:1786-1795`), so normal refresh no longer holds auth/source and later upgrades to the live pool.

Borrowed reclaim no longer implements an independent order: it delegates to that same transaction with the ticket's exact owning store (`agent/credential_pool_reclaim.py:305-316`) and keeps validation, refresh, durable CAS publication, and live publication inside it (`agent/credential_pool_reclaim.py:386-443`). For Anthropic, the transaction takes the active auth store, then the distinct owner/root store if needed, then the `claude_code` or `hermes_pkce` singleton (`agent/credential_pool.py:1538-1558`). Root refresh is therefore `root owner -> source`; a borrowed profile is `profile active -> root owner -> source`. No reviewed path takes the source and then waits for the owner or takes auth/source and then waits for the same live pool. Nested calls are safe reentrances: the pool uses `RLock`, while auth/source locks are path-keyed and reentrant per thread (`hermes_cli/auth.py:575-604`, `hermes_cli/auth.py:642-654`).

The deterministic contracts exercise the production refresh and ticket methods, not a parallel model: normal deferred refresh versus borrowed reclaim (`tests/agent/test_pool_revert_after_cooldown.py:980-1105`) and root Anthropic refresh versus borrowed reclaim (`tests/agent/test_pool_revert_after_cooldown.py:1193-1301`). Their endpoint calls are stubbed, but live-pool ordering, owner/source acquisition, real file publication, winner state, and timeout-sensitive interleavings are exercised. The explicit order assertion also covers `pool -> active -> owner -> source` (`tests/agent/test_pool_revert_after_cooldown.py:922-977`).

**Verdict: ADDRESSED.**

### Important 2 — tokenless `claude_code` loser cannot adopt the winner: ADDRESSED

On a newer healthy durable revision, adoption now permits the intentionally tokenless `claude_code` row, re-reads the authoritative Claude source while the enclosing source lock is held, and rejects a source token whose fingerprint disagrees with the durable row (`agent/credential_pool_reclaim.py:318-362`). It then merges the source token pair/expiry into the durable peer object, preserving the peer's winning revision/status metadata, updates only the losing live pool, and marks the ticket committed without rewriting the winner's row (`agent/credential_pool_reclaim.py:357-366`).

The winning durable write still passes through borrowed-secret sanitization (`agent/credential_pool_reclaim.py:119-149`; `agent/credential_persistence.py:13-22`, `agent/credential_persistence.py:72-116`), so the fingerprint/revision/status can identify the winner while raw Claude tokens remain absent. The two-profile contract uses two real pools and real auth/source files, runs both real reclaim commits concurrently, and proves one refresh POST, identical returned rotated pairs, a healthy revision-1 root row with the rotated access-token fingerprint, no raw tokens in that row, and no profile-local pool fork (`tests/agent/test_pool_revert_after_cooldown.py:1304-1397`).

**Verdict: ADDRESSED.**

## New Breakage

No new Critical or Important breakage found in the three-file delta. Publication remains owner-locked and CAS-checked before live replacement (`agent/credential_pool_reclaim.py:394-443`); the loser performs no competing durable write (`agent/credential_pool_reclaim.py:397-400`). The ordinary refresh behavioral change is limited to making the pool boundary outermost for single-use refreshes (`agent/credential_pool.py:1561-1602`); selection still defers those refreshes until after traversal (`agent/credential_pool.py:2048-2056`, `agent/credential_pool.py:2088-2159`). This preserves the policy-off/upstream selection path while applying the credential-safety correction at the owner component, as permitted by the binding plan.

## Out-of-Scope

- The previously deferred target-only read-only Minor remains deferred and was not reassessed.
- Compressor findings were already closed, were untouched by this delta, and were not reopened.
- No live runtime/configuration, real credential, external endpoint, merge, push, restart, or promotion assessment was performed.
- Test-suite execution was not repeated; the report's `933 passed, 4 skipped` and narrower repetitions are accepted only as recorded execution evidence.

## Verdict

**READY.** Both accepted Important residuals are **ADDRESSED**, and the reviewed delta introduces no new Critical/Important defect. This verdict applies only to the isolated HZ-009 candidate and does not authorize live promotion or runtime action.
