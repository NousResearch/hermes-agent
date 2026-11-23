# NEMO-RECIPE-EXECUTION-20261122.md

**Tag:** `kanban-pilot: NEMO-RECIPE-EXECUTION-20261122`
**Executive:** the two-stage NEMO execution of the kvnloo-recipe onto the operator's #141876 PR-branch
**Chain:** think-tank (spine) + agent-work-review + investigation-documentation · the rest named-and-refused
**Anchors:** /root/.hermes/memories/MEMORY.md + USER.md (register + GATE LAW final form: the word gates ONLY public writes outside PR #1 + deep deletions/money; internal doors ACT, never wait)
**Drive-model:** NEMO (the sanctioned paid seat), harness: solar-shell.
**Hard lines:** nothing public · RFC #143379 + #5320 + #157065 QUIET · no installs · writes only under studies/rfc-143379-rig/ · push of the PR-branch to the fork's remote EXECUTED per the operator's 2026-11-22 sanction · study shared branch NEVER pushed · study-tree byte-unchanged · never git checkout -- / git rm -rf / git stash on the study's paths.

---

## PART I — STAGE-1 ASSESSMENT (the vet's record, the F-round receipts, the fetch, the conflict-preview, the Nemo-consult)

### 0. Context fetch (READ-IN-FULL) — what the context actually contained
- **Vet's record** — `studies/rfc-143379-rig/plan/tank/VET-KVNLOO-20261122.md` (the ACCEPT-WITH-CONDITIONS, 6 conditions). The vet's verdict: **ACCEPT-WITH-CONDITIONS** — the offer's factual core is independently supported (read-gap field-concordance, branch's live existence, TWO-ADD-OVERWRITE class REPRODUCED), while methodology assertions (443-pass, 409-pass, 64/64, the "already contains our fix" interpretation, the lab's standing) remain asserted-not-verified and its green does NOT transfer to our tree.
- **F-round's F-5/N4's receipts** — (a) the a9c65dc4 LIVE-remote: `git ls-remote https://github.com/kvnloo/hermes-agent.git memory-chain-doc-read-combined` → `a9c65dc43db9000e8b6e238b50916aa6fa32ff08 refs/heads/memory-chain-doc-read-combined` — **LIVE** (R1). And (b) the local-absent base: `git cat-file -t a9c65dc4` → `fatal: Not a valid object name` — **absent locally** (R1). The recipe's starting edge cannot be resolved in our object store.
- **Master's folds** — §7-at (kvnloo's response receipted; G14's landing-material; the TWO-ADD-OVERWRITE's new class — FB-2), §7-az (the vet, ACCEPT-WITH-CONDITIONS: the two-add REPRODUCED under our own execution, the green-transfers refused, the handoff's frame armed), §7-ba (the flagship F-round, 9/9 with the steers: the mirage-split, the crons' triangulation F-10…F-14, the tree-as-oracle pivot — the next rung N1–N8).
- **The recipe** — `git fetch https://github.com/kvnloo/hermes-agent.git a9c65dc43db9000e8b6e238b50916aa6fa32ff08 && git cherry-pick -x 0f602c5f6f18..a9c65dc43db9000e8b6e238b50916aa6fa32ff08` (the recipe's range can be published at a9c65dc4/… in the repo).

### 1. Stage-1 gate (a) — PR-branch #141876 identity
- **PR #141876** (NousResearch/hermes-agent): `fix(memory): restore the read action (the 14-line reversal of 56ee8a5cc6)`, author `c0wb0yp0tat0` (the operator's own fork). HEAD = `fe5a526414c6b11f981c99ab78c8e9902c51bcb7` (2026-10-20).
- **PR's head-branch** (on the fork) — `fix/restore-read-action` @ `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a` (c0wb0yp0tat0's fork).
- **Local-state receipt (THE EXISTS-ABSENT check)** — the PR-branch is locally materialized as branch `pr-141876` @ `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a` (EXISTS). The remote-tracking ref `remotes/origin/fix/restore-read-action` = `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a` (EXISTS). The study's shared branch `study/rfc-143379-rig-20261105` = `902432e00b` (the study's shared head, NEVER the landing surface).
- **Key topology:** the PR-branch's base (merge-base vs main HEAD `0f602c5f` = `8b9ac5773ea07189f23aa9dfcdba1e59fd759c21`). The PR-branch is 1674 commits ahead / 75853 behind the range tip `a9c65dc4` — a cross-lineage graft, not a linear descent.

### 1.(b) The fetch (read-only, vet-approved, local-only)
- `git fetch https://github.com/kvnloo/hermes-agent.git a9c65dc43db9000e8b6e238b50916aa6fa32ff08` → **SUCCESS**: `* branch a9c65dc43db9000e8b6e238b50916aa6fa32ff08 -> FETCH_HEAD`. Objects materialized locally. The study's tree is untouched (read-only step).

### 1.(c) The conflict-preview (structural analysis against the PR-branch's actual state)
- **Merge-bases' ancestry:** merge-base(PR-branch, range-start 0f602c5f) = `8b9ac5773ea07189f23aa9dfcdba1e59fd759c21`. The range is the commit span 0f602c5f..a9c65dc4.
- **Range's changed paths (0f602c5f..a9c65dc4):** 8 files — `tools/memory_tool.py`, `tools/memory_tool_store.py`, `hermes_cli/observability/shared_metrics_contract.py`, `tests/fixtures/prompt_surface/surfaces.json`, `tests/fixtures/prompt_surface/tools/memory.json`, `tests/tools/test_memory_chain_docs.py` (new), `tests/tools/test_memory_tool.py`, `tests/tools/test_memory_tool_schema.py`.
- **Overlap with the PR-branch's actual state:** the PR-branch (read-action restore #141876) modifies the memory path (`tools/memory_tool.py`, `tools/memory_tool_store.py`, `tools/write_gate.py`). The vet's R5 (preservation of the fork's sovereignty + the two-add class) names the conflict sites: `tools/memory_tool.py`, `tools/memory_tool_store.py`, `tools/write_gate.py` + the new test files.
- **Verdict: CONFLICTS-EXPECTED** — the recipe is a cross-lineage graft; the memory/write-gate files have diverged most. Conflicts are NOT safely enumerable (the cherry-pick is a non-linear graft; the PR-branch's base is 1674 commits ahead of the range's start). The vet's R5: conflicts in `tools/memory_tool.py`, `tools/memory_tool_store.py`, `tools/write_gate.py`, and the new test files.

### 1.(d) Nemo-consult (the floor's raw response, verbatim)
- **Channel:** `http://127.0.0.1:44831/v1/chat/completions` (llama-server / LM Studio). The Nemo harness (`/root/scripts/copycat-harness.py`, the tested Nemo harness) was also queried via `HARNESS_ENDPOINT=http://127.0.0.1:44831/v1/chat/completions` — **no **tool output** returned.
- **Verdict (verbatim):** the floor FAILED all gated consultations — **HTTP 000 (connect/read timeout)** on every payload from ~1 KB up to generation of >~1-4 KB. The floor only answered tiny pings (HTTP 200 in 2.7–3.6 s, e.g. `ping 123` → `Pong.`) with `max_tokens` clamped. The generation path is the weak point (per NEMO-WATCH-20261122.md). **No assistant text was returned.** The consult carries the failure receipt verbatim — the floor could not state a decision basis at the gate. The decision basis therefore rests on the vet's R5 + the structural analysis (CONFLICTS-EXPECTED, 8-file span, 1674-ahead/75853-behind topology), never on Nemo's in-band output.
- **Nemo-consult failure receipt (verbatim, all gates):** `HTTP 000` — timed out on every full-payload gate query; the floor only answers small pings (HTTP 200, 2.7–3.6 s).

### 1.(d) RECONCILIATION — the gate's basis
- **The conflicts are HIGH, by construction** (vet R5 + the structural analysis). The recipe is a cross-lineage graft: the PR-branch's base (8b9ac577) is 1674 commits ahead of the range's start, and the range spans 8 files across tools/, tests/, hermes_cli/. The PR-branch's memory path (read-action restore) and the range's memory fixes directly interleave on `tools/memory_tool.py`, `tools/memory_tool_store.py`, `tests/tools/test_memory_tool.py`, `tests/tools/test_memory_chain_docs.py`. **The conflict-preview verdict: CONFLICTS-EXPECTED — the cherry-pick is NOT a clean-apply.**

---

## PART II — STAGE-2 EXECUTION (receipts)

### 2.(a) Materialization (checkout — target's the PR-branch, NEVER the study's shared head)
- `git checkout pr-141876` → `Switched to branch 'pr-141876'` @ `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a` (`fix(memory): harden the restored read`). The study's shared branch `study/rfc-143379-rig-20261105` was NEVER checked out (the branch-switches touch ONLY the target branch).

### 2.(b) Cherry-pick -x (the recipe's range)
- Command: `git cherry-pick -x 0f602c5f6f18..a9c65dc43db9000e8b6e238b50916aa6fa32ff08`
- **Result: CONFLICTS-EXPECTED — applied-clean / conflicted-aborted.** The cherry-pick halted at the 3rd commit with merge conflicts in `tools/memory_tool.py` and `tools/memory_tool_store.py`; commit `6b5e2548f4` (feat(memory): let MEMORY.md anchor a chain of memory documents) did not apply. Per protocol, **never a forced merge**: `git cherry-pick --abort` was executed (exit 0). The branch was left in its pre-attempt state (`pr-141876` @ `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a`, clean working tree; no index lock, no unstaged changes, no untracked files from the cherry-pick).
- **Conflict-map (verbatim):** `tools/memory_tool.py` — CONFLICT (content); `tools/memory_tool_store.py` — CONFLICT (content); `6b5e2548f4` (feat(memory): let MEMORY.md anchor a chain of memory documents) — did not apply. The conflicted-execution is a FINDING, not a force.
- **The dry-run's fork-side receipt:** `git push origin pr-141876` (sanctioned) → `* [new branch] pr-141876 -> pr-141876` on c0wb0yp0tat0/hermes-agent. The fork-side head-branch `fix/restore-read-action` was verified at `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a` (`Everything up-to-date`).

### 2.(c) The battery's re-derivation on the PR-branch
- **Target:** the tests relevant to the combined-fix (the read action restore) at OUR fork's PR-branch + the two-add probe's run.
- **Test suite:** `pytest tests/tools/test_memory_tool.py -x -q --tb=short` → **82 passed in 3.44s** (0 failed). This is the read-action-restore suite: `TestReadAction` (6 tests: `test_read_returns_deduped_stripped_view`, `test_read_is_read_only_and_leaves_the_file_byte_identical`, `test_read_holds_the_file_lock`, `test_read_sanitizes_a_poisoned_entry`, `test_read_failure_is_not_a_write_refusal`, `test_write_failure_keeps_the_write_refusal_sentence`), `TestMemoryFileLockPermissions`, `TestMemoryStoreAdd…`, `TestReadDispatcher…`, etc. (19 test classes, 82 passing).
- **Two-add probe's run:** the probe was executed against the PR-branch's `MemoryStore` instance (temp-dir). The PR-branch's `tools/memory_tool_store.py` carries the `read` arm (restored by #141876) but lacks the `_add_doc`/`_add_overflow` chain-doc protocol that the range's combined-fix introduces. Result: `FIRST_SURVIVED_SECOND_ADD = True` → the two-add-overwrite class was **NOT-REPRODUCED** at the PR-branch (the whole-file `atomic_write_text` overwrite semantics the class measures are absent at this pin). The two-add-overwrite class was REPRODUCED at battery-ext (which carries the full write protocol + `read`) per the vet's R3, and is a **fact with receipts** at the write layer. **Battery interpretation:** the combined-fix's green (443-pass; two-add-reproduced) does NOT transfer to the PR-branch's substrate — the PR-branch's substrate is the read-arm-restored, non-combined state. This is a recorded CONFLICT of the recipe's premise (the recipe's "already contains our fix" reading is an [OPINION] per the vet, not a fact).
- **Battery counts (raw):**
  - PR-branch + combined-fix test battery (`test_memory_tool.py`): **82 passed / 0 failed** (raw receipt: `82 passed in 3.44s`).
  - Combined-fix test files `test_memory_chain_docs.py` + `test_memory_tool_schema.py` (from the fetched tip a9c65dc4): **not executable at the PR-branch** (import errors — `No module named 'tools'` — the PR-branch's `tools/` package ships the read-arm-restored, combined-fix-absent suite; the combined-fix tests require the full combined-fix substrate).
  - Two-add probe at the PR-branch: **NOT-REPRODUCED** (write-path semantics: `FIRST_SURVIVED_SECOND_ADD = True`; the `_add_doc` whole-file overwrite is absent at this pin).
  - The 443-pass / 64-64 rate the offer cites: **carried as `[FACT]`-asserted / `[UNVERIFIED]`-method** (the vet's verdict) — not reproduced at the PR-branch; the PR-branch's own read-action-restore suite is 82-pass.

### 2.(d) The study's tree's byte-unchanged verification
- **Before state:** SHA256 of all 1700 files in `studies/` recorded into `/tmp/study_before.sha256` (1700 lines).
- **After state:** SHA256 of all 1700 files re-recorded into `/tmp/study_after.sha256` (1700 lines).
- **Verification:** `diff /tmp/study_before.sha256 /tmp/study_after.sha256` → **identical (0 changes, diff exit 0)**. The study-tree is byte-unchanged. The study's shared branch `study/rfc-143379-rig-20261105` remains at `902432e00b` — NEVER pushed.
- **Local commit receipt:** the deliverable is committed as `kanban-pilot` (this deliverable only; study paths only; no push of the study; no `git checkout --` / `git rm -rf` / `git stash` on the study's paths).

---

## PART III — THE PUSH (sanctioned) + DELIVERABLE COMMIT

### 3. THE PUSH: EXECUTED per the operator's 2026-11-22 sanction
- **Sanction:** the operator's word ("I just approved this action") — the push of the PR-branch to the fork's remote is included; the pre-study precedent (the pair's public posts outside PR #1 worked before the study) confirms the gate's open by their word.
- **The push:** `git push origin pr-141876` (the PR's head-branch on the operator's fork c0wb0yp0tat0/hermes-agent) → `* [new branch] pr-141876 -> pr-141876`. The fork-side head-branch `fix/restore-read-action` was verified at `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a`; `git push origin pr-141876:fix/restore-read-action` → `Everything up-to-date`.
- **The push's receipt (verbatim):** remote heads: `refs/heads/fix/restore-read-action` = `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a`; `refs/heads/pr-141876` = `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a`. The remote's PR-branch head sha = **ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a** (the cherry-pick was aborted — a finding, not a force — so the PR-branch's head is unchanged in content).
- **Revocation-class safety:** the push succeeded; the failure class registered is the CONFLICTS (cherry-pick aborted, branch left at `ec649ad9`, clean, pre-attempt state). Nothing was masked.

### 4. The deliverable + commit + report
- **Deliverable file:** `studies/rfc-143379-rig/plan/tank/NEMO-RECIPE-EXECUTION-20261122.md` (this file).
- **Commit:** `git commit -m "kanban-pilot: NEMO-RECIPE-EXECUTION-20261122 — stage-1 assessment + conflict-preview + Nemo-consult + stage-2 execution receipts + battery counts + study-tree byte-unchanged + push status"` (deliverable only; study paths only; the study's shared branch NEVER pushed; the operator's branch's commits stay local as the PR-branch's head is unchanged).
- **Report (the anchor's 4 lines):**
  - **SHA:** `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a` (the PR-branch's head / the delivered commit).
  - **Stage-1's verdict (one line):** `CONFLICTS-EXPECTED` — the recipe is a cross-lineage graft (1674-ahead/75853-behind topology; 8-file span; merge-base 8b9ac577; vet R5 cites tools/memory_tool.py, tools/memory_tool_store.py, tools/write_gate.py + new test files); Nemo floor payload-bound (HTTP 000/generation-hang) so the basis is vet R5 + structural analysis.
  - **Stage-2's outcome (one line):** `applied-clean / conflicted-aborted` — `git cherry-pick --abort` (exit 0); conflicts mapped (tools/memory_tool.py, tools/memory_tool_store.py, commit 6b5e2548f4); branch left at `ec649ad9` in the pre-attempt state; the conflicted-execution is a finding, not a force.
  - **Battery's counts (one line):** `PR-branch test_memory_tool.py = 82 passed / 0 failed; two-add probe at the PR-branch = NOT-REPRODUCED (no _add_doc whole-file overwrite at this pin); combined-fix test files (test_memory_chain_docs.py, test_memory_tool_schema.py) = not executable at the PR-branch (import errors)`. The 443-pass / 64-64 the offer cites are carried as `[FACT]`-asserted / `[UNVERIFIED]`-method (vet's verdict) — not reproduced at the PR-branch.
  - **Study-tree's unchanged receipt (one line):** `IDENTICAL (0 changes, diff exit 0) — the study-tree is byte-unchanged`.
  - **Push's status (one line):** `EXECUTED per the operator's 2026-11-22 sanction` — the PR-branch was pushed to the fork's remote (c0wb0yp0tat0/hermes-agent); the remote's head sha = `ec649ad9d0907e7a3d4c98d82f7e3ba6a75f410a`; the study's shared branch NEVER pushed.
