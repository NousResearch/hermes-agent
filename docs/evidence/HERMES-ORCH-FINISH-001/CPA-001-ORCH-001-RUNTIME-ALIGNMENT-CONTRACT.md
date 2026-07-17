# CPA-001 ORCH-001 Runtime Alignment Contract

## Inputs and identity ledger
- Worktree: `C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-orch-control-plane-alignment`
- Branch: `runtime/hermes-orch-control-plane-alignment`
- Candidate identity set:
  - live base: `e9b8ae6be137abead6d19ed8a67c523f8c527096`
  - accepted source-stack head: `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79`
  - pre-repair evidence commit: `80304e0aaae1d8d45d0bc27897d17d07132affeb`
  - final repair commit: `$(git log -1 --format=%H -- docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md)`
- This document is repaired-only evidence text; it does not include a post-repair self-hash.

## Accepted serial commit provenance (inspected and non-amended)
The four accepted serial commits were cherry-picked into the candidate with this order:
1. `c50e6bf76d0ffcba20655ab9f7960f961c561640` (source)
   - rewritten local commit: `7fe018a19`
   - stable patch-id: `38b39d594d335f8255664560f5a404cf53ed681b`
2. `f093334045fe33a899897bc051622c206ba6154c` (source)
   - rewritten local commit: `cd63ad6b3`
   - stable patch-id: `6812153f5e0a79f68a3ae47d71eaca1636bdb936`
3. `142ba8d19f5c3f21cea7817075482ab6ee8330fa` (source)
   - rewritten local commit: `25a6db143`
   - stable patch-id: `7bb1ba3e883675e32b8c690a894e240bc7d4fb2b`
4. `8f300d27650a3af891307f00f574c270a448b6ca` (source)
   - rewritten local commit: `5ea4a27d9`
   - stable patch-id: `a75322572a80eba461c5c38d3fb199e2f9a739ff`

## Change scope checked from live base to pre-repair evidence commit
Observed `git diff --name-status e9b8ae6be137abead6d19ed8a67c523f8c527096..80304e0aaae1d8d45d0bc27897d17d07132affeb`:
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-A-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-B-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-C-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-D-REPORT.md`
- `hermes_cli/kanban.py`
- `hermes_cli/kanban_db.py`
- `tools/kanban_tools.py`
- `tests/hermes_cli/test_kanban_notify_inheritance.py`
- `tests/hermes_cli/test_kanban_task_admission.py`
- `tests/hermes_cli/test_kanban_task_contract.py`
- `tests/hermes_cli/test_kanban_task_inspection.py`
- `tests/tools/test_kanban_child_policy.py`

Total observed pre-repair path count from this range: 15.

## Exact source / test files in scope
### Source files
- `hermes_cli/kanban_db.py`
- `hermes_cli/kanban.py`
- `tools/kanban_tools.py`

### Test files
- `tests/hermes_cli/test_kanban_task_contract.py`
- `tests/hermes_cli/test_kanban_task_admission.py`
- `tests/hermes_cli/test_kanban_notify_inheritance.py`
- `tests/hermes_cli/test_kanban_task_inspection.py`
- `tests/tools/test_kanban_child_policy.py`

## Exact functions/capabilities
- `hermes_cli/kanban_db.py`
  - `create_task`
  - `normalize_task_contract`
  - `serialize_task_contract`
  - `deserialize_task_contract`
  - `resolve_admission_enforce_mode`
  - `evaluate_admission`
  - `assert_child_creation_allowed`
  - `inspect_task_admission`
  - `get_task_contract`
  - `_task_has_notification_subscription`
  - `_inherit_notify_subs_unlocked`
- `hermes_cli/kanban.py`
  - task contract CLI entry points and persistence wiring for task creation contract flow
- `tools/kanban_tools.py`
  - `_handle_create` (deterministic child-policy enforcement before child insertion)

## Contract behavior invariants required
- `create_task(contract=...)` must remain supported.
- Contract migration compatibility with existing `tasks.contract` content must be preserved.
- Contract read/write path must remain normalized and schema compatible.
- Child creation rejection for `allow_child_creation=false` remains deterministic before insertion in tool path.
- Parent notification subscriptions must still inherit where legacy data is missing.
- Admission evaluation and enforcement remains deterministic and explicit.
- No behavior changes were introduced by the evidence-only repair.

## Runtime truth snapshot (timestamped)
- Timestamped operational observation: `2026-07-17T15:37:23Z`.
- `default_launcher` PID `11896` exists.
- `default python` PID `19416` exists and is a child of `11896`.
- Both PIDs report cwd `C:/Users/fallo/AppData/Local/hermes`.
- `PID 23900` is absent.
- `PID 19416` process `PYTHONPATH`: `C:/Users/fallo/AppData/Local/hermes/hermes-agent;C:/Users/fallo/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages`.
- `Editable finder` maps top-level packages to live source.
- These facts are live-router proof for current platform routing, not proof of candidate execution.
- Candidate test/runtime execution uses candidate-first `PYTHONPATH`:
  `C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-orch-control-plane-alignment`

## Verified command context for test evidence
- Builder focused lane previously recorded: `64 passed` across the required five files.
- First checker invocation under sanitized profile failed before collection because `pytest` in the profile-specific Python311 user-site was unavailable. This is a verifier-environment setup issue (tooling), not evidence of product regression.
- Prescribed next checker harness command (no claim of independent pass yet):
  ```bash
  PYTHONPATH='C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-orch-control-plane-alignment;C:/Users/fallo/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages' C:/Python314/python.exe -m pytest \
    tests/hermes_cli/test_kanban_task_contract.py \
    tests/hermes_cli/test_kanban_task_admission.py \
    tests/hermes_cli/test_kanban_notify_inheritance.py \
    tests/hermes_cli/test_kanban_task_inspection.py \
    tests/tools/test_kanban_child_policy.py
  ```

## SHA-256 evidence snapshots for current pre-repair CPA documents
- `CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md`: `d417cdf84eb468cd82742849522605b292e56a072f9ff76ac01db1b548f63318`
- `CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md`: `3f010edae44c0ea8c0686b6ce5ca8d0eadd00765e58b346d974d6d8f8529da88`
- `CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`: `2bbe344e324527334df914619f4211a83c349cf47c1aec22bd9e1ca60eb61c47`

These are pre-repair evidence file hashes only. Post-commit hashes must be re-supplied by the final handoff output, not embedded here.

## Additional operational disclosures
- default and builder-grok `kanban.auto_decompose` values are `false`.
- all-board read-only title query found no matching cards with tags/names:
  `CPA`, `HKNIR`, `HOF-020A`, `HOF-020-R1`, `HOF-029`.
- live tracked-source check was clean while only untracked evidence drift existed in live worktree snapshot:
  - `docs/evidence/HERMES-ORCH-001-REPORT.md`
  - `docs/evidence/HERMES-ORCH-FINISH-001-HOF-000-RECOVERY.md`
  - `docs/evidence/HERMES-ORCH-FINISH-001-RUNTIME-INVENTORY.md`
  - `docs/evidence/HERMES-ORCH-FINISH-001/`
- no live source/config DB edits, gateway restarts, Telegram canary, remote operation, or activation performed by this repair pass.
