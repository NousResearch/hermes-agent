# CPA-001 ORCH-001 Runtime Alignment Contract

## Inputs and baseline
- Worktree: `C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-orch-control-plane-alignment`
- Branch: `runtime/hermes-orch-control-plane-alignment`
- Candidate parent (`base`): `e9b8ae6be137abead6d19ed8a67c523f8c527096`
- Candidate HEAD: `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79`
- Candidate state before CPA docs: clean, no uncommitted edits.
- This work is isolated to the control-plane alignment worktree; no live source/config/DB/profile edits.

## Accepted serial commit set (inspected)
1. `c50e6bf76d0ffcba20655ab9f7960f961c561640` (source commit)
   - Rewritten local commit: `7fe018a19`
   - Stable patch-id: `38b39d594d335f8255664560f5a404cf53ed681b`
2. `f093334045fe33a899897bc051622c206ba6154c` (source commit)
   - Rewritten local commit: `cd63ad6b3`
   - Stable patch-id: `6812153f5e0a79f68a3ae47d71eaca1636bdb936`
3. `142ba8d19f5c3f21cea7817075482ab6ee8330fa` (source commit)
   - Rewritten local commit: `25a6db143`
   - Stable patch-id: `7bb1ba3e883675e32b8c690a894e240bc7d4fb2b`
4. `8f300d27650a3af891307f00f574c270a448b6ca` (source commit)
   - Rewritten local commit: `5ea4a27d9`
   - Stable patch-id: `a75322572a80eba461c5c38d3fb199e2f9a739ff`

## Exact source files and test files in scope
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

## Exact functions/capabilities required and covered
- `hermes_cli/kanban_db.py`
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
  - task contract CLI entry points and persistence paths used by task creation
- `tools/kanban_tools.py`
  - `_handle_create` (deterministic child-policy enforcement before insert)

## Rollback boundary
- Candidate is rooted directly at live-base `e9b8ae6be137abead6d19ed8a67c523f8c527096`.
- `e9b8...` contains no tracked source deltas relative to this branch's live-alignment baseline; only untracked evidence artifacts were observed in that environment in this lane.
- Rollback, if required, is to fast-fallback to `e9b8...` with no partial migration/state mutation introduced by these docs.

## Test evidence and exclusions
- Focused tests run: 
  - `tests/hermes_cli/test_kanban_task_contract.py`
  - `tests/hermes_cli/test_kanban_task_admission.py`
  - `tests/hermes_cli/test_kanban_notify_inheritance.py`
  - `tests/hermes_cli/test_kanban_task_inspection.py`
  - `tests/tools/test_kanban_child_policy.py`
- Results: `64 passed`
- Exclusions:
  - No full repository test suite
  - No runtime invocation, gateway restart, Telegram, remote ops, DB writes outside test fixtures

## Observed runtime/config facts (recorded for traceability)
- `default launcher` routing: `11896`
- `default python` routing: `19416`
- Python and interpreter import path observed as editable/edit-map through `PYTHONPATH` into `C:/Users/fallo/AppData/Local/hermes/hermes-agent`.
- No gateway activation or restart performed in this lane.

## Contract boundary assertions
- `create_task(contract=...)` support preserved.
- Contract migration compatibility retained with existing `tasks.contract` rows.
- Contract serialization remains normalized (`dict`-style contract field in DB)
- Child creation policy enforces deterministic rejection when `allow_child_creation=false` before child insertion.
- Notification subscription inheritance retained.
- Deterministic admission behavior retained.
