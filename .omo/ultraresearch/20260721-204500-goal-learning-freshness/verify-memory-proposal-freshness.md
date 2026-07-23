# Verification — memory proposal freshness and provenance replay

## Executed evidence

```text
python -m pytest tests/tools/test_write_approval.py tests/agent/test_learning_mutations.py -q
84 passed, 1 skipped

python -m pytest tests/tools/test_memory_tool.py -q
85 passed

python -m pytest tests/tools/test_write_approval.py tests/tools/test_memory_tool.py tests/agent/test_learning_mutations.py --basetemp C:\\tmp\\hermes-verify-evidence-basetemp-20260721-210500 tests/agent/test_verification_evidence.py -q
198 passed, 1 skipped
```

## Confirmed claims

| Claim | Verdict | Evidence |
| --- | --- | --- |
| A V2 memory proposal binds its reviewed record and target revision. | CONFIRMED | `test_memory_pending_v2_binds_review_to_target_revision` |
| A stale add, replace, or remove performs no mutation and is not requeued. | CONFIRMED | parameterized `test_stale_memory_pending_is_terminal_and_never_mutates_live_state` |
| Legacy or tampered memory records are terminal no-ops. | CONFIRMED | `test_legacy_memory_pending_is_terminal_without_replay`; `test_modified_memory_pending_is_terminal_without_replay` |
| An approved background skill retains background-created attribution. | CONFIRMED | `test_background_skill_approval_preserves_original_provenance` |
| Goal outcome evidence remains green when its temporary workspace is outside the home Git ancestor. | CONFIRMED | 29 outcome-evidence tests in the combined run |

`git diff --check` and `python -m compileall` over every modified Python file
also completed successfully.  `ruff` is not installed in this worktree's
Python environment, so no lint-pass claim is made.
