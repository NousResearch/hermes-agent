# Kanban Finish Gate and Handoff

Hermes Kanban can be used as a generic problem-solving loop, not just a task
list. A task is easier to trust and continue when the card says what success
means, the run records the evidence, and failed attempts leave a clear repair
trail.

## Task Card Contract

Use these sections in task bodies:

- **Goal**: the user-facing outcome.
- **Approach**: the intended path, kept short.
- **Acceptance criteria**: concrete checks that decide whether the work is done.
- **Evidence required**: proof the worker must leave behind.
- **Out of scope**: what should not be changed or pursued.

## Evidence Metadata

Store completion evidence in `task_runs.metadata` through
`kanban_complete(metadata=...)` or `hermes kanban complete --metadata`.

Recommended keys:

```json
{
  "changed_files": [],
  "commands_run": [],
  "tests": [],
  "acceptance": [],
  "artifacts": [],
  "decisions": [],
  "open_questions": [],
  "critic_review": [],
  "temp_files": [],
  "cleanup": {},
  "repair_loop": {},
  "hypothesis_tests": []
}
```

Use `[]` when a category is intentionally empty. Existing free-form metadata is
still allowed; the finish gate reports gaps but does not require a schema
migration.

## Temp File Ledger

Temporary files should be recorded as ledger objects, not bare paths:

```json
{
  "title": "session s123 parser scratch",
  "path": "C:/path/to/tmp.json",
  "session_id": "s123",
  "task_id": "t_abcd",
  "run_id": 42,
  "purpose": "compare parser output before patch",
  "created_during": "evidence collection",
  "disposition": "delete_on_verified_success",
  "cleanup_status": "deleted",
  "keep_reason": ""
}
```

Successful work should delete temporary files when safe. Failed or blocked work
should keep useful scratch files and explain why.

## Commands

- `hermes kanban check <task_id>` prints the evidence report and then appends
  the same handoff prompt produced by `prompt-next`. Use `--no-prompt-next`
  when a script needs the report only.
- `hermes kanban check <task_id> --strict` exits non-zero when required evidence
  is missing, while still printing the automatic handoff prompt by default.
- `hermes kanban prompt-next <task_id>` prints only the continuation prompt that
  the next session or worker can use to continue.
- `hermes kanban complete <task_id>` remains warning-only for compatibility.
- `hermes kanban complete <task_id> --require-evidence` blocks completion when
  required evidence is missing and prints the continuation prompt immediately.
  Use `--no-prompt-next` to suppress that prompt in automation.

## Repair Loop

When work fails, record:

- why it failed,
- which acceptance criteria remain open,
- which strategies should not be repeated,
- the next strategy,
- whether another attempt is allowed under `kanban.failure_limit`.

`hypothesis_tests` and `critic_review` can guide the next attempt, but they do
not replace real verification from `commands_run` or `tests`.
