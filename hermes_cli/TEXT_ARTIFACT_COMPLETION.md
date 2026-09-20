# Text artifact completion contracts

At task creation, `--completion-contract 'text-artifact-v1:JSON'` declares a
required full-text handoff independently of the task body and worker output.
The JSON object has these fields:

```json
{"metadata_key":"advice_artifact","assignee":"architecture","identity":{"schema_version":1,"repository":"owner/repo","item_number":70,"source_revision_key":"source-digest","snapshot_revision":"git-commit","role":"architecture"},"max_characters":4000,"evidence_source":"hermes_completion_metadata"}
```

Identity entries are exact typed scalar comparisons. Hermes adds the generated
`task_id` to the expected identity. `metadata[metadata_key]` must include all
identity entries, that task ID, `complete: true`, `truncated: false`, and nonempty
`text` within the declared Unicode character bound. Its text must differ from the
short completion summary. An explicit `evidence_source` must match the declared
source; omission uses the declared source. Task assignee and active run profile
must match the contract as well. Completion requires a current, open run; an
unclaimed task cannot use a synthetic manual completion, even with `--force`.

The shared database completion function validates these requirements inside its
terminal write transaction. A rejected tool call returns an actionable error,
leaves the run open and does not release dependant tasks. File attachments do not
substitute for the structured artifact. CLI, tool and automatic callers share the
gate, including `--force`. A worker may correct its handoff in the same run or
block if it cannot produce the artifact.

The contract cannot be replaced through completion metadata or body edits.
Idempotent creation refuses a changed contract, including dropping it. Existing
local-only/PR tasks and historical completed runs are unchanged. This enforces
handoff structure and declared identity; it does not establish semantic quality,
independent review, implementation approval, or authorize a model retry.
