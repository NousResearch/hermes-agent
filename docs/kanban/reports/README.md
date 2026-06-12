# Kanban worker reports

Kanban workers that need review should write a durable JSON handoff report and
then leave a short task summary containing this marker:

```text
review_required: <report_path>
<one- or two-sentence summary of what changed and why review is needed>
```

`<report_path>` should be repo-relative when the report lives in the task
workspace, or an absolute path only when the report is outside the repository.
The marker is intentionally plain text so it works in CLI summaries, comments,
dashboard cards, and future reviewer automation without a DB status migration.

## Files

- [`schemas/worker-handoff-report.schema.json`](schemas/worker-handoff-report.schema.json)
  defines the worker report contract.
- [`schemas/reviewer-verdict.schema.json`](schemas/reviewer-verdict.schema.json)
  defines the reviewer verdict contract.
- [`samples/worker-handoff-report.example.json`](samples/worker-handoff-report.example.json)
  is a complete worker handoff example.
- [`samples/reviewer-verdict.example.json`](samples/reviewer-verdict.example.json)
  is a complete reviewer verdict example.

## Worker handoff rules

1. Write the report before completing or commenting on the task.
2. Include evidence that the reviewer can reproduce locally: commands with exit
   codes, file paths, pre-existing dirty files, schema changes, residual risks,
   and the route used to do the work.
3. Mark the task summary or comment with `review_required: <report_path>` plus a
   short human-readable summary.
4. Do not encode lifecycle state in the marker. The Kanban task remains in its
   normal CLI/DB status; the marker only points reviewers to the report.

## Reviewer verdict rules

Reviewer verdict JSON must use one of these values:

- `PASS` — the handoff is accepted.
- `REQUEST_CHANGES` — specific worker changes are required before acceptance.
- `PARTIAL` — part of the work is accepted, but documented follow-up remains.
- `BLOCKED` — review cannot finish because evidence, access, or prerequisites
  are missing.
- `NEEDS_HUMAN_APPROVAL` — the next step is credentialed, destructive,
  production-facing, or otherwise requires explicit human approval.

Required automation fields are intentionally unambiguous:

- `review_task_id` is the reviewer task that produced the verdict. It must
  match the `review_task_id` argument passed to `apply_review_verdict()`.
- `source_task_id` is the worker/source task being reviewed and the task that
  `apply_review_verdict()` completes, blocks, or uses for follow-up routing.
- `task_id` is only a legacy alias for `source_task_id`; new automation should
  emit both only when they are identical.
- `safe_to_merge`, `safe_to_deploy`, `blocking_findings`,
  `non_blocking_findings`, `evidence`, and `next_action` are required so a
  transition engine can act without scraping prose.

Reviewers should write the verdict as a separate JSON file, then comment with a
short summary and the verdict path.
