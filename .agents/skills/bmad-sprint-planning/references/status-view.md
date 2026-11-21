# Status View

When the user wants to know where the sprint stands ("show sprint status", "where are we"), run:

```
uv run {skill-root}/scripts/sprint_plan.py status \
  --status-file {implementation_artifacts}/sprint-status.yaml --date "{date}"
```

`{date}` is `MM-DD-YYYY HH:MM`. The script computes everything: counts by status (legacy values like `drafted` mapped transparently and reported in `legacy_mapped`), risk flags (stale file, orphaned stories, in-progress epics without stories, stories waiting in review, unrecognized keys), open action items from retrospectives, and the next recommended action by fixed priority — resume in-progress → review what's in review → start the next ready or backlog story → run an open retrospective → all done. If the file is missing, the script says so — offer to run sprint planning to create it.

Render the JSON as a compact summary in `{communication_language}`: counts, risks, open action items, and the recommendation with its story key. Offer to run the recommended skill. Surface `illegal` and `unrecognized` entries and the script's `warnings` (malformed action items land there); if the user gives corrections, apply them via the fix flow rather than ad-hoc edits. No time estimates — status, risks, and next steps only.

If the script errors — malformed YAML, a hand-edited structure it can't parse, anything — do not stop at the error. Read `sprint-status.yaml` yourself, apply best judgment to give the user the same summary (counts, risks, next recommended action), tell them the deterministic path failed and why, and offer the fix flow (`fix-sprint-status.md`) so the script works next time.
