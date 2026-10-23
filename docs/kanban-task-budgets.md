# Kanban task budgets

`hermes kanban create --max-iterations N --max-retries N` stores positive,
task-scoped caps. `--max-iterations` is passed to the native worker as
`--max-turns N`; an omitted value keeps the profile/global setting. A run
snapshots the iteration cap at claim time, so later edits do not rewrite its
evidence. For `goal_mode`, this cap applies to each goal turn: the finite
maximum native calls for one run is `max_iterations × effective_goal_max_turns`
(the task value or the goals default of 20), shown as `max_api_calls_per_run`.

Use `hermes kanban set-limits TASK --max-iterations N --max-retries N` to edit
an existing card. Pass `none` to clear one override. `max-retries=1` blocks on
the first terminal failure; together with a finite iteration cap this bounds a
retry batch without changing the scheduler.

`kanban_show` returns the full task body, current/tail run state, and newest
comments by default. It reports ID cursors for older comments, runs and events;
pass `before_comment_id`, `before_run_id`, or `before_event_id` to page by
structure. `detail: "full"` is the explicit complete audit-history read.
