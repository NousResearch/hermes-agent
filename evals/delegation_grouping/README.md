# Delegation result-grouping A/B

Does a parent that will answer only after every subagent finishes get the results in ONE message, and does a
parent that must act on each result as it lands still get them one at a time? Run with
`delegation.independent_completions: true`, the only configuration where the model makes that choice.

- `worker.py <arm> <parent_model> <task> <rep> <out.json>`: one cell in its own process. Temp `HERMES_HOME`,
  `isolate_host`, a real parent `AIAgent` (toolsets `delegation` + `file`) and real background children on
  `DG_CHILD_MODEL`. Between turns the worker does what the CLI drain does: it claims every delivered completion
  unit and feeds each one back to the parent as its own new turn.
- `run.py <parent_model> <reps>`: the battery, resume-safe per cell; `run.py --report` prints the tables;
  `run.py --schema <tree> <on|off>` prints the `delegate_task` definition a tree builds (for token pricing).

Arms are checkouts named by `DG_<ARM>_TREE` (`DG_BASE_TREE`, `DG_CAND_TREE`); pass other names with `--arms=`.

## Battery

| task | class | expected completion units |
|---|---|---|
| `all_join_ids`, `all_slowest`, `all_summary_file`, `all_implicit_table` | the answer needs every result | 1 |
| `each_write_as_lands`, `each_stream_to_user` | every result is acted on as it lands | 3 |
| `single_control` | one subagent | 1 |

## Scoring (runtime accounting, never model prose)

- `units`: completion units the registry delivered. `delivered as expected` = `units == expected`.
- `parent turns after dispatch`: turns the parent spent on re-entering results (one per unit).
- `task success`: an oracle on the final reply or on files in the workspace (ids, numbers and counts are
  seeded per cell).
- Tokens: the parent's summed session usage plus every child's; dollars are usage x the endpoint's list prices.
- A cell with no model turn exits 3 and is rerun, never scored. Errors and wall timeouts score 0 on their arm.

Results of the validating run are under `results/`.
