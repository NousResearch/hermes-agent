# Validating run, 2026-11-09

- Arms: `base` = `origin/main` at `e0f07847b8f3`; `cand` = `84d1c1a3a7a3` (the schema change).
- Parents: `anthropic/claude-fable-5.2` and `openai/gpt-6-luna`; every child on `openai/gpt-6-luna`. Nous Portal
  inference route, `delegation.independent_completions: true`, 7 tasks x 3 reps x 2 arms x 2 parents = 84 cells.
- 84/84 cells produced a model turn; none errored or timed out, so no cell is excluded or rerun.
- Dollars are provider usage x the route's list prices (`pricing.json`), not a reconciled invoice.

Regenerate: `DG_RESULTS=evals/delegation_grouping/results/2026-11-09 python evals/delegation_grouping/run.py --report`

## parent anthropic/claude-fable-5.2

| task | arm | delivered as expected | task success | units | parent turns after dispatch | parent model calls | tokens per cell (parent + children) | $ per cell |
|---|---|---|---|---|---|---|---|---|
| all_join_ids | base | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 43,748.0 | 0.0657 |
| all_join_ids | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 42,063.3 | 0.0800 |
| all_slowest | base | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 43,676.7 | 0.0681 |
| all_slowest | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 44,198.0 | 0.0887 |
| all_summary_file | base | 3/3 | 3/3 | 1.0 | 1.0 | 4.0 | 58,383.0 | 0.1130 |
| all_summary_file | cand | 3/3 | 3/3 | 1.0 | 1.0 | 4.0 | 54,167.3 | 0.1020 |
| all_implicit_table | base | 2/3 | 3/3 | 1.7 | 1.7 | 3.7 | 52,721.3 | 0.0969 |
| all_implicit_table | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 50,988.3 | 0.0924 |
| each_write_as_lands | base | 3/3 | 3/3 | 3.0 | 3.0 | 8.0 | 91,848.3 | 0.1208 |
| each_write_as_lands | cand | 3/3 | 3/3 | 3.0 | 3.0 | 8.0 | 90,141.0 | 0.1177 |
| each_stream_to_user | base | 3/3 | 3/3 | 3.0 | 3.0 | 5.0 | 60,488.0 | 0.0778 |
| each_stream_to_user | cand | 3/3 | 3/3 | 3.0 | 3.0 | 5.0 | 60,309.7 | 0.0781 |
| single_control | base | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 26,050.7 | 0.0395 |
| single_control | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 25,953.0 | 0.0401 |
| **all tasks** | base | 20/21 | 21/21 | 1.7 | 1.7 | 4.2 | 53,845.1 | 0.0831 |
| **all tasks** | cand | 21/21 | 21/21 | 1.6 | 1.6 | 4.1 | 52,545.8 | 0.0856 |

| class | arm | delivered as expected | task success | tokens per cell | $ per cell |
|---|---|---|---|---|---|
| answer after all (1 unit expected) | base | 11/12 | 12/12 | 49,632.2 | 0.0859 |
| answer after all (1 unit expected) | cand | 12/12 | 12/12 | 47,854.2 | 0.0908 |
| act on each (3 units expected) | base | 6/6 | 6/6 | 76,168.2 | 0.0993 |
| act on each (3 units expected) | cand | 6/6 | 6/6 | 75,225.3 | 0.0979 |
| single task (control) | base | 3/3 | 3/3 | 26,050.7 | 0.0395 |
| single task (control) | cand | 3/3 | 3/3 | 25,953.0 | 0.0401 |

## parent openai/gpt-6-luna

| task | arm | delivered as expected | task success | units | parent turns after dispatch | parent model calls | tokens per cell (parent + children) | $ per cell |
|---|---|---|---|---|---|---|---|---|
| all_join_ids | base | 0/3 | 3/3 | 3.0 | 3.0 | 6.3 | 58,061.7 | 0.0037 |
| all_join_ids | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.7 | 40,891.0 | 0.0033 |
| all_slowest | base | 0/3 | 3/3 | 3.0 | 3.0 | 5.7 | 54,345.3 | 0.0036 |
| all_slowest | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.3 | 39,252.7 | 0.0033 |
| all_summary_file | base | 0/3 | 3/3 | 3.0 | 3.0 | 9.3 | 86,150.7 | 0.0049 |
| all_summary_file | cand | 3/3 | 3/3 | 1.0 | 1.0 | 7.0 | 69,382.3 | 0.0046 |
| all_implicit_table | base | 0/3 | 3/3 | 3.0 | 3.0 | 5.0 | 50,119.7 | 0.0036 |
| all_implicit_table | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 37,728.7 | 0.0034 |
| each_write_as_lands | base | 3/3 | 3/3 | 3.0 | 3.0 | 10.3 | 89,216.0 | 0.0045 |
| each_write_as_lands | cand | 3/3 | 3/3 | 3.0 | 3.0 | 10.3 | 87,847.3 | 0.0044 |
| each_stream_to_user | base | 3/3 | 3/3 | 3.0 | 3.0 | 5.0 | 50,087.3 | 0.0036 |
| each_stream_to_user | cand | 3/3 | 3/3 | 3.0 | 3.0 | 6.0 | 57,113.7 | 0.0038 |
| single_control | base | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 20,968.7 | 0.0015 |
| single_control | cand | 3/3 | 3/3 | 1.0 | 1.0 | 3.0 | 20,965.0 | 0.0015 |
| **all tasks** | base | 9/21 | 21/21 | 2.7 | 2.7 | 6.4 | 58,421.3 | 0.0036 |
| **all tasks** | cand | 21/21 | 21/21 | 1.6 | 1.6 | 5.2 | 50,454.4 | 0.0035 |

| class | arm | delivered as expected | task success | tokens per cell | $ per cell |
|---|---|---|---|---|---|
| answer after all (1 unit expected) | base | 0/12 | 12/12 | 62,169.3 | 0.0039 |
| answer after all (1 unit expected) | cand | 12/12 | 12/12 | 46,813.7 | 0.0036 |
| act on each (3 units expected) | base | 6/6 | 6/6 | 69,651.7 | 0.0040 |
| act on each (3 units expected) | cand | 6/6 | 6/6 | 72,480.5 | 0.0041 |
| single task (control) | base | 3/3 | 3/3 | 20,968.7 | 0.0015 |
| single task (control) | cand | 3/3 | 3/3 | 20,965.0 | 0.0015 |

Usage x list price over every recorded cell: $3.69
