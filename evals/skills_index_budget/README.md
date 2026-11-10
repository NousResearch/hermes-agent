# skills_index_budget — live A/B for the budgeted skills index

Does the model still load the right skill when that skill is listed by name only?
`skills.index_budget_percent` lets the least-used skills lose their description once the index
outgrows its share of the context window (`agent/skills_index_budget.py`). This harness rebuilds a
real library (names, descriptions, usage counters, file ages) in a throwaway `HERMES_HOME` and runs
real in-process `AIAgent`s against it, one subprocess per cell, host side effects sandboxed by
`../core_tool_deferral/sandbox.py`.

Arms run the same tree; only the config differs:

| arm | config | index |
|---|---|---|
| `full` | `index_budget_percent: 0` | every skill described (the default) |
| `tiered` | `index_budget_percent: 1`, window pinned to 200K | least-used skills names-only, up to half |

## Inputs

- `SIB_LIBRARY`: JSON list of `{"rel", "name", "desc", "mtime", "usage"}` exported from a library
  (`library.py` documents the shape). A real library is private data: keep the export outside the repo.
- `SIB_TASKS`: JSON map `task_id -> {"kind", "skill", "prompt"}`. `kind` is `scrubbed` (the prompt
  matches exactly one skill that the tiered arm lists by name only), `kept` (matches one skill both
  arms describe) or `none` (`"skill": null`, matches no skill). Pick the `scrubbed` skills from
  `measure.py`'s `*.names_only.json`, and mix names that explain themselves with names that do not.

## Running

```bash
export SIB_TREE=$PWD SIB_LIBRARY=<library.json> SIB_TASKS=<tasks.json>
export SIB_PYTHON=<the install's committed env python> SIB_RESULTS=<scratch>/results
python3 evals/skills_index_budget/measure.py $SIB_LIBRARY <scratch>/measure 200000:1 1000000:1   # what gets scrubbed
python3 evals/skills_index_budget/run.py qwen/qwen3.8-27b 3 --parallel=6
python3 evals/skills_index_budget/run.py --report
```

`measure.py` needs `PYTHONPATH=$SIB_TREE` and the Hermes env python; it writes `renders.json`
(label -> index text) for a provider token counter.

Inference defaults to the Nous Portal agent key, read fresh from `$HERMES_HOME/auth.json` for every
cell (read-only; the grant is never refreshed or copied). Set `SIB_INFERENCE_URL` +
`SIB_INFERENCE_KEY` for any other OpenAI-compatible endpoint.

## Scoring

Per cell, from the transcript's tool calls (never model prose): the expected skill was loaded with
`skill_view` (`scrubbed`, `kept`), or no skill was loaded (`none`); every other loaded skill counts
as a wrong load. Each cell also records whether the expected skill was listed and described in the
prompt the agent actually built, model calls, and usage summed over the session. The agent is capped
at 3 model calls and the terminal and file-writing tools answer with a fixed refusal on both arms:
the question is routing, not task completion, and a prompt like "free up disk space" must never
reach the host. Errored cells score 0 and stay in the
denominator; exit 3 (no model turn at all) is an infra failure and is never written.
