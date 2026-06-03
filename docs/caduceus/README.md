# Caduceus

**A native deep-planning mode for Hermes.** One switch turns Hermes into a
senior-engineer planner: it drives a visible to-do plan, raises reasoning
effort, delegates where it helps, and escalates to a deterministic multi-agent
**workflow engine** (the *Loom*) when you ask. An optional **Auto Router** sends
each delegated worker to the cheapest model that can do that subtask.

> Off by default · session-scoped · additive · fully tested.

```
/caduceus on        # plan-and-drive mode (bare /caduceus toggles)
# ...give it a multi-step task; watch the live to-do plan
# say "workflow" to fan out across subagents on the Loom
/caduceus auto on   # optional: per-task worker model routing
/caduceus status
```

## Docs in this folder

| Doc | What |
|---|---|
| [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md) | The contribution summary + surgical integration map (start here). |
| [`USER_GUIDE.md`](USER_GUIDE.md) | How to use it + the full `caduceus.*` config reference. |
| [`DESIGN.md`](DESIGN.md) | The full design record (architecture, prompt stack, the Loom, the Theater). |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | File-by-file implementation notes. |
| [`PARITY.md`](PARITY.md) | Behavioral-parity matrix vs a reference deep-planning loop. |
| [`evidence/`](evidence/) | The reference to-do-loop contract (ground truth for parity). |
| [`eval/`](eval/) | Runnable offline self-tests + a live A/B harness. |

## At a glance

- **Three layers:** the *deep-planning loop* (always-on behaviour when the mode
  is on), the *Loom* (opt-in parallel workflow engine), and the *Auto Router*
  (opt-in per-task worker model selection).
- **New, isolated modules:** `agent/caduceus.py`, `agent/auto_router.py`,
  `agent/workflow/`, `tools/workflow_tool.py`.
- **Tiny core touch-points:** every hot-path edit is additive and a no-op when
  the mode is off (see the integration map in `PR_DESCRIPTION.md`).
- **Tests:** 77 feature tests + 280 existing regression tests green; a runnable
  parity eval and router self-test under [`eval/`](eval/).

## Verify locally

```bash
# feature + engine unit tests
pytest tests/caduceus/ tests/workflow/test_loom_offline.py -q

# offline evidence (no network/keys)
python3 docs/caduceus/eval/parity_eval.py            # to-do-loop discipline rubric
python3 docs/caduceus/eval/auto_router_selftest.py   # router selection core
```
