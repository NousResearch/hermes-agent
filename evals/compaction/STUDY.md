# Summary and original evidence

Does appending a few Jev-selected original evidence blocks restore facts omitted
by a frozen summary? This fixture study compares a generic handoff (**S**) with
Hermes's structured summary (**H**), using the same evidence selection for both.

[Results and figure](results/SCORECARD-2026-10-19-study.md) ·
[Runner](study.py) · [Evidence and grading](study_cases.py)

## Experimental setting

**Use strict small-budget summaries to create compression pressure.** The body
allowance is 20% of rendered evidence tokens, clamped to **64–256 tokens**.

| Item | Setting |
| --- | --- |
| Data | 14 frozen synthetic tasks: 4 primary, 10 short-source / query-only controls |
| Primary tasks | `payment`, `checkout_504`, `server_error`, `build_token` |
| Text model | `qwen3.8-max`; temperature 0, thinking disabled |
| Selector | `jev-latest`; record the resolved response version |
| Conditions | Full history; S/H alone; S/H + Jev Top 1–5 |
| Repetitions | 3 summary/ranking rounds × 2 answers per context |
| Total | **546 contexts / 1,092 answers** |
| Score | Exact JSON fields and types; equal task weights; no LLM judge |

These are 14 tasks, not 1,092 independent observations. The corpus, fixed exams,
gold answers and source bytes are hash-verified. Collection-final reports and
private metadata are excluded; no new trajectories or questions are generated.

## Paired comparison

1. **Summarize.** S receives the task and evidence blocks. H uses Hermes's
   structured summary generator, legacy serialization and the task as focus.
   Each summary gets at most three length-only attempts.
2. **Rank once per round.** Jev receives only the original task and tool-result
   bodies, ten blocks per request. It never sees summaries, exams or gold answers.
3. **Append originals.** S and H keep their entire summary unchanged and share
   the same Top N blocks. Sort by probability, break ties by ID, then restore
   source order. Keep whole blocks; no thresholds, skipped ranks or truncation.
4. **Freeze, then score.** Prepare all contexts before any scored answers.
   Failures remain in fixed denominators; average answers, rounds, then tasks.
   Report the primary tasks and controls separately.

Top 0 means summary alone. Top N counts tool-result body blocks with paths and
source offsets, not complete tool calls. Long results are grouped by related
events or text segments. Evidence grows with N; inputs are not length-matched.

<details>
<summary>Budget, generation and scoring details</summary>

- Budget tokenizer: `Qwen/Qwen3.5-9B`, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with `tokenizers==0.23.3`.
  This is a budget proxy; reported answer inputs use actual API usage.
- For attempt `a = 0..2`, S's word target is
  `max(12, int(allowance * 0.4 * 0.8**a))`. H requests
  `max(12, int(allowance * 0.8**a))` tokens and at most
  `max(18, int(requested_budget * 0.35))` words. H preserves Historical Task
  Snapshot and Critical Context. Accept only bodies within the original allowance.
- S requests up to 4,096 output tokens; answers request 2,400 and JSON object
  format. H has no separate API output cap. No answer-based retries or Full fallback.
- The complete input ceiling includes system/task/exam content, the five largest
  whole blocks, the summary allowance and 96 tokens for reserve/framing. Raise
  it before answers if serialized S conditions need more space. Check actual
  usage against the ceiling. H alone must also be shorter than paired Full;
  record shortening separately for fusion conditions.
- Check nested JSON types and ordered arrays exactly. Only the payment task's
  status-code array is order-insensitive. See [the grader](study_cases.py).

</details>

## Reproduce

Use an isolated `HERMES_HOME`. Configure `auxiliary.compression` for
`qwen3.8-max`, your custom endpoint, `api_key_env: OPENAI_API_KEY`,
`reasoning_effort: none`, `extra_body: {enable_thinking: false}` and
`max_concurrency: 4`. Supply `OPENAI_API_KEY` and `TYPESAFE_API_KEY` as secrets.
Provision `tokenizers==0.23.3` through Hermes PM in an isolated environment.

Pass the frozen corpus and budget tokenizer as local files. The corpus is not
included in this checkout; its SHA-256 is
`47fc6d4138fa54fa10236293dd121aae578c5d229c9ba586b64fcf132579dca7`.
Raw histories and provider artifacts stay outside Git.

```bash
python -m evals.compaction.study --live \
  --corpus /path/to/corpus/corpus.json \
  --tokenizer /path/to/tokenizer.json \
  --out /path/to/new/study-output

# Render the saved aggregates; only plotting requires matplotlib.
python -m evals.compaction.plot_study /path/to/study-output/aggregate.json \
  --out /path/to/summary-evidence.svg
```

The runner saves the protocol, original evidence, requests/responses, frozen
contexts, selections, every answer and grade, usage and aggregates.

This panel measures information retention under the stated conditions. It does
not measure agent task completion, isolate Jev's ranking advantage over other
selectors or equally long summaries, or establish an optimal N.

Experiment adapted from [ReMe’s context compaction study](https://github.com/agentscope-ai/ReMe/blob/main/docs/en/context_compaction.md), reproduced independently in Hermes Agent.
