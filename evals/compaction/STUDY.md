# Summary and original-evidence study

This fixture experiment asks whether a few Jev-selected original evidence units
can restore exact facts omitted by a frozen summary. It uses Hermes's summary
generator and auxiliary model routing, plus the existing Jev HTTP transport.
It does not measure agent task completion.

## Experimental setting

Use strict small-budget summaries to create compression pressure. The summary
body allowance is 20% of rendered evidence tokens, clamped to 64–256 tokens.
The budget tokenizer is `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`; it estimates budgets, not Qwen 3.8's
actual billing. Actual answering input sizes come from API usage.

The frozen corpus contains 14 synthetic fixture tasks: four primary tasks
(`payment`, `checkout_504`, `server_error`, `build_token`) and ten short-source
or query-only controls. Reuse the original collected histories, fixture bytes,
questions and gold answers, verified by hashes. Exclude collection-final reports
and private chat metadata. No new trajectories or generated questions are used.

- **Full**: original public history plus the fixed exam.
- **S**: a structured handoff generated from the task and evidence units.
- **H**: Hermes's structured summary generator with the task as focus, legacy
  serialization, a body budget and total word constraint. Preserve Historical
  Task Snapshot and Critical Context; omit other sections when necessary.
- **S/H + Jev Top 1–5**: the corresponding unchanged summary plus whole original
  evidence units. Top 0 is the summary alone.

S and H each get at most three length-only attempts, without answer feedback.
S's word target is `max(12, int(allowance * 0.4 * 0.8**attempt))`.
H requests `max(12, int(allowance * 0.8**attempt))` tokens and at most
`max(18, int(requested_budget * 0.35))` words. Accept only bodies within the
original allowance; a failed generation is recorded, never replaced with Full.

Evidence units are tool-result body blocks with paths and exact source offsets;
long results are grouped by related events or text segments. Top N counts these
blocks, not complete tool calls. Jev sees the original task and evidence bodies,
in batches of ten,
but never summaries, exam questions or gold answers. Rank by descending
probability, break ties by ID and restore source order. Select strict Top N
without thresholds, skipped units or partial blocks. One ranking is shared by
both summaries and all N values in a round. This is an increasing-evidence
comparison, not an equal-length comparison.

Run three independent summary/ranking rounds, then answer twice per context:
**14 × 3 × 13 = 546 contexts, 1,092 answers**. Freeze every input before any
scored answer. Qwen `qwen3.8-max` uses temperature zero and disabled thinking.
S requests up to 4,096 output tokens; answers request 2,400 and JSON object
format. H has no separate API output cap; its body allowance is checked locally.
Jev requests `jev-latest`; record the resolved service version in raw responses.

The complete answering input ceiling includes protected system/task/exam
content, the five largest whole evidence units, the summary allowance and 96
tokens for reserve/framing. Raise it before answering if serialized S conditions
need more space. Check actual input usage against that ceiling. H alone must
also be shorter than paired Full; record shortening separately for fusion arms.

Score exact JSON fields, including nested types and ordered arrays. Only the
payment task's status-code array is order-insensitive. No LLM judge or
answer-based retries. Failures stay in fixed denominators. Average trials,
rounds and then tasks equally; report primary and control cohorts separately.

## Running

Use an isolated `HERMES_HOME` with `auxiliary.compression` configured for
`qwen3.8-max`, the custom endpoint and `api_key_env: OPENAI_API_KEY`.
Set `reasoning_effort: none`, `extra_body: {enable_thinking: false}` and
`max_concurrency: 4`. Supply `OPENAI_API_KEY` and `TYPESAFE_API_KEY` as secrets.
The evaluation imports no external memory framework.

The evaluation environment also needs `tokenizers==0.23.3`; provision it through
Hermes PM in an isolated environment. Pass the frozen corpus and tokenizer as
local files. Raw histories and provider artifacts stay outside Git; the corpus
is not distributed in this checkout. Its SHA-256 is
`47fc6d4138fa54fa10236293dd121aae578c5d229c9ba586b64fcf132579dca7`.

```bash
python -m evals.compaction.study --live \
  --corpus /path/to/corpus/corpus.json \
  --tokenizer /path/to/tokenizer.json \
  --out /path/to/new/study-output
```

The output includes the protocol, source units, auxiliary request arguments, resolved routes and responses,
frozen contexts, selections, every answer and rule check, usage, and aggregates.
This small synthetic panel can establish an information-retention effect under
the stated conditions. Without other selectors or an equally long summary
control, it cannot attribute all improvement specifically to Jev ranking or
establish an optimal N.
