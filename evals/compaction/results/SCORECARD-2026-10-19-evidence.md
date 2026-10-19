# Frozen-summary evidence augmentation: synthetic pipeline smoke

## Result

**The native baseline and every augmentation arm scored 100% recall. No quality
gain was observed.** This validates the paired pipeline on two seed variants of
one simple synthetic template; it is not evidence of effectiveness on real tasks
or of Jev outperforming recency. The fixture explicitly asks to remember deploy
codes, which the native summaries already preserved.

## Protocol

- Source: `fixtures.synthetic_transcript(n_turns=600, seed=7|11)` from this
  repository. Each source has 1,203 messages and 379,202 chars/4 estimated tokens.
- Host model: `qwen3.8-max`; native `current` policy, full `compress(force=True)`.
  No summary-prompt or summary-length adapter. Summary, answer, query and judge
  calls use the same model via `auxiliary.compression`, with thinking disabled.
- Five generated/cached recall questions per source; Jev never sees the exam or
  gold. Qwen's JSONL question output is normalized without changing its content.
- Jev: first-party System One, requested `jev-latest`, returned `jev-1.13.0`.
- One frozen native output and one ranking per source. Baseline, Jev Top 1/3
  and recent-first Top 1/3, each both closed-book and with one recovery round:
  **20 conditions, 100 answers**. These are not 20 independent tasks.
- 2,000 estimated-token evidence ceiling; 2,000-character candidate chunks.
  Whole-unit prefix packing and source-order rendering; no threshold tuning.

| Condition | Added evidence tokens, each seed | Closed-book recall | With recovery |
|---|---:|---:|---:|
| Native baseline | 0 | 100% | 100% |
| + Jev Top 1 | 45 | 100% | 100% |
| + Jev Top 3 | 101 | 100% | 100% |
| + recent-first Top 1 | 133 | 100% | 100% |
| + recent-first Top 3 | 763 | 100% | 100% |

Evidence sizes include quotation framing and use the harness's chars/4 estimate.
The selectors share a ceiling, not equal realized sizes. Each requested count was
actually selected. All ten conditions per source referenced the same frozen
artifact hash; native summary errors were null.

| Shared preparation | Seed 7 | Seed 11 |
|---|---:|---:|
| Native summary API calls | 1 | 1 |
| Native summary API input tokens | 25,473 | 25,473 |
| Native summary API output tokens | 5,621 | 14,988 |
| Native retained-context estimate | 18,470 | 26,119 |
| Original candidate units / valid scores | 1,561 / 1,561 | 1,561 / 1,561 |
| Jev requests | 23 | 23 |
| Jev reported input / output tokens | 410,969 / 34,688 | 410,969 / 34,688 |
| Jev ranking wall time | 32.414 s | 33.721 s |
| Failed batches / invalid answers / skipped units | 0 / 0 / 0 | 0 / 0 / 0 |

Preparation is shared across the ten conditions, not billed ten times. The
provider did not report Jev dollar cost, and no summary price was available, so
cost fields are null. Each source used 125 paired exam calls (answers, judges and
recovery queries); setup probes and question generation are outside that count.
The two different summary lengths are retained as generated, not length-matched.

## Reproduction

Use an isolated `HERMES_HOME` with `auxiliary.compression` configured for
`qwen3.8-max` on your custom endpoint (`OPENAI_API_KEY`, `OPENAI_BASE_URL`), plus
`TYPESAFE_API_KEY`. Set `reasoning_effort: none` and
`extra_body: {enable_thinking: false}` on that auxiliary route.

```python
import json
from pathlib import Path
from evals.compaction.fixtures import synthetic_transcript

root = Path("/tmp/hermes-evidence-smoke")
root.mkdir(exist_ok=True)
for seed in (7, 11):
    (root / f"seed{seed}.json").write_text(json.dumps({
        "messages": synthetic_transcript(n_turns=600, seed=seed),
    }))
```

For each source, run (substitute `11` for the second seed):

```bash
python -m evals.compaction.runner \
  --transcript /tmp/hermes-evidence-smoke/seed7.json \
  --model qwen3.8-max --policies current --questions 5 \
  --evidence-top-n 1 3 --evidence-budget 2000 \
  --evidence-provider typesafe \
  --out /tmp/hermes-evidence-smoke/run7
```

Fresh questions, summaries, scores and answers can differ between runs. Freeze
and retain local artifacts for paired analysis. An effectiveness study needs a
predeclared, diverse real-session corpus and repeated trials; report all planned
cases, including baseline-perfect ones. Continuous-compaction measurements are
still needed for cache and agent task-completion claims. This smoke establishes
neither those benefits nor an optimal N. Production defaults are unchanged.
