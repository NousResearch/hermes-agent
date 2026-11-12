# session_search Schema A/B Eval

Live tool-use A/B harness measuring whether changes to the `session_search`
tool schema (description/param diets, response hints) affect a model's
ability to actually use the tool. Built for PR #95570 (schema diet
1,570 → 695 tok/call), where the question was: "does moving the teaching
essay out of the schema and into response hints confuse models?"

Unlike the readtool/browser evals, this one does not run the full AIAgent —
it runs a minimal agent loop where the ONLY variable between arms is
`tools/session_search_tool.py` extracted from two git refs. Everything else
(seeded DB, tasks, oracles, system prompt, temperature) is held constant.

## What it measures

Six tasks against a deterministic seeded session DB (plus a second
"work"-profile DB), each with a programmatic oracle — no LLM judging:

| task | shape exercised | oracle |
|---|---|---|
| `t1_discover` | discovery | answer contains `pglogical` |
| `t2_scroll` | forced forward scroll — fact planted OUTSIDE the ±5 window and outside bookends | `statement_timeout` + `45` |
| `t3_broaden` | AND-query miss → must broaden (OR / fewer terms); the two query nouns never co-occur in one message | port `3000` |
| `t4_link` | verbatim `@session:` link emission | link present, NOT backticked/markdown |
| `t5_profile` | `@session:work/<id>` profile link resolution (read shape) | `vault` + `90` |
| `t6_browse` | browse shape | ≥3 recent-session topics named |

Metrics per run: oracle pass, tool-call count, malformed/errored calls,
first-call prompt tokens (measures the schema itself), total tokens, wall.

## Running

```bash
# Arms are git refs; the runner extracts tools/session_search_tool.py
# from each and imports them side by side.
python3 evals/session_search_schema/runner.py \
    --base origin/main --cand HEAD \
    --model qwen/qwen3-coder-30b-a3b-instruct --reps 3

python3 evals/session_search_schema/report.py results/<model>.jsonl
```

Requires `OPENROUTER_API_KEY` in `~/.hermes/.env` (or env). The seeded DB is
rebuilt fresh in a temp dir per invocation; nothing touches your real
`state.db`.

Rules of engagement (hermesbench discipline):

- 3 reps minimum; n=1 cell differences are noise — pull the transcript
  (`calls` + `final` in the JSONL) before diagnosing any miss.
- Provider noise (zero tool calls AND empty final) gets one retry, applied
  identically to both arms; retries are logged.
- Report per-task x/N for BOTH arms with the same denominators. Never
  exclude runs from one arm only.
- Weak/mid models are the signal; frontier models mask schema ergonomics.

## Reference results (PR #95570, 2026-08-26)

108 runs, 3 models × 6 tasks × 3 reps × 2 arms
(base `2b8b4542e` = pre-diet main, cand `d8a78a4dc` = diet):

| model | base | diet | avg tok/task |
|---|---|---|---|
| qwen3-coder-30b | 16/18 | 18/18 | 11.1k → 7.1k |
| gpt-5.6-luna | 18/18 | 17/18 | 5.4k → 3.7k |
| gpt-5.6-terra | 15/18 | 17/18 | 8.0k → 5.0k |
| **total** | 49/54 | **52/54** | 7.0k → 5.3k |

Findings: diet arm held/gained accuracy; scroll `hint` measurably helped the
paging task; one 1/9 luna markdown-link miss on the diet arm; both arms
surfaced the pre-existing `around_message_id=0` falsy-sentinel bug
(issue #94792 / PR #79118).
