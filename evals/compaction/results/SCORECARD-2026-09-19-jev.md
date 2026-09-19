# fast-jev-compaction vs Hermes compaction — 3-transcript scorecard (2026-09-19)

Question asked: does https://github.com/tamaratran/fast-jev-compaction ("replace the
compaction summary with Jev decisions: score every tool call/result, drop or truncate
stale ones, keep everything else verbatim") beat our compressor on remaining tokens,
compaction cost and recall accuracy?

Harness: `evals/compaction/runner.py` with the new `engine: jev` arm
(`evals/compaction/jev_arm.py`, a Python port of the plugin over OpenRouter's Decisions
API, `~typesafe/jev-latest` → served as `typesafe/jev-1.13-20260917`). Three real 500K-token
lineage prefixes from state.db (PR review campaign, system-prompt token analysis, SIGSEGV
fix), 15-question recall exam each, same bank for every arm, answered and judged by the
configured `auxiliary.compression` route (gemini-3.8-flash via Nous). A fourth transcript
(541 tool calls in 500K) could not be fitted into Jev's 25K-token state ceiling even at the
last fitting stage — the plugin throws there and Claude Code falls back to its built-in
summary; recorded as `jev_fallback`, not scored.

## Results (recall % @ retained tokens; compaction cost and wall time per event)

| policy | prreview | sysprompt | sigsegv | AVG | compaction $ | compaction s |
|---|---|---|---|---|---|---|
| current (main) | 36.7 @ 40K | 50.0 @ 64K | 43.3 @ 61K | **43.3 @ 55K** | $0.061 | 36.9 |
| lean | 33.3 @ 40K | 53.3 @ 65K | 36.7 @ 61K | 41.1 @ 55K | $0.062 | 33.4 |
| jev (plugin defaults) | 70.0 @ 50K | 63.3 @ 112K | 93.3 @ 181K | **75.5 @ 115K** | $0.007 | 1.4 |
| jev_tail40 (40 pinned rows) | 73.3 @ 71K | 63.3 @ 179K | 93.3 @ 213K | 76.6 @ 154K | $0.006 | 1.4 |
| jev_t15 (threshold 0.15) | 93.3 @ 372K | 76.7 @ 332K | 100.0 @ 484K | 90.0 @ 396K | $0.007 | 1.4 |
| jev_top60k (Jev-ranked, 60K tool budget) | 70.0 @ 113K | 70.0 @ 173K | 93.3 @ 243K | 77.8 @ 176K | $0.007 | 1.5 |
| recent_top60k (recency-ranked, same budget) | 76.7 @ 114K | 63.3 @ 174K | 93.3 @ 245K | 77.8 @ 177K | $0 | 0.0 |

`current` and `lean` are the same code path on today's main (lean tail is the default), so
their 2–7 pt spread on identical context is the exam noise floor (15 questions ≈ ±3.3 pts).
Per-question paired comparison, jev vs current across 45 questions: 17 wins, 1 loss, 27 ties.

## Findings

1. **Jev's default arm is +32 pts recall (75.5 vs 43.3) at 2.1× the retained tokens
   (115K vs 55K), for 1/9 the compaction cost ($0.007 vs $0.061) in 1/25 the time
   (1.4 s vs 37 s).** On the one transcript where the sizes are comparable (prreview,
   50K vs 40K) it still wins 70.0 vs 36.7.

2. **The recall gain is verbatim text, not Jev's judgment.** At the plugin's 0.5 threshold
   Jev's `keep_result` never exceeded 0.20 (median 0.15) and `keep_call` topped out at 0.50,
   so it dropped 100% of the 851 unpinned candidates across all three transcripts
   (kept=0, result-truncated=0). The default `jev` arm is therefore behaviourally identical
   to "delete every old tool call + result, keep every user/assistant row verbatim". The
   facts the summary loses (delegation ids, root causes, config keys, exact error strings)
   sat in assistant text the whole time.

3. **At a matched budget Jev's ranking ties plain recency: 77.8 vs 77.8.** Keeping 60K
   tokens of tool pairs ranked by `keep_result` (jev_top60k) vs ranked by position
   (recent_top60k) gives the same average; per transcript it is +6.7 / −6.7 / 0, inside the
   noise floor. Lowering the threshold to 0.15 (jev_t15) reaches 90% but retains 396K of
   500K — that is not compaction.

4. **The state ceiling does not fit Hermes scale.** Jev's 32K window forces the whole
   history into 25K tokens; at 500K every transcript needed the harshest fitting stages
   ("old calls compacted/merged", "old messages collapsed") and one of four could not fit at
   all. The plugin is designed for Claude Code's ~200K compaction point; a 1M-window Hermes
   session compacting at 500K+ will fall back to the summary regularly, and once the tool
   results are gone a second compaction has nothing left to remove.

5. **Cost shape.** Jev: 5–7 requests per compaction, ~125–200K input tokens total at
   $0.042/M ≈ $0.005–0.008, ~1.5 s wall. Our summary: one gemini-3.8-flash call over
   52–60K input tokens ≈ $0.06, 25–49 s. Both are noise against the per-turn cost of the
   retained context that follows (115K vs 55K tokens on every subsequent turn).

## What this suggests for the compressor

The cheap win is not a new decision model but a retention rule the data supports: **keep
user/assistant text verbatim and drop/truncate old tool results before anything is
summarised.** We already have that layer (`_prune_old_tool_results`, phase 1); today it is
followed by a summary that also rewrites assistant text, and that rewrite is where the
recall goes. A "prune-only until the tail budget is reached, summarise only the remainder"
posture would capture most of Jev's gain at zero extra cost. If a scoring model is wanted
for the prune ranking, Jev is fast and cheap enough (1.5 s, < 1¢) but this data shows no
signal over recency at equal budget; re-test before wiring it in.

## Method notes

- Transcripts reconstructed with `scripts/reconstruct_lineage.py` from a state.db copy;
  not committed. Question banks generated from the region current compaction summarises
  (the most conservative boundary) and cached per transcript+cap so every arm answers the
  identical exam.
- The `jev` arm counts rows (Hermes has one `role: tool` row per result), so
  `preserve_recent_messages: 6` pins fewer turns than in Claude Code; `jev_tail40` widens
  it to roughly lean's 25K tail and changes nothing (+1 pt).
- Eval spend for the whole run (question generation, 634 answer/judge calls): 43.5M input
  tokens ≈ $33.6 at gemini-3.8-flash list, through Nous inference. Jev spend across all
  arms: $0.08 via OpenRouter.
