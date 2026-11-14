# fast-jev-compaction vs Hermes compaction — 3-transcript scorecard (2026-09-19)

## Verdict

**Do not adopt Jev, and do not adopt its retention rule either.** Against what we actually ship
(summary + one session_search round-trip) Jev loses: 75.5% @ 115K vs 78.9% @ 55K. Its
closed-book recall gain over the bare summary is real but
it is bought with 2.1× the retained context re-billed on every turn and with compactions that
arrive ever more often (each one a prompt-cache break); our summary frees ~90% per event and
then stays cache-warm for a long stretch. Programmatic tool-result removal as the primary
compaction multiplies cache breaks; it is the wrong trade for Hermes.

- The +32 pt recall win is entirely "keep user/assistant text verbatim, delete old tool
  output". Jev itself dropped 100% of 851 candidates at its default threshold and, at a
  matched token budget, ranked no better than plain recency (77.8 vs 77.8).
- Jev-only compaction is a one-way ratchet: the text floor grows every cycle and is never
  reduced, so freed-per-compaction decays (89% → 55%, 76% → 20%, 63% → 8% over 32–40
  cycles) and a text-heavy session on a 200K-window host hit a hard stop after 0.4M tokens
  of work (floor ≥ threshold, 0% freed). A summary path is still required; Jev could only be
  a pre-pass.
- The 32K request window forces the whole history into 25K tokens: in every measured cycle
  the ceiling was binding and Jev decided on tool name + 60-char input + result size with
  message texts replaced by `[… N chars omitted …]`. >~500 tool calls between compactions
  cannot fit at all (1 of 4 transcripts never got a first compaction).
- What it does buy: 1.4 s and < 1¢ per compaction vs 37 s and 6¢, and verbatim retention
  of everything not a tool result.

What the data does point at: the facts our summary loses (delegation ids, root causes,
config keys, exact error strings) sat in assistant text. That is a summariser-retention
target (anchor index / identifier capture), not a reason to change compaction cadence.

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
| current (main), closed-book | 36.7 @ 40K | 50.0 @ 64K | 43.3 @ 61K | 43.3 @ 55K | $0.061 | 36.9 |
| **current + session_search recovery** (what we ship) | 76.7 @ 40K | 90.0 @ 63K | 70.0 @ 62K | **78.9 @ 55K** | $0.061 | 36.9 |
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

0. **Against the shipping mechanism (`current+recovery`, 78.9% @ 55K) Jev's default arm
   loses on recall AND retains 2.1× the tokens.** The closed-book `current` row below is the
   summary with its recovery pointer unused; findings 1–3 compare against that weaker arm.

1. **Jev's default arm is +32 pts closed-book recall (75.5 vs 43.3) at 2.1× the retained tokens
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

Nothing structural. Keeping text verbatim wins the closed-book exam but at 2.1× retained
tokens per turn and a compaction cadence that tightens every cycle (prompt cache broken far
more often); the summary's one-time 90% reduction is the better trade for a long-lived
cached conversation. The actionable residue is summariser quality: the misses were exact
identifiers in assistant text, which the anchor index is meant to capture — check its
coverage on these three banks before touching anything else.

## Repeated compaction: how many cycles does Jev-only compaction survive?

`scripts/jev_cycles.py` feeds a lineage chronologically and compacts with Jev (plugin
defaults) every time the estimate crosses the threshold; 500K ≈ our 1M-window posture, 160K
≈ a 200K-window host. Raw JSON per run in `results/jev-cycles-2026-09-19/`. Jev spend for all
six runs: $0.60.

| run | cycles | raw session consumed | freed per cycle | text floor | end state |
|---|---|---|---|---|---|
| prreview @500K | 32 | 11.4M (whole lineage) | 89% → 55% | 43K → 139K | still working |
| sysprompt @500K | 40 | 8.6M of 23.3M | 76% → 20% | 91K → 240K | degrading |
| sigsegv @500K | 40 | 5.2M of 8.6M | 63% → 8% | 175K → 359K | 40K freed/cycle; wall ≈ 8M |
| prreview @160K | 60 | 3.9M of 11.4M | 87% → 22% | 12K → 87K | compacting every ~14 rows |
| sigsegv @160K | 11 | 0.42M of 8.6M | 52% → 0% | 85K → 151K | **STUCK** (floor ≥ threshold) |
| afff57 @500K | 0 | — | — | — | fallback on cycle 1 (541 calls) |

Reading: every cycle has candidates (new tool calls arrive between compactions and Jev
drops ~all of them; 200+ cycles, zero orphaned call/result pairs), so "nothing left to
remove" never happens. What runs out is headroom: each cycle frees at most
`threshold − floor`, and the floor (all user/assistant rows) only grows. Well before the
hard stop the session runs permanently near the window: sigsegv @500K at cycle 40 compacts
every ~40K tokens of new work with every turn billed at ~460K input.

## The 32K window in practice

- `state_tok` was 24,8xx–25,000 in every one of ~180 measured cycles: the ceiling always
  binds. Fitting stages reached at 500K: "old messages collapsed" (texts → `[… N chars
  omitted …]`), "old calls compacted" (one line per call, input cut to 60 chars), "old
  messages left out" (text-only old rows removed from the state), "old calls merged". Jev
  never sees tool result contents (by design) and, at these stages, barely sees message
  text either — it decides on tool name, input stub, result size and the last 3 user prompts.
- Hard limit: ~500+ unpinned tool calls between compactions cannot fit (afff57, 541 calls);
  the plugin throws and the host falls back to its summary. The text floor does not affect
  fit (old text rows are left out of the state); the call count does.
- Requests: full state resent per batch of ~40–80 calls → 5–8 concurrent requests per
  cycle at 500K, 1–2 s, $0.005–0.008.
- Untouchable content: anything in assistant text (pasted logs, long analyses) is never
  compacted, which is exactly what makes the floor grow.

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
