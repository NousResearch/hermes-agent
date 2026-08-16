# Compaction v2 — 4-transcript scorecard (2026-08-15, anchor-index build)

Four real 500K-token lineage transcripts from state.db (sweep campaign, GUI
desktop work, PR-merge campaign, ACP/PR review), 15-question recall exam
each. "recovery" = one session_search round-trip (FTS5+BM25 sim) against the
archived region. Lean build includes: 25K clamped tail, tail tool demotion,
chunked digests (noise-filtered, pristine tool contents), mechanical anchor
index, verbatim user messages, recovery footer, upgraded summarizer prompt.

## Results (recall % @ retained tokens)

policy            sweep          gui            prmerge        acp            AVG
uncompacted       93.3 @ 500K    96.7 @ 500K    96.7 @ 500K   100.0 @ 500K   96.7
current           93.3*@ 176K    26.7*@ 156K    33.3 @ 155K    30.0 @ 160K   45.8 @ 162K
lean              40.0 @  62K    60.0 @  41K    23.3 @  44K    36.7 @  50K   40.0 @  49K
lean+recovery     70.0 @  62K    80.0 @  41K    43.3 @  45K    80.0 @  50K   68.3 @  49K

* sweep/gui current scores are from the previous question banks (same
  transcripts; banks regenerated in the 4-way run). prmerge/acp are clean
  same-bank comparisons across all arms.

## Findings

1. LEAN+RECOVERY BEATS CURRENT BY +22.5pts ON AVERAGE (68.3 vs 45.8) AT 3.3x
   FEWER TOKENS (49K vs 162K). It wins on 3 of 4 transcripts and loses only
   sweep — the one transcript where current's fat tail got lucky with
   restated facts (93.3 is bank-inflated luck; see finding 3 of the previous
   scorecard).

2. THE ANCHOR INDEX FIXED THE NEEDLE-FACT CLASS. GUI closed-book went
   23.3 -> 60.0 and GUI+recovery 46.7 -> 80.0 after mechanically indexing
   exact identifiers (SHAs, ids, paths, error strings) instead of trusting
   the summarizer with them. ACP+recovery hit 80.0.

3. TWO FRESH TRANSCRIPTS CONFIRM CURRENT IS WEAK, NOT STRONG: 33.3 and 30.0
   at ~157K retained. The original sweep 93.3 was restatement luck, not
   policy quality. Current's average is 45.8% for 162K tokens — lean+recovery
   is 22 points better for less than a third of the spend.

4. prmerge IS THE HARD CASE for everyone (96.7 ceiling, best policy 43.3):
   1.1M-token lineage truncated at 500K, dense multi-PR state. Recovery
   misses there are mostly query formulation. Headroom, not a blocker.

5. Goal check (Teknium): tail = max(10K, 2.5%) ✓; summaries scoped to the
   compacted region only ✓ (sentinel tripwire test); session_search pointer ✓
   (+20-43pts measured); better accuracy AND more savings than current ✓
   (+22.5pts at 0.30x tokens).

## Recommendation

Ship lean as opt-in (compression.tail_mode: lean, legacy default), harness as
the permanent gate. Iterate prmerge-class recall behind the flag (query
mining, per-epoch anchor windows) before default flip.
