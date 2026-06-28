# D2 Dedup Threshold Calibration — Result (DD-3 gate)

**Date:** 2026-06-27 · **Harness:** `eval/dedup_threshold_sweep.py` · **Fixture:**
`eval/fixtures/dedup_pairs.jsonl` (65 pairs, 3 arms, hand-authored + independently labeled
BEFORE this run). Embedder: `text-embedding-3-small` (the store's embedder). **Verdict: PASS.**

## The headline finding (design-validating)
On this store, the cosine distributions of **reworded-same-fact** and **contradiction** pairs
**overlap**, so there is NO cosine threshold that catches paraphrase-dupes without also
swallowing contradictions:

| arm | min | median | max |
|---|---|---|---|
| reworded_same (n=30) | 0.576 | 0.811 | **0.921** |
| high_cosine_distinct / contradictions (n=20) | 0.614 | 0.880 | **0.989** |
| low_sim_distinct (n=15) | 0.071 | 0.192 | 0.386 |

The **contradiction max (0.989) is HIGHER than the reword-dup max (0.921)** — value-flip
contradictions ("freshness weight 0.02"→"0.10" = 0.989; "DEDUP threshold 0.95"→"0.85" = 0.988;
"GPU 5090"→"3090" = 0.961) embed *more* similarly than genuine paraphrases. Sweep:

```
 IDENT   catch  false-merge  contra-swallow
 0.950    0.0%            6              6
 0.985    0.0%            2              2
 0.990    0.0%            0              0     <- first zero-swallow point, but catch=0
 0.995    0.0%            0              0     <- chosen (margin above max contra 0.9892)
```

**At every threshold with non-zero reword-catch, contra-swallow is also non-zero.** The lowest
threshold that swallows zero contradictions (0.99+) catches zero reword-dupes. Cosine auto-skip
is therefore **unsafe and near-useless** on a fidelity-first store.

## Resolution (folded into code + matches DD-1)
- **`DEDUP_COSINE_IDENTICAL = 0.995`** — a near-verbatim safety belt sitting *above all observed
  contradictions* (max 0.9892). In practice Tier-1 exact-hash already catches what 0.995 would.
- Tier-2 therefore **effectively never auto-skips**; the ambiguous band **always WRITES** (DD-1:
  dropping the newer fact is unrecoverable; an extra row is reversible).
- **Real semantic dedup is deferred to Tier-4** (the LLM contradiction-reconciler), which is the
  only mechanism that can separate "same fact" from "contradiction" — exactly the irreducibly-LLM
  judgment the ladder reserved for it. Tier-4 is default-OFF; until it's on, the store grows
  slowly with no false-merges, which is the correct fidelity-first posture.

## Why this is a PASS, not a failure
The eval's job is to establish the **safe operating point from real data**, not to make a guessed
threshold green. It did: it proved the guessed 0.95/0.985 band was unsafe (would swallow 6 / 2
contradictions respectively), and pinned a provably-safe belt (0.995) with the auto-skip burden
correctly shifted off cosine. This is the prd-plan "a knob the spec wanted to tune turns out to
need a different mechanism" corollary — caught before shipping, by the eval.
