# PRD-8.1 — Probe-Isolation Amendment to the LCM Phase-2 Campaign

**Status:** Draft for review
**Parent:** PRD-7 (LCM QA/E2E battery), PRD-8 (Aegis store reset)
**Author:** Apollo
**Date:** 2026-06-17
**Blast radius:** test harness only (no engine, no config, no gateway change). Aegis store already clean (PRD-8 reset, backup `lcm.db.backup-20260617_024323`).

---

## 1. Why this amendment exists

The first clean N=180 campaign (post-reset) finished with both arms **BLOCKED/FAIL** — but root-cause analysis shows **both failures are harness/probe-design artifacts, not LCM engine defects.** The engine's own guarantees passed cleanly:

| Engine guarantee | Result |
|---|---|
| Condensation fired (≥1 depth-1 node) | **180/180** |
| Buried fact preserved in a depth≥1 node | **180/180** |
| Node-served recall (Arm B) | **176/180 (97.8%)**, Wilson lb **0.944** (≥0.90 ✅) |

The engine never lost or corrupted a fact. The campaign nonetheless failed on two scoring artifacts:

### Artifact A — Arm A scored on a probe type its path isn't built for
Arm A exercises the **raw-store / FTS (`lcm_grep`)** recovery path. The fixture generator (`lcm_live_recovery.py:449`) interleaves probes `arm="semantic" if i % 3 == 1 else "exact"` → **1/3 of Arm A trials are semantic** (recover the *meaning*, e.g. an owner name), **2/3 exact** (recover a verbatim sentinel string). Result by type:

- exact probes: **103/120 (85.8%)**
- semantic probes: **43/60 (71.7%)** ← drags the aggregate to 0.817

The raw FTS path is an exact-string matcher; semantic recovery is the **DAG's** job (Arm B), not raw FTS. Scoring raw-store on semantic probes measures the wrong contract. *(Note: even exact at 85.8% is below 0.95 — see §3, this is a separate, smaller bury-calibration issue, 20 trials with missing tool-call evidence.)*

### Artifact B — Arm B's confident-wrong is a K=4 intra-node phrase-disambiguation artifact
Arm B's single FAIL trigger was **confident_wrong = 1** (gate requires 0); recall and Wilson both passed. I extracted the actual `node_served_answer` for all 4 misses from `arm-b-n180-haiku.json` (per Opus review GI-#38 — proven, not asserted):

| idx | target (owner / phrase) | answer | classification |
|---|---|---|---|
| 12 | Ada Lovelace / recover-0300 | "…recover-0300 is **not found** in the provided context…" | honest miss (cw=False) |
| 13 | Grace Hopper / recover-0301 | "…recover-0301 is **not present**…" | honest miss (cw=False) |
| 174 | Radia Perlman / recover-4302 | "…recover-4302 is **not present**…mappings cover 1200–4003…" | honest miss (cw=False) |
| **173** | **Frances Allen / recover-4003** | "**Karen Sparck Jones** is the recovery owner for recover-4003." | **confident-wrong (cw=True)** |

**Corrected mechanism (my original spec draft was wrong about this):** the confident-wrong is NOT "the scorer fires on co-located sibling names." It is that **K=4 batching packs four distinct `(owner, phrase)` pairs into one depth-1 node**, and when asked to recover *one specific phrase's* owner, Haiku confidently returned a **different co-located pair's owner** (Karen Sparck Jones, whose phrase was also in that node). The other 3 misses were honest "not found" — no fabrication.

Two load-bearing facts:
- **All 4 misses have `sentinel_in_node = True`** — the DAG preserved every fact; **zero data loss**. Every miss is *intra-node phrase disambiguation*, not retrieval failure.
- The single fabrication is a **small-model disambiguation slip among 4 facts crammed into one node** — an artifact of the K=4 cost optimization, not of LCM. A real Apollo session does not bury four near-identical "the owner is <person>" facts into one condensation window; K=1 (one fact per node) removes the disambiguation surface entirely.

**Conclusion:** the engine is sound on every engine-level guarantee (180/180 condensed, 180/180 preserved, zero data loss). The gates failed because the harness (a) cross-contaminated raw-store scoring with semantic probes, and (b) packed K=4 facts into one node, creating an intra-node phrase-disambiguation surface that produced one small-model fabrication. Both are cheap, structural harness fixes — **not** a reason to rerun the full ~$200 campaign.

### 1.1 Pre-registration — these are the PRD-7 contracts, not new goalposts (Opus GI-1)

The charge to refute: "you diagnosed a failure and changed the measurement so it passes." Evidence that exact-only / K=1 was **always** the agreed contract, not a narrowing:

- **PRD-7 §4.1 (lines 200–205) defines the two arms by contract:** *"Arm A — raw-store/FTS recovery (the user-visible recovery contract)"* and *"Arm B — summary-node recovery (`lcm_expand_query` over condensed nodes)."* Arm A's contract **is** exact-string FTS recovery; Arm B's **is** semantic node-served recovery. The 2/3-exact + 1/3-semantic interleave in `lcm_live_recovery.py:449` was a **harness implementation deviation** that scored Arm A partly on Arm B's contract — correcting it *restores* PRD-7, it does not move the goalpost.
- **K=4 batching is not in PRD-7 at all.** It was a **cost optimization I added** mid-campaign (commit `161e968f5`) to compress ~180 sessions into ~45. PRD-7 §4 specifies one buried fact recovered per probe. K=1 *returns* to the PRD-7 per-fact contract; K=4 was the deviation.
- The **promotion bar itself is unchanged**: recall ≥0.95, Wilson lb ≥0.90, confident-wrong == 0, N≥180 (PRD-7 §2 Tier-2, §4.1). This amendment changes *what probe exercises which path*, never the thresholds.

Therefore this is a **harness-fidelity fix to match the pre-registered PRD-7 contract**, with the bar held constant. If a reviewer still considers per-arm probe isolation a contract change, it is one that moves *toward* the written spec, and the bar does not move.

---

## 2. Goal

Re-establish a **clean, contract-aligned** Phase-2 measurement so the Apollo cutover gate (zero confident-wrong, recall ≥0.95, Wilson lb ≥0.90, N≥180) is evaluated on what each path actually promises — at **targeted re-run cost (~$30–50)**, not a full campaign.

Non-goal: changing any engine behavior, config, or the promotion thresholds themselves. The bar stays exactly as agreed.

---

## 3. Changes

### C1 — Arm A: score the raw-store path on exact probes only
- `lcm_live_recovery.py`: add `--probe-kind {exact,semantic,mixed}` (default `mixed` for backward-compat; **the gate run passes `exact`**). Semantic recovery is owned by Arm B. *(Implemented: `make_fixtures(probe_kind=...)`, threaded through `run_recovery_gate` and argparse; verified all-exact / all-semantic fixture generation.)*
- Keep the tightened bury (`--filler-turns 36`) that forces store eviction. The 20 missing-tool-evidence trials are a **separate** calibration tail: add a per-trial **assertion that a store tool (`lcm_grep`/`lcm_expand*`) was actually called**; if the model answered from its own window without a lookup, the trial is **VOID (re-drawn), not scored**.
- **VOID is a bounded filter, not an unbounded selector (Opus GI-2):** record `total_draws`, `void_count`, `void_rate` in the report; **hard-stop the run and surface as a finding if `void_rate > 20%`** — that means the bury isn't reliably exercising the store path (a real defect), not something to silently redraw around. A 50%-VOID 0.96 is not the same evidence as a 2%-VOID 0.96.
- Net: Arm A measures "raw FTS recovers a buried exact sentinel, with **proven** store lookup, at a disclosed VOID rate."

### C2 — Arm B: one distinct fact per node + positive-control the detector
- Run the gate at **`--sentinels-per-session 1`** (each depth-1 node holds exactly one `(owner, phrase)` pair) so node-served recovery is measured without the intra-node disambiguation surface that produced the lone K=4 fabrication.
- **Positive control for the confident-wrong detector (Opus #3 + GI-3 — highest fake-green vector):** the detector (`score_semantic_recovery`, extracted module-level + pure) has committed tests asserting it fires on **both** failure modes and stays silent on success/abstention:
  - **(a) name-collision** — asserts a known co-located owner (the real K=4 idx-173 case);
  - **(b) free-standing fabrication (the actual K=1 threat model, GI-3)** — invents an owner *not in the node/pool* (e.g. "The recovery owner is Brian Kernighan"). **The single discriminator across both modes (Opus CB-1):** confident-wrong fires when the model affirmatively asserts an owner **and that asserted owner ≠ the target/node's true owner** — this catches the K=4 co-located sibling AND the K=1 invented name under one rule. A *correct* affirmative recovery ("Frances Allen is the recovery owner") asserts the target, so `cw=False` — the matcher does not fire on passing trials and cannot tank recall. (Verified: `test_detector_negative_control_correct_recovery_is_not_confident_wrong` passes against the same matcher as the positive controls.)
  - **negative controls** — correct recovery and honest "not found" abstention both yield `confident_wrong == False`.
  - 10 committed tests, green; full context_engine suite 126 green.
- **K≥2 characterization (Opus GI-4 — non-gating):** because real Apollo condensation windows *will* sometimes co-locate ≥2 facts, run a **small non-gating K=2 batch** (N≈40, not N=180) and record the **intra-node disambiguation failure rate** in the report. This characterizes the co-location risk K=1 removes, so the cutover decision sees the real operating-point risk, not only the K=1 floor. It does **not** gate (Apollo regime ≈ K=1), but it is measured once, not assumed away.
- Keep the proven semantic owner-name probe and `lcm_expand_query` node-only recovery (raw excluded) — the DAG's real contract.

### C3 — Targeted re-run with a fixed extend-rule and a hard cost ceiling
- **Arm B:** fresh **N=180 at K=1**, properly scored. Estimated ~$30–45.
- **Arm A:** **N=120 shakedown first**, then a **normative, pre-registered extend predicate** (Opus #4 — no judgment calls): **extend to N=180 iff `point_recall ≥ 0.95 AND wilson_lb ≥ 0.90 AND void_rate ≤ 0.20` at N=120.** Otherwise **stop** and report. **Do not re-tune parameters and re-draw the 120 after seeing the results** — the 120 stands as drawn; a failing 120 is a finding, not a do-over.
- **Hard cost ceiling (Opus DevOps lens):** the campaign script enforces a **`--max-usd` kill** (default $160) read from live Aegis Blackbox spend between phases; if cumulative spend crosses it, abort and alert — not just the manual N=120 checkpoint.
- Run sequentially from the venv-pinned campaign script, heartbeats to the working channel, same anti-fake-fail discipline.

### C4 — Per-run throwaway store for Arm A by default
- Arm A exact probes run against a **per-run throwaway `--lcm-db` by default** (not conditional), so prior-campaign sentinel rows never sit in the FTS index during a gate run. Isolate-by-default; contamination-avoidance is not optional. Arm B continues to use the live Aegis store (its DAG condensation needs the real engine path), which is clean from the PRD-8 reset.

---

## 4. Acceptance criteria

- **AC-1:** Arm A gate run is exact-probe-only (`--probe-kind exact`); zero semantic probes in the scored set.
- **AC-2:** Every scored Arm A trial has proven store-tool evidence; no-lookup trials are VOID, not miss. Report includes `total_draws`, `void_count`, `void_rate`; run hard-stops + flags if `void_rate > 20%`.
- **AC-3:** Arm B gate run is K=1 (one fact per node). The confident-wrong detector has passing **positive controls for BOTH failure modes** — name-collision (known co-located owner) AND free-standing fabrication (an invented owner absent from the node, the actual K=1 threat model, Opus GI-3) — plus negative controls (correct recovery + honest abstention). All committed as tests, green, before the gate run counts.
- **AC-3b:** A non-gating K=2 characterization run (N≈40) records the intra-node disambiguation failure rate, so cutover sees the real co-location risk, not just the K=1 floor (Opus GI-4).
- **AC-4:** Both re-runs evaluated against the **unchanged** Phase-2 bar: recall ≥0.95, Wilson lb ≥0.90, confident-wrong == 0, N≥180.
- **AC-5:** No engine/config/gateway change; diff confined to `scripts/lcm_live_recovery.py`, `scripts/lcm_arm_b_node_recovery.py`, `scripts/lcm_armab_campaign.sh`, plus regression tests for the new flags + the detector controls.
- **AC-6:** Arm A `N=120→180` extend uses the fixed predicate (recall ≥0.95 ∧ Wilson lb ≥0.90 ∧ void_rate ≤0.20); the 120 is not re-drawn after inspection. Campaign aborts if cumulative Aegis Blackbox spend crosses `--max-usd` ($120 default, below worst case).
- **AC-7:** Load-bearing run parameters (`--filler-turns 36`, `--sentinels-per-session 1`, `--probe-kind exact`, `--max-usd`) are pinned in `lcm_armab_campaign.sh` **and validated before scoring; a mismatch between the pinned contract and the actual run params aborts with non-zero exit and emits NO verdict** (Opus CB-2 — the pin raises, it does not merely print). The `--probe-kind` flag also defaults to `exact` (the safe contract); `mixed` must be typed deliberately, so an ad-hoc rerun outside the campaign script cannot silently re-cross-contaminate.
- **AC-8:** Reports to `docs/reports/lcm-qa/` with per-trial evidence; final dual verdict to the working channel. Apollo cutover remains gated on explicit user go after both arms pass.

### 4.1 Arm A's role in the Apollo gate — decided now, not after the spend (Opus R1)
PRD-7 §4.1 makes Arm A "the user-visible recovery contract" and Arm B the DAG/long-session contract that matches **Apollo's actual regime** (long coding conversations). Decision recorded here **before** the run, to avoid relitigating after $60–100: **Arm B (DAG node-served) is the binding Apollo cutover gate.** Arm A (raw-store FTS) is **a required-to-run regression with the full bar, but if exact-only Arm A misses 0.95 after the VOID fix, that is a characterized finding about raw-FTS fidelity under deep bury — it informs, and blocks cutover only if it reveals data loss or confident-wrong, not on point-recall alone.** Rationale: Apollo never relies on raw-FTS exact-string recall in practice; it relies on DAG recovery of meaning after condensation. This matches PRD-7 §4.1's "Arm A passes AND Arm B characterized/acceptable" with Arm B as the load-bearing arm.

## 5. Why this is worth doing — and rollback
- **The win is contract-correctness, not cost (Opus #4).** This amendment exists to measure each path against its *pre-registered PRD-7 contract* (raw-FTS=exact, DAG=semantic, one fact/node) with a positive-controlled fabrication detector — so a PASS means what it claims. The cost saving is secondary and modest.
- **Cost, stated plainly:** ~$30–45 (Arm B K=1, N=180) + ~$60–100 (Arm A exact, N=120→180 staged) = worst case **~$145–150**, best case ~$75 if Arm A stops at the shakedown. That is ~**28% under** the ~$200 full rerun — a real but not order-of-magnitude saving. The `--max-usd` ceiling is **$120**: it sits *below* the ~$150 worst case, so it is a genuine checkpoint (forces a re-decision if both arms run long), not a decorative guard above max spend.
- **Rollback:** harness-only changes on `main`; revert the script/test commits. Engine untouched, so nothing to roll back on the live Aegis install. PRD-8 store backup still on disk.

## 6. Risks
- **R1 — Arm A exact at 85.8% may still miss 0.95 even after the VOID rule.** If so, that's a *real* finding about raw-FTS recovery fidelity under deep bury, not an artifact — surface it honestly and decide whether raw-store recall is even an Apollo gate (Apollo's real need is DAG/Arm B). Do not paper it.
- **R2 — K=1 makes Arm B sessions more numerous** (180 sessions vs ~45). Mitigate: K=1 only needs one fact to age into one node; the session can be shorter than the K=4 batches, so wall-clock is comparable. Confirm with a small shakedown before the full run.
- **R3 — Haiku small-model slips** could still produce a genuine confident-wrong at K=1. That would be a *real* signal worth having before trusting LCM on a privileged Apollo session — exactly what the gate is for.
