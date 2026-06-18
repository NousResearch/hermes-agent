# Independent Senior Review (Opus)

## Verdict: APPROVE WITH CHANGES

## Critical Blockers (severity-ordered, cite section/evidence)

**None rise to BLOCK.** Blast radius is harness-only (header; AC-5), engine/config/gateway untouched, rollback is `git revert`. The five Pass-1 required changes are genuinely closed, not papered — confirmed below. Two *new* gate-integrity gaps surface on close reading of the amendment's own additions:

**GI-3 — AC-3's positive control proves the detector *fires*, but not that it fires *for the right reason* at K=1 (§3 C2, AC-3).** The Pass-1 fix landed: a positive control now injects a known wrong-owner answer and asserts `confident_wrong == True`. But the original K=4 fabrication (idx 173) fired because Haiku returned a **co-located sibling's** owner. At K=1 there are **no siblings in the node** — so the positive control must inject a wrong owner that is *not* otherwise present in the K=1 context, i.e. the detector must catch a free-standing fabrication, not a name-collision. If the unit test's "known wrong-owner" happens to seed a name that the detector only catches via co-location heuristics, the K=1 gate is testing a code path the K=1 run will never hit. **Required: the AC-3 positive control must assert the detector fires on a wrong owner that does NOT appear anywhere in the single-fact node — proving free-standing-fabrication detection, the actual K=1 threat model.**

**GI-4 — Removing the disambiguation surface (K=1) does not characterize the risk it removes; it hides it (§1 Artifact B, C2, R3).** The amendment's own conclusion is that the lone fabrication was a *small-model disambiguation slip among 4 co-located facts*. K=1 is justified as matching the real Apollo regime ("a real Apollo session does not bury four near-identical facts into one node") — fair. But Apollo's condensation windows **will** sometimes co-locate ≥2 facts. K=1 measures the floor, not the operating point. R3 acknowledges a genuine confident-wrong is still possible. **Required (not BLOCK, but must be written down before the spend): record K=4 (or K=2) as a *characterized, non-gating* data point — "intra-node disambiguation failure rate at K=N" — so the cutover decision knows the real co-location risk instead of only the K=1 floor.** Otherwise §1's framing ("K=1 removes the disambiguation surface entirely") quietly redefines the gate to the easiest regime and the Apollo session inherits an unmeasured fabrication surface.

## Required Changes

1. **(GI-3) Tighten AC-3's positive control to the K=1 threat model** — assert the detector fires on a wrong owner absent from the single-fact node (free-standing fabrication), not merely on any wrong-owner string. As written, AC-3 could pass while testing the wrong failure mode.

2. **(GI-4) Add a non-gating K≥2 characterization point.** A small co-located run (need not be N=180) quantifying intra-node disambiguation failure, recorded in the report, so cutover is decided on the real co-location risk, not the K=1 floor. R3 names the risk; the plan must *measure* it once, not assume it away.

3. **Verify the §1 Artifact-B co-location claim is now actually evidenced, not asserted** (Pass-1 Residual #3). The table at §1 *does* now show extracted `node_served_answer` per miss and confirms idx 173's answer ("Karen Sparck Jones") is a co-located sibling — this closes the prior open question. **Keep this evidence in-repo (`arm-b-n180-haiku.json`) referenced from the report**, not just inline in the PRD, so the next reviewer can re-derive it.

4. **Correct the cost framing (Pass-1 Residual: cost asymmetry).** Worst case is ~$145–150 vs ~$200 full = **~28% savings, not the order-of-magnitude the §1/§5 framing implies**, and the `--max-usd $160` ceiling sits *above* the $150 worst case — so it never bites on the normal path and is a runaway-guard only. State plainly: **the win is contract-correctness, not cost** (Pass-1 said this; §5 still leads with cost). Lower `--max-usd` to ~$120 if it's meant to be a real checkpoint rather than decorative.

## Lens Notes (one line each)

- **Architecture:** Arm A=exact-FTS / Arm B=semantic-DAG split is the right contract; §1.1's PRD-7 §4.1 citation (lines 200–205) now grounds it — GI-1 genuinely closed.
- **Security/identity-isolation:** N/A — throwaway test data, no credential/privilege/cross-agent surface; Aegis store clean per PRD-8.
- **DevOps/SRE:** Rollback clean; `--max-usd` kill added (C3) — but set above worst-case spend, so it's a guard not a gate (see Required #4).
- **Implementation/maintainability:** `--probe-kind` defaults to `mixed` for back-compat but the *gate* passes `exact` — good, provided AC-7's report-header assertion actually fails the run on drift, not just prints (verify the assert is hard, not advisory).
- **QA:** Tests become real gates **only if** GI-3 lands; the positive control closed Pass-1 #3 but must target the K=1 failure mode, not name-collision.
- **Config-drift:** AC-7 pins the four magic numbers and asserts them in the report header — Pass-1 config-drift change genuinely closed; confirm the assertion is enforcing (raises), not cosmetic.

## Residual Risks / Open Questions

- **Pass-1 closure scorecard:** GI-1 closed (§1.1 PRD-7 citation) · GI-2 closed (C1/AC-2 void_rate report + 20% hard-stop) · Req-3 positive control closed (AC-3) but see GI-3 · Req-4 fixed extend predicate closed (C3/AC-6, normative) · Req-5 throwaway DB now default closed (C4). **All five Pass-1 items genuinely addressed.**
- **R1 decided before spend (Pass-1 Residual #1 closed):** §4.1 now binds Arm B as the cutover gate and demotes Arm A point-recall to informational unless it reveals data loss / confident-wrong — the relitigation risk is pre-empted. Good.
- **R2 wall-clock still an assumption:** 180 K=1 sessions vs ~45 K=4 — "comparable" is unvalidated; the shakedown (R2) must report *time and cost*, and the $30–45 Arm B estimate is unverified for 4× the session count. Don't let a slow/expensive shakedown silently extend.
- **VOID interaction with extend predicate (AC-6):** void_rate ≤0.20 gates the N=120→180 extend, but a run sitting at 15–20% VOID passes the extend while signaling the bury barely exercises the path — watch that a high-but-sub-threshold VOID isn't quietly carrying a thin scored sample.
- **`mixed` default is a latent foot-gun:** a future invocation that forgets `--probe-kind exact` silently re-cross-contaminates Arm A. AC-7 pins it for the campaign script, but the *flag default* still points at the wrong contract for ad-hoc runs — consider defaulting `exact` and making `mixed` explicit.