# Independent Senior Review (Opus)

## Verdict: APPROVE WITH CHANGES

## Critical Blockers (severity-ordered, cite section/evidence)

**None rise to BLOCK.** Blast radius is test-harness-only (header; AC-5), engine/config/gateway untouched, rollback is `git revert` of script commits. The amendment cannot cause an outage. That said, the following are *gate-integrity* blockers — if unaddressed they reintroduce exactly the fake-green risk this amendment claims to remove:

**GI-1 — The amendment redesigns the probe distribution *and* relaxes the scoring rule on the same arms that just failed, then proposes to re-run only those arms. This is motivated-stopping risk (§1, §3).** You diagnosed two failures and are changing the measurement so they pass. That can be legitimate (a wrong contract was scored) — but the document never shows the *pre-registered* contract that says Arm A was always exact-only and Arm B was always K=1. If the K=4/mixed design was the originally-agreed Phase-2 contract (PRD-7), then narrowing it now is moving the goalposts, not fixing a harness bug. **Required: cite the PRD-7 section that defines what each arm was contractually supposed to measure.** Without that citation, an adversarial reader cannot distinguish "harness bug" from "score until green."

**GI-2 — VOID-and-redraw (C1, AC-2) is an unbounded selection filter with no cap or disclosure floor.** If the model frequently answers from its own window without a store lookup, you silently redraw until you accumulate N=180 trials that *happened* to do a lookup. That biases the sample toward easy/long-bury trials and can mask a real "the path doesn't get exercised" defect. **Required: cap VOID rate (e.g. abort/flag if >X% of draws VOID) and report VOID count alongside N in the final report.** A 50%-VOID run that scores 0.96 is not the same evidence as a 2%-VOID run that scores 0.96.

## Required Changes

1. **(GI-1) Pre-registration citation.** Add the PRD-7 reference proving exact-only / K=1 was always the intended contract per arm, OR explicitly label this as a *contract change* requiring re-approval of the Phase-2 design — not a harness fix. As written, §1's "harness artifact" framing is asserted, not evidenced against the original spec.

2. **(GI-2) Bound the VOID filter.** Add to AC-2: report total draws, VOID count, VOID rate; hard-stop/flag threshold (suggest VOID >20% ⇒ the path-exercise assumption is itself broken, surface as a finding per R1's discipline).

3. **C2/AC-3 — "validated to not fire" is untested as written.** AC-3 says the K=1 confident-wrong detector is "validated to not fire on a correct single-owner recovery." That's the *negative* case. **Add a positive control: inject a known wrong-owner answer at K=1 and assert the detector DOES fire.** Otherwise C2 could silently disable the only gate that actually failed — a detector that never fires trivially yields confident_wrong==0 (AC-4). This is the single highest fake-green vector in the doc.

4. **§3 C3 staged Arm A (N=120→180) — define the extend decision as a fixed rule, not judgment.** "If clearly failing, stop and reassess" is unfalsifiable. State the exact go/extend predicate at N=120 (it's implied — recall≥0.95 ∧ Wilson lb≥0.90 — make it normative and add: do not re-tune and re-draw the 120 after seeing them).

5. **C4 — per-run throwaway `--lcm-db` should be the default for Arm A, not conditional.** "If pollution is a concern" leaves prior-campaign sentinel rows in the FTS index as a maybe. For a clean gate, isolate by default; don't make contamination-avoidance optional.

## Lens Notes (one line each)

- **Architecture:** Sound — clean Arm A (raw FTS = exact) / Arm B (DAG = semantic) contract split is the *right* separation; the bug is that they were ever cross-scored.
- **Security/identity-isolation:** N/A to this diff; no credential, no privilege, no cross-agent surface touched — Aegis store is throwaway test data.
- **DevOps/SRE:** Rollback is clean (revert script commits, engine untouched); but no healthcheck/abort contract for a *runaway cost* re-run beyond the manual N=120 gate — add a hard $ ceiling kill, not just a checkpoint.
- **Implementation/maintainability:** Adding `--probe-kind` with default `exact` is fine, but defaulting a *gate* script to the narrower contract risks future runs silently never testing semantic again — comment the line and the PRD reference at `:449`.
- **QA:** The tests become real gates *only if* the positive-control (Required #3) lands; without it, K=1 + relaxed detector is a happy-path rubber-stamp.
- **Config-drift:** `--filler-turns 36`, `--sentinels-per-session 1`, `--probe-kind exact` are now load-bearing magic numbers — pin them in `lcm_armab_campaign.sh` and assert them in the report header, or the next run drifts off-contract silently.

## Residual Risks / Open Questions

- **R1 is the real one and the doc handles it honestly:** if Arm A exact still misses 0.95 post-VOID, is raw-store recall even an Apollo gate? **Open question the doc raises but doesn't resolve: decide *now* whether Arm A gates Apollo at all, or you'll relitigate it after spending $60–100.** Apollo's stated real need is Arm B/DAG — if Arm A is informational, say so before the run, not after.
- **K=1 changes the workload shape (R2):** 180 distinct sessions vs ~45 batched. "Wall-clock comparable" is an assumption, not a measurement — the shakedown must confirm both cost *and* time, and the ~$30–45 estimate is unvalidated for 4× the session count.
- **Co-location claim (§1 Artifact B) is plausible but unproven in-doc:** you assert all 4 misses contained *sibling* names (legit co-location) vs *non-sibling* names (real hallucination). Did any of the 4 contain an owner name *not* in that node? That single check decides whether B is truly an artifact — include the evidence, don't assert it.
- **Cost asymmetry:** worst case ~$150 vs the "~$200 full campaign" you're avoiding — the savings are thinner than the framing implies if Arm A runs full N=180. The win is contract-correctness, not cost; lead with that.