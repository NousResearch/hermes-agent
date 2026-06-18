# Independent Senior Review (Opus)

## Verdict: APPROVE WITH CHANGES

## Critical Blockers (severity-ordered, cite section/evidence)

**None rise to BLOCK.** Pass-3 convergence check: GI-3 and GI-4 from Pass-2 are genuinely closed in spec text — AC-3 now names both failure modes (name-collision + free-standing fabrication, "an invented owner absent from the node"), and AC-3b/C2 add the non-gating K=2 characterization (N≈40). Blast radius remains harness-only (header, AC-5). Two integrity gaps survive — both are *verification-of-closure* gaps, not new design defects, which is exactly the failure mode adversarial review exists to catch: **the spec asserts tests are green; it does not let this reviewer re-derive that they gate the right thing.**

**CB-1 — AC-3/GI-3 closure is asserted in prose but the free-standing-fabrication test's *discriminating power* is unproven (§3 C2, AC-3).** The spec says the detector "matches affirmative 'the recovery owner is <X>' assertions regardless of pool membership." That is the correct threat model — but a detector that fires on *any* "the recovery owner is X" string will also fire on a **correct** affirmative recovery, which would make `confident_wrong` fire on Arm B's passing trials and tank recall. The negative controls (correct recovery → cw=False) and the positive control (free-standing fab → cw=True) must therefore both pass *against the same matcher*, which means the matcher is doing **pool-membership-aware** classification, not pure string-shape matching. The prose ("regardless of pool membership") and the negative control ("correct recovery yields cw=False") are in tension on their face. This is almost certainly resolved correctly in the actual code (the matcher checks the asserted owner against the node's true owner) — but **the spec's own description of the matcher contradicts its own negative control**, and a reviewer cannot certify a fake-green vector on a contradictory description. **Required: state the actual discriminator (asserted-owner ≠ node's-true-owner ⇒ cw=True) and confirm the 10 committed tests include the correct-affirmative-recovery negative control passing against that exact matcher.**

**CB-2 — AC-7's "asserted in each report header" is the load-bearing config-drift guard, and Pass-2 Required twice flagged it must be *enforcing not cosmetic* — the spec still does not say it raises (§AC-7, §C3).** Pass-2 explicitly asked to "verify the assert is hard, not advisory" and "confirm the assertion is enforcing (raises), not cosmetic." The amendment text added the pin (AC-7 lists the four params) but **still describes the mechanism as "asserted in each report header"** without stating the failure behavior. A header that *prints* `--probe-kind=mixed` when the contract demands `exact` is a record of the drift, not a guard against it. This is the single highest-value config-drift control in the doc and its enforcement semantics remain unspecified. **Required: AC-7 must state that a parameter mismatch *aborts the run and voids the report* (non-zero exit, no verdict emitted), not merely records the value.**

## Required Changes

1. **(CB-1) Resolve the matcher-description contradiction.** State the discriminator explicitly: confident-wrong fires when the model affirmatively asserts an owner **and that owner ≠ the node's true owner** (covering both the K=4 co-located-sibling case and the K=1 free-standing-fabrication case under one rule). Confirm the committed negative control — a *correct* affirmative recovery → `cw=False` — passes against that same matcher, so the detector cannot tank recall on passing trials. Without this, AC-3's "green" is uninterpretable.

2. **(CB-2) Make AC-7 enforcing.** Change "asserted in each report header" to "**validated before scoring; a mismatch aborts with non-zero exit and emits no verdict.**" Pin must raise, not print. This was Pass-2 Required twice; close it in text.

3. **(Carried, Pass-2 Residual) Default `--probe-kind` to `exact`, make `mixed` explicit.** The `mixed` default remains a latent foot-gun for any ad-hoc invocation outside the pinned campaign script. AC-7 protects the campaign run; it does not protect a manual rerun. Flip the default to the safe contract; require `--probe-kind mixed` to be typed deliberately.

4. **(Carried, Pass-2 Residual R2) The Arm B shakedown must report wall-clock *and* cost before the full N=180 K=1 run.** "Comparable wall-clock" (R2) and the $30–45 estimate are both unvalidated at 4× session count. The shakedown gates the full run on measured time+spend, not the estimate.

## Lens Notes (one line each)

- **Architecture:** Arm A=exact-FTS / Arm B=semantic-DAG split is correct and now PRD-7-grounded (§1.1); convergence on the contract is real.
- **Security/identity-isolation:** N/A — throwaway test data, no credential/privilege/cross-agent surface; Aegis store clean (PRD-8).
- **DevOps/SRE:** `--max-usd` lowered to $120 below the ~$150 worst case (§5/AC-6) — now a genuine checkpoint; rollback clean (revert harness commits).
- **Implementation/maintainability:** `--probe-kind` default still `mixed` — back-compat convenience at the cost of a contract foot-gun; flip it (Required #3).
- **QA:** Tests are real gates **iff** CB-1 holds — a detector whose description contradicts its negative control cannot be certified green from the spec alone.
- **Config-drift:** AC-7 pins the four magic numbers but its enforcement semantics are still unstated — the one control that must raise is described as if it prints (CB-2).

## Residual Risks / Open Questions

- **Pass-2 closure scorecard:** GI-3 closed in *spec text* (AC-3 names free-standing fabrication) — but see CB-1, the description self-contradicts · GI-4 closed (AC-3b/C2 K=2 N≈40 non-gating) · Req-3 evidence-in-repo closed (`arm-b-n180-haiku.json` referenced) · Req-4 cost framing closed (§5 leads with "contract-correctness, not cost", $120 ceiling below worst case). **Three of four Pass-2 items genuinely closed; GI-3's closure is textual but contradictory — CB-1 must resolve before green.**
- **The review is now converging on verification, not design.** Both surviving blockers (CB-1, CB-2) are "the spec claims green but doesn't let me re-derive the gate is real" — the design is sound; the remaining risk is fake-green. This is the correct terminal state for adversarial review, and both are closable with text edits + one confirmed test assertion, no rework.
- **VOID/extend interaction (Pass-2 carried):** a run sitting at 15–20% VOID still passes the AC-6 extend predicate while signaling the bury barely exercises the path — a high-but-sub-threshold VOID can quietly carry a thin scored sample. Watch the scored-N after VOID subtraction, not just void_rate ≤ 0.20.
- **R2 wall-clock still an assumption** until the shakedown reports it (Required #4) — don't let a slow/expensive K=1 shakedown silently extend into the full run.
- **Open question for cutover, not this PRD:** AC-3b characterizes K=2 disambiguation failure but nothing maps that measured rate to a go/no-go threshold. If K=2 shows (say) 5% intra-node fabrication, what does cutover do with that number? The characterization is necessary; the decision rule for it is still unwritten — fine to defer, but name it as deferred.