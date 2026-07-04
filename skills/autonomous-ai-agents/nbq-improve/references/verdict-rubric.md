# Verdict rubric

Apply these rules mechanically to the pre-registered gate.

- **Broad-win guard:** Wins must be ≥ 2× losses across the gate's paired comparisons. Otherwise the
  verdict is no-adopt.
- **Mean > SE:** Do not move a default on a within-noise ablation win. The effect must clear its own
  standard error.
- **Borderline = keep (no-adopt):** Follow the #28 precedent: a directionally right result that is
  not a broad win stays off.
- **Cost-ceiling veto:** A result win that exceeds the pre-registered efficiency budget is no-adopt,
  or ships as adopt-with-knob-off-by-default when that disposition was pre-registered. Cost is
  **multi-dimensional**: pre-register a ceiling for EACH axis — wall, tokens, and calls — and treat a
  bust on ANY one axis as a veto. Do not collapse cost to a single scalar (a wall-neutral but
  token-heavy change, or one that adds a hidden extra model call, must not pass on wall alone).
- **Proxy-vs-objective gate selection:** Use the OBJECTIVE outcome harness (`outcome_eval.py`-class
  ground truth) for anything touching elicitation or generation. Use the #25-style two-arm realized
  scan for exposure/lens-only changes.

## Commit-message contract

Every iteration commit MUST include sections named `ATTEMPTED`, `WHY`, and `RESULT`. In a verdict
commit, `RESULT` must include the actual gate numbers—not only a verdict word—plus Δresult, Δcost,
and the verdict. A build-stage, pre-gate commit may put `build stage, gate pending` in `RESULT`, but
it must be followed by a separate verdict commit after the gate runs.
