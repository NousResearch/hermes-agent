"""COA scoring profiles for AMDP.

Two profiles, so the report can show the fix side-by-side:

* ``v0_paranoid`` — the original formula from the plan (§3.3 note):
      Score = 1.0*alignment - 0.15*Σ(severity²) - 0.5*fragility*staleness_norm
  Defect: Σ(severity²) grows with the NUMBER of risks. A thorough war-gamer
  that surfaces 7 risks is punished harder than a lazy one that names 2, the
  penalty (45–120) dwarfs the 1–10 alignment term, every score goes negative,
  and the decision collapses to "fewest high-severity risks" — alignment stops
  mattering.

* ``v1_balanced`` — the default. Penalize how SEVERE the risks are (mean +
  worst-case), not how MANY. Count still nudges (a plan with many failure modes
  is mildly worse) but is capped and weighted lightly, so a diligent critic no
  longer sinks a good plan. Everything stays on alignment's 0–10 scale, so a
  one-point alignment difference actually moves the ranking.

      risk_index  = 0.55*mean_sev + 0.35*max_sev + 0.10*count_factor   # 0..1
      Score       = alignment - W_RISK*risk_index - W_FRAG*fragility*staleness_norm
      W_RISK = 5.0, W_FRAG = 2.0

  where sev is normalized severity (severity/5) and
  count_factor = min(1, n_risks/6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

W_RISK = 5.0
W_FRAG = 2.0
# Fragility matters even when the state reading is fresh: a plan that breaks if
# the believed state is slightly wrong is dangerous during multi-step execution
# because the world drifts WHILE the plan runs, not only when the intake was old.
# FRAG_BASE is the share of the fragility penalty that applies on perfectly fresh
# state; staleness amplifies the rest up to full weight. (Before this, the whole
# fragility term was multiplied by staleness_norm ≈ 0 on fresh state and did
# nothing — the critic's fragility judgement was silently discarded.)
FRAG_BASE = 0.35
MIX = (0.55, 0.35, 0.10)  # risk_index = mix·(mean_sev, max_sev, count_factor)
COUNT_SATURATION = 6  # risks beyond this don't add more count penalty


@dataclass
class Scored:
    score: float
    components: dict[str, float]


def _risks(review: dict[str, Any]) -> list[int]:
    out = []
    for r in review.get("risks") or []:
        try:
            out.append(int(r.get("severity_1to5", 0)))
        except (TypeError, ValueError):
            continue
    return [s for s in out if s > 0]


def score_v0_paranoid(
    review: dict[str, Any], *, staleness_norm: float
) -> Scored:
    alignment = float(review.get("alignment_1to10", 0.0))
    sev_sq = sum(s * s for s in _risks(review))
    fragility = float(review.get("fragility_0to1", 0.0))
    align_term = 1.0 * alignment
    risk_term = -0.15 * sev_sq
    frag_term = -0.5 * fragility * staleness_norm
    return Scored(
        score=align_term + risk_term + frag_term,
        components={
            "alignment": round(align_term, 3),
            "risk_penalty": round(risk_term, 3),
            "fragility_penalty": round(frag_term, 3),
            "sev_sq": round(sev_sq, 2),
        },
    )


def score_v1_params(
    review: dict[str, Any],
    *,
    staleness_norm: float,
    w_risk: float = W_RISK,
    w_frag: float = W_FRAG,
    frag_base: float = FRAG_BASE,
    mix: tuple[float, float, float] = MIX,
) -> Scored:
    """Parameterized balanced scorer. Defaults ARE the v1_balanced profile;
    the sensitivity sweep calls this with perturbed weights."""
    alignment = float(review.get("alignment_1to10", 0.0))
    fragility = float(review.get("fragility_0to1", 0.0))
    sevs = _risks(review)
    n = len(sevs)
    if n:
        mean_sev = (sum(sevs) / n) / 5.0
        max_sev = max(sevs) / 5.0
    else:
        mean_sev = max_sev = 0.0
    count_factor = min(1.0, n / COUNT_SATURATION)
    risk_index = mix[0] * mean_sev + mix[1] * max_sev + mix[2] * count_factor
    risk_penalty = w_risk * risk_index
    # Fragility is a live signal on fresh state (frag_base) and amplifies with
    # staleness up to full weight.
    frag_penalty = w_frag * fragility * (frag_base + (1.0 - frag_base) * staleness_norm)
    score = alignment - risk_penalty - frag_penalty
    return Scored(
        score=score,
        components={
            "alignment": round(alignment, 3),
            "risk_penalty": round(-risk_penalty, 3),
            "fragility_penalty": round(-frag_penalty, 3),
            "risk_index": round(risk_index, 3),
            "mean_sev": round(mean_sev, 3),
            "max_sev": round(max_sev, 3),
            "n_risks": n,
        },
    )


def score_v1_balanced(review: dict[str, Any], *, staleness_norm: float) -> Scored:
    return score_v1_params(review, staleness_norm=staleness_norm)


PROFILES = {
    "v0_paranoid": score_v0_paranoid,
    "v1_balanced": score_v1_balanced,
}

DEFAULT_PROFILE = "v1_balanced"


def score(
    review: dict[str, Any], *, staleness_norm: float, profile: str = DEFAULT_PROFILE
) -> Scored:
    fn = PROFILES.get(profile, score_v1_balanced)
    return fn(review, staleness_norm=staleness_norm)


def decide_v1(items: list[dict[str, Any]], *, staleness_norm: float, **params: Any) -> str | None:
    """Winner under the balanced scorer with given params. ``items`` is a list of
    {"coa_id", "n_dispatches", "review"}. Ties break toward fewer dispatches."""
    best = None
    best_key = None
    for it in items:
        s = score_v1_params(it["review"], staleness_norm=staleness_norm, **params).score
        key = (s, -it["n_dispatches"])
        if best_key is None or key > best_key:
            best_key, best = key, it["coa_id"]
    return best


def sensitivity(items: list[dict[str, Any]], *, staleness_norm: float) -> dict[str, Any]:
    """How robust is the balanced decision? Re-decide under an 81-point grid of
    perturbed weights (risk/frag weights, frag baseline, risk_index mixture) and
    report the fraction that keep the same winner. Pure recompute — no model calls."""
    base = decide_v1(items, staleness_norm=staleness_norm)
    grid: list[dict[str, Any]] = []
    for wr in (W_RISK * 0.6, W_RISK, W_RISK * 1.4):
        for wf in (W_FRAG * 0.5, W_FRAG, W_FRAG * 2.0):
            for fb in (0.2, FRAG_BASE, 0.6):
                for mix in ((0.70, 0.20, 0.10), MIX, (0.40, 0.50, 0.10)):
                    grid.append({"w_risk": wr, "w_frag": wf, "frag_base": fb, "mix": mix})
    same = 0
    challengers: dict[str, int] = {}
    for p in grid:
        w = decide_v1(items, staleness_norm=staleness_norm, **p)
        if w == base:
            same += 1
        else:
            challengers[w] = challengers.get(w, 0) + 1
    return {
        "base_choice": base,
        "n_perturbations": len(grid),
        "stable": same,
        "stability": round(same / len(grid), 3) if grid else 0.0,
        "challengers": challengers,
    }
