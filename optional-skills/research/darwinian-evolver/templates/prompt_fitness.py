"""Prompt-evolution fitness template.

Copy this file into your experiment directory as ``fitness.py`` and
replace the scoring body. The decorator metadata tells the evolver how
to run you — ``timeout_s`` caps each evaluation, ``held_out_frac``
reserves a split for the reward-hacking guard, and ``objectives=None``
signals single-objective (the function returns a float).

The example below rewards brevity and penalises prompts that omit an
explicit instruction verb. Replace it with your own metric: LLM-as-judge
comparisons, automatic scoring against a reference, precision/recall on
a downstream task, etc.
"""

from __future__ import annotations

from evolver_sdk import fitness_spec


_ACTION_VERBS = (
    "summarize", "list", "extract", "translate", "classify",
    "rewrite", "compress", "explain", "answer",
)


@fitness_spec(held_out_frac=0.2, timeout_s=30)
def fitness(candidate: str, context: dict) -> float:
    """Return a score in ``[0, 1]``; higher is better.

    Parameters
    ----------
    candidate : str
        The prompt being evaluated.
    context : dict
        ``context["seed"]`` — deterministic RNG seed.
        ``context["held_out"]`` — True during reward-hacking re-eval.
        ``context["fidelity"]`` — successive-halving rung (1.0 = full).
    """
    # Penalise anything longer than 120 characters.
    length_score = max(0.0, 1.0 - max(0, len(candidate) - 120) / 120)

    # Bonus if the prompt contains an explicit action verb.
    low = candidate.lower()
    verb_bonus = 0.2 if any(v in low for v in _ACTION_VERBS) else 0.0

    return min(1.0, length_score + verb_bonus)
