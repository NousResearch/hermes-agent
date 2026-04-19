"""Summarize-in-10-words fitness for the packaged demo.

Rewards prompts that:

  1. Are close to 10 words long (penalises deviation linearly).
  2. Mention ``one`` / ``single`` / ``exactly`` — signals that the
     prompt itself instructs brevity, not just happens to be brief.
  3. Stay below 140 characters (cheap-to-serve heuristic).

The scoring is fully deterministic so ``evolver run`` is cheap to
repeat — the LLM is only called by the mutation operators, not by the
fitness function. That keeps this example usable even when the user's
endpoint is a small local model.
"""

from __future__ import annotations

import re

from evolver_sdk import fitness_spec


_BREVITY_KEYWORDS = ("one", "single", "exactly", "precisely")
_TARGET_WORDS = 10


@fitness_spec(held_out_frac=0.2, timeout_s=5)
def fitness(candidate: str, context: dict) -> float:
    tokens = re.findall(r"\b\w+\b", candidate)
    word_distance = abs(len(tokens) - _TARGET_WORDS) / _TARGET_WORDS
    word_score = max(0.0, 1.0 - word_distance)

    low = candidate.lower()
    brevity_bonus = 0.2 if any(k in low for k in _BREVITY_KEYWORDS) else 0.0

    length_penalty = max(0.0, len(candidate) - 140) / 140
    length_score = max(0.0, 1.0 - length_penalty)

    # Weighted sum keeps the output in [0, 1] for the default display.
    return min(1.0, 0.7 * word_score + 0.2 * brevity_bonus + 0.1 * length_score)
