"""Constitutional critic — reward-hacking detector.

The held-out guard (see ``evaluator.held_out_guard``) catches one
failure mode: candidates that fit the training split but not the
held-out split. It cannot catch candidates that *generalise* their
hack — e.g. a prompt that flatters any LLM judge, or a regex that
matches every positive example by accepting essentially everything.

This module pairs a second LLM with a short constitution (rule-set) so
that, after each generation, the top-K candidates are inspected for
signs of gaming the evaluation. The critic emits a structured JSON
record ``{risk ∈ [0,1], evidence, signal_tags}``. Candidates with
``risk > threshold`` take a penalty on their **archive-placement**
score — the raw fitness row in SQLite is never rewritten, so the audit
trail stays intact.

References
----------
Bai et al. (2022)    "Constitutional AI: Harmlessness from AI
                      Feedback." arXiv:2212.08073.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from llm import LLMClient


_DEFAULT_CONSTITUTION_PATH = (
    Path(__file__).resolve().parent.parent / "templates" / "constitution.md"
)


def load_default_constitution() -> str:
    if _DEFAULT_CONSTITUTION_PATH.exists():
        return _DEFAULT_CONSTITUTION_PATH.read_text(encoding="utf-8")
    # Fallback text keeps the critic self-contained if the templates
    # directory is missing (e.g., partial checkout).
    return (
        "You are a reward-hacking detector. Review the candidate for signs "
        "of gaming the evaluation and return STRICT JSON "
        '{"risk": <0..1>, "evidence": "<sentence>", "signal_tags": [...]}.'
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


_JSON_BLOCK = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json(raw: str) -> Optional[dict]:
    """Return the first JSON object found in *raw*, or None on failure.

    LLMs occasionally wrap structured output in code fences or sprinkle
    commentary before / after. We try the whole payload first, then
    the largest balanced-brace match.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for match in _JSON_BLOCK.finditer(raw):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f:   # NaN
        return 0.0
    return max(lo, min(hi, f))


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


@dataclass
class CriticReview:
    risk:        float = 0.0
    evidence:    str   = ""
    signal_tags: list[str] = field(default_factory=list)
    model:       str   = ""
    raw:         str   = ""    # original LLM response, for audit


@dataclass
class ConstitutionalCritic:
    """LLM-backed reward-hacking detector.

    The critic is decoupled from the main evolutionary loop so it can
    run on a cheaper model (set via ``model_override``) than the one
    generating offspring.
    """

    client:          LLMClient
    constitution:    str = ""
    threshold:       float = 0.5
    model_override:  Optional[str] = None
    temperature:     float = 0.0

    def __post_init__(self) -> None:
        if not self.constitution:
            self.constitution = load_default_constitution()

    def _system_prompt(self) -> str:
        return (
            self.constitution
            + "\n\nReturn STRICT JSON only — no prose, no code fences.\n"
              "Schema: {\"risk\": <float 0..1>, \"evidence\": \"<string>\","
              " \"signal_tags\": [<string>...]}."
        )

    async def review(
        self,
        candidate: str,
        *,
        fitness_value: Any = None,
        task_description: str = "",
        seed: Optional[int] = None,
    ) -> CriticReview:
        """Score one candidate against the constitution.

        ``fitness_value`` and ``task_description`` are optional hints
        passed into the user prompt; providing them helps the critic
        distinguish "high score because candidate is genuinely good"
        from "high score because candidate games the metric".
        """
        user = (
            (f"Task: {task_description}\n\n" if task_description else "")
            + (f"Observed fitness: {fitness_value}\n\n" if fitness_value is not None else "")
            + f"Candidate:\n{candidate}"
        )
        model_arg = self.model_override or self.client.model
        # Temporarily override the client's model for this call only.
        saved_model = self.client.model
        try:
            self.client.model = model_arg
            raw = await self.client.complete(
                self._system_prompt(), user,
                seed=seed, temperature=self.temperature,
                operator="critic",
            )
        finally:
            self.client.model = saved_model

        parsed = _extract_json(raw) or {}
        risk = _clamp(parsed.get("risk", 0.0))
        evidence = str(parsed.get("evidence", "") or "")[:500]
        tags = parsed.get("signal_tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t)[:64] for t in tags[:8]]

        return CriticReview(
            risk=risk,
            evidence=evidence,
            signal_tags=tags,
            model=model_arg,
            raw=raw,
        )

    def penalty(self, review: CriticReview, fitness: float) -> float:
        """Compute the placement penalty for a reviewed candidate.

        Returns ``risk * fitness`` only when ``risk`` crosses the
        threshold; below the threshold we pass through unchanged so
        honest candidates aren't unfairly dragged down by judge noise.
        """
        if review.risk < self.threshold:
            return 0.0
        return review.risk * float(fitness)
