"""Track token usage and estimated cost per session.

Provides a CostTracker class that accumulates usage across API calls
and computes estimated costs based on known model pricing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


# Approximate pricing per 1M tokens (USD)
_COST_RATES: Dict[str, Dict[str, float]] = {
    "claude-opus":    {"input": 15.0,  "output": 75.0},
    "claude-sonnet":  {"input": 3.0,   "output": 15.0},
    "claude-haiku":   {"input": 0.25,  "output": 1.25},
    "gpt-4o":         {"input": 2.50,  "output": 10.0},
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
}
_DEFAULT_RATE = {"input": 1.0, "output": 5.0}


def _match_rate(model_name: str) -> Dict[str, float]:
    """Find the best matching cost rate for a model name."""
    name = model_name.lower()
    for key, rate in _COST_RATES.items():
        if key in name:
            return rate
    return _DEFAULT_RATE


@dataclass
class _ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


class CostTracker:
    """Accumulates token usage and estimated cost across API calls."""

    def __init__(self) -> None:
        self._by_model: Dict[str, _ModelUsage] = {}
        self._total_prompt: int = 0
        self._total_completion: int = 0
        self._total_cost: float = 0.0

    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str = "unknown",
    ) -> None:
        """Record usage from a single API call."""
        if model_name not in self._by_model:
            self._by_model[model_name] = _ModelUsage()

        entry = self._by_model[model_name]
        entry.prompt_tokens += prompt_tokens
        entry.completion_tokens += completion_tokens
        entry.calls += 1

        self._total_prompt += prompt_tokens
        self._total_completion += completion_tokens

        rate = _match_rate(model_name)
        cost = (
            prompt_tokens * rate["input"] / 1_000_000
            + completion_tokens * rate["output"] / 1_000_000
        )
        self._total_cost += cost

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict with totals and per-model breakdown."""
        breakdown = {}
        for model, usage in self._by_model.items():
            rate = _match_rate(model)
            model_cost = (
                usage.prompt_tokens * rate["input"] / 1_000_000
                + usage.completion_tokens * rate["output"] / 1_000_000
            )
            breakdown[model] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                "calls": usage.calls,
                "cost_usd": round(model_cost, 6),
            }

        return {
            "total_prompt_tokens": self._total_prompt,
            "total_completion_tokens": self._total_completion,
            "total_tokens": self._total_prompt + self._total_completion,
            "total_cost_usd": round(self._total_cost, 6),
            "breakdown": breakdown,
        }

    def format_summary(self) -> str:
        """Return a human-readable cost summary string."""
        summary = self.get_summary()
        if not summary["total_tokens"]:
            return "No usage recorded yet."

        lines = [
            "💰 Session Cost Summary",
            f"  Total tokens: {summary['total_tokens']:,}",
            f"    Input:  {summary['total_prompt_tokens']:,}",
            f"    Output: {summary['total_completion_tokens']:,}",
            f"  Estimated cost: ${summary['total_cost_usd']:.4f}",
        ]

        if len(summary["breakdown"]) > 1:
            lines.append("  Per model:")
            for model, info in summary["breakdown"].items():
                lines.append(
                    f"    {model}: {info['total_tokens']:,} tokens, "
                    f"${info['cost_usd']:.4f} ({info['calls']} calls)"
                )

        return "\n".join(lines)
