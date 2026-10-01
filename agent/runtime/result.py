"""Typed run result returned by ``MultiStepLoop.run()``.

This is the structure callers of ``AIAgent.run()`` should consume going
forward. The existing dict-shaped return is preserved for backward
compatibility via ``RunResult.to_legacy_dict()``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class Timing:
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Result of a complete ``MultiStepLoop.run()``.

    ``steps`` is a tuple of plain dicts (one per step) so callers can serialize
    without importing the step dataclasses. ``state`` is the shared scratch
    dict the loop exposed to callbacks during execution.
    """

    output: Any
    steps: tuple[dict[str, Any], ...] = ()
    state: dict[str, Any] = field(default_factory=dict)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    timing: Timing = field(default_factory=Timing)
    completed: bool = True
    termination_reason: str = "final_answer"

    def to_legacy_dict(self) -> dict[str, Any]:
        """Shape consumed by existing ``batch_runner.py`` paths."""
        return {
            "output": self.output,
            "completed": self.completed,
            "termination_reason": self.termination_reason,
            "input_tokens": self.token_usage.input_tokens,
            "output_tokens": self.token_usage.output_tokens,
            "duration_s": self.timing.duration_s,
            "steps": list(self.steps),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
