"""
Hermes Core Reasoning Package.
Handles iterative observation, error recovery, and self-correction.
"""


class ReasoningEngine:
    def evaluate_observation(self, observation: str) -> bool:
        """Returns True if observation indicates success, False if correction needed."""
        return "error" not in observation.lower()


__all__ = ["ReasoningEngine"]
