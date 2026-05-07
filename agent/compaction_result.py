"""Lightweight metrics for one context-compaction event.

Stored on the engine as ``last_compaction_result`` so the gateway,
status line, and ``/usage`` can surface what just happened without
changing the ``compress()`` return signature (which the
ContextEngine ABC and any plugin implementations rely on).
"""

from dataclasses import dataclass
from typing import Literal

TriggerReason = Literal["token", "turn", "message", "manual"]


@dataclass(frozen=True)
class CompactionResult:
    original_messages: int
    compacted_messages: int
    original_tokens: int
    compacted_tokens: int
    operations_deduped: int
    triggered_by: TriggerReason

    @property
    def token_reduction_pct(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        delta = self.original_tokens - self.compacted_tokens
        return round(delta / self.original_tokens * 100, 1)

    @property
    def message_reduction_pct(self) -> float:
        if self.original_messages <= 0:
            return 0.0
        delta = self.original_messages - self.compacted_messages
        return round(delta / self.original_messages * 100, 1)

    def summary_line(self) -> str:
        return (
            f"compaction: {self.original_tokens:,} → {self.compacted_tokens:,} tokens "
            f"({self.token_reduction_pct:.0f}% saved), "
            f"{self.original_messages} → {self.compacted_messages} msgs, "
            f"deduped {self.operations_deduped}, "
            f"trigger={self.triggered_by}"
        )
