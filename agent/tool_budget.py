"""Thread-safe per-agent budgets for model-requested tool calls and results.

The primitive is intentionally independent of any particular tool, plugin, or
workflow. A caller reserves a call before dispatching it and truncates the
result before adding that result to the conversation context.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


class ToolBudget:
    """Cumulative, thread-safe limits for one agent's tool execution.

    ``None`` means that dimension is unlimited. Call reservations and output
    allocations are atomic so concurrent tool dispatch cannot oversubscribe a
    finite budget.
    """

    def __init__(
        self,
        max_calls: Optional[int] = None,
        max_output_chars: Optional[int] = None,
    ) -> None:
        self.max_calls = self._validate_limit("max_calls", max_calls)
        self.max_output_chars = self._validate_limit(
            "max_output_chars", max_output_chars
        )
        self._calls_used = 0
        self._output_chars_used = 0
        self._exhaustion_reason: Optional[str] = None
        self._lock = threading.Lock()

    @staticmethod
    def _validate_limit(name: str, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-negative integer or None")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer or None")
        return value

    def _record_exhaustion_locked(self) -> None:
        """Record the first finite limit reached while holding ``_lock``."""
        if self._exhaustion_reason is not None:
            return
        if self.max_calls is not None and self._calls_used >= self.max_calls:
            self._exhaustion_reason = "max_calls"
        elif (
            self.max_output_chars is not None
            and self._output_chars_used >= self.max_output_chars
        ):
            self._exhaustion_reason = "max_output_chars"

    def reserve_call(self) -> bool:
        """Atomically reserve one tool call for execution.

        Returns ``False`` when either configured dimension is already
        exhausted. A failed reservation never increments the call counter.
        """
        with self._lock:
            if self.max_calls is not None and self._calls_used >= self.max_calls:
                self._exhaustion_reason = self._exhaustion_reason or "max_calls"
                return False
            if (
                self.max_output_chars is not None
                and self._output_chars_used >= self.max_output_chars
            ):
                self._exhaustion_reason = self._exhaustion_reason or "max_output_chars"
                return False
            self._calls_used += 1
            self._record_exhaustion_locked()
            return True

    # Explicit alias for callers that prefer the try-* naming convention.
    try_reserve_call = reserve_call

    def reserve_output(self, requested_chars: int) -> int:
        """Atomically allocate output characters and return the granted count."""
        if isinstance(requested_chars, bool) or not isinstance(requested_chars, int):
            raise TypeError("requested_chars must be a non-negative integer")
        if requested_chars < 0:
            raise ValueError("requested_chars must be a non-negative integer")

        with self._lock:
            if self.max_output_chars is None:
                granted = requested_chars
            else:
                remaining = max(0, self.max_output_chars - self._output_chars_used)
                granted = min(requested_chars, remaining)
            self._output_chars_used += granted
            self._record_exhaustion_locked()
            return granted

    reserve_output_chars = reserve_output

    def truncate_output(self, content: Any) -> Any:
        """Truncate textual tool output to the exact remaining allocation.

        Non-string content is returned unchanged because multimodal tool
        results contain structured image/audio blocks that cannot safely be
        sliced as text. Textual tool results, which are the budget's contract,
        are allocated and truncated atomically.
        """
        if not isinstance(content, str):
            return content
        granted = self.reserve_output(len(content))
        return content[:granted]

    @property
    def calls_used(self) -> int:
        with self._lock:
            return self._calls_used

    @property
    def output_chars_used(self) -> int:
        with self._lock:
            return self._output_chars_used

    @property
    def remaining_calls(self) -> Optional[int]:
        with self._lock:
            if self.max_calls is None:
                return None
            return max(0, self.max_calls - self._calls_used)

    @property
    def remaining_output_chars(self) -> Optional[int]:
        with self._lock:
            if self.max_output_chars is None:
                return None
            return max(0, self.max_output_chars - self._output_chars_used)

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return (
                self.max_calls is not None
                and self._calls_used >= self.max_calls
            ) or (
                self.max_output_chars is not None
                and self._output_chars_used >= self.max_output_chars
            )

    @property
    def is_exhausted(self) -> bool:
        """Compatibility-friendly boolean spelling for callers/readers."""
        return self.exhausted

    @property
    def exhaustion_reason(self) -> Optional[str]:
        with self._lock:
            if self._exhaustion_reason is not None:
                return self._exhaustion_reason
            self._record_exhaustion_locked()
            return self._exhaustion_reason

    @property
    def usage(self) -> dict[str, Any]:
        with self._lock:
            self._record_exhaustion_locked()
            exhausted = (
                self.max_calls is not None
                and self._calls_used >= self.max_calls
            ) or (
                self.max_output_chars is not None
                and self._output_chars_used >= self.max_output_chars
            )
            return {
                "max_calls": self.max_calls,
                "max_output_chars": self.max_output_chars,
                "calls_used": self._calls_used,
                "output_chars_used": self._output_chars_used,
                "remaining_calls": (
                    None
                    if self.max_calls is None
                    else max(0, self.max_calls - self._calls_used)
                ),
                "remaining_output_chars": (
                    None
                    if self.max_output_chars is None
                    else max(0, self.max_output_chars - self._output_chars_used)
                ),
                "exhausted": exhausted,
                "exhaustion_reason": self._exhaustion_reason,
            }

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable usage snapshot."""
        return self.usage


__all__ = ["ToolBudget"]
