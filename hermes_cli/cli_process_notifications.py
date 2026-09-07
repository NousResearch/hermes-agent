"""Structured process-notification input for the interactive CLI."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _ProcessNotificationBatch:
    """Keep drained events identifiable until the CLI is about to run the turn."""

    notifications: tuple[tuple[dict[str, Any], str], ...]

    def render(self, registry: Any) -> str | None:
        """Render one model input, dropping completions consumed after queueing."""
        messages = [
            message
            for event, message in self.notifications
            if not (
                event.get("type") == "completion"
                and registry.is_completion_consumed(event.get("session_id", ""))
            )
        ]
        if not messages:
            return None
        if len(messages) == 1:
            return messages[0]
        header = (
            f"[IMPORTANT: {len(messages)} background notifications are ready. "
            "Treat these results as one completion batch and send at most one "
            "consolidated response. If a notification does not change the "
            "current conclusion, absorb it silently.]"
        )
        return "\n\n".join((header, *messages))
