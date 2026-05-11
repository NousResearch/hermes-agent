from __future__ import annotations

from collections import defaultdict


class ToolLimiter:
    """Enforces per-session tool usage limits."""

    def __init__(self, limits: dict[str, int]) -> None:
        self._limits = limits
        self._counts: defaultdict[str, int] = defaultdict(int)

    @staticmethod
    def _normalize(tool_name: str) -> str:
        """Normalize tool names — maps all browser_* variants to 'browser'.

        Case-insensitive prefix check to prevent bypass via mixed case
        (e.g., 'Browser_Navigate' or 'BROWSER_click').
        """
        lower = tool_name.lower()
        if lower.startswith("browser_"):
            return "browser"
        return lower

    def check(self, tool_name: str) -> bool:
        """Return True if the tool call is allowed.

        - If the tool has no limit entry, it's DENIED by default (secure default).
        - If the limit is 0, the tool is disabled → False.
        - If the limit is -1, the tool is unlimited → True.
        - Otherwise, allowed if count < limit.
        """
        normalized = self._normalize(tool_name)
        if normalized not in self._limits:
            return False  # Deny unknown tools by default for security
        limit = self._limits[normalized]
        if limit == 0:
            return False
        if limit < 0:
            return True  # -1 means unlimited
        return self._counts[normalized] < limit

    def record(self, tool_name: str) -> None:
        """Record a tool usage, incrementing the count."""
        normalized = self._normalize(tool_name)
        self._counts[normalized] += 1

    def remaining(self, tool_name: str) -> int | None:
        """Return remaining calls for a tool, or None if unlimited."""
        normalized = self._normalize(tool_name)
        if normalized not in self._limits:
            return 0  # Unknown tool = denied
        limit = self._limits[normalized]
        if limit == 0:
            return 0
        if limit < 0:
            return None  # Unlimited
        return max(0, limit - self._counts[normalized])

    def denial_message(self, tool_name: str) -> str:
        """Return a human-readable denial message for a tool."""
        normalized = self._normalize(tool_name)
        if normalized not in self._limits:
            return f"Tool '{tool_name}' is not permitted in this session."
        limit = self._limits[normalized]
        if limit == 0:
            return f"Tool '{normalized}' is disabled for this session."
        return (
            f"Tool '{normalized}' limit reached: "
            f"{self._counts[normalized]}/{limit} calls used."
        )
