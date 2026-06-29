from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BudgetTracker:
    """Encapsulates token counters and estimated cost tracking for a session."""
    session_total_tokens: int = 0
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_cache_read_tokens: int = 0
    session_cache_write_tokens: int = 0
    session_reasoning_tokens: int = 0
    session_api_calls: int = 0
    session_estimated_cost_usd: float = 0.0
    session_cost_status: str = "unknown"
    session_cost_source: str = "none"

    def reset(self) -> None:
        """Reset all session-scoped token counters to 0 for a fresh session."""
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"

    def update_from_usage(self, usage: dict, api_calls: int = 1) -> None:
        """Update counters from a normalized usage dictionary."""
        self.session_api_calls += api_calls
        if not usage:
            return
            
        self.session_prompt_tokens += usage.get("prompt_tokens", 0)
        self.session_completion_tokens += usage.get("completion_tokens", 0)
        self.session_total_tokens += usage.get("total_tokens", 0)
        self.session_input_tokens += usage.get("input_tokens", 0)
        self.session_output_tokens += usage.get("output_tokens", 0)
        self.session_cache_read_tokens += usage.get("cache_read_tokens", 0)
        self.session_cache_write_tokens += usage.get("cache_write_tokens", 0)
        self.session_reasoning_tokens += usage.get("reasoning_tokens", 0)

    def add_cost(self, cost_usd: float, cost_status: str = "known", cost_source: str = "api") -> None:
        """Add to the estimated cost tracker."""
        self.session_estimated_cost_usd += cost_usd
        self.session_cost_status = cost_status
        self.session_cost_source = cost_source
