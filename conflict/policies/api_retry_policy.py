"""
API Retry & Fallback Policies — Conflict-driven error recovery.

Migrated from agent/conversation_loop.py _perform_api_call retry/fallback logic.
Responsibility: Classify API errors, decide retry strategy, apply fallback models.
Uses conflict/resolver for multi-constraint arbitration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Error Classification ────────────────────────────────────────────────────

class ErrorCategory(Enum):
    """API error categories for retry strategy selection."""
    RATE_LIMIT = "rate_limit"
    CONTEXT_LENGTH = "context_length"
    AUTH_FAILURE = "auth_failure"
    MALFORMED_RESPONSE = "malformed_response"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class RetryDirective(Enum):
    """What the retry arbiter decided to do."""
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RETRY_WITH_COMPRESSION = "retry_with_compression"
    RETRY_WITH_LENGTH_BOOST = "retry_with_length_boost"
    FALLBACK_MODEL = "fallback_model"
    ABORT = "abort"


@dataclass
class ClassifiedError:
    """Result of error classification."""
    category: ErrorCategory
    retryable: bool
    should_compress: bool
    should_rotate: bool
    should_fallback: bool
    reason: str
    status_code: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


# ─── Retry State ──────────────────────────────────────────────────────────────

@dataclass
class RetryState:
    """Mutable retry state carried across retry attempts."""
    restart_with_compressed_messages: bool = False
    restart_with_length_continuation: bool = False
    restart_with_fallback: bool = False
    compression_attempts: int = 0
    length_continue_retries: int = 0
    truncated_tool_call_retries: int = 0


# ─── Retry Policy ──────────────────────────────────────────────────────────────

class APIRetryPolicy:
    """
    API retry and fallback policy arbiter.

    Uses ConflictResolver for multi-constraint arbitration when
    budget, priority, and model selection conflict.
    """

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_MAX_COMPRESSION = 3
    DEFAULT_MAX_LENGTH_CONTINUE = 3
    DEFAULT_MAX_TRUNCATED_TOOL_CALL = 3

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        self.max_retries = max_retries

    def classify_error(
        self,
        error: Exception,
        status_code: Optional[int] = None,
        error_body: str = "",
        retry_count: int = 0,
    ) -> ClassifiedError:
        """
        Classify an API error and determine retry strategy.

        This is the core decision logic migrated from conversation_loop.py.
        """
        from conflict.resolver import ConflictResolver, ConflictEvent

        reason = getattr(error, "reason", None) or str(error)
        category = self._categorize(status_code, reason, error_body)

        retryable = category in {
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.SERVER_ERROR,
            ErrorCategory.MALFORMED_RESPONSE,
        }
        should_compress = category == ErrorCategory.CONTEXT_LENGTH
        should_rotate = category == ErrorCategory.RATE_LIMIT
        should_fallback = (
            category == ErrorCategory.CONTEXT_LENGTH
            and retry_count >= self.max_retries
        )

        # Use ConflictResolver for budget/priority arbitration
        if should_compress or should_fallback:
            resolver = ConflictResolver()
            event = ConflictEvent(
                source_module="AGENTS",
                conflict_type="api_retry_policy",
                options={
                    "compress": should_compress,
                    "fallback": should_fallback,
                    "retry_immediate": retryable and retry_count < self.max_retries,
                }
            )
            resolution = resolver.resolve(event)
            should_compress = resolution.winner_value.get("compress", should_compress)
            should_fallback = resolution.winner_value.get("fallback", should_fallback)

        return ClassifiedError(
            category=category,
            retryable=retryable,
            should_compress=should_compress,
            should_rotate=should_rotate,
            should_fallback=should_fallback,
            reason=reason,
            status_code=status_code,
            retry_count=retry_count,
            max_retries=self.max_retries,
        )

    def _categorize(
        self,
        status_code: Optional[int],
        reason: str,
        error_body: str,
    ) -> ErrorCategory:
        """Categorize error based on status code and error message."""
        if status_code == 429:
            return ErrorCategory.RATE_LIMIT
        if status_code == 401 or status_code == 403:
            return ErrorCategory.AUTH_FAILURE
        if status_code == 413:
            return ErrorCategory.CONTEXT_LENGTH
        if status_code and 500 <= status_code < 600:
            return ErrorCategory.SERVER_ERROR
        if "context" in reason.lower() or "length" in reason.lower():
            return ErrorCategory.CONTEXT_LENGTH
        if "json" in reason.lower() or "parse" in reason.lower():
            return ErrorCategory.MALFORMED_RESPONSE
        if "rate" in reason.lower():
            return ErrorCategory.RATE_LIMIT
        return ErrorCategory.UNKNOWN

    def get_retry_directive(
        self,
        classified: ClassifiedError,
        retry_state: RetryState,
    ) -> RetryDirective:
        """Determine what action to take based on classified error."""
        if not classified.retryable:
            return RetryDirective.ABORT

        if classified.should_fallback:
            return RetryDirective.FALLBACK_MODEL

        if classified.should_compress:
            if retry_state.compression_attempts < self.DEFAULT_MAX_COMPRESSION:
                return RetryDirective.RETRY_WITH_COMPRESSION
            return RetryDirective.ABORT

        if retry_state.length_continue_retries < self.DEFAULT_MAX_LENGTH_CONTINUE:
            return RetryDirective.RETRY_WITH_LENGTH_BOOST

        if classified.retry_count < classified.max_retries:
            return RetryDirective.RETRY_WITH_BACKOFF

        return RetryDirective.ABORT

    def build_fallback_options(
        self,
        current_model: str,
        current_provider: str,
    ) -> list[dict]:
        """Build list of fallback model options for ConflictResolver."""
        # Standard fallback chain
        fallbacks = [
            {"model": current_model, "provider": current_provider, "priority": 0},
        ]

        # Model-specific fallbacks
        if "claude" in current_model.lower():
            fallbacks.append({"model": "claude-sonnet-4", "provider": current_provider, "priority": 1})
        elif "gpt" in current_model.lower():
            fallbacks.append({"model": "gpt-4o-mini", "provider": current_provider, "priority": 1})

        return fallbacks


# ─── Backoff Calculator ────────────────────────────────────────────────────────

def calculate_backoff(retry_count: int, base: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay with jitter."""
    import random
    delay = min(base * (2 ** retry_count), max_delay)
    jitter = delay * 0.1 * random.random()
    return delay + jitter


# ─── Rate Limit Helpers ───────────────────────────────────────────────────────

def parse_rate_limit_reset(error_body: str) -> Optional[float]:
    """Parse rate limit reset time from error response."""
    import re
    patterns = [
        r'x-ratelimit-reset:\s*(\d+)',
        r'"reset"\s*:\s*(\d+)',
        r'retry[-_\s]after[-_\s]?\d*:\s*(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, error_body, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def should_backoff_from_rate_limit(
    current_backoff: float,
    reset_time: Optional[float] = None,
) -> bool:
    """Determine if we should backoff based on rate limit state."""
    if reset_time and time.time() < reset_time:
        return True
    return current_backoff > 0