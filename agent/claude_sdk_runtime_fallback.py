"""Replay-safe provider hand-off for the claude-agent-sdk runtime.

``ClaudeSdkTurnEffects`` (the observable-effects ledger that decides whether a
failed turn may be replayed on another provider) and the failover-reason
classifier. Extracted from ``claude_sdk_runtime.py``; the consumer half of the
contract stays in ``agent.turn_runtime_handoff``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


@dataclass(frozen=True)
class ClaudeSdkTurnEffects:
    """Replay-safety ledger for one SDK provider attempt."""

    tool: bool = False
    streamed: bool = False
    projected: bool = False
    interrupted: bool = False
    mutated: bool = False

    @property
    def replay_safe(self) -> bool:
        return not any(asdict(self).values())

    def as_result_dict(self) -> Dict[str, bool]:
        return asdict(self)


_SDK_PROVIDER_FAILOVER_REASONS = frozenset(
    {
        "auth",
        "auth_permanent",
        "rate_limit",
        "upstream_rate_limit",
        "overloaded",
        "server_error",
        "timeout",
    }
)


def _sdk_provider_failover_reason(agent, error: str, fatal_reason: Optional[str]):
    """Return one canonical provider handoff reason, or ``None``.

    SDK failures are serialized into result text, so recover a status code when
    present and delegate classification to the shared API error taxonomy.  The
    explicit allowlist is intentionally narrower than ``should_fallback``:
    local/configuration errors, request-shape failures, policy refusals, and
    billing-safety guards must remain terminal on this runtime.
    """
    from agent.error_classifier import FailoverReason, classify_api_error

    if fatal_reason:
        try:
            reason = FailoverReason(str(fatal_reason))
        except ValueError:
            return None
        return reason if reason.value in _SDK_PROVIDER_FAILOVER_REASONS else None

    text = str(error or "").strip()
    if not text:
        return None

    class _SerializedSdkProviderError(RuntimeError):
        status_code: Optional[int] = None

    exc = _SerializedSdkProviderError(text)
    match = re.search(
        r"(?:api\s+error|status(?:_code)?|http)\s*[:=]?\s*(\d{3})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        exc.status_code = int(match.group(1))
    classified = classify_api_error(
        exc,
        provider=str(getattr(agent, "provider", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    return (
        classified.reason
        if classified.reason.value in _SDK_PROVIDER_FAILOVER_REASONS
        else None
    )
