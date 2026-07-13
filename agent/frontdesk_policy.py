"""Deprecated shim — use :mod:`agent.concierge_policy`.

Kept so mid-flight imports and older tests keep working while surfaces
migrate to Concierge naming.
"""

from __future__ import annotations

from agent.concierge_policy import *  # noqa: F403
from agent.concierge_policy import (  # noqa: F401
    ACK_TOKENS_EN,
    ACK_TOKENS_KO,
    STOP_TOKENS_EN,
    STOP_TOKENS_KO,
    ConciergeConfidence,
    ConciergePolicyDecision,
    ConciergeRecommendation,
    ConciergeSignal,
    FrontdeskConfidence,
    FrontdeskPolicyDecision,
    FrontdeskRecommendation,
    FrontdeskSignal,
    classify_request,
    fingerprint,
)

__all__ = [
    "ConciergeRecommendation",
    "ConciergeConfidence",
    "ConciergeSignal",
    "ConciergePolicyDecision",
    "FrontdeskRecommendation",
    "FrontdeskConfidence",
    "FrontdeskSignal",
    "FrontdeskPolicyDecision",
    "classify_request",
    "fingerprint",
    "STOP_TOKENS_EN",
    "STOP_TOKENS_KO",
    "ACK_TOKENS_EN",
    "ACK_TOKENS_KO",
]
