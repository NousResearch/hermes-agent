"""safety_redline — generic peer safety state machine.

A deterministic state machine for any external consumer that needs to
gate incoming requests after repeated upstream API failures. The
four-state cascade (HEALTHY / WARN / PAUSED / HARD_PAUSE) follows
the well-known "3 -> paused / 4 -> hard_pause / 5-minute cooldown"
convention. Pure stdlib, zero new runtime dependencies.

Use cases:
- external peer drivers that drive the gateway
- plugin hosts that make repeated outbound API calls
- any wrapper layer that needs to back off after repeated failures

Public surface:
    SafetyRedline           -- pure-Python state machine
    SafetyRedlineProtocol   -- line-delimited JSON protocol adapter
    SafetyConfig            -- tunable thresholds + cooldown
    make_protocol           -- convenience builder
"""

from .redline import (
    SafetyState,
    SafetyRedline,
    SafetyConfig,
    REDLINE_VERSION,
)
from .protocol import SafetyRedlineProtocol, make_protocol

__all__ = [
    "SafetyState",
    "SafetyRedline",
    "SafetyConfig",
    "SafetyRedlineProtocol",
    "make_protocol",
    "REDLINE_VERSION",
] 
