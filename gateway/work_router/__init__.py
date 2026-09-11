"""Fail-closed, durable work-conversation routing primitives.

The module is intentionally opt-in.  It does not change gateway behaviour until a
caller injects a validated :class:`WorkRouter` into an adapter.
"""

from .config import (
    DEFAULT_CHANNEL_ID,
    DEFAULT_MEETING_CHANNEL_ALLOWLIST,
    DEFAULT_PROFILES,
    DEFAULT_SINCLAIR_USER_ID,
    BotProfile,
    BotRegistry,
    RouterConfig,
    RouterConfigError,
)
from .meeting import (
    MeetingPreflightResult,
    PremiseCheckResult,
    PreflightParseError,
    PreflightSourceResult,
    SensitiveLookupRequest,
    parse_meeting_preflight_result,
    preflight_response_contract_examples,
)
from .models import (
    CanonicalEvent,
    MeetingCandidate,
    MeetingRoundState,
    MeetingState,
    RouteAction,
    RouteDecision,
    RouterThreadState,
)
from .rules import (
    DEMIAN_GUIDANCE,
    build_meeting_preflight_prompt,
    is_meeting_stop_request,
    route,
    route_meeting_control,
    transition_advisor_completion,
)
from .service import AckGate, WorkRouter
from .store import MeetingStopTransition, RouterStore, StoreError
from .worker import WorkRouterWorker

__all__ = [
    "AckGate",
    "BotProfile",
    "BotRegistry",
    "CanonicalEvent",
    "DEFAULT_CHANNEL_ID",
    "DEFAULT_MEETING_CHANNEL_ALLOWLIST",
    "DEFAULT_PROFILES",
    "DEFAULT_SINCLAIR_USER_ID",
    "DEMIAN_GUIDANCE",
    "MeetingCandidate",
    "MeetingPreflightResult",
    "MeetingRoundState",
    "MeetingState",
    "MeetingStopTransition",
    "PremiseCheckResult",
    "PreflightParseError",
    "PreflightSourceResult",
    "SensitiveLookupRequest",
    "RouteAction",
    "RouteDecision",
    "RouterConfig",
    "RouterConfigError",
    "RouterStore",
    "RouterThreadState",
    "StoreError",
    "WorkRouter",
    "WorkRouterWorker",
    "build_meeting_preflight_prompt",
    "is_meeting_stop_request",
    "parse_meeting_preflight_result",
    "preflight_response_contract_examples",
    "route",
    "route_meeting_control",
    "transition_advisor_completion",
]
