"""Platform specs for oral social-control request parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gateway.config import Platform
from gateway.qq_intents import (
    _looks_like_qq_social_policy_candidate,
    _looks_like_qq_social_request_list_query,
)
from gateway.qq_social_control_requests import (
    looks_like_qq_social_policy_query,
    match_qq_social_control_request,
    match_qq_social_request_type,
    qq_social_policy_notify_target,
)


@dataclass(frozen=True)
class SocialControlRequestPlatformSpec:
    platform: Platform
    request_matcher: Callable[..., tuple[dict[str, object] | None, str | None]]
    looks_like_request_list_query: Callable[[str], bool]
    looks_like_policy_candidate: Callable[[str], bool]
    looks_like_policy_query: Callable[[str], bool]
    request_type_matcher: Callable[[str], str]
    notify_target_resolver: Callable[[Any, str], str | None]


def build_qq_social_control_request_platform_spec(
    *,
    request_matcher: Callable[..., tuple[dict[str, object] | None, str | None]] | None = None,
    looks_like_request_list_query: Callable[[str], bool] | None = None,
    looks_like_policy_candidate: Callable[[str], bool] | None = None,
    looks_like_policy_query: Callable[[str], bool] | None = None,
    request_type_matcher: Callable[[str], str] | None = None,
    notify_target_resolver: Callable[[Any, str], str | None] | None = None,
) -> SocialControlRequestPlatformSpec:
    return SocialControlRequestPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        request_matcher=request_matcher or match_qq_social_control_request,
        looks_like_request_list_query=looks_like_request_list_query
        or _looks_like_qq_social_request_list_query,
        looks_like_policy_candidate=looks_like_policy_candidate
        or _looks_like_qq_social_policy_candidate,
        looks_like_policy_query=looks_like_policy_query or looks_like_qq_social_policy_query,
        request_type_matcher=request_type_matcher or match_qq_social_request_type,
        notify_target_resolver=notify_target_resolver or qq_social_policy_notify_target,
    )


QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC = build_qq_social_control_request_platform_spec()
