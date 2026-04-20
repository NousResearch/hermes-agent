"""Platform specs for oral group-moderation request parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gateway.config import Platform
from gateway.qq_group_moderation_requests import (
    extract_qq_oral_moderation_duration_seconds,
    extract_qq_oral_moderation_reason,
    extract_qq_oral_moderation_user_query,
    match_qq_group_moderation_action,
    match_qq_group_moderation_request,
)


@dataclass(frozen=True)
class GroupModerationRequestPlatformSpec:
    platform: Platform
    request_matcher: Callable[..., tuple[dict[str, object] | None, str | None]]
    action_matcher: Callable[[str], str]
    user_query_extractor: Callable[[str], str]
    reason_extractor: Callable[[str], str]
    duration_extractor: Callable[[str], int | None]


def build_qq_group_moderation_request_platform_spec(
    *,
    request_matcher: Callable[..., tuple[dict[str, object] | None, str | None]] | None = None,
    action_matcher: Callable[[str], str] | None = None,
    user_query_extractor: Callable[[str], str] | None = None,
    reason_extractor: Callable[[str], str] | None = None,
    duration_extractor: Callable[[str], int | None] | None = None,
) -> GroupModerationRequestPlatformSpec:
    return GroupModerationRequestPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        request_matcher=request_matcher or match_qq_group_moderation_request,
        action_matcher=action_matcher or match_qq_group_moderation_action,
        user_query_extractor=user_query_extractor or extract_qq_oral_moderation_user_query,
        reason_extractor=reason_extractor or extract_qq_oral_moderation_reason,
        duration_extractor=duration_extractor or extract_qq_oral_moderation_duration_seconds,
    )


def build_weixin_group_moderation_request_platform_spec(
    *,
    request_matcher: Callable[..., tuple[dict[str, object] | None, str | None]] | None = None,
    action_matcher: Callable[[str], str] | None = None,
    user_query_extractor: Callable[[str], str] | None = None,
    reason_extractor: Callable[[str], str] | None = None,
    duration_extractor: Callable[[str], int | None] | None = None,
) -> GroupModerationRequestPlatformSpec:
    return GroupModerationRequestPlatformSpec(
        platform=Platform.WEIXIN,
        request_matcher=request_matcher or match_qq_group_moderation_request,
        action_matcher=action_matcher or match_qq_group_moderation_action,
        user_query_extractor=user_query_extractor or extract_qq_oral_moderation_user_query,
        reason_extractor=reason_extractor or extract_qq_oral_moderation_reason,
        duration_extractor=duration_extractor or extract_qq_oral_moderation_duration_seconds,
    )


QQ_GROUP_MODERATION_REQUEST_PLATFORM_SPEC = build_qq_group_moderation_request_platform_spec()
WEIXIN_GROUP_MODERATION_REQUEST_PLATFORM_SPEC = build_weixin_group_moderation_request_platform_spec()


def get_group_moderation_request_platform_spec(platform: Platform) -> GroupModerationRequestPlatformSpec:
    if platform is Platform.WEIXIN:
        return WEIXIN_GROUP_MODERATION_REQUEST_PLATFORM_SPEC
    return QQ_GROUP_MODERATION_REQUEST_PLATFORM_SPEC
