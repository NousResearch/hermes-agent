"""Platform specs for oral group runtime status loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gateway.config import Platform
from gateway.group_runtime_status_service import (
    build_qq_group_runtime_status_details,
    build_weixin_group_runtime_status_details,
)
from gateway.qq_group_policies import get_group_policy as get_qq_group_policy
from gateway.qq_intel_assignments import get_group_monitoring_overlay
from gateway.weixin_group_archive import WeixinGroupArchiveStore
from gateway.weixin_group_policies import get_group_policy as get_weixin_group_policy


@dataclass(frozen=True)
class GroupRuntimeStatusPlatformSpec:
    platform: Platform
    load_status_details: Callable[[str], dict[str, Any]]


def build_qq_group_runtime_status_platform_spec(
    *,
    get_group_policy_fn: Callable[[str], dict[str, Any]] | None = None,
    get_group_monitoring_overlay_fn: Callable[[str], dict[str, Any] | None] | None = None,
) -> GroupRuntimeStatusPlatformSpec:
    get_group_policy_fn = get_group_policy_fn or get_qq_group_policy
    get_group_monitoring_overlay_fn = get_group_monitoring_overlay_fn or get_group_monitoring_overlay
    return GroupRuntimeStatusPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        load_status_details=lambda target: build_qq_group_runtime_status_details(
            target,
            get_group_policy_fn=get_group_policy_fn,
            get_group_monitoring_overlay_fn=get_group_monitoring_overlay_fn,
        ),
    )


def build_weixin_group_runtime_status_platform_spec(
    *,
    get_group_policy_fn: Callable[[str], dict[str, Any]] | None = None,
    describe_group_reporting_fn: Callable[..., dict[str, Any]] | None = None,
) -> GroupRuntimeStatusPlatformSpec:
    get_group_policy_fn = get_group_policy_fn or get_weixin_group_policy
    describe_group_reporting_fn = (
        describe_group_reporting_fn or WeixinGroupArchiveStore().describe_group_reporting
    )
    return GroupRuntimeStatusPlatformSpec(
        platform=Platform.WEIXIN,
        load_status_details=lambda target: build_weixin_group_runtime_status_details(
            target,
            get_group_policy_fn=get_group_policy_fn,
            describe_group_reporting_fn=describe_group_reporting_fn,
        ),
    )
