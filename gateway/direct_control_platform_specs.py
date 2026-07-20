"""Platform-specific config for direct admin gateway shortcuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gateway.config import Platform
from gateway.group_target_intents import (
    extract_qq_group_target,
    extract_recent_target_from_history,
    extract_weixin_group_target,
)
from gateway.group_runtime_status_platform_specs import (
    build_qq_group_runtime_status_platform_spec,
    build_weixin_group_runtime_status_platform_spec,
)
from gateway.qq_intents import _QQ_VISIBLE_NAME_ALIASES
from gateway.send_runtime_service import (
    extract_recent_send_target_from_history as shared_extract_recent_send_target_from_history,
)
from gateway.send_intents import (
    extract_qq_inline_send_target_and_message,
    extract_weixin_inline_send_target_and_message,
)


@dataclass(frozen=True)
class AdminSendPlatformSpec:
    platform: Platform
    platform_label: str
    error_prefix: str
    inline_extractor: Callable[[str], tuple[str | None, str | None]]
    history_target_extractor: Callable[[Any, list[dict[str, Any]] | None], str]
    direct_target_extractor: Callable[[Any, str], str]
    query_prompt_formatter: Callable[[str], str]
    target_formatter: Callable[[str], str]
    reply_target_normalizer: Callable[[Any], str]


@dataclass(frozen=True)
class AdminGroupControlPlatformSpec:
    platform: Platform
    platform_label: str
    error_prefix: str
    target_extractor: Callable[[Any, str], str]
    missing_target_message: str
    admin_action_label: str
    collect_only_action: str
    target_key: str
    strip_group_prefix: bool
    unresolved_target_guard: Callable[[str], bool] | None = None


@dataclass(frozen=True)
class AdminGroupRuntimeStatusSpec:
    platform: Platform
    target_extractor: Callable[[Any, str], str]
    history_target_extractor: Callable[[Any, list[dict[str, Any]] | None], str]
    status_loader: Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class AdminGroupModerationPlatformSpec:
    platform: Platform
    platform_label: str
    error_prefix: str
    target_extractor: Callable[[Any, str], str | None]
    missing_target_message: str
    admin_action_label: str
    current_group_target_formatter: Callable[[str], str]
    reply_target_normalizer: Callable[[Any], str]


QQ_ADMIN_SEND_SPEC = AdminSendPlatformSpec(
    platform=Platform.QQ_NAPCAT,
    platform_label="QQ 群",
    error_prefix="QQ 发消息执行失败",
    inline_extractor=extract_qq_inline_send_target_and_message,
    history_target_extractor=lambda source, history: shared_extract_recent_send_target_from_history(
        source,
        history,
        target_extractor=extract_qq_group_target,
    ),
    direct_target_extractor=extract_qq_group_target,
    query_prompt_formatter=lambda target_label: (
        f"可以。把要发的内容直接发我，或者一句话说“往 QQ 群 {target_label} 发：xxx”。"
    ),
    target_formatter=lambda target: (
        f"qq_napcat:{target}" if str(target).startswith("group:") else str(target)
    ),
    reply_target_normalizer=lambda value: str(value or "")
    .replace("qq_napcat:group:", "")
    .replace("group:", "")
    .strip(),
)


WEIXIN_ADMIN_SEND_SPEC = AdminSendPlatformSpec(
    platform=Platform.WEIXIN,
    platform_label="微信群",
    error_prefix="微信发消息执行失败",
    inline_extractor=extract_weixin_inline_send_target_and_message,
    history_target_extractor=lambda source, history: shared_extract_recent_send_target_from_history(
        source,
        history,
        target_extractor=extract_weixin_group_target,
    ),
    direct_target_extractor=extract_weixin_group_target,
    query_prompt_formatter=lambda target_label: (
        f"可以。把要发的内容直接发我，或者一句话说“往 微信群 {target_label} 发：xxx”。"
    ),
    target_formatter=lambda target: (
        str(target) if str(target).startswith("weixin:") else f"weixin:{str(target)}"
    ),
    reply_target_normalizer=lambda value: str(value or "").replace("weixin:", "").strip(),
)


QQ_ADMIN_GROUP_CONTROL_SPEC = AdminGroupControlPlatformSpec(
    platform=Platform.QQ_NAPCAT,
    platform_label="QQ 群",
    error_prefix="QQ 群监听控制执行失败",
    target_extractor=extract_qq_group_target,
    missing_target_message="要切群监听/日报，请直接说清群号，或者在目标群里明确说“这个群”。",
    admin_action_label="调整 QQ 群监听/日报策略",
    collect_only_action="enable_collect_only",
    target_key="group_id",
    strip_group_prefix=True,
    unresolved_target_guard=lambda body: any(marker in body for marker in ("情报员", "员工"))
    or any(alias in body for alias in _QQ_VISIBLE_NAME_ALIASES),
)


WEIXIN_ADMIN_GROUP_CONTROL_SPEC = AdminGroupControlPlatformSpec(
    platform=Platform.WEIXIN,
    platform_label="微信群",
    error_prefix="微信群监听控制执行失败",
    target_extractor=extract_weixin_group_target,
    missing_target_message="要切微信群监听/日报，请直接说清 chatroom，或者在目标群里明确说“这个群”。",
    admin_action_label="调整微信群监听/日报策略",
    collect_only_action="collect_only",
    target_key="chat_id",
    strip_group_prefix=False,
)


QQ_GROUP_RUNTIME_STATUS_PLATFORM_SPEC = build_qq_group_runtime_status_platform_spec()
WEIXIN_GROUP_RUNTIME_STATUS_PLATFORM_SPEC = build_weixin_group_runtime_status_platform_spec()


QQ_ADMIN_GROUP_RUNTIME_STATUS_SPEC = AdminGroupRuntimeStatusSpec(
    platform=Platform.QQ_NAPCAT,
    target_extractor=extract_qq_group_target,
    history_target_extractor=lambda source, history: extract_recent_target_from_history(
        source,
        history,
        extractor=extract_qq_group_target,
    ),
    status_loader=lambda target: QQ_GROUP_RUNTIME_STATUS_PLATFORM_SPEC.load_status_details(target),
)


WEIXIN_ADMIN_GROUP_RUNTIME_STATUS_SPEC = AdminGroupRuntimeStatusSpec(
    platform=Platform.WEIXIN,
    target_extractor=extract_weixin_group_target,
    history_target_extractor=lambda source, history: extract_recent_target_from_history(
        source,
        history,
        extractor=extract_weixin_group_target,
    ),
    status_loader=lambda target: WEIXIN_GROUP_RUNTIME_STATUS_PLATFORM_SPEC.load_status_details(target),
)


QQ_ADMIN_GROUP_MODERATION_SPEC = AdminGroupModerationPlatformSpec(
    platform=Platform.QQ_NAPCAT,
    platform_label="QQ 群",
    error_prefix="QQ 群管理执行失败",
    target_extractor=extract_qq_group_target,
    missing_target_message="要禁言/踢人，请直接说清群号，或者在目标群里明确说“这个群”。",
    admin_action_label="操作 QQ 群禁言/踢人",
    current_group_target_formatter=lambda chat_id: f"group:{str(chat_id).strip()}",
    reply_target_normalizer=lambda value: str(value or "").replace("group:", "").strip(),
)


WEIXIN_ADMIN_GROUP_MODERATION_SPEC = AdminGroupModerationPlatformSpec(
    platform=Platform.WEIXIN,
    platform_label="微信群",
    error_prefix="微信群管理执行失败",
    target_extractor=extract_weixin_group_target,
    missing_target_message="要禁言/踢人，请直接说清 chatroom，或者在目标群里明确说“这个群”。",
    admin_action_label="操作微信群禁言/踢人",
    current_group_target_formatter=lambda chat_id: str(chat_id or "").strip(),
    reply_target_normalizer=lambda value: str(value or "").replace("weixin:", "").strip(),
)
