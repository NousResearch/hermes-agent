"""Direct admin-control router extracted from GatewayRunner."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from gateway.config import Platform
from gateway.direct_control_event_runtime_service import (
    build_admin_platform_text_context,
    extract_platform_text_event_context,
)
from gateway.group_control_intents import (
    looks_like_group_runtime_status_query as looks_like_shared_group_runtime_status_query,
    resolve_oral_report_delivery_target,
)
from gateway.group_control_runtime_service import (
    match_admin_platform_group_control_request,
    run_admin_group_control_shortcut,
)
from gateway.group_moderation_runtime_service import (
    format_admin_qq_group_moderation_reply,
    match_admin_qq_group_moderation_request as shared_match_admin_qq_group_moderation_request,
    run_admin_qq_group_moderation_shortcut,
)
from gateway.group_reply_formatters import (
    format_admin_group_control_reply,
    format_admin_send_reply,
)
from gateway.group_runtime_status_runtime_service import (
    try_handle_admin_platform_group_runtime_status as shared_try_handle_admin_platform_group_runtime_status,
)
from gateway.group_runtime_status_service import (
    build_qq_group_runtime_status_details,
    build_weixin_group_runtime_status_details,
)
from gateway.group_target_intents import (
    extract_qq_group_target,
    extract_recent_target_from_history,
    extract_weixin_group_target,
)
from gateway.platforms.base import MessageEvent, MessageType
from gateway.qq_group_policies import get_group_policy
from gateway.qq_intel_runtime_service import (
    format_admin_qq_intel_control_reply as shared_format_admin_qq_intel_control_reply,
    match_admin_qq_intel_control_request as shared_match_admin_qq_intel_control_request,
    run_admin_qq_intel_control_shortcut,
)
from gateway.qq_intel_assignments import get_group_monitoring_overlay, list_intel_workers
from gateway.qq_social_runtime_service import (
    format_admin_qq_social_control_reply as shared_format_admin_qq_social_control_reply,
    match_admin_qq_social_control_request as shared_match_admin_qq_social_control_request,
    run_admin_qq_social_control_shortcut,
)
from gateway.send_runtime_service import (
    execute_send_shortcut_tool,
    extract_recent_send_target_from_history as shared_extract_recent_send_target_from_history,
    match_admin_platform_send_request as shared_match_admin_platform_send_request,
    run_admin_send_shortcut,
)
from gateway.send_intents import (
    extract_qq_inline_send_target_and_message,
    extract_weixin_inline_send_target_and_message,
)
from gateway.session import SessionSource
from gateway.weixin_group_archive import WeixinGroupArchiveStore
from gateway.weixin_group_policies import get_group_policy as get_weixin_group_policy

logger = logging.getLogger(__name__)


DIRECT_CONTROL_ROUTER_METHODS = frozenset(
    {
        "_extract_recent_group_target_from_history",
        "_match_admin_qq_send_request",
        "_format_admin_qq_send_reply",
        "_try_handle_admin_qq_send_shortcut",
        "_match_admin_weixin_send_request",
        "_format_admin_weixin_send_reply",
        "_try_handle_admin_weixin_send_shortcut",
        "_match_admin_platform_send_request",
        "_extract_platform_text_event_body",
        "_try_handle_admin_send_shortcut_common",
        "_match_admin_qq_intel_control_request",
        "_format_admin_qq_intel_control_reply",
        "_try_handle_admin_qq_intel_control",
        "_try_handle_admin_qq_group_runtime_status",
        "_load_qq_group_runtime_status_details",
        "_try_handle_admin_weixin_group_runtime_status",
        "_load_weixin_group_runtime_status_details",
        "_try_handle_admin_platform_group_runtime_status",
        "_resolve_oral_report_delivery_target",
        "_match_admin_qq_group_control_request",
        "_format_admin_qq_group_control_reply",
        "_try_handle_admin_qq_group_control",
        "_match_admin_weixin_group_control_request",
        "_format_admin_weixin_group_control_reply",
        "_try_handle_admin_weixin_group_control",
        "_run_qq_group_control_tool",
        "_run_weixin_group_control_tool",
        "_try_handle_admin_group_control_common",
        "_match_admin_platform_group_control_request",
        "_match_admin_qq_group_moderation_request",
        "_format_admin_qq_group_moderation_reply",
        "_try_handle_admin_qq_group_moderation",
        "_match_admin_qq_social_control_request",
        "_format_admin_qq_social_control_reply",
        "_try_handle_admin_qq_social_control",
    }
)


class DirectControlRouter:
    """Own direct admin shortcut parsing/execution outside GatewayRunner."""

    def __init__(self, owner: Any):
        self.owner = owner

    @staticmethod
    def _extract_platform_text_event_body(
        event: MessageEvent,
        *,
        platform: Platform,
    ) -> tuple[SessionSource | None, str]:
        return extract_platform_text_event_context(event, platform=platform)

    @staticmethod
    def _extract_recent_group_target_from_history(
        source: SessionSource,
        conversation_history: Optional[list[dict[str, Any]]],
        extractor,
    ) -> str:
        return extract_recent_target_from_history(
            source,
            conversation_history,
            extractor=extractor,
        )

    @staticmethod
    def _resolve_oral_report_delivery_target(
        source: SessionSource,
        message_text: str,
        *,
        prefer_dm: bool,
    ) -> str:
        del source
        return resolve_oral_report_delivery_target(message_text, prefer_dm=prefer_dm)

    def _match_admin_platform_send_request(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        platform: Platform,
        inline_extractor,
        history_target_extractor,
        direct_target_extractor,
        query_prompt_formatter,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=platform,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        source = context["source"]
        body = context["body"]
        if source is None:
            return None, None
        if not context["admin_ids_configured"]:
            return None, None
        if not context["is_admin_user"]:
            return None, None
        return shared_match_admin_platform_send_request(
            source=source,
            body=body,
            conversation_history=conversation_history,
            admin_ids_configured=True,
            is_admin_user=True,
            inline_extractor=inline_extractor,
            history_target_extractor=history_target_extractor,
            direct_target_extractor=direct_target_extractor,
            query_prompt_formatter=query_prompt_formatter,
        )

    def _match_admin_qq_send_request(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_send_request(
            event,
            conversation_history=conversation_history,
            platform=Platform.QQ_NAPCAT,
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
        )

    @staticmethod
    def _format_admin_qq_send_reply(tool_args: dict[str, Any]) -> str:
        return format_admin_send_reply(
            tool_args,
            platform_label="QQ 群",
            target_normalizer=lambda value: str(value or "")
            .replace("qq_napcat:group:", "")
            .replace("group:", "")
            .strip(),
        )

    def _match_admin_weixin_send_request(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_send_request(
            event,
            conversation_history=conversation_history,
            platform=Platform.WEIXIN,
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
        )

    @staticmethod
    def _format_admin_weixin_send_reply(tool_args: dict[str, Any]) -> str:
        return format_admin_send_reply(
            tool_args,
            platform_label="微信群",
            target_normalizer=lambda value: str(value or "").replace("weixin:", "").strip(),
        )

    def _try_handle_admin_send_shortcut_common(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        matcher,
        target_formatter,
        error_prefix: str,
        reply_formatter,
    ) -> str | None:
        tool_args, shortcut_error = matcher(event, conversation_history=conversation_history)
        return run_admin_send_shortcut(
            tool_args=tool_args,
            shortcut_error=shortcut_error,
            tool_runner=lambda current_tool_args: execute_send_shortcut_tool(
                current_tool_args,
                target_formatter=target_formatter,
            ),
            error_prefix=error_prefix,
            reply_formatter=reply_formatter,
            logger=logger,
        )

    def _try_handle_admin_qq_send_shortcut(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> str | None:
        return self._try_handle_admin_send_shortcut_common(
            event,
            conversation_history=conversation_history,
            matcher=self._match_admin_qq_send_request,
            target_formatter=lambda target: (
                f"qq_napcat:{target}" if str(target).startswith("group:") else str(target)
            ),
            error_prefix="QQ 发消息执行失败",
            reply_formatter=self._format_admin_qq_send_reply,
        )

    def _try_handle_admin_weixin_send_shortcut(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> str | None:
        return self._try_handle_admin_send_shortcut_common(
            event,
            conversation_history=conversation_history,
            matcher=self._match_admin_weixin_send_request,
            target_formatter=lambda target: (
                str(target) if str(target).startswith("weixin:") else f"weixin:{str(target)}"
            ),
            error_prefix="微信发消息执行失败",
            reply_formatter=self._format_admin_weixin_send_reply,
        )

    def _match_admin_qq_intel_control_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=Platform.QQ_NAPCAT,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        source = context["source"]
        body = context["body"]
        known_worker_names = {
            str(item.get("worker_name") or "").strip()
            for item in list_intel_workers()
            if isinstance(item, dict) and str(item.get("worker_name") or "").strip()
        }
        return shared_match_admin_qq_intel_control_request(
            source=source,
            body=body,
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            looks_like_joined_group_list_query=self.owner._looks_like_joined_group_list_query,
            known_worker_names=known_worker_names,
            report_target_resolver=self._resolve_oral_report_delivery_target,
        )

    def _format_admin_qq_intel_control_reply(self, tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return shared_format_admin_qq_intel_control_reply(
            tool_args,
            result,
            status_label_formatter=self.owner._format_intel_worker_status_label,
            unique_report_targets_fn=self.owner._unique_report_targets,
        )

    def _try_handle_admin_qq_intel_control(self, event: MessageEvent) -> str | None:
        tool_args, shortcut_error = self._match_admin_qq_intel_control_request(event)

        try:
            from tools.qq_control_tool import qq_control_tool

            return run_admin_qq_intel_control_shortcut(
                tool_args=tool_args,
                shortcut_error=shortcut_error,
                tool_runner=lambda current_tool_args: (
                    (lambda raw: json.loads(raw) if isinstance(raw, str) else (raw or {}))(
                        qq_control_tool(current_tool_args)
                    )
                ),
                reply_formatter=self._format_admin_qq_intel_control_reply,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Admin QQ oral intel control shortcut bootstrap failed: %s", exc)
            return f"QQ 情报员控制执行失败：{exc}"

    def _load_qq_group_runtime_status_details(self, target: str) -> dict[str, Any]:
        return build_qq_group_runtime_status_details(
            target,
            get_group_policy_fn=get_group_policy,
            get_group_monitoring_overlay_fn=get_group_monitoring_overlay,
        )

    def _load_weixin_group_runtime_status_details(self, target: str) -> dict[str, Any]:
        return build_weixin_group_runtime_status_details(
            target,
            get_group_policy_fn=get_weixin_group_policy,
            describe_group_reporting_fn=WeixinGroupArchiveStore().describe_group_reporting,
        )

    def _try_handle_admin_platform_group_runtime_status(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        platform: Platform,
        target_extractor,
        history_target_extractor,
        status_loader,
    ) -> str | None:
        context = build_admin_platform_text_context(
            event,
            platform=platform,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        return shared_try_handle_admin_platform_group_runtime_status(
            source=context["source"],
            body=context["body"],
            conversation_history=conversation_history,
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            looks_like_group_runtime_status_query=looks_like_shared_group_runtime_status_query,
            target_extractor=target_extractor,
            history_target_extractor=history_target_extractor,
            status_loader=status_loader,
        )

    def _try_handle_admin_qq_group_runtime_status(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> str | None:
        return self._try_handle_admin_platform_group_runtime_status(
            event,
            conversation_history=conversation_history,
            platform=Platform.QQ_NAPCAT,
            target_extractor=extract_qq_group_target,
            history_target_extractor=lambda source, history: self._extract_recent_group_target_from_history(
                source,
                history,
                extract_qq_group_target,
            ),
            status_loader=self._load_qq_group_runtime_status_details,
        )

    def _try_handle_admin_weixin_group_runtime_status(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> str | None:
        return self._try_handle_admin_platform_group_runtime_status(
            event,
            conversation_history=conversation_history,
            platform=Platform.WEIXIN,
            target_extractor=extract_weixin_group_target,
            history_target_extractor=lambda source, history: self._extract_recent_group_target_from_history(
                source,
                history,
                extract_weixin_group_target,
            ),
            status_loader=self._load_weixin_group_runtime_status_details,
        )

    def _match_admin_platform_group_control_request(
        self,
        event: MessageEvent,
        *,
        platform: Platform,
        target_extractor,
        missing_target_message: str,
        admin_action_label: str,
        collect_only_action: str,
        unresolved_target_guard=None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=platform,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        source = context["source"]
        body = context["body"]
        return match_admin_platform_group_control_request(
            source=source,
            body=body,
            target_extractor=target_extractor,
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            missing_target_message=missing_target_message,
            admin_only_message=(
                self.owner._admin_only_message(source, admin_action_label)
                if source is not None
                else ""
            ),
            collect_only_action=collect_only_action,
            report_target_resolver=self._resolve_oral_report_delivery_target,
            unresolved_target_guard=unresolved_target_guard,
        )

    def _match_admin_qq_group_control_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_group_control_request(
            event,
            platform=Platform.QQ_NAPCAT,
            target_extractor=extract_qq_group_target,
            missing_target_message="要切群监听/日报，请直接说清群号，或者在目标群里明确说“这个群”。",
            admin_action_label="调整 QQ 群监听/日报策略",
            collect_only_action="enable_collect_only",
            unresolved_target_guard=lambda body: any(marker in body for marker in ("情报员", "员工")),
        )

    @staticmethod
    def _format_admin_qq_group_control_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return format_admin_group_control_reply(
            tool_args,
            result,
            platform_label="QQ 群",
            target_key="group_id",
            collect_only_action="enable_collect_only",
            strip_group_prefix=True,
        )

    def _match_admin_weixin_group_control_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_group_control_request(
            event,
            platform=Platform.WEIXIN,
            target_extractor=extract_weixin_group_target,
            missing_target_message="要切微信群监听/日报，请直接说清 chatroom，或者在目标群里明确说“这个群”。",
            admin_action_label="调整微信群监听/日报策略",
            collect_only_action="collect_only",
        )

    @staticmethod
    def _format_admin_weixin_group_control_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return format_admin_group_control_reply(
            tool_args,
            result,
            platform_label="微信群",
            target_key="chat_id",
            collect_only_action="collect_only",
            strip_group_prefix=False,
        )

    @staticmethod
    def _run_qq_group_control_tool(tool_args: dict[str, Any]) -> dict[str, Any]:
        from tools.qq_control_tool import qq_control_tool

        raw = qq_control_tool(tool_args)
        return json.loads(raw) if isinstance(raw, str) else (raw or {})

    @staticmethod
    def _run_weixin_group_control_tool(tool_args: dict[str, Any]) -> dict[str, Any]:
        from tools.weixin_control_tool import weixin_control_tool

        raw = weixin_control_tool(tool_args)
        return json.loads(raw) if isinstance(raw, str) else (raw or {})

    def _try_handle_admin_group_control_common(
        self,
        event: MessageEvent,
        *,
        matcher,
        tool_runner,
        error_prefix: str,
        reply_formatter,
    ) -> str | None:
        tool_args, shortcut_error = matcher(event)
        return run_admin_group_control_shortcut(
            tool_args=tool_args,
            shortcut_error=shortcut_error,
            tool_runner=tool_runner,
            error_prefix=error_prefix,
            reply_formatter=reply_formatter,
            logger=logger,
        )

    def _try_handle_admin_qq_group_control(self, event: MessageEvent) -> str | None:
        return self._try_handle_admin_group_control_common(
            event,
            matcher=self._match_admin_qq_group_control_request,
            tool_runner=self._run_qq_group_control_tool,
            error_prefix="QQ 群监听控制执行失败",
            reply_formatter=self._format_admin_qq_group_control_reply,
        )

    def _try_handle_admin_weixin_group_control(self, event: MessageEvent) -> str | None:
        return self._try_handle_admin_group_control_common(
            event,
            matcher=self._match_admin_weixin_group_control_request,
            tool_runner=self._run_weixin_group_control_tool,
            error_prefix="微信群监听控制执行失败",
            reply_formatter=self._format_admin_weixin_group_control_reply,
        )

    def _match_admin_qq_group_moderation_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=Platform.QQ_NAPCAT,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        return shared_match_admin_qq_group_moderation_request(
            source=context["source"],
            body=context["body"],
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            admin_only_message=self.owner._admin_only_message(context["source"], "操作 QQ 群禁言/踢人"),
        )

    @staticmethod
    def _format_admin_qq_group_moderation_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return format_admin_qq_group_moderation_reply(tool_args, result)

    def _try_handle_admin_qq_group_moderation(self, event: MessageEvent) -> str | None:
        tool_args, shortcut_error = self._match_admin_qq_group_moderation_request(event)

        try:
            from tools.qq_control_tool import qq_control_tool

            return run_admin_qq_group_moderation_shortcut(
                tool_args=tool_args,
                shortcut_error=shortcut_error,
                tool_runner=lambda current_tool_args: (
                    (lambda raw: json.loads(raw) if isinstance(raw, str) else (raw or {}))(
                        qq_control_tool(current_tool_args)
                    )
                ),
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Admin QQ oral moderation shortcut bootstrap failed: %s", exc)
            return f"QQ 群管理执行失败：{exc}"

    def _match_admin_qq_social_control_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=Platform.QQ_NAPCAT,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        return shared_match_admin_qq_social_control_request(
            source=context["source"],
            body=context["body"],
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            admin_only_message=self.owner._admin_only_message(context["source"], "处理 QQ 社交请求"),
        )

    @staticmethod
    def _format_admin_qq_social_control_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return shared_format_admin_qq_social_control_reply(tool_args, result)

    def _try_handle_admin_qq_social_control(self, event: MessageEvent) -> str | None:
        tool_args, shortcut_error = self._match_admin_qq_social_control_request(event)

        try:
            from tools.qq_social_tool import qq_social_tool

            return run_admin_qq_social_control_shortcut(
                tool_args=tool_args,
                shortcut_error=shortcut_error,
                tool_runner=lambda current_tool_args: (
                    (lambda raw: json.loads(raw) if isinstance(raw, str) else (raw or {}))(
                        qq_social_tool(current_tool_args)
                    )
                ),
                reply_formatter=self._format_admin_qq_social_control_reply,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Admin QQ social shortcut bootstrap failed: %s", exc)
            return f"QQ 社交控制执行失败：{exc}"
