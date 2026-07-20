"""Direct admin-control router extracted from GatewayRunner."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from gateway.config import Platform
from gateway.direct_control_platform_specs import (
    AdminGroupControlPlatformSpec,
    AdminGroupModerationPlatformSpec,
    AdminGroupRuntimeStatusSpec,
    AdminSendPlatformSpec,
    QQ_ADMIN_GROUP_CONTROL_SPEC,
    QQ_ADMIN_GROUP_MODERATION_SPEC,
    QQ_ADMIN_GROUP_RUNTIME_STATUS_SPEC,
    QQ_ADMIN_SEND_SPEC,
    WEIXIN_ADMIN_GROUP_CONTROL_SPEC,
    WEIXIN_ADMIN_GROUP_MODERATION_SPEC,
    WEIXIN_ADMIN_GROUP_RUNTIME_STATUS_SPEC,
    WEIXIN_ADMIN_SEND_SPEC,
)
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
    match_admin_platform_group_moderation_request,
    run_admin_platform_group_moderation_shortcut,
)
from gateway.group_reply_formatters import (
    format_admin_group_control_reply,
    format_admin_send_reply,
)
from gateway.group_runtime_status_runtime_service import (
    try_handle_admin_platform_group_runtime_status as shared_try_handle_admin_platform_group_runtime_status,
)
from gateway.group_runtime_status_service import unique_report_targets as shared_unique_report_targets
from gateway.group_target_intents import extract_recent_target_from_history
from gateway.intel_worker_platform_specs import load_known_qq_intel_worker_names
from gateway.platforms.base import MessageEvent, MessageType
from gateway.qq_intel_runtime_service import (
    format_admin_qq_intel_control_reply as shared_format_admin_qq_intel_control_reply,
    match_admin_qq_intel_control_request as shared_match_admin_qq_intel_control_request,
    run_admin_qq_intel_control_shortcut,
)
from gateway.qq_social_runtime_service import (
    format_admin_qq_social_control_reply as shared_format_admin_qq_social_control_reply,
    match_admin_qq_social_control_request as shared_match_admin_qq_social_control_request,
    run_admin_qq_social_control_shortcut,
)
from gateway.send_runtime_service import (
    execute_send_shortcut_tool,
    match_admin_platform_send_request as shared_match_admin_platform_send_request,
    run_admin_send_shortcut,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


DIRECT_CONTROL_ROUTER_METHODS = frozenset(
    {
        "_try_handle_admin_qq_send_shortcut",
        "_try_handle_admin_weixin_send_shortcut",
        "_match_admin_qq_intel_control_request",
        "_try_handle_admin_qq_intel_control",
        "_try_handle_admin_qq_group_runtime_status",
        "_try_handle_admin_weixin_group_runtime_status",
        "_try_handle_admin_qq_group_control",
        "_try_handle_admin_weixin_group_control",
        "_try_handle_admin_qq_group_moderation",
        "_try_handle_admin_weixin_group_moderation",
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
        spec: AdminSendPlatformSpec,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=spec.platform,
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
            inline_extractor=spec.inline_extractor,
            history_target_extractor=spec.history_target_extractor,
            direct_target_extractor=spec.direct_target_extractor,
            query_prompt_formatter=spec.query_prompt_formatter,
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
            spec=QQ_ADMIN_SEND_SPEC,
        )

    @staticmethod
    def _format_admin_send_reply(
        tool_args: dict[str, Any],
        *,
        spec: AdminSendPlatformSpec,
    ) -> str:
        return format_admin_send_reply(
            tool_args,
            platform_label=spec.platform_label,
            target_normalizer=spec.reply_target_normalizer,
        )

    @staticmethod
    def _format_admin_qq_send_reply(tool_args: dict[str, Any]) -> str:
        return DirectControlRouter._format_admin_send_reply(tool_args, spec=QQ_ADMIN_SEND_SPEC)

    def _match_admin_weixin_send_request(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_send_request(
            event,
            conversation_history=conversation_history,
            spec=WEIXIN_ADMIN_SEND_SPEC,
        )

    @staticmethod
    def _format_admin_weixin_send_reply(tool_args: dict[str, Any]) -> str:
        return DirectControlRouter._format_admin_send_reply(tool_args, spec=WEIXIN_ADMIN_SEND_SPEC)

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
            target_formatter=QQ_ADMIN_SEND_SPEC.target_formatter,
            error_prefix=QQ_ADMIN_SEND_SPEC.error_prefix,
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
            target_formatter=WEIXIN_ADMIN_SEND_SPEC.target_formatter,
            error_prefix=WEIXIN_ADMIN_SEND_SPEC.error_prefix,
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
        return shared_match_admin_qq_intel_control_request(
            source=context["source"],
            body=context["body"],
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            looks_like_joined_group_list_query=self.owner._looks_like_joined_group_list_query,
            known_worker_names=load_known_qq_intel_worker_names(),
            report_target_resolver=self._resolve_oral_report_delivery_target,
        )

    def _format_admin_qq_intel_control_reply(self, tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return shared_format_admin_qq_intel_control_reply(
            tool_args,
            result,
            status_label_formatter=self.owner._format_intel_worker_status_label,
            unique_report_targets_fn=shared_unique_report_targets,
        )

    def _try_handle_admin_qq_intel_control(self, event: MessageEvent) -> str | None:
        tool_args, shortcut_error = self._match_admin_qq_intel_control_request(event)

        try:
            return run_admin_qq_intel_control_shortcut(
                tool_args=tool_args,
                shortcut_error=shortcut_error,
                tool_runner=lambda current_tool_args: self._run_messaging_control_tool(
                    current_tool_args,
                    platform=Platform.QQ_NAPCAT,
                ),
                reply_formatter=self._format_admin_qq_intel_control_reply,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Admin QQ oral intel control shortcut bootstrap failed: %s", exc)
            return f"QQ 情报员控制执行失败：{exc}"

    def _load_qq_group_runtime_status_details(self, target: str) -> dict[str, Any]:
        return QQ_ADMIN_GROUP_RUNTIME_STATUS_SPEC.status_loader(target)

    def _load_weixin_group_runtime_status_details(self, target: str) -> dict[str, Any]:
        return WEIXIN_ADMIN_GROUP_RUNTIME_STATUS_SPEC.status_loader(target)

    def _try_handle_admin_platform_group_runtime_status(
        self,
        event: MessageEvent,
        *,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        spec: AdminGroupRuntimeStatusSpec,
        status_loader,
    ) -> str | None:
        context = build_admin_platform_text_context(
            event,
            platform=spec.platform,
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
            target_extractor=spec.target_extractor,
            history_target_extractor=spec.history_target_extractor,
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
            spec=QQ_ADMIN_GROUP_RUNTIME_STATUS_SPEC,
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
            spec=WEIXIN_ADMIN_GROUP_RUNTIME_STATUS_SPEC,
            status_loader=self._load_weixin_group_runtime_status_details,
        )

    def _match_admin_platform_group_control_request(
        self,
        event: MessageEvent,
        *,
        spec: AdminGroupControlPlatformSpec,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=spec.platform,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        source = context["source"]
        body = context["body"]
        return match_admin_platform_group_control_request(
            source=source,
            body=body,
            target_extractor=spec.target_extractor,
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            missing_target_message=spec.missing_target_message,
            admin_only_message=(
                self.owner._admin_only_message(source, spec.admin_action_label)
                if source is not None
                else ""
            ),
            collect_only_action=spec.collect_only_action,
            report_target_resolver=self._resolve_oral_report_delivery_target,
            unresolved_target_guard=spec.unresolved_target_guard,
        )

    def _match_admin_qq_group_control_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_group_control_request(
            event,
            spec=QQ_ADMIN_GROUP_CONTROL_SPEC,
        )

    @staticmethod
    def _format_admin_group_control_reply(
        tool_args: dict[str, Any],
        result: dict[str, Any],
        *,
        spec: AdminGroupControlPlatformSpec,
    ) -> str:
        return format_admin_group_control_reply(
            tool_args,
            result,
            platform_label=spec.platform_label,
            target_key=spec.target_key,
            collect_only_action=spec.collect_only_action,
            strip_group_prefix=spec.strip_group_prefix,
        )

    @staticmethod
    def _format_admin_qq_group_control_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return DirectControlRouter._format_admin_group_control_reply(
            tool_args,
            result,
            spec=QQ_ADMIN_GROUP_CONTROL_SPEC,
        )

    def _match_admin_weixin_group_control_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_group_control_request(
            event,
            spec=WEIXIN_ADMIN_GROUP_CONTROL_SPEC,
        )

    @staticmethod
    def _format_admin_weixin_group_control_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        return DirectControlRouter._format_admin_group_control_reply(
            tool_args,
            result,
            spec=WEIXIN_ADMIN_GROUP_CONTROL_SPEC,
        )

    @staticmethod
    def _parse_tool_json_result(raw: Any) -> dict[str, Any]:
        return json.loads(raw) if isinstance(raw, str) else (raw or {})

    @classmethod
    def _run_messaging_control_tool(
        cls,
        tool_args: dict[str, Any],
        *,
        platform: Platform,
    ) -> dict[str, Any]:
        from tools.messaging_control_tool import messaging_control_tool

        payload = dict(tool_args)
        payload.setdefault("platform", platform.value)
        raw = messaging_control_tool(payload)
        return cls._parse_tool_json_result(raw)

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
            tool_runner=lambda current_tool_args: self._run_messaging_control_tool(
                current_tool_args,
                platform=QQ_ADMIN_GROUP_CONTROL_SPEC.platform,
            ),
            error_prefix=QQ_ADMIN_GROUP_CONTROL_SPEC.error_prefix,
            reply_formatter=self._format_admin_qq_group_control_reply,
        )

    def _try_handle_admin_weixin_group_control(self, event: MessageEvent) -> str | None:
        return self._try_handle_admin_group_control_common(
            event,
            matcher=self._match_admin_weixin_group_control_request,
            tool_runner=lambda current_tool_args: self._run_messaging_control_tool(
                current_tool_args,
                platform=WEIXIN_ADMIN_GROUP_CONTROL_SPEC.platform,
            ),
            error_prefix=WEIXIN_ADMIN_GROUP_CONTROL_SPEC.error_prefix,
            reply_formatter=self._format_admin_weixin_group_control_reply,
        )

    def _match_admin_platform_group_moderation_request(
        self,
        event: MessageEvent,
        *,
        spec: AdminGroupModerationPlatformSpec,
    ) -> tuple[dict[str, Any] | None, str | None]:
        context = build_admin_platform_text_context(
            event,
            platform=spec.platform,
            configured_admin_user_ids_fn=self.owner._configured_admin_user_ids,
            is_admin_user_fn=self.owner._is_admin_user,
        )
        return match_admin_platform_group_moderation_request(
            source=context["source"],
            body=context["body"],
            admin_ids_configured=context["admin_ids_configured"],
            is_admin_user=context["is_admin_user"],
            admin_only_message=self.owner._admin_only_message(context["source"], spec.admin_action_label),
            spec=spec,
        )

    def _match_admin_qq_group_moderation_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_group_moderation_request(
            event,
            spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
        )

    def _match_admin_weixin_group_moderation_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return self._match_admin_platform_group_moderation_request(
            event,
            spec=WEIXIN_ADMIN_GROUP_MODERATION_SPEC,
        )

    def _try_handle_admin_platform_group_moderation(
        self,
        event: MessageEvent,
        *,
        matcher,
        tool_runner,
        spec: AdminGroupModerationPlatformSpec,
    ) -> str | None:
        tool_args, shortcut_error = matcher(event)
        return run_admin_platform_group_moderation_shortcut(
            tool_args=tool_args,
            shortcut_error=shortcut_error,
            tool_runner=tool_runner,
            logger=logger,
            spec=spec,
        )

    def _try_handle_admin_qq_group_moderation(self, event: MessageEvent) -> str | None:
        try:
            return self._try_handle_admin_platform_group_moderation(
                event,
                matcher=self._match_admin_qq_group_moderation_request,
                tool_runner=lambda current_tool_args: self._run_messaging_control_tool(
                    current_tool_args,
                    platform=QQ_ADMIN_GROUP_MODERATION_SPEC.platform,
                ),
                spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
            )
        except Exception as exc:
            logger.warning("Admin QQ oral moderation shortcut bootstrap failed: %s", exc)
            return f"{QQ_ADMIN_GROUP_MODERATION_SPEC.error_prefix}：{exc}"

    def _try_handle_admin_weixin_group_moderation(self, event: MessageEvent) -> str | None:
        try:
            return self._try_handle_admin_platform_group_moderation(
                event,
                matcher=self._match_admin_weixin_group_moderation_request,
                tool_runner=lambda current_tool_args: self._run_messaging_control_tool(
                    current_tool_args,
                    platform=WEIXIN_ADMIN_GROUP_MODERATION_SPEC.platform,
                ),
                spec=WEIXIN_ADMIN_GROUP_MODERATION_SPEC,
            )
        except Exception as exc:
            logger.warning("Admin Weixin oral moderation shortcut bootstrap failed: %s", exc)
            return f"{WEIXIN_ADMIN_GROUP_MODERATION_SPEC.error_prefix}：{exc}"

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
            return run_admin_qq_social_control_shortcut(
                tool_args=tool_args,
                shortcut_error=shortcut_error,
                tool_runner=lambda current_tool_args: self._run_messaging_control_tool(
                    current_tool_args,
                    platform=Platform.QQ_NAPCAT,
                ),
                reply_formatter=self._format_admin_qq_social_control_reply,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("Admin QQ social shortcut bootstrap failed: %s", exc)
            return f"QQ 社交控制执行失败：{exc}"
