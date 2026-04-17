"""Direct admin-control router extracted from GatewayRunner."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from gateway.config import Platform
from gateway.group_control_intents import (
    looks_like_group_runtime_status_query as looks_like_shared_group_runtime_status_query,
    resolve_oral_report_delivery_target,
)
from gateway.group_control_requests import match_group_control_request
from gateway.group_control_runtime_service import (
    match_admin_platform_group_control_request,
    run_admin_group_control_shortcut,
)
from gateway.group_reply_formatters import (
    format_admin_group_control_reply,
    format_admin_send_reply,
    format_group_runtime_status_reply,
)
from gateway.group_runtime_status_service import (
    build_qq_group_runtime_status_details,
    build_weixin_group_runtime_status_details,
)
from gateway.group_runtime_status_requests import match_group_runtime_status_request
from gateway.group_target_intents import (
    extract_qq_group_target,
    extract_recent_target_from_history,
    extract_weixin_group_target,
)
from gateway.platforms.base import MessageEvent, MessageType
from gateway.qq_group_moderation_requests import (
    extract_qq_oral_moderation_duration_seconds,
    extract_qq_oral_moderation_reason,
    extract_qq_oral_moderation_user_query,
    match_qq_group_moderation_action,
    match_qq_group_moderation_request,
)
from gateway.qq_group_policies import get_group_policy
from gateway.qq_intel_assignments import get_group_monitoring_overlay, list_intel_workers
from gateway.qq_intel_control_requests import (
    extract_qq_oral_intel_hire_objective,
    extract_qq_worker_name,
    looks_like_qq_intel_worker_context,
    match_qq_intel_control_request,
)
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
from gateway.send_intents import (
    extract_qq_inline_send_target_and_message,
    extract_send_confirmation_message,
    extract_weixin_inline_send_target_and_message,
    looks_like_send_confirmation,
    looks_like_send_query,
)
from gateway.send_requests import match_send_request
from gateway.session import SessionSource
from gateway.weixin_group_archive import WeixinGroupArchiveStore
from gateway.weixin_group_policies import get_group_policy as get_weixin_group_policy

logger = logging.getLogger(__name__)


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
        source = getattr(event, "source", None)
        if getattr(source, "platform", None) != platform:
            return None, ""
        if event.get_command():
            return None, ""
        if getattr(event, "message_type", None) != MessageType.TEXT:
            return None, ""
        if getattr(event, "media_urls", None):
            return None, ""

        body = str(getattr(event, "text", "") or "").strip()
        if not body:
            return None, ""
        return source, body

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

    def _extract_recent_send_target_from_history(
        self,
        source: SessionSource,
        conversation_history: Optional[list[dict[str, Any]]],
        *,
        target_extractor,
    ) -> str:
        return extract_recent_target_from_history(
            source,
            conversation_history,
            extractor=target_extractor,
            predicate=lambda item, content: (
                str(item.get("role") or "").strip().lower() == "user"
                and looks_like_send_query(content)
            ),
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

    def _match_admin_send_request_common(
        self,
        *,
        source: SessionSource,
        body: str,
        conversation_history: Optional[list[dict[str, Any]]],
        inline_extractor,
        history_target_extractor,
        direct_target_extractor,
        query_prompt_formatter,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return match_send_request(
            source=source,
            body=body,
            conversation_history=conversation_history,
            inline_extractor=inline_extractor,
            history_target_extractor=history_target_extractor,
            direct_target_extractor=direct_target_extractor,
            looks_like_send_query=looks_like_send_query,
            looks_like_send_confirmation=looks_like_send_confirmation,
            extract_send_confirmation_message=extract_send_confirmation_message,
            query_prompt_formatter=query_prompt_formatter,
        )

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
        source, body = self._extract_platform_text_event_body(event, platform=platform)
        if source is None:
            return None, None
        if not self.owner._configured_admin_user_ids(getattr(source, "platform", None)):
            return None, None
        if not self.owner._is_admin_user(source):
            return None, None
        return self._match_admin_send_request_common(
            source=source,
            body=body,
            conversation_history=conversation_history,
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
            history_target_extractor=lambda source, history: self._extract_recent_send_target_from_history(
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
            history_target_extractor=lambda source, history: self._extract_recent_send_target_from_history(
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

    @staticmethod
    def _run_send_shortcut_tool(tool_args: dict[str, Any], *, target_formatter):
        from tools.send_message_tool import send_message_tool

        target = str(tool_args.get("target") or "").strip()
        message = str(tool_args.get("message") or "").strip()
        raw = send_message_tool(
            {
                "action": "send",
                "target": target_formatter(target),
                "message": message,
            }
        )
        return json.loads(raw) if isinstance(raw, str) else (raw or {})

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
        if shortcut_error:
            return shortcut_error
        if not tool_args:
            return None

        try:
            result = self._run_send_shortcut_tool(
                tool_args,
                target_formatter=target_formatter,
            )
        except Exception as exc:
            logger.warning("%s: %s", error_prefix, exc)
            return f"{error_prefix}：{exc}"

        if result.get("error"):
            return str(result["error"])
        return reply_formatter(tool_args)

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
        source, body = self._extract_platform_text_event_body(
            event,
            platform=Platform.QQ_NAPCAT,
        )
        if source is None:
            return None, None
        known_worker_names = {
            str(item.get("worker_name") or "").strip()
            for item in list_intel_workers()
            if isinstance(item, dict) and str(item.get("worker_name") or "").strip()
        }
        return match_qq_intel_control_request(
            source=source,
            body=body,
            admin_ids_configured=bool(self.owner._configured_admin_user_ids(getattr(source, "platform", None))),
            is_admin_user=self.owner._is_admin_user(source),
            looks_like_joined_group_list_query=self.owner._looks_like_joined_group_list_query,
            extract_worker_name=extract_qq_worker_name,
            looks_like_worker_context=looks_like_qq_intel_worker_context,
            known_worker_names=known_worker_names,
            target_extractor=extract_qq_group_target,
            report_target_resolver=self._resolve_oral_report_delivery_target,
            hire_objective_extractor=extract_qq_oral_intel_hire_objective,
        )

    def _format_admin_qq_intel_control_reply(self, tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        action = str(tool_args.get("action") or "").strip().lower()
        if action == "list_joined_groups":
            groups = list(result.get("groups") or [])
            if not groups:
                return "当前还没查到已加入的 QQ 群。"
            lines = ["当前已加入的 QQ 群："]
            for item in groups[:20]:
                if not isinstance(item, dict):
                    continue
                group_id = str(item.get("group_id") or "").strip()
                group_name = str(item.get("group_name") or group_id).strip()
                lines.append(f"- {group_name} ({group_id})")
            return "\n".join(lines)

        worker = result.get("worker") or {}
        worker_name = str(worker.get("worker_name") or tool_args.get("worker_name") or "").strip()
        status_label = self.owner._format_intel_worker_status_label(worker.get("status"))
        if action == "hire_worker":
            target_group = str(
                worker.get("target_group_id")
                or worker.get("target_group_ref")
                or tool_args.get("target_group")
                or ""
            ).replace("group:", "").strip()
            return f"已安排情报员 {worker_name} 去 QQ 群 {target_group} 执行任务。当前状态：{status_label}。"
        if action == "pause_worker":
            return f"情报员 {worker_name} 已暂停。当前状态：{status_label}。"
        if action == "resume_worker":
            return f"情报员 {worker_name} 已恢复任务。当前状态：{status_label}。"
        if action == "stop_worker":
            return f"情报员 {worker_name} 已停用。当前状态：{status_label}。"
        if action == "run_report_now":
            delivery = str(
                (result.get("delivery") or {}).get("target")
                or tool_args.get("manual_report_target")
                or ""
            ).strip()
            if delivery:
                return f"已让情报员 {worker_name} 立即汇报，发送到 {delivery}。"
            return f"已让情报员 {worker_name} 立即汇报。"

        group_id = str(worker.get("target_group_id") or "").strip()
        group_name = str(worker.get("target_group_name") or "").strip()
        objective = str(worker.get("objective") or "").strip()
        lines = [f"情报员 {worker_name} 当前状态：{status_label}。"]
        if group_id or group_name:
            label = group_name or group_id
            if group_id and group_name and group_id != group_name:
                label = f"{group_name} ({group_id})"
            lines.append(f"目标群：{label}")
        if objective:
            lines.append(f"任务：{objective}")
        daily_targets = self.owner._unique_report_targets([worker.get("daily_report_target")])
        manual_targets = self.owner._unique_report_targets([worker.get("manual_report_target")])
        if bool(worker.get("daily_report_enabled")) and daily_targets:
            lines.append(f"日报目标：{', '.join(daily_targets)}")
        if manual_targets:
            lines.append(f"立即汇报目标：{', '.join(manual_targets)}")
        last_error = str(worker.get("last_error") or "").strip()
        if last_error:
            lines.append(f"备注：{last_error}")
        return "\n".join(lines)

    def _try_handle_admin_qq_intel_control(self, event: MessageEvent) -> str | None:
        tool_args, shortcut_error = self._match_admin_qq_intel_control_request(event)
        if shortcut_error:
            return shortcut_error
        if not tool_args:
            return None

        try:
            from tools.qq_control_tool import qq_control_tool

            raw = qq_control_tool(tool_args)
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception as exc:
            logger.warning("Admin QQ oral intel control shortcut failed: %s", exc)
            return f"QQ 情报员控制执行失败：{exc}"

        if result.get("error"):
            return str(result["error"])
        return self._format_admin_qq_intel_control_reply(tool_args, result)

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
        source, body = self._extract_platform_text_event_body(event, platform=platform)
        if source is None:
            return None
        target = match_group_runtime_status_request(
            source=source,
            body=body,
            conversation_history=conversation_history,
            admin_ids_configured=bool(self.owner._configured_admin_user_ids(getattr(source, "platform", None))),
            is_admin_user=self.owner._is_admin_user(source),
            looks_like_group_runtime_status_query=looks_like_shared_group_runtime_status_query,
            target_extractor=target_extractor,
            history_target_extractor=history_target_extractor,
        )
        if not target:
            return None
        return format_group_runtime_status_reply(**status_loader(target))

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

    def _match_admin_group_control_request_common(
        self,
        *,
        source: SessionSource,
        body: str,
        target: str | None,
        missing_target_message: str,
        admin_action_label: str,
        collect_only_action: str,
        report_target_resolver,
    ) -> tuple[dict[str, Any] | None, str | None]:
        return match_group_control_request(
            source=source,
            body=body,
            target=target,
            admin_ids_configured=bool(self.owner._configured_admin_user_ids(getattr(source, "platform", None))),
            is_admin_user=self.owner._is_admin_user(source),
            missing_target_message=missing_target_message,
            admin_only_message=self.owner._admin_only_message(source, admin_action_label),
            collect_only_action=collect_only_action,
            report_target_resolver=report_target_resolver,
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
        source, body = self._extract_platform_text_event_body(event, platform=platform)
        return match_admin_platform_group_control_request(
            source=source,
            body=body,
            target_extractor=target_extractor,
            admin_ids_configured=bool(self.owner._configured_admin_user_ids(getattr(source, "platform", None)))
            if source is not None
            else False,
            is_admin_user=self.owner._is_admin_user(source) if source is not None else False,
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
        source, body = self._extract_platform_text_event_body(
            event,
            platform=Platform.QQ_NAPCAT,
        )
        if source is None:
            return None, None
        return match_qq_group_moderation_request(
            source=source,
            body=body,
            admin_ids_configured=bool(self.owner._configured_admin_user_ids(getattr(source, "platform", None))),
            is_admin_user=self.owner._is_admin_user(source),
            admin_only_message=self.owner._admin_only_message(source, "操作 QQ 群禁言/踢人"),
            action_matcher=match_qq_group_moderation_action,
            target_extractor=extract_qq_group_target,
            user_query_extractor=extract_qq_oral_moderation_user_query,
            reason_extractor=extract_qq_oral_moderation_reason,
            duration_extractor=extract_qq_oral_moderation_duration_seconds,
        )

    @staticmethod
    def _format_admin_qq_group_moderation_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        action = str(result.get("action") or tool_args.get("action") or "").strip().lower()
        group_id = str(result.get("group_id") or tool_args.get("target") or "").replace("group:", "").strip()
        member_name = str(
            result.get("member_name")
            or tool_args.get("user_query")
            or result.get("user_id")
            or "目标成员"
        ).strip()
        reason = str(result.get("reason") or tool_args.get("reason") or "").strip()
        if action == "mute_user":
            duration_seconds = int(result.get("duration_seconds") or tool_args.get("duration_seconds") or 0)
            line = f"已把 QQ 群 {group_id} 的 {member_name} 禁言 {duration_seconds} 秒。"
        else:
            line = f"已把 QQ 群 {group_id} 的 {member_name} 踢出。"
        if reason:
            line += f" 原因：{reason}。"
        return line

    def _try_handle_admin_qq_group_moderation(self, event: MessageEvent) -> str | None:
        tool_args, shortcut_error = self._match_admin_qq_group_moderation_request(event)
        if shortcut_error:
            return shortcut_error
        if not tool_args:
            return None

        try:
            from tools.qq_control_tool import qq_control_tool

            raw = qq_control_tool(tool_args)
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception as exc:
            logger.warning("Admin QQ oral moderation shortcut failed: %s", exc)
            return f"QQ 群管理执行失败：{exc}"

        if result.get("error"):
            return str(result["error"])
        return self._format_admin_qq_group_moderation_reply(tool_args, result)

    def _match_admin_qq_social_control_request(
        self,
        event: MessageEvent,
    ) -> tuple[dict[str, Any] | None, str | None]:
        source, body = self._extract_platform_text_event_body(
            event,
            platform=Platform.QQ_NAPCAT,
        )
        if source is None:
            return None, None
        return match_qq_social_control_request(
            source=source,
            body=body,
            admin_ids_configured=bool(self.owner._configured_admin_user_ids(getattr(source, "platform", None))),
            is_admin_user=self.owner._is_admin_user(source),
            admin_only_message=self.owner._admin_only_message(source, "处理 QQ 社交请求"),
            looks_like_request_list_query=_looks_like_qq_social_request_list_query,
            looks_like_policy_candidate=_looks_like_qq_social_policy_candidate,
            looks_like_policy_query=looks_like_qq_social_policy_query,
            request_type_matcher=match_qq_social_request_type,
            notify_target_resolver=qq_social_policy_notify_target,
        )

    @staticmethod
    def _format_admin_qq_social_control_reply(tool_args: dict[str, Any], result: dict[str, Any]) -> str:
        action = str(tool_args.get("action") or "").strip().lower()
        if action == "list_requests":
            requests = list(result.get("requests") or [])
            request_type = str(tool_args.get("request_type") or "").strip().lower()
            if not requests:
                if request_type == "friend":
                    return "当前没有待处理的 QQ 好友申请。"
                if request_type == "group":
                    return "当前没有待处理的 QQ 加群/邀请申请。"
                return "当前没有待处理的 QQ 社交申请。"

            if request_type == "friend":
                lines = ["当前待处理的 QQ 好友申请："]
            elif request_type == "group":
                lines = ["当前待处理的 QQ 加群/邀请申请："]
            else:
                lines = ["当前待处理的 QQ 社交申请："]
            for item in requests[:10]:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("request_key") or "").strip()
                user_id = str(item.get("user_id") or "").strip()
                group_id = str(item.get("group_id") or "").strip()
                comment = str(item.get("comment") or "").strip()
                line = f"- {key}"
                if user_id:
                    line += f" | 用户 {user_id}"
                if group_id:
                    line += f" | 群 {group_id}"
                if comment:
                    line += f" | 备注：{comment}"
                lines.append(line)
            return "\n".join(lines)

        policy = result.get("policy") or {}
        lines = ["QQ 社交自动处理策略已更新：" if action == "set_social_policy" else "QQ 社交自动处理策略："]
        enabled_label = "已开启" if action == "set_social_policy" else "开"
        disabled_label = "已关闭" if action == "set_social_policy" else "关"
        lines.append(
            f"- 好友申请自动通过：{enabled_label if bool(policy.get('auto_approve_friend_requests')) else disabled_label}"
        )
        lines.append(
            f"- 加群申请自动通过：{enabled_label if bool(policy.get('auto_approve_group_add_requests')) else disabled_label}"
        )
        lines.append(
            f"- 群邀请自动通过：{enabled_label if bool(policy.get('auto_approve_group_invites')) else disabled_label}"
        )
        notify_target = str(policy.get("notify_target") or "").strip()
        if notify_target:
            lines.append(f"- 通知目标：{notify_target}")
        return "\n".join(lines)

    def _try_handle_admin_qq_social_control(self, event: MessageEvent) -> str | None:
        tool_args, shortcut_error = self._match_admin_qq_social_control_request(event)
        if shortcut_error:
            return shortcut_error
        if not tool_args:
            return None

        try:
            from tools.qq_social_tool import qq_social_tool

            raw = qq_social_tool(tool_args)
            result = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception as exc:
            logger.warning("Admin QQ social shortcut failed: %s", exc)
            return f"QQ 社交控制执行失败：{exc}"

        if result.get("error"):
            return str(result["error"])
        return self._format_admin_qq_social_control_reply(tool_args, result)
