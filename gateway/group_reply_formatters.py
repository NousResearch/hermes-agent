"""Shared operator-facing reply formatters for group runtime/control shortcuts."""

from __future__ import annotations

from typing import Any


def format_group_runtime_status_reply(
    *,
    platform_label: str,
    target_label: str,
    effective_mode: str,
    can_reply_in_group: bool,
    archive_enabled: bool,
    daily_report_enabled: bool,
    daily_targets: list[str],
    manual_targets: list[str],
    worker_names: list[str] | None = None,
) -> str:
    lines = [f"{platform_label} {target_label} 当前模式：{effective_mode}。"]
    lines.append(f"群里主动说话：{'能' if can_reply_in_group else '不能'}。")
    lines.append(f"归档：{'开' if archive_enabled else '关'}。")
    lines.append(f"日报：{'开' if daily_report_enabled else '关'}。")
    if daily_targets:
        lines.append(f"日报目标：{', '.join(daily_targets)}")
    if manual_targets:
        lines.append(f"立即汇报目标：{', '.join(manual_targets)}")
    if worker_names:
        lines.append(f"当前监听情报员：{', '.join(worker_names)}")
    else:
        lines.append("当前没有活跃情报员在监听。")
    return "\n".join(lines)


def format_admin_group_control_reply(
    tool_args: dict[str, Any],
    result: dict[str, Any],
    *,
    platform_label: str,
    target_key: str,
    collect_only_action: str,
    strip_group_prefix: bool,
) -> str:
    def _normalize_target(value: Any) -> str:
        text = str(value or "").strip()
        if strip_group_prefix:
            return text.replace("group:", "")
        return text

    if tool_args.get("action") == "deliver_report":
        report = result.get("report") or {}
        delivery = (result.get("delivery") or {}).get("target") or tool_args.get("delivery_target") or ""
        target_value = _normalize_target(report.get(target_key) or tool_args.get("target") or "")
        subject = f"{platform_label} {target_value}".strip()
        if delivery:
            return f"已把 {subject} 的汇报发到 {delivery}。"
        return f"已把 {subject} 的汇报发出。"

    policy = result.get("policy") or {}
    target_value = _normalize_target(policy.get(target_key) or tool_args.get("target") or "")
    subject = f"{platform_label} {target_value}".strip()
    parts: list[str] = []
    mode = str(policy.get("mode") or tool_args.get("mode") or "").strip().lower()
    action = str(tool_args.get("action") or "").strip().lower()
    if mode == "collect_only" or action == collect_only_action:
        parts.append(f"已把 {subject} 切到监听采集模式")
    elif mode == "disabled" or action == "disable_group":
        parts.append(f"已停止 {subject} 的监听采集")
    elif (
        mode == "default"
        and (
            action == "resume_chat"
            or (
                tool_args.get("archive_enabled") is False
                and "mode" in tool_args
            )
        )
    ):
        parts.append(f"已停止 {subject} 的监听采集，恢复正常聊天")

    if "daily_report_enabled" in policy or "daily_report_enabled" in tool_args:
        if bool(policy.get("daily_report_enabled", tool_args.get("daily_report_enabled"))):
            daily_target = str(
                policy.get("daily_report_target")
                or tool_args.get("daily_report_target")
                or ""
            ).strip()
            manual_target = str(
                policy.get("manual_report_target")
                or tool_args.get("manual_report_target")
                or ""
            ).strip()
            if daily_target:
                parts.append(f"日报已开启，发送到 {daily_target}")
            else:
                parts.append("日报已开启")
            if manual_target and manual_target != daily_target:
                parts.append(f"立即汇报发到 {manual_target}")
        else:
            parts.append("日报已关闭")

    if not parts:
        parts.append(f"{subject} 策略已更新")
    return "，".join(parts) + "。"


def format_admin_send_reply(
    tool_args: dict[str, Any],
    *,
    platform_label: str,
    target_normalizer,
) -> str:
    target = target_normalizer(tool_args.get("target"))
    message = str(tool_args.get("message") or "").strip()
    if target:
        return f"已发到 {platform_label} {target}：{message}"
    return f"已发出消息：{message}"
