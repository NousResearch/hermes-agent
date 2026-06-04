"""告警消息格式化器。"""

from datetime import datetime
from typing import Any, Dict, List, Optional


# 严重等级标签
LEVEL_TAGS = {
    "P0": {"emoji": "🚨", "label": "紧急", "color": "red"},
    "P1": {"emoji": "⚠️", "label": "警告", "color": "yellow"},
    "P2": {"emoji": "📊", "label": "关注", "color": "blue"},
    "P3": {"emoji": "📋", "label": "信息", "color": "green"},
}


def determine_alert_level(status_code: int) -> str:
    """根据巡检状态码确定告警级别。"""
    if status_code >= 2:
        return "P0"
    elif status_code >= 1:
        return "P1"
    return "P3"


def format_feishu_card(
    component: str,
    status_code: int,
    checks: List[Dict[str, Any]],
    summary: str,
    level: Optional[str] = None,
    timestamp: Optional[str] = None,
    ai_diagnosis: Optional[str] = None,
) -> Dict[str, Any]:
    """格式化飞书交互式卡片消息。

    返回飞书卡片 JSON 结构。
    """
    if level is None:
        level = determine_alert_level(status_code)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tag_info = LEVEL_TAGS.get(level, LEVEL_TAGS["P3"])

    # 标题
    title = f"{tag_info['emoji']} MES {component.upper()} 巡检 - {tag_info['label']}"

    # 构建检查项摘要
    check_lines = []
    for c in checks:
        status_icon = "✅" if c.get("status_code", 0) == 0 else ("⚠️" if c.get("status_code", 0) == 1 else "🚨")
        name = c.get("name", "unknown")
        message = c.get("message", "")
        value = c.get("value", "")
        threshold = c.get("threshold", "")

        line = f"{status_icon} **{name}**"
        if value != "":
            line += f": {value}"
        if threshold != "":
            line += f" (阈值: {threshold})"
        if message:
            line += f"\n   {message}"
        check_lines.append(line)

    check_text = "\n".join(check_lines) if check_lines else "无检查项"

    # 分隔线
    separator = "━" * 24

    # 构建正文
    body_parts = [
        f"⏰ **时间**: {timestamp}",
        f"🏷 **组件**: {component}",
        f"📊 **状态**: {summary}",
        separator,
        "**检查详情**:",
        check_text,
    ]

    if ai_diagnosis:
        body_parts.extend([separator, "🧠 **AI 诊断**:", ai_diagnosis])

    body_text = "\n".join(body_parts)

    # 飞书卡片结构
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"content": title, "tag": "plain_text"},
            "template": tag_info["color"],
        },
        "elements": [
            {
                "tag": "div",
                "text": {"content": body_text, "tag": "lark_md"},
            }
        ],
    }

    return card


def format_text_message(
    component: str,
    status_code: int,
    checks: List[Dict[str, Any]],
    summary: str,
    level: Optional[str] = None,
    timestamp: Optional[str] = None,
    ai_diagnosis: Optional[str] = None,
) -> str:
    """格式化纯文本消息（飞书卡片降级用）。"""
    if level is None:
        level = determine_alert_level(status_code)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tag_info = LEVEL_TAGS.get(level, LEVEL_TAGS["P3"])
    separator = "━" * 24

    lines = [
        f"{tag_info['emoji']} MES {component.upper()} 巡检 - {tag_info['label']}",
        "",
        f"⏰ 时间: {timestamp}",
        f"🏷 组件: {component}",
        f"📊 状态: {summary}",
        separator,
    ]

    for c in checks:
        status_icon = "✅" if c.get("status_code", 0) == 0 else ("⚠️" if c.get("status_code", 0) == 1 else "🚨")
        name = c.get("name", "unknown")
        message = c.get("message", "")
        value = c.get("value", "")
        line = f"{status_icon} {name}"
        if value != "":
            line += f": {value}"
        if message:
            line += f" — {message}"
        lines.append(line)

    if ai_diagnosis:
        lines.extend([separator, "🧠 AI 诊断:", ai_diagnosis])

    return "\n".join(lines)


def format_memory_entry(
    component: str,
    status_code: int,
    summary: str,
    checks: List[Dict[str, Any]],
    root_cause: str = "",
    fix_action: str = "",
    elapsed_seconds: float = 0,
) -> str:
    """格式化记忆条目（用于写入 Hermes Memory）。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    failed_checks = [c for c in checks if c.get("status_code", 0) > 0]

    lines = [
        f"[{timestamp}] MES 故障案例",
        f"组件: {component}",
        f"状态: {['正常', '告警', '严重'][min(status_code, 2)]}",
        f"摘要: {summary}",
    ]

    if failed_checks:
        lines.append("异常项:")
        for c in failed_checks:
            lines.append(f"  - {c.get('name')}: {c.get('message', '')}")

    if root_cause:
        lines.append(f"根因: {root_cause}")
    if fix_action:
        lines.append(f"修复: {fix_action}")
    if elapsed_seconds:
        lines.append(f"耗时: {elapsed_seconds:.1f}s")

    return "\n".join(lines)
