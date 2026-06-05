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
    node_grouped: bool = False,
) -> Dict[str, Any]:
    """格式化飞书交互式卡片消息。

    返回飞书卡片 JSON 结构。
    """
    if level is None:
        level = determine_alert_level(status_code)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tag_info = LEVEL_TAGS.get(level, LEVEL_TAGS["P3"])

    title = f"{tag_info['emoji']} MES {component.upper()} 巡检 - {tag_info['label']}"

    if node_grouped:
        check_text = _format_checks_by_node(checks)
    else:
        check_text = _format_checks_flat(checks)

    separator = "━" * 24

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


def _format_checks_flat(checks: List[Dict[str, Any]]) -> str:
    """平铺格式化检查项。"""
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
    return "\n".join(check_lines) if check_lines else "无检查项"


def _format_checks_by_node(checks: List[Dict[str, Any]]) -> str:
    """按节点分组格式化检查项。"""
    node_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in checks:
        node_name = c.get("details", {}).get("node_name", "")
        if node_name:
            node_map.setdefault(node_name, []).append(c)
        else:
            node_map.setdefault("_no_node", []).append(c)

    lines = []
    for node_name, node_checks in node_map.items():
        if node_name == "_no_node":
            for c in node_checks:
                lines.append(_format_single_check(c, indent=""))
            continue

        worst = max(c.get("status_code", 0) for c in node_checks) if node_checks else 0
        node_icon = "✅" if worst == 0 else ("⚠️" if worst == 1 else "🚨")
        lines.append(f"{node_icon} **{node_name}**")
        for c in node_checks:
            lines.append(_format_single_check(c, indent="  "))

    return "\n".join(lines) if lines else "无检查项"


def _format_single_check(c: Dict[str, Any], indent: str = "") -> str:
    """格式化单个检查项。"""
    status_icon = "✅" if c.get("status_code", 0) == 0 else ("⚠️" if c.get("status_code", 0) == 1 else "🚨")
    name = c.get("name", "unknown")
    message = c.get("message", "")
    value = c.get("value", "")
    threshold = c.get("threshold", "")

    line = f"{indent}{status_icon} **{name}**"
    if value != "":
        line += f": {value}"
    if threshold != "":
        line += f" (阈值: {threshold})"
    if message:
        line += f"\n{indent}   {message}"
    return line


def format_text_message(
    component: str,
    status_code: int,
    checks: List[Dict[str, Any]],
    summary: str,
    level: Optional[str] = None,
    timestamp: Optional[str] = None,
    ai_diagnosis: Optional[str] = None,
    node_grouped: bool = False,
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

    if node_grouped:
        lines.append(_format_checks_by_node_text(checks))
    else:
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


def _format_checks_by_node_text(checks: List[Dict[str, Any]]) -> str:
    """按节点分组格式化检查项（纯文本版）。"""
    node_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in checks:
        node_name = c.get("details", {}).get("node_name", "")
        if node_name:
            node_map.setdefault(node_name, []).append(c)
        else:
            node_map.setdefault("_no_node", []).append(c)

    lines = []
    for node_name, node_checks in node_map.items():
        if node_name == "_no_node":
            for c in node_checks:
                lines.append(_format_single_check_text(c))
            continue

        worst = max(c.get("status_code", 0) for c in node_checks) if node_checks else 0
        node_icon = "✅" if worst == 0 else ("⚠️" if worst == 1 else "🚨")
        lines.append(f"{node_icon} {node_name}")
        for c in node_checks:
            lines.append(f"  {_format_single_check_text(c)}")

    return "\n".join(lines) if lines else "无检查项"


def _format_single_check_text(c: Dict[str, Any]) -> str:
    """格式化单个检查项（纯文本版）。"""
    status_icon = "✅" if c.get("status_code", 0) == 0 else ("⚠️" if c.get("status_code", 0) == 1 else "🚨")
    name = c.get("name", "unknown")
    message = c.get("message", "")
    value = c.get("value", "")
    line = f"{status_icon} {name}"
    if value != "":
        line += f": {value}"
    if message:
        line += f" — {message}"
    return line


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
