"""结构化拒绝反馈模板 — 包含原因、替代方案、信任升级路径。

三个级别：
  HARDLINE   — 无条件拒绝（系统保护）
  DENY       — 可覆盖拒绝（需要用户确认理解风险）
  ESCALATE   — 需要用户手动决策
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict


def build_deny_message(
    tool_name: str,
    args: Dict[str, Any],
    reason: str,
    verdict: str = "DENY",
) -> Dict[str, str]:
    """构造结构化拒绝消息。

    主 Agent 收到此消息后可以：
    1. 理解为什么被拒
    2. 知道替代方案
    3. 选择覆盖（回复"确认"或特定短语）
    """
    approval_id = f"apr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    lines = []
    lines.append("=" * 60)
    lines.append("SAFETY REVIEW: DENIED")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Tool:    {tool_name}")
    lines.append(f"Reason:  {reason}")
    lines.append(f"Ref:     {approval_id}")
    lines.append("")

    if verdict == "HARDLINE":
        lines.append("⚠️  HARD PROTECTION — 此操作被无条件拦截")
        lines.append("    系统关键路径保护，不可覆盖。")
    else:
        lines.append("💡 如何覆盖此拦截：")
        lines.append(f'    回复 "确认 允许 {approval_id}"  → 本次放行')
        lines.append('    回复 "总是允许 此操作"           → 永久信任此模式')

    # 添加替代方案提示
    suggestions = _get_suggestions(tool_name, args, reason)
    if suggestions:
        lines.append("")
        lines.append("🔀 替代方案：")
        for s in suggestions:
            lines.append(f"    • {s}")

    lines.append("")
    return {"action": "block", "message": "\n".join(lines)}


def build_hardline_message(
    context: str,
    rule_id: str,
    description: str,
) -> Dict[str, str]:
    """构造硬保护拒绝消息。"""
    approval_id = f"hardline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    lines = [
        "=" * 60,
        "SAFETY REVIEW: HARD PROTECTION — BLOCKED",
        "=" * 60,
        "",
        f"Operation: {context}",
        f"Rule:      {rule_id}",
        f"Reason:    {description}",
        "",
        "⚠️  此操作被系统硬保护拦截，不可覆盖。",
        "    这些路径/操作是系统关键基础设施，",
        "    对它们的修改可能导致数据丢失或系统不可用。",
        f"    拦截ID: {approval_id}",
        "",
    ]
    return {"action": "block", "message": "\n".join(lines)}


def _get_suggestions(
    tool_name: str, args: Dict[str, Any], reason: str
) -> list[str]:
    """根据工具类型和原因生成替代方案。"""
    suggestions = []

    if tool_name in {"write_file", "patch"}:
        path = args.get("path", "")
        if "/etc/" in path:
            suggestions.append("写入项目目录 → 审核后通过 terminal 部署到系统路径")
            suggestions.append(f"terminal: sudo cp <project_path> {path}")
        elif ".env" in path.split("/")[-1].lower():
            suggestions.append("仅修改 .env.example 模板文件")
            suggestions.append("敏感密钥通过 hermes config set 配置，不直接编辑 .env")

    elif tool_name == "terminal":
        command = args.get("command", "")
        if "rm" in command and "-rf" in command:
            suggestions.append("先运行 ls 确认目标路径")
            suggestions.append("用 rm (不带 f) 逐文件删除，出问题时好回滚")
            suggestions.append("用 mv 移到 /tmp 而非直接删除，确认后再清理")

    elif tool_name == "delegate_task":
        suggestions.append("将破坏性操作拆分成独立的小任务，逐个审批")
        suggestions.append("在子 agent 的 goal 中明确声明安全边界")

    return suggestions
