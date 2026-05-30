"""阶段一：危险信号提取 — 不做拒绝，仅为 LLM prompt 提供上下文。

设计原则：
  第一层不做硬编码 DENY。所有"看起来危险"的操作全部交给 LLM 语义判断。
  本模块只提取危险信号作为 LLM 的参考上下文，不替 LLM 做决定。

对于 terminal 命令：
  导入系统 tools.approval 的 HARDLINE / DANGEROUS 正则检测，
  提取匹配到的规则描述 + pattern_key（用于 approve_session 预标记）。

返回:
  dict — 注入 LLM prompt 的上下文信息
    {
      "signals": [str, ...],              # 危险信号描述（给 LLM 看）
      "dangerous_pattern_keys": [str, ..],# 匹配到的 DANGEROUS pattern key（不含 HARDLINE）
      "tool_name": str,
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ── 敏感路径（仅作为上下文提示，不做硬编码拒绝）───────────────────
_SENSITIVE_PATHS = (
    "/etc/",
    "/boot/",
    "/sys/",
    "/proc/",
    "/dev/",
    "~/.ssh/",
    "~/.gnupg/",
)

_SENSITIVE_FILE_NAMES = frozenset({
    ".env", "config.yaml", "id_rsa", "id_ed25519",
    "id_rsa.pub", "authorized_keys",
})

# ── delegate_task 危险关键词（作为上下文提示）─────────────────────
_DANGER_KEYWORDS = (
    "delete all", "rm -rf /", "format disk",
    "wipe system", "destroy everything",
)


def _get_sensitive_signals_for_write(path: str) -> List[str]:
    """提取写操作的敏感信号（只返回描述，不做拒绝）。"""
    import os
    signals: List[str] = []
    expanded = os.path.expanduser(path)

    for sp in _SENSITIVE_PATHS:
        if expanded.startswith(os.path.expanduser(sp)):
            signals.append(f"目标路径在系统关键目录下 ({sp}...)")
            break

    basename = path.split("/")[-1].lower()
    if basename in _SENSITIVE_FILE_NAMES:
        signals.append(f"目标文件是敏感文件 ({basename})，可能包含密钥或配置")

    return signals


def _get_terminal_risk_signals(command: str) -> Tuple[List[str], List[str]]:
    """对 terminal 命令做 dangerous/hardline 正则检测。

    导入系统 tools.approval 的检测函数，复用 12 HARDLINE + 47 DANGEROUS 正则。

    Returns:
        (signals, dangerous_pattern_keys)
          - signals: 人类可读的风险描述（注入 LLM prompt）
          - dangerous_pattern_keys: 匹配到的 DANGEROUS pattern keys（用于 approve_session 预标记）
            注意：HARDLINE 的 key 不包含在内——HARDLINE 永远不应被预批准
    """
    signals: List[str] = []
    pattern_keys: List[str] = []
    try:
        from tools.approval import (
            detect_dangerous_command,
            detect_hardline_command,
        )

        # Hardline 检测（毁灭性命令 — 不做预标记，永远无条件拦截）
        is_hardline, hardline_desc = detect_hardline_command(command)
        if is_hardline:
            signals.append(f"⚠️ 触发 HARDLINE 模式: {hardline_desc}")

        # Dangerous 检测（可审批的危险模式 — 保留 pattern_key 用于预标记）
        is_dangerous, pattern_key, description = detect_dangerous_command(command)
        if is_dangerous:
            signals.append(f"⚠️ 触发危险模式: {description}")
            pattern_keys.append(pattern_key)

        if not signals:
            signals.append("未匹配到已知危险模式（可能是安全的常规操作）")

    except ImportError:
        signals.append("（无法加载系统危险命令检测模块，请人工判断）")
    except Exception as exc:
        signals.append(f"（危险命令检测出错: {exc}，请人工判断）")

    return signals, pattern_keys


def _get_delegate_risk_signals(goal: str) -> List[str]:
    """提取 delegate_task 的危险信号。"""
    signals: List[str] = []
    if goal and any(kw in goal.lower() for kw in _DANGER_KEYWORDS):
        signals.append("子任务目标包含明确的破坏性操作关键词")
    return signals


def extract_context(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """提取工具调用的危险信号上下文。

    不做硬编码拒绝。所有信号仅作为 LLM prompt 的参考信息。

    Returns:
        {
            "signals": [str, ...],                # 危险信号描述（给 LLM 看）
            "dangerous_pattern_keys": [str, ...], # terminal 的 DANGEROUS pattern key（用于预标记）
            "tool_name": str,
        }
    """
    signals: List[str] = []
    dangerous_pattern_keys: List[str] = []

    if tool_name in {"write_file", "patch"}:
        path = args.get("path", "")
        if path:
            signals.extend(_get_sensitive_signals_for_write(path))

    elif tool_name == "terminal":
        command = args.get("command", "")
        if command:
            sigs, pks = _get_terminal_risk_signals(command)
            signals.extend(sigs)
            dangerous_pattern_keys = pks

    elif tool_name == "delegate_task":
        goal = args.get("goal", "")
        if goal:
            signals.extend(_get_delegate_risk_signals(goal))

    return {
        "signals": signals,
        "dangerous_pattern_keys": dangerous_pattern_keys,
        "tool_name": tool_name,
    }
