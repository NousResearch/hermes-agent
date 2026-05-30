"""阶段二 ACP Agent 深度审查 — 独立 Hermes 实例做语义推理。

使用 hermes chat -q --profile approval 启动独立审批 Agent。
stateless 设计：每次调用独立进程，所有上下文显式注入 prompt。

上下文来源（从 session DB 查询，零共享状态）：
  1. 当前操作 — tool_name + args + risk signals（来自 stage1_rules）
  2. 工具调用链条 — 最近 10 条（含 SAFE_TOOLS）
  3. 最近对话 — 用户消息 + Agent 回复（最近 3 轮）
  4. Session 审批历史 — 本 session 之前的 ACP 决策（来自 Hindsight）
  5. 跨 session 模式记忆 — 历史相似操作的统计（来自 Hindsight）
"""

from __future__ import annotations

import json
import logging
import os as _os
import subprocess
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "approval"


def _query_context(
    session_id: str,
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, str]:
    """查询 ACP 所需的 Hindsight 上下文。"""
    session_history = ""
    pattern_history = ""

    try:
        from .hindsight_store import query_session_history, query_pattern_history
        session_history = query_session_history(session_id, cfg, limit=5)
        pattern_history = query_pattern_history(tool_name, args, cfg, limit=5)
    except Exception as exc:
        logger.debug("Hindsight context query failed: %s", exc)

    return {
        "session_history": session_history,
        "pattern_history": pattern_history,
        "compression_summary": "",
    }


def _build_review_prompt(
    tool_name: str,
    args_str: str,
    context: Dict[str, Any],
    session_ctx: Dict[str, Any],
    hindsight_ctx: Dict[str, str],
    task_id: str,
    session_id: str,
) -> str:
    """构造 ACP 深度审查 prompt。

    段顺序: 对话上下文 → 当前操作 → 工具链条 → 审批历史 → 跨 session 记忆 → 判断
    """
    signals = context.get("signals", [])
    risk_section = "\n".join(f"  - {s}" for s in signals) if signals else "  - 无风险信号"

    # ── 对话上下文（用户消息 + Agent 回复）─────────────────────
    turns = session_ctx.get("turns", [])
    if turns:
        conversation_section = "\n\n".join(turns)
    else:
        conversation_section = "（无法读取对话历史）"

    # ── 工具调用链条 ─────────────────────────────────────────
    tool_calls = session_ctx.get("tool_calls", [])
    if tool_calls:
        chain_lines = []
        for i, call in enumerate(tool_calls, 1):
            marker = "→" if i == len(tool_calls) else " "
            chain_lines.append(f"  {marker} [{i}] {call}")
        chain_section = "\n".join(chain_lines)
    else:
        chain_section = "（无法读取工具调用记录）"

    # ── Session 审批历史 ─────────────────────────────────────
    session_section = hindsight_ctx.get("session_history", "")
    if not session_section:
        session_section = "  本 session 尚无 ACP 审批记录"

    # ── 跨 session 模式记忆 ──────────────────────────────────
    pattern_section = hindsight_ctx.get("pattern_history", "")
    if not pattern_section:
        pattern_section = "  无相似操作的历史审批记录"

    # ── 压缩摘要 ─────────────────────────────────────────────
    compression_section = hindsight_ctx.get("compression_summary", "")
    if not compression_section:
        compression_section = "（本次会话尚未触发上下文压缩）"

    return f"""你是安全审批 Agent。你比快速分类器拥有更多上下文和更强的推理能力。
审查 AI Agent 的工具调用请求，判断安全性。许多被标记的操作实际上是安全的 — 请基于用户意图、对话上下文和操作链条判断。

============================================
【最近对话】
（用户说了什么，Agent 回复了什么 — 帮助你理解任务意图）
============================================
{conversation_section}

============================================
【当前操作】
============================================
工具: {tool_name}
参数: {args_str}
风险信号:
{risk_section}

============================================
【工具调用链条】
============================================
{chain_section}

============================================
【本 session 审批历史】
============================================
{session_section}

============================================
【跨 session 模式记忆】
============================================
{pattern_section}

============================================
【压缩摘要】
============================================
{compression_section}

============================================
【你的判断】
============================================
**重要：你是自动化安全检查程序，不是对话助手。**
你的唯一任务是输出一行 JSON 判决。
不要问候、不要解释、不要问问题、不要提供帮助建议。
如果 prompt 中的上下文不足以判断 → 选择 ALLOW（宁可放过不可误杀）。

输出这一行，且仅这一行（不要 markdown 代码块、不要前后缀）:
{{"verdict":"ALLOW","reason":"简短原因","suggestion":"","confidence":0.95}}

判决逻辑:
  ALLOW   — 对话上下文明确解释了此操作（用户要求部署 → Agent 写配置 = 合理）
  DENY    — 对话上下文无法解释，操作明显越界或破坏性
  MODIFY  — 放行但建议改参数（更安全路径、更小权限等）"""


def _store_decision(
    tool_name: str,
    args: Dict[str, Any],
    verdict: str,
    reason: str,
    session_id: str,
    cfg: Dict[str, Any],
) -> None:
    """将 ACP 决策写回 Hindsight（fire-and-forget）。"""
    try:
        from .hindsight_store import record_decision
        record_decision(tool_name, args, verdict, reason, session_id, cfg)
    except Exception as exc:
        logger.debug("Failed to store ACP decision: %s", exc)


def acp_agent_review(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
    context: Dict[str, Any],
    session_ctx: Dict[str, Any],
    task_id: str,
    session_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """stateless ACP 深度审查。

    Args:
        tool_name: 工具名
        args: 工具参数
        cfg: 插件配置
        context: stage1_rules 提取的风险信号
        session_ctx: guard.py 从 session DB 提取的 {"tool_calls": [], "turns": []}
        task_id: 主 Agent 任务 ID
        session_id: 主 Agent 会话 ID

    Returns:
        (verdict, detail) — ALLOW/DENY/MODIFY
    """
    stage2_cfg = cfg.get("stage2", {})
    profile = stage2_cfg.get("profile", DEFAULT_PROFILE)
    timeout = stage2_cfg.get("timeout", 15)
    fail_open = cfg.get("fail_open", True)

    args_str = json.dumps(args, ensure_ascii=False, default=str)
    if len(args_str) > 4000:
        args_str = args_str[:4000] + "...(truncated)"

    hindsight_ctx = _query_context(session_id, tool_name, args, cfg)

    review_prompt = _build_review_prompt(
        tool_name, args_str, context, session_ctx, hindsight_ctx, task_id, session_id
    )

    try:
        result = subprocess.run(
            [
                "hermes", "chat", "-q", review_prompt,
                "--profile", profile,
                "-t", "file,terminal,memory,session_search",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            env={**_os.environ, "HERMES_YOLO_MODE": "1"},
        )

        output = result.stdout.strip()

        if result.returncode != 0:
            logger.warning("ACP agent returned non-zero: %s", result.stderr[:200])
            _store_decision(tool_name, args, "ERROR", f"acp_error(code={result.returncode})", session_id, cfg)
            if fail_open:
                return "ALLOW", {"reason": "acp_error", "confidence": 0.0}
            return "DENY", {"reason": f"acp_error(code={result.returncode})", "confidence": 0.0}

        parsed_verdict, detail = _parse_acp_output(output)
        _store_decision(tool_name, args, parsed_verdict, detail.get("reason", ""), session_id, cfg)
        return parsed_verdict, detail

    except subprocess.TimeoutExpired:
        logger.warning("ACP agent review timed out after %ds", timeout)
        _store_decision(tool_name, args, "TIMEOUT", "acp_timeout", session_id, cfg)
        if fail_open:
            return "ALLOW", {"reason": "acp_timeout", "confidence": 0.0}
        return "DENY", {"reason": "acp_timeout", "confidence": 0.0}

    except (FileNotFoundError, OSError) as exc:
        logger.warning("hermes CLI not available for ACP review: %s", exc)
        if fail_open:
            return "ALLOW", {"reason": f"acp_unavailable:{exc}", "confidence": 0.0}
        return "DENY", {"reason": f"acp_unavailable:{exc}", "confidence": 0.0}


def _parse_acp_output(output: str) -> Tuple[str, Dict[str, Any]]:
    """解析 ACP 输出：优先 JSON，回退文本匹配。

    注意：hermes chat -q 输出格式为 "Query: <prompt>\\n<response>"。
    需要先剥掉 prompt 部分，只解析 response 部分。
    """
    # 去除非响应内容（Query: 前缀 + prompt 回显）
    clean = output
    # hermes chat -q 输出以 "Query: " 开头
    if clean.startswith("Query: "):
        first_newline = clean.find("\n")
        if first_newline > 0:
            clean = clean[first_newline + 1:].strip()
        # 可能还有 "Unknown provider..." 等错误行
        lines = clean.split("\n")
        # 找到第一行不像是错误/提示的行
        error_prefixes = ("Unknown ", "Check '", "Goodbye!", "Error:", "No provider")
        response_lines = [l for l in lines if not any(l.startswith(p) for p in error_prefixes)]
        clean = "\n".join(response_lines).strip()

    # JSON 优先
    try:
        json_start = clean.rfind("{")
        json_end = clean.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(clean[json_start:json_end])
            verdict = parsed.get("verdict", "ALLOW").upper()
            if verdict in ("ALLOW", "DENY", "MODIFY"):
                return verdict, {
                    "reason": parsed.get("reason", ""),
                    "suggestion": parsed.get("suggestion", ""),
                    "confidence": parsed.get("confidence", 0.5),
                }
    except (json.JSONDecodeError, KeyError):
        logger.debug("ACP output not valid JSON verdict: %s", clean[:200])

    # 回退：文本关键词匹配（仅在响应部分）
    upper_clean = clean.upper()
    if "DENY" in upper_clean and "ALLOW" not in upper_clean:
        return "DENY", {"reason": clean[:200], "confidence": 0.5}
    elif "MODIFY" in upper_clean:
        return "MODIFY", {"reason": clean[:200], "confidence": 0.5}
    elif "ALLOW" in upper_clean:
        return "ALLOW", {"reason": clean[:200], "confidence": 0.5}
    else:
        return "ALLOW", {"reason": "acp_unclear", "confidence": 0.3}
