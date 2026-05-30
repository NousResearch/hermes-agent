"""pre_tool_call 主回调调度器 — 两阶段审批入口。

接管所有工具（含 terminal/process）。

调用流：
  SAFE_TOOLS → 直接放行 (0ms)
  其余       → 提取上下文（stage1_rules.extract_context）
               ↓
            阶段一 LLM 快速分类（~500ms）
              ALLOW    → 放行
              ESCALATE → 阶段二 ACP 深度审查 (3-8s)
                          ALLOW → 放行
                          DENY  → 拒绝 + 结构化反馈
                          故障  → return None（交系统处理）

设计原则：
  第一层不做硬编码 DENY。危险信号仅作为 LLM 上下文注入，
  LLM 只输出 ALLOW/ESCALATE。DENY 的决定权在阶段二 ACP。

会话上下文：
  从 Hermes session DB (SessionDB.get_messages) 直接查询：
    - 工具调用链条（所有工具，含 SAFE_TOOLS）
    - 最近对话（用户消息 + Assistant 回复）
  零共享状态，天然并发安全。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 总是安全的工具（只读，跳过全部审查）─────────────────────────
_SAFE_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
    "session_search", "browser_snapshot", "browser_console",
    "browser_get_images", "vision_analyze", "clarify",
    "skills_list", "skill_view", "hindsight_recall", "hindsight_reflect",
    "lcm_grep", "lcm_describe", "lcm_expand", "lcm_expand_query",
    "lcm_status", "lcm_doctor", "lcm_load_session",
})

# ── 配置缓存 ─────────────────────────────────────────────────────
_config_cache: Optional[Dict[str, Any]] = None
_config_disable: bool = False


def _load_config() -> Dict[str, Any]:
    global _config_cache, _config_disable
    if _config_disable:
        return {"enabled": False}
    if _config_cache is not None:
        return _config_cache
    try:
        from hermes_cli.config import load_config
        config = load_config()
        guard_cfg = config.get("plugin_guard", {})
        if not isinstance(guard_cfg, dict):
            guard_cfg = {}
        _config_cache = guard_cfg
        if not guard_cfg.get("enabled", False):
            _config_disable = True
        return guard_cfg
    except Exception:
        _config_disable = True
        return {"enabled": False}


def _summarize_tool_args(args: Dict[str, Any]) -> str:
    """将工具参数压缩为简短文本。"""
    if not args:
        return ""
    if "command" in args:
        cmd = str(args["command"])
        return cmd[:80] + "..." if len(cmd) > 80 else cmd
    if "path" in args:
        return str(args["path"])
    if "goal" in args:
        goal = str(args["goal"])
        return goal[:80] + "..." if len(goal) > 80 else goal
    for key in ("url", "query", "target", "action", "pattern"):
        val = args.get(key)
        if val:
            return str(val)[:80]
    return ""


def _get_session_context(
    session_id: str,
    tool_count: int = 10,
    turn_count: int = 3,
) -> Dict[str, Any]:
    """从 Hermes session DB 获取会话上下文。

    返回:
      {
        "tool_calls": ["termial git status", "read_file nginx.conf", ...] (最近 tool_count 条),
        "turns": ["👤 User: ...", "🤖 Agent: ...", "  → Tool: ..."] (最近 turn_count 轮),
      }
    """
    result: Dict[str, Any] = {"tool_calls": [], "turns": []}
    if not session_id:
        return result

    try:
        from hermes_state import SessionDB
        db = SessionDB()
        messages = db.get_messages(session_id)
    except Exception as exc:
        logger.debug("Session DB query failed: %s", exc)
        return result

    # ── 提取工具调用链条 ────────────────────────────────────────
    tool_calls_raw: List[str] = []
    for msg in reversed(messages):
        tc_list = msg.get("tool_calls")
        if msg.get("role") == "assistant" and tc_list:
            for tc in tc_list:
                name = tc.get("name", tc.get("function", {}).get("name", ""))
                if not name:
                    continue
                tc_args = tc.get("args", tc.get("function", {}).get("arguments", {}))
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except (json.JSONDecodeError, TypeError):
                        tc_args = {}
                summary = _summarize_tool_args(tc_args)
                tool_calls_raw.append(f"{name} {summary}" if summary else name)
                if len(tool_calls_raw) >= tool_count:
                    break
        if len(tool_calls_raw) >= tool_count:
            break
    result["tool_calls"] = list(reversed(tool_calls_raw))

    # ── 提取最近对话轮次 ────────────────────────────────────────
    # 将 messages 按 user→assistant(tool_calls)→tool_results 分组
    turns: List[str] = []
    current_turn: List[str] = []
    user_count = 0

    for msg in reversed(messages):
        role = msg.get("role", "")

        if role == "user":
            user_count += 1
            content = str(msg.get("content", ""))
            text = content[:200] + "..." if len(content) > 200 else content
            # 用户消息是回合起点 → 翻转并插入
            if current_turn:
                turns.insert(0, "\n".join(reversed(current_turn)))
                current_turn = []
            current_turn.insert(0, f"👤 User: {text}")
            if user_count >= turn_count:
                break

        elif role == "assistant":
            content = str(msg.get("content") or "")
            if content:
                text = content[:150] + "..." if len(content) > 150 else content
                current_turn.insert(0, f"🤖 Agent: {text}")
            # 工具调用已在 tool_calls 中提取，这里不重复

    # 最后一个未完成的回合
    if current_turn:
        turns.insert(0, "\n".join(reversed(current_turn)))

    result["turns"] = turns
    return result


def pre_tool_call_handler(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> Optional[Dict[str, str]]:
    """pre_tool_call 钩子回调。"""
    cfg = _load_config()
    if not cfg.get("enabled", False):
        return None

    tool_args = args if isinstance(args, dict) else {}

    if tool_name in _SAFE_TOOLS:
        return None

    from .stage1_rules import extract_context
    context = extract_context(tool_name, tool_args, cfg)

    # ── Terminal 优化：无风险信号 → 跳过 LLM（0ms）────────────────
    # 注意：HARDLINE 匹配的信号以 "⚠️ 触发 HARDLINE" 开头，不算"无信号"
    #       这类命令不应快跳——让 LLM 看到，且系统第二层会做最终拦截
    if tool_name == "terminal":
        signals = context.get("signals", [])
        has_real_risk = any(
            "⚠️" in s and "HARDLINE" not in s
            for s in signals
        )
        if not has_real_risk:
            # 无 DANGEROUS 匹配，直接放行。系统 HARDLINE 在第二层兜底
            return None

    from .stage1_llm import llm_classify
    verdict = llm_classify(tool_name, tool_args, cfg, context, task_id, session_id)
    if verdict == "ALLOW":
        # ── Terminal 优化：LLM 已批准 → 预写 approve_session 标记 ──
        #     这样系统 check_all_command_guards 运行时，
        #     is_approved() 返回 True → 跳过 DANGEROUS 检测和第二次 LLM
        if tool_name == "terminal":
            pattern_keys = context.get("dangerous_pattern_keys", [])
            if pattern_keys:
                try:
                    from tools.approval import (
                        approve_session, get_current_session_key,
                    )
                    sk = get_current_session_key()
                    for pk in pattern_keys:
                        approve_session(sk, pk)
                    logger.debug(
                        "Pre-approved %d dangerous patterns "
                        "for session %s: %s",
                        len(pattern_keys), sk, pattern_keys,
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to pre-approve patterns: %s", exc
                    )
        return None

    cfg_stage2 = cfg.get("stage2", {})
    if not cfg_stage2.get("enabled", True):
        if cfg.get("fail_open", True):
            return None
        return None

    # 从 session DB 获取完整上下文（工具链条 + 对话）
    session_ctx = _get_session_context(session_id)

    try:
        from .stage2_acp import acp_agent_review
        verdict, detail = acp_agent_review(
            tool_name, tool_args, cfg, context, session_ctx, task_id, session_id
        )
    except Exception as exc:
        logger.warning("Stage2 ACP failed: %s (fail_open=%s)",
                       exc, cfg.get("fail_open", True))
        if cfg.get("fail_open", True):
            return None
        return None

    if verdict == "ALLOW":
        return None
    else:
        from .feedback import build_deny_message
        return build_deny_message(tool_name, tool_args,
                                  detail.get("reason", "acp_deny"), verdict)
