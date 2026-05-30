"""pre_tool_call main callback dispatcher — two-stage approval entry point.

Covers all tools (including terminal/process).

Call flow:
  SAFE_TOOLS → direct pass (0ms)
  Others      → extract context (stage1_rules.extract_context)
                ↓
              Stage 1 LLM fast classification (~500ms)
                ALLOW    → pass
                ESCALATE → Stage 2 ACP deep review (3-8s)
                            ALLOW → pass
                            DENY  → block + structured feedback
                            Error → return None (delegate to system)

Design principles:
  Stage 1 never makes hardcoded DENY decisions. Risk signals are only
  injected as LLM context. LLM only outputs ALLOW/ESCALATE.
  DENY authority resides in Stage 2 ACP.

Session context:
  Queried directly from Hermes session DB (SessionDB.get_messages):
    - Tool call chain (all tools, including SAFE_TOOLS)
    - Recent conversation (user messages + Agent replies)
  Zero shared state, naturally concurrency-safe.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Always-safe tools (read-only, skip all review) ─────────────────
_SAFE_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
    "session_search", "browser_snapshot", "browser_console",
    "browser_get_images", "vision_analyze", "clarify",
    "skills_list", "skill_view", "hindsight_recall", "hindsight_reflect",
    "lcm_grep", "lcm_describe", "lcm_expand", "lcm_expand_query",
    "lcm_status", "lcm_doctor", "lcm_load_session",
})

# ── Config cache ───────────────────────────────────────────────────
_config_cache: Optional[Dict[str, Any]] = None
_config_cache_time: float = 0
_config_cache_ttl: float = 30  # seconds — refresh config periodically
_config_disable: bool = False


def _load_config() -> Dict[str, Any]:
    global _config_cache, _config_disable, _config_cache_time
    if _config_disable:
        return {"enabled": False}
    # Refresh cache if TTL expired — allows config changes without gateway restart
    if _config_cache is not None and (time.monotonic() - _config_cache_time) < _config_cache_ttl:
        return _config_cache
    try:
        from hermes_cli.config import load_config
        config = load_config()
        guard_cfg = config.get("plugin_guard", {})
        if not isinstance(guard_cfg, dict):
            guard_cfg = {}
        _config_cache = guard_cfg
        _config_cache_time = time.monotonic()
        _config_disable = False  # reset on refresh — don't carry stale disable state
        if not guard_cfg.get("enabled", False):
            _config_disable = True
        return guard_cfg
    except Exception:
        _config_disable = True
        return {"enabled": False}


def _summarize_tool_args(args: Dict[str, Any]) -> str:
    """Compress tool args to a short text summary."""
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
    """Get session context from Hermes session DB.

    Returns:
      {
        "tool_calls": ["terminal git status", "read_file nginx.conf", ...] (last tool_count),
        "turns": ["User: ...", "Agent: ...", "  Tool: ..."] (last turn_count rounds),
      }
    """
    result: Dict[str, Any] = {"tool_calls": [], "turns": []}
    # Upstream bug: invoke_tool() does not pass session_id to the
    # pre_tool_call hook.  Walk the call stack to discover it from
    # AIAgent._invoke_tool's `self` parameter.  No monkey-patching,
    # no system file modifications — just read what's already there.
    effective_id = session_id
    if not effective_id:
        try:
            import inspect
            for frame_info in inspect.stack():
                if frame_info.function == "_invoke_tool":
                    agent = frame_info.frame.f_locals.get("self")
                    if agent is not None:
                        sid = getattr(agent, "session_id", None)
                        if sid:
                            effective_id = sid
                            logger.info("call-stack session_id discovery: %s", effective_id)
                            break
        except Exception:
            pass
    if not effective_id:
        return result

    try:
        from hermes_state import SessionDB
        db = SessionDB()
        messages = db.get_messages(effective_id)
    except Exception as exc:
        logger.debug("Session DB query failed: %s", exc)
        return result

    # ── Extract tool call chain ───────────────────────────────────
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

    # ── Extract recent conversation turns ─────────────────────────
    # Group messages by user→assistant(tool_calls)→tool_results
    turns: List[str] = []
    current_turn: List[str] = []
    user_count = 0

    for msg in reversed(messages):
        role = msg.get("role", "")

        if role == "user":
            user_count += 1
            content = str(msg.get("content", ""))
            text = content[:200] + "..." if len(content) > 200 else content
            # User message is turn start → reverse and insert
            if current_turn:
                turns.insert(0, "\n".join(reversed(current_turn)))
                current_turn = []
            current_turn.insert(0, f"User: {text}")
            if user_count >= turn_count:
                break

        elif role == "assistant":
            content = str(msg.get("content") or "")
            if content:
                text = content[:150] + "..." if len(content) > 150 else content
                current_turn.insert(0, f"Agent: {text}")
            # Tool calls already extracted in tool_calls, not repeated here

    # Last incomplete turn
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
    """pre_tool_call hook callback."""
    cfg = _load_config()
    if not cfg.get("enabled", False):
        return None

    tool_args = args if isinstance(args, dict) else {}

    if tool_name in _SAFE_TOOLS:
        return None

    from .stage1_rules import extract_context
    context = extract_context(tool_name, tool_args, cfg)

    # ── Terminal optimization: no risk signals → skip LLM (0ms) ────
    # Note: HARDLINE signals start with "WARNING: HARDLINE pattern triggered"
    #       — these should NOT be considered "real risk" for fast-path,
    #       since LLM cannot override HARDLINE anyway.
    if tool_name == "terminal":
        signals = context.get("signals", [])
        has_real_risk = any(
            "WARNING" in s and "HARDLINE" not in s
            for s in signals
        )
        if not has_real_risk:
            # No DANGEROUS match, pass through. System HARDLINE is the safety net.
            return None

    from .stage1_llm import llm_classify
    verdict = llm_classify(tool_name, tool_args, cfg, context, task_id, session_id)
    if verdict == "ALLOW":
        # ── Terminal optimization: LLM approved → pre-write approve_session marker ──
        #     So when system check_all_command_guards runs,
        #     is_approved() returns True → skips DANGEROUS check and second LLM
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
    if not cfg_stage2.get("enabled", False):
        return None

    # Get full context from session DB (tool chain + conversation)
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
