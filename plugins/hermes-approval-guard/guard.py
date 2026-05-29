"""pre_tool_call main dispatch — two-stage approval entry point.

Responsibility split:
  terminal commands → skip, defer to existing approvals.mode (smart/manual)
  other tools (write_file/patch/delegate_task/execute_code) → plugin approval

Call flow:
  SAFE_TOOLS → direct allow (0ms)
  HARDLINE   → direct deny + structured feedback (<1ms)
  remainder  → Stage 1 LLM fast classification (~500ms)
  ESCALATE   → Stage 2 ACP Agent deep review (3-8s)

LLM unavailable: fail_open=True → ALLOW (newly-protected tools, no regression)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Always-safe tools (read-only, skip all review) ─────────────────────
_SAFE_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
    "session_search", "browser_snapshot", "browser_console",
    "browser_get_images", "vision_analyze", "clarify",
    "skills_list", "skill_view", "hindsight_recall", "hindsight_reflect",
    "lcm_grep", "lcm_describe", "lcm_expand", "lcm_expand_query",
    "lcm_status", "lcm_doctor", "lcm_load_session",
})

# ── terminal tools handled by existing approvals.mode, plugin does not intercept ──
_TERMINAL_DELEGATED = frozenset({"terminal", "process"})

# ── Config cache ───────────────────────────────────────────────────────
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


def pre_tool_call_handler(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> Optional[Dict[str, str]]:
    """pre_tool_call hook callback.

    Returns:
        None — allow
        {"action": "block", "message": "..."} — block execution
    """
    cfg = _load_config()
    if not cfg.get("enabled", False):
        return None

    tool_args = args if isinstance(args, dict) else {}

    # ── SAFE_TOOLS → direct allow (0ms) ──────────────────────────────
    if tool_name in _SAFE_TOOLS:
        return None

    # ── terminal commands → skip, defer to existing approvals.mode ───
    if tool_name in _TERMINAL_DELEGATED:
        return None

    # ── Stage 1: fast path (rule engine) ──────────────────────────────
    from .stage1_rules import fast_path
    result = fast_path(tool_name, tool_args, cfg)
    if result is not None:
        return result  # HARDLINE deny

    # ── Stage 1 continued: LLM fast classification ───────────────────
    from .stage1_llm import llm_classify
    verdict = llm_classify(tool_name, tool_args, cfg, task_id, session_id)
    if verdict == "ALLOW":
        return None
    elif verdict == "DENY":
        from .feedback import build_deny_message
        return build_deny_message(tool_name, tool_args, "stage1_llm", verdict)
    # ESCALATE → Stage 2

    # ── Stage 2: ACP Agent deep review ────────────────────────────────
    cfg_stage2 = cfg.get("stage2", {})
    if not cfg_stage2.get("enabled", True):
        # Stage 2 not enabled, LLM unsure → follow fail_open policy
        if cfg.get("fail_open", True):
            return None  # Allow: don't block when newly-protected tools fail
        from .feedback import build_deny_message
        return build_deny_message(
            tool_name, tool_args, "escalate_no_stage2", "DENY")

    try:
        from .stage2_acp import acp_agent_review
        verdict, detail = acp_agent_review(tool_name, tool_args, cfg, task_id, session_id)
    except Exception as exc:
        logger.warning("Stage2 ACP failed: %s (fail_open=%s)",
                       exc, cfg.get("fail_open", True))
        if cfg.get("fail_open", True):
            return None
        from .feedback import build_deny_message
        return build_deny_message(tool_name, tool_args,
                                  f"acp_error:{exc}", "DENY")

    if verdict == "ALLOW":
        return None
    else:
        from .feedback import build_deny_message
        return build_deny_message(tool_name, tool_args,
                                  detail.get("reason", "acp_deny"), verdict)
