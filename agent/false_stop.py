"""Post-tool-round false-stop detection guard.

When an LLM produces finish_reason=stop with text content but NO tool calls,
and the text indicates intent to continue (e.g., a colon-preamble that lost its
tool_calls, or a narrated continuation), the conversation loop nudges the model
to issue the actual tool call instead of silently ending the turn (#42503).

This is the POST-tool-round sibling of intent_ack_continuation (which fires at
turn START before any tool calls). The two are complementary, not overlapping.

This module is policy-only: it returns a bounded synthetic nudge so the
conversation loop continues instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Optional


# Intent markers: the text declares it is ABOUT TO do work rather
# than reporting a finished result.
_FALSE_STOP_INTENT_MARKERS = (
    # CJK (production-hardened — the real differentiator vs #57610)
    "继续", "接下来", "下一步", "然后", "先读", "先看", "先把",
    "接着看", "再看", "继续看", "再读", "再看下", "随后", "接着",
    "马上", "即将", "然后开始", "先快速", "先确认", "先检查",
    "再实现", "开始", "下一步骤", "先看下", "先梳理",
    # English (balanced for global project, matching action-verb patterns
    # from looks_like_codex_intermediate_ack)
    "continue", "next", "let me", "then read", "read the rest",
    "carry on", "proceed", "i'll", "i will", "now",
    "then i'll", "proceeding to", "moving on to",
)

# Genuine-completion guard: skip candidates that already read as a finished
# deliverable even if they mention next steps.
_FALSE_STOP_DONE_MARKERS = (
    "已完成", "全部完成", "已搞定", "done", "finished",
    "结论如下", "总结如下", "如上", "以上是",
)

# Waiting-for-user guard: skip candidates where the model is genuinely idle.
_FALSE_STOP_WAIT_MARKERS = (
    "等你", "等待", "仍在跑", "正在跑", "仍在运行",
    "等确认", "等决定", "还在跑", "还在运行", "继续等待",
    "waiting for", "still running", "pending",
)

# User-directed / question guard: the model is deferring agency to the user
# or asking a question — structurally turn-ending.
_FALSE_STOP_QUESTION_MARKERS = (
    "你决定", "你来决定", "由你决定", "取决于你",
    "建议你", "你可以", "请确认", "你确认", "你看",
    "如何？", "怎么样？", "要不要", "是否需要", "还是",
)

# CJK sentence-ending punctuation for Signature B's ending gate.
# ASCII "." is intentionally excluded — English completions ending in "."
# would over-trigger. Signature B is CJK-only for the ending-set gate;
# English false-stops are caught via Signature A (colon-preamble) or the
# intent markers alone when content ends with CJK punctuation.
_CJK_ENDINGS = ("。", "！", "？", ";", "…")

_MAX_SIGNATURE_A_LENGTH = 120
_MAX_SIGNATURE_B_LENGTH = 200
_MAX_ATTEMPTS = 2
_TOOL_ROUND_LOOKBACK = 8


def _session_is_messaging_surface() -> bool:
    """Whether this turn is delivered over a human messaging channel."""
    try:
        from gateway.session_context import session_is_messaging_surface
        return session_is_messaging_surface()
    except Exception:
        return False


def false_stop_detection_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return whether false-stop detection is enabled.

    Precedence: an explicit ``HERMES_FALSE_STOP_DETECTION`` env var wins, then
    an explicit ``agent.false_stop_detection`` config value. The config default
    is ``"auto"`` (see ``DEFAULT_CONFIG``) — surface-aware: ON for interactive
    coding surfaces (CLI, TUI, desktop) and programmatic callers, OFF for
    conversational messaging surfaces (Telegram, Discord, etc.) where the
    nudge narrative would reach a human as chat noise. An explicit bool forces
    the behavior in either direction. A missing or unrecognized value falls
    back to the surface-aware ``"auto"`` default.
    """
    env = os.environ.get("HERMES_FALSE_STOP_DETECTION")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    cfg_val = agent_cfg.get("false_stop_detection") if isinstance(agent_cfg, dict) else None
    if isinstance(cfg_val, bool):
        return cfg_val
    if isinstance(cfg_val, str):
        token = cfg_val.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        if token == "auto":
            return not _session_is_messaging_surface()
    return not _session_is_messaging_surface()


def _was_in_tool_round(messages: list, lookback: int = _TOOL_ROUND_LOOKBACK) -> bool:
    """Check if a recent assistant message (within lookback) carried tool_calls."""
    return any(
        isinstance(_m, dict)
        and _m.get("role") == "assistant"
        and _m.get("tool_calls")
        for _m in (messages or [])[-lookback:]
    )


def build_false_stop_nudge(
    content: str,
    messages: list,
    attempts: int,
) -> Optional[str]:
    """Return a synthetic nudge if the model false-stopped, else None.

    Two detection signatures (either triggers a nudge):

    Signature A — lost tool-call preamble:
        Short text (<120 chars) ending with a colon (":" or "：") produced
        after a tool round. Likely the start of a tool-call narration that
        lost its tool_calls. The tool-round gate prevents standalone
        answers like "清单如下：" from triggering.

    Signature B — narrated continuation:
        Text (≤200 chars) ending with CJK sentence punctuation (。！？;…),
        produced after a tool round, containing at least one intent marker,
        and NOT containing done/wait/question markers.

    Both signatures require that a recent assistant message (within the last
    8 messages) carried tool_calls — this is the ``_was_in_tool_round`` gate.

    Bounded to _MAX_ATTEMPTS (2) nudges per turn. Returns None when:
    - content is empty
    - attempts >= _MAX_ATTEMPTS
    - neither signature matches
    """
    if not content or attempts >= _MAX_ATTEMPTS:
        return None

    stripped = content.strip()
    if not stripped:
        return None

    in_tool_round = _was_in_tool_round(messages)

    # Signature A: short colon-preamble — likely a truncated tool-call narration.
    signature_a = (
        len(stripped) < _MAX_SIGNATURE_A_LENGTH
        and stripped[-1] in (":", "：")
        and in_tool_round
    )

    # Signature B: narrated continuation with intent markers, guarded by
    # done/wait/question exclusions.
    signature_b = (
        len(stripped) <= _MAX_SIGNATURE_B_LENGTH
        and stripped[-1] in _CJK_ENDINGS
        and in_tool_round
        and any(_mk in stripped.lower() for _mk in _FALSE_STOP_INTENT_MARKERS)
        and not any(_dk in stripped for _dk in _FALSE_STOP_DONE_MARKERS)
        and not any(_wk in stripped for _wk in _FALSE_STOP_WAIT_MARKERS)
        and not any(_qk in stripped for _qk in _FALSE_STOP_QUESTION_MARKERS)
    )

    if not (signature_a or signature_b):
        return None

    return (
        "Your previous response ended as if you were about to "
        "call a tool, but no tool call was made. Continue the "
        "task by issuing the actual tool call now. Do not "
        "narrate intent — make the call directly."
    )


__all__ = ["build_false_stop_nudge", "false_stop_detection_enabled"]
