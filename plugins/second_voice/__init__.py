"""second_voice plugin — opt-in "Second Voice" step-reflection guardrail.

Registers a ``tool_execution`` middleware. When enabled, for a configurable allowlist of tools it
asks an auxiliary LLM (strict, isolated context window) whether the proposed tool call is logical,
complete, and compliant. On REDO it blocks the tool and feeds a critique back so the executor
corrects course; on ESCALATE (or after ``max_consecutive_rejections`` REDOs) it blocks with an
"ask the user" error. It fails CLOSED on any LLM error or ambiguous verdict so a gated tool never
executes on a verdict the referee did not explicitly clear.

Opt-in: the middleware only runs for enabled plugins — ``hermes plugins enable second_voice``. All
behavior is config-gated through ``plugins.entries.second_voice.settings``.
"""

from __future__ import annotations

import logging
from typing import Any

from . import reflection as r

logger = logging.getLogger(__name__)

_breaker = r.Breaker()


def _reflect(cfg: r.Config, tool_name: str, args: Any, instruction: str):
    """Run the aux-LLM step reflection; fail CLOSED (escalate) on any LLM error or ambiguity."""
    from agent.auxiliary_client import call_llm

    user_prompt = r.build_user_prompt(tool_name, args, instruction)
    call_kwargs: dict[str, Any] = dict(
        task="second_voice",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        messages=[
            {"role": "system", "content": r.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    if cfg.model:
        call_kwargs["model"] = cfg.model
    try:
        response = call_llm(**call_kwargs)
    except Exception as exc:
        # Fail closed: we could not check the step, so a gated tool must not run unverified.
        logger.warning("second_voice reflection LLM failed (%s); failing closed (escalate)", exc)
        return "escalate", f"cannot evaluate the step ({type(exc).__name__})"
    raw = ""
    try:
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("second_voice response parse failed: %s", exc)
    return r.parse_verdict(raw)


def _on_tool_execution(ctx: Any, **kwargs: Any) -> Any:
    """``tool_execution`` middleware: reflect on gated tools; block with a critique on REDO/ESCALATE."""
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args")
    next_call = kwargs.get("next_call")
    session_id = kwargs.get("session_id", "")
    turn_id = kwargs.get("turn_id", "")

    if not callable(next_call):
        raise RuntimeError(
            "second_voice: tool_execution middleware received a non-callable next_call"
        )

    cfg = r.Config.from_ctx(ctx)
    if not cfg.reflect_tools or tool_name not in cfg.reflect_tools:
        return next_call(args)

    instruction = r.gather_instruction(session_id, cfg.max_instruction_chars)
    verdict, reason = _reflect(cfg, tool_name, args, instruction)

    if verdict == "approve":
        _breaker.reset(session_id, turn_id)
        return next_call(args)

    if verdict == "escalate":
        _breaker.reset(session_id, turn_id)
        return r.block_result(f"Second Voice escalation: {reason}", escalate=True)

    # REDO: block and feed the critique back so the executor corrects course.
    _breaker.bump(session_id, turn_id)
    if _breaker.should_escalate(session_id, turn_id, cfg.max_consecutive_rejections):
        _breaker.reset(session_id, turn_id)
        return r.block_result(
            "Second Voice: repeated rejections — stop and ask the user before proceeding. "
            f"({reason})",
            escalate=True,
        )
    return r.block_result(f"Second Voice blocked this step: {reason}")


def register(ctx) -> None:
    """Plugin entry point — register the tool-execution step-reflection middleware."""
    ctx.register_middleware("tool_execution", lambda **kw: _on_tool_execution(ctx, **kw))
