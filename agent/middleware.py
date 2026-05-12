"""Built-in AgentLoop middleware implementations.

Each middleware class extracts one concern from the inline while-loop body
in AIAgent.run_conversation(), giving it a testable, composable home.

Phase 3 middleware:
  - SteerDrainMiddleware: pre-API-call /steer injection into tool messages
  - StepCallbackMiddleware: gateway step event emission
  - SkillNudgeMiddleware: skill nudge counter tracking

Phase 4 middleware:
  - MessageSanitizationMiddleware: tool-call argument + role-alternation repair
  - ApiMessageBuilder: construct per-API-call message list with context injection
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from agent.loop import LoopContext, MiddlewareBase

if TYPE_CHECKING:
    pass  # avoid circular import with run_agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SteerDrainMiddleware
# ---------------------------------------------------------------------------

class SteerDrainMiddleware(MiddlewareBase):
    """Drain pending /steer text and inject it into the last tool message.

    Mirrors the logic at lines 12016-12064 of run_conversation():
    before each API call, check if the user sent a /steer during the
    previous model think. If so, append it to the last tool-role message
    so the model sees the guidance on THIS iteration.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def before_iteration(
        self,
        ctx: LoopContext,
        iteration: int,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if messages is None:
            return
        drain_fn = getattr(self._agent, "_drain_pending_steer", None)
        if drain_fn is None:
            return

        steer_text = drain_fn()
        if not steer_text:
            return

        # Walk backwards to find last tool message
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict) and msg.get("role") == "tool":
                marker = f"\n\nUser guidance: {steer_text}"
                existing = msg.get("content", "")
                if isinstance(existing, str):
                    msg["content"] = existing + marker
                else:
                    # Multimodal content blocks
                    try:
                        blocks = list(existing) if existing else []
                        blocks.append({"type": "text", "text": marker})
                        msg["content"] = blocks
                    except Exception:
                        pass
                logger.debug(
                    "SteerDrainMiddleware: injected steer into tool msg at index %d",
                    i,
                )
                return

        # No tool message found -- put it back for post-tool-execution drain
        _lock = getattr(self._agent, "_pending_steer_lock", None)
        if _lock is not None:
            with _lock:
                existing = getattr(self._agent, "_pending_steer", None)
                if existing:
                    self._agent._pending_steer = existing + "\n" + steer_text
                else:
                    self._agent._pending_steer = steer_text
        else:
            existing = getattr(self._agent, "_pending_steer", None)
            self._agent._pending_steer = (
                (existing + "\n" + steer_text) if existing else steer_text
            )


# ---------------------------------------------------------------------------
# StepCallbackMiddleware
# ---------------------------------------------------------------------------

class StepCallbackMiddleware(MiddlewareBase):
    """Fire step_callback for gateway hooks (agent:step event).

    Mirrors the logic at lines 11982-12008 of run_conversation():
    scans backward through messages for the last assistant tool_calls
    and builds a prev_tools summary, then calls step_callback(iteration, prev_tools).
    """

    def __init__(self, step_callback: Optional[Callable] = None) -> None:
        self._callback = step_callback

    def before_iteration(
        self,
        ctx: LoopContext,
        iteration: int,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if self._callback is None:
            return
        if messages is None:
            return

        try:
            prev_tools = self._extract_prev_tools(messages)
            self._callback(iteration, prev_tools)
        except Exception as exc:
            logger.debug(
                "StepCallbackMiddleware error (iteration %s): %s",
                iteration, exc,
            )

    @staticmethod
    def _extract_prev_tools(messages: List[Dict[str, Any]]) -> List[Dict]:
        """Scan backward for the last assistant message with tool_calls
        and build a summary of the tool results that followed it."""
        for idx, msg in enumerate(reversed(messages)):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                fwd_start = len(messages) - idx
                results_by_id = {}
                for tm in messages[fwd_start:]:
                    if tm.get("role") != "tool":
                        break
                    tcid = tm.get("tool_call_id")
                    if tcid:
                        results_by_id[tcid] = tm.get("content", "")
                return [
                    {
                        "name": tc["function"]["name"],
                        "result": results_by_id.get(tc.get("id")),
                        "arguments": tc["function"].get("arguments"),
                    }
                    for tc in msg["tool_calls"]
                    if isinstance(tc, dict)
                ]
        return []


# ---------------------------------------------------------------------------
# SkillNudgeMiddleware
# ---------------------------------------------------------------------------

class SkillNudgeMiddleware(MiddlewareBase):
    """Track tool-calling iterations for skill nudge.

    Mirrors the logic at lines 12010-12014 of run_conversation():
    increments a counter each iteration (when skill_manage is available)
    so the agent can decide when to nudge the model to save a skill.
    """

    def __init__(
        self,
        nudge_interval: int = 0,
        has_skill_manage: bool = False,
    ) -> None:
        self._nudge_interval = nudge_interval
        self._has_skill_manage = has_skill_manage
        self.iters_since_skill: int = 0

    def before_iteration(self, ctx: LoopContext, iteration: int, **kwargs) -> None:
        if self._nudge_interval > 0 and self._has_skill_manage:
            self.iters_since_skill += 1

    def reset(self) -> None:
        """Reset the counter (e.g. when skill_manage is actually used)."""
        self.iters_since_skill = 0


# ---------------------------------------------------------------------------
# MessageSanitizationMiddleware (Phase 4)
# ---------------------------------------------------------------------------

class MessageSanitizationMiddleware(MiddlewareBase):
    """Sanitize tool-call arguments and repair role-alternation before API calls.

    Mirrors the inline logic at lines 12027-12053 of run_conversation():
    calls _sanitize_tool_call_arguments and _repair_message_sequence on
    the messages list before each API call, logging any repairs.
    """

    def __init__(self, agent: Any, logger: Optional[Any] = None) -> None:
        self._agent = agent
        self._logger = logger

    def before_iteration(
        self,
        ctx: LoopContext,
        iteration: int,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if messages is None:
            return

        log = self._logger or getattr(self._agent, "logger", None) or logging.getLogger(__name__)

        # Sanitize corrupted tool_call arguments
        sanitize_fn = getattr(self._agent, "_sanitize_tool_call_arguments", None)
        if sanitize_fn is not None:
            repaired_tc = sanitize_fn(
                messages,
                logger=log,
                session_id=getattr(self._agent, "session_id", None),
            )
            if repaired_tc > 0:
                log.info(
                    "Sanitized %s corrupted tool_call arguments before request (session=%s)",
                    repaired_tc,
                    getattr(self._agent, "session_id", None) or "-",
                )

        # Repair malformed role-alternation
        repair_fn = getattr(self._agent, "_repair_message_sequence", None)
        if repair_fn is not None:
            repaired_seq = repair_fn(messages)
            if repaired_seq > 0:
                log.info(
                    "Repaired %s message-alternation violations before request (session=%s)",
                    repaired_seq,
                    getattr(self._agent, "session_id", None) or "-",
                )


# ---------------------------------------------------------------------------
# ApiMessageBuilder (Phase 4)
# ---------------------------------------------------------------------------

class ApiMessageBuilder(MiddlewareBase):
    """Build per-API-call message list with context injection.

    Mirrors the inline logic at lines 12055-12098 of run_conversation():
    iterates over messages, strips internal fields, copies reasoning,
    injects memory/plugin context into the user message, and produces
    a clean api_messages list suitable for the LLM API call.

    The result is stored in ``last_api_messages`` after each before_iteration.
    """

    def __init__(
        self,
        agent: Any,
        current_turn_user_idx: int = 0,
        memory_prefetch: str = "",
        plugin_context: str = "",
        memory_fence_fn: Optional[Callable] = None,
    ) -> None:
        self._agent = agent
        self._current_turn_user_idx = current_turn_user_idx
        self._memory_prefetch = memory_prefetch
        self._plugin_context = plugin_context
        self._memory_fence_fn = memory_fence_fn
        self.last_api_messages: List[Dict[str, Any]] = []

    def before_iteration(
        self,
        ctx: LoopContext,
        iteration: int,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if messages is None:
            self.last_api_messages = []
            return

        api_messages = []
        copy_reasoning_fn = getattr(self._agent, "_copy_reasoning_content_for_api", None)
        should_sanitize_fn = getattr(self._agent, "_should_sanitize_tool_calls", None)
        sanitize_fn = getattr(self._agent, "_sanitize_tool_calls_for_strict_api", None)

        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # Inject ephemeral context into the current turn's user message
            if idx == self._current_turn_user_idx and msg.get("role") == "user":
                injections = []
                if self._memory_prefetch:
                    fenced = self._memory_prefetch
                    if self._memory_fence_fn is not None:
                        fenced = self._memory_fence_fn(self._memory_prefetch)
                    if fenced:
                        injections.append(fenced)
                if self._plugin_context:
                    injections.append(self._plugin_context)
                if injections:
                    base = api_msg.get("content", "")
                    if isinstance(base, str):
                        api_msg["content"] = base + "\n\n" + "\n\n".join(injections)

            # Copy reasoning content for the API
            if copy_reasoning_fn is not None:
                copy_reasoning_fn(msg, api_msg)

            # Strip internal fields
            api_msg.pop("reasoning", None)
            api_msg.pop("finish_reason", None)
            api_msg.pop("_thinking_prefill", None)

            # Strip Codex Responses API fields for strict providers
            if should_sanitize_fn is not None and should_sanitize_fn():
                if sanitize_fn is not None:
                    sanitize_fn(api_msg)

            api_messages.append(api_msg)

        self.last_api_messages = api_messages


# ---------------------------------------------------------------------------
# ApiMessageFinalizer (Phase 5)
# ---------------------------------------------------------------------------

class ApiMessageFinalizer(MiddlewareBase):
    """Finalize the API message list before the LLM call.

    Mirrors the inline logic at lines 12043-12131 of run_conversation():
    - Prepends system prompt (base + ephemeral)
    - Injects prefill messages after system prompt
    - Applies Anthropic prompt caching if configured
    - Calls _sanitize_api_messages (orphan tool result cleanup)
    - Calls _drop_thinking_only_and_merge_users
    - Strips content whitespace for consistent KV cache prefixes
    - Normalizes tool_call JSON to sorted compact form
    - Strips surrogate characters

    Operates on a copy -- never mutates the input messages.
    """

    def __init__(
        self,
        agent: Any,
        active_system_prompt: str = "",
        prompt_cache_fn: Optional[Callable] = None,
        surrogate_sanitize_fn: Optional[Callable] = None,
    ) -> None:
        self._agent = agent
        self._active_system_prompt = active_system_prompt
        self._prompt_cache_fn = prompt_cache_fn
        self._surrogate_sanitize_fn = surrogate_sanitize_fn
        self.last_finalized_messages: List[Dict[str, Any]] = []

    def before_iteration(
        self,
        ctx: LoopContext,
        iteration: int,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if messages is None:
            self.last_finalized_messages = []
            return

        import copy as _copy
        import json as _json

        api_messages = _copy.deepcopy(messages)

        # System prompt assembly
        effective_system = self._active_system_prompt or ""
        ephemeral = getattr(self._agent, "ephemeral_system_prompt", "")
        if ephemeral:
            effective_system = (effective_system + "\n\n" + ephemeral).strip()

        if effective_system:
            api_messages.insert(0, {"role": "system", "content": effective_system})

        # Prefill messages
        prefill = getattr(self._agent, "prefill_messages", None)
        if prefill:
            sys_offset = 1 if effective_system else 0
            for idx, pfm in enumerate(prefill):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Prompt caching
        if getattr(self._agent, "_use_prompt_caching", False) and self._prompt_cache_fn:
            api_messages = self._prompt_cache_fn(
                api_messages,
                cache_ttl=getattr(self._agent, "_cache_ttl", None),
                native_anthropic=getattr(self._agent, "_use_native_cache_layout", False),
            )

        # Sanitize orphaned tool results
        sanitize_fn = getattr(self._agent, "_sanitize_api_messages", None)
        if sanitize_fn is not None:
            api_messages = sanitize_fn(api_messages)

        # Drop thinking-only assistant turns
        drop_fn = getattr(self._agent, "_drop_thinking_only_and_merge_users", None)
        if drop_fn is not None:
            api_messages = drop_fn(api_messages)

        # Strip content whitespace
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()

        # Normalize tool_call JSON
        repair_fn = getattr(self._agent, "_repair_tool_call_arguments", None)
        for am in api_messages:
            tcs = am.get("tool_calls")
            if not tcs:
                continue
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict) and "function" in tc:
                    try:
                        args_obj = _json.loads(tc["function"]["arguments"])
                        tc = {**tc, "function": {
                            **tc["function"],
                            "arguments": _json.dumps(
                                args_obj, separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }}
                    except Exception:
                        if repair_fn is not None:
                            tc["function"]["arguments"] = repair_fn(
                                tc["function"]["arguments"],
                                tc["function"].get("name", "?"),
                            )
                new_tcs.append(tc)
            am["tool_calls"] = new_tcs

        # Strip surrogate characters
        if self._surrogate_sanitize_fn is not None:
            self._surrogate_sanitize_fn(api_messages)

        self.last_finalized_messages = api_messages
