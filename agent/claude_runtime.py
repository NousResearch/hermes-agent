"""Claude CLI runtime — Phase 2b multi-turn session resume.

Drives one turn through a ``claude -p`` subprocess when
``agent.api_mode == "claude_cli"``. Mirrors ``agent.codex_runtime``'s
``run_codex_app_server_turn`` early-return shape so the conversation loop
treats the result as a normal completed turn.

Phase 2a: Hermes tools via MCP (``hermes_tools_mcp_server``). Tool
round-trip is internal to ``claude -p``; this module projects tool
activity into Hermes UI via the event bridge.

Phase 2b: multi-turn context via Claude's own session:
  * Lazy ``agent._claude_cli_session`` (one ClaudeCliSession per AIAgent /
    Hermes conversation) — same lifetime model as ``agent._codex_session``.
  * Turn 1: ``--session-id <uuid>``; turn 2+: ``--resume <uuid>``.
  * Mapping lives on the ClaudeCliSession; process-local reuse across
    Hermes turns. Stable cwd for Claude session files.
  * Missing resume → fresh session once; hard failures retire the session.

Phase 2c: host-global concurrency semaphore (``claude_cli_concurrency``)
caps concurrent ``claude -p`` spawns across all Hermes profiles; aux
HTTP (title / Hermes compression / metadata) is skipped or silenced —
Claude owns native session compaction via ``--resume``.

TODO(later): doctor checks, full transcript pre-seed for resumed
Hermes histories.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_claude_cli_usage(agent, turn) -> dict[str, Any]:
    """Translate Claude CLI result usage into Hermes session accounting.

    Mirrors ``_record_codex_app_server_usage`` closely so Max-subscription
    turns still tick session_api_calls and token counters.
    """
    agent.session_api_calls += 1

    usage = getattr(turn, "token_usage_last", None) or getattr(turn, "usage", None)
    # token_usage_last is already codex-shaped; raw usage is Claude-shaped.
    if isinstance(usage, dict) and (
        "inputTokens" in usage or "outputTokens" in usage
    ):
        input_tokens = _coerce_usage_int(usage.get("inputTokens"))
        cache_read_tokens = _coerce_usage_int(usage.get("cachedInputTokens"))
        output_tokens = _coerce_usage_int(usage.get("outputTokens"))
        reasoning_tokens = _coerce_usage_int(usage.get("reasoningOutputTokens"))
        reported_total = _coerce_usage_int(usage.get("totalTokens"))
    elif isinstance(usage, dict) and usage:
        input_tokens = _coerce_usage_int(
            usage.get("input_tokens") or usage.get("prompt_tokens")
        )
        cache_read_tokens = _coerce_usage_int(
            usage.get("cache_read_input_tokens") or usage.get("cache_read_tokens")
        )
        output_tokens = _coerce_usage_int(
            usage.get("output_tokens") or usage.get("completion_tokens")
        )
        reasoning_tokens = 0
        reported_total = input_tokens + cache_read_tokens + output_tokens
    else:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode="subscription_included",
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "claude_cli api-call persistence failed (session=%s): %s",
                    agent.session_id,
                    exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=0,
        reasoning_tokens=reasoning_tokens,
        raw_usage=usage if isinstance(usage, dict) else {},
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = reported_total or canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": 0,
        "reasoning_tokens": reasoning_tokens,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
        except Exception:
            logger.debug("claude_cli usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_reasoning_tokens += reasoning_tokens

    cost_result = estimate_usage_cost(
        agent.model,
        canonical_usage,
        provider=agent.provider,
        base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    if cost_result.amount_usd is not None:
        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=float(cost_result.amount_usd)
                if cost_result.amount_usd is not None
                else None,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included"
                else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude_cli token persistence failed (session=%s): %s",
                agent.session_id,
                exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": float(cost_result.amount_usd)
        if cost_result.amount_usd is not None
        else None,
        "cost_status": cost_result.status,
        "cost_source": cost_result.source,
        "runtime": "claude_cli",
    }


def make_claude_cli_event_bridge(agent) -> Callable[[dict], None]:
    """Bridge claude_cli projector events into Hermes stream + tool UI.

    Text deltas, assistant completion, and MCP tool.started / tool.completed
    (mirrors codex_runtime's mcpToolCall projection: strip hermes-tools
    namespacing so the user sees bare Hermes tool names).
    """

    def _fire_text_delta(params: dict) -> None:
        text = params.get("delta") or params.get("text") or ""
        if not isinstance(text, str) or not text:
            return
        fn = getattr(agent, "_fire_stream_delta", None)
        if fn is None:
            return
        try:
            fn(text)
        except Exception:
            logger.debug("_fire_stream_delta raised", exc_info=True)

    def _fire_assistant_completed(params: dict) -> None:
        text = params.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            return
        if not getattr(agent, "show_commentary", True):
            return
        emit = getattr(agent, "_emit_interim_assistant_message", None)
        if emit is None:
            return
        try:
            emit({"role": "assistant", "content": text})
        except Exception:
            logger.debug(
                "_emit_interim_assistant_message raised", exc_info=True
            )

    def _tool_preview(args: Any) -> Any:
        if not isinstance(args, dict) or not args:
            return None
        for key in ("command", "query", "path", "url", "pattern", "code"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val[:120]
        # First short string value as fallback preview.
        for val in args.values():
            if isinstance(val, str) and val.strip():
                return val[:120]
        return None

    def _fire_tool_progress(phase: str, params: dict) -> None:
        """Emit tool.started / tool.completed via agent.tool_progress_callback.

        Signature mirrors codex_runtime:
          tool.started  → (event, name, preview, args)
          tool.completed → (event, name, None, None, is_error=..., result=...)
        """
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is None:
            return
        name = params.get("name") or "unknown"
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            if phase == "started":
                cb("tool.started", name, _tool_preview(args), args)
            else:
                result = params.get("result")
                result_text = (
                    result
                    if isinstance(result, str)
                    else (str(result) if result is not None else "")
                )
                cb(
                    "tool.completed",
                    name,
                    None,
                    None,
                    is_error=bool(params.get("is_error")),
                    result=result_text,
                )
        except TypeError:
            # Older callback signatures take fewer args — best-effort.
            try:
                cb("tool." + phase if not phase.startswith("tool.") else phase, name)
            except Exception:
                logger.debug("tool_progress_callback raised", exc_info=True)
        except Exception:
            logger.debug("tool_progress_callback raised", exc_info=True)

    def on_event(note: dict) -> None:
        if not isinstance(note, dict):
            return
        method = note.get("method") or ""
        params = note.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if method == "claude/text_delta":
            _fire_text_delta(params)
            return
        if method == "claude/assistant_completed":
            _fire_assistant_completed(params)
            return
        if method == "claude/tool_started":
            _fire_tool_progress("started", params)
            return
        if method == "claude/tool_completed":
            _fire_tool_progress("completed", params)
            return

    return on_event


def _extract_system_prompt(agent, messages: List[Dict[str, Any]]) -> Optional[str]:
    """Best-effort system prompt for --append-system-prompt-file."""
    # Prefer an explicit agent-composed system message if present on the
    # conversation (first system role), else agent.system_prompt / similar.
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif isinstance(block, str):
                        parts.append(block)
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
    for attr in ("system_prompt", "_system_prompt", "system_message"):
        val = getattr(agent, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _retire_claude_cli_session(agent, reason: str = "") -> None:
    """Close and drop agent._claude_cli_session (mirrors codex retire path)."""
    session = getattr(agent, "_claude_cli_session", None)
    if session is None:
        return
    if reason:
        logger.warning("claude_cli session retired: %s", reason)
    try:
        session.close()
    except Exception:
        pass
    agent._claude_cli_session = None


def run_claude_cli_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Claude CLI runtime path. Hands one turn to ``claude -p``.

    Called from conversation_loop when agent.api_mode == "claude_cli".
    Returns the same dict shape as the chat_completions / codex_app_server path.

    Phase 2b: reuses ``agent._claude_cli_session`` across turns (create on
    first turn with ``--session-id``, resume later with ``--resume``) so
    Claude retains conversation context. Mirrors codex app-server's
    ``agent._codex_session`` lifetime.
    """
    from agent.transports.claude_cli import (
        ClaudeCliConcurrencyError,
        ClaudeCliError,
    )
    from agent.transports.claude_cli_session import (
        ClaudeCliSession,
        resolve_claude_cli_oauth_token,
    )

    try:
        from agent.runtime_cwd import resolve_agent_cwd

        cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
    except Exception:
        cwd = None

    try:
        token = resolve_claude_cli_oauth_token(agent=agent)
    except ClaudeCliError as exc:
        logger.exception("claude_cli token resolution failed")
        return {
            "final_response": str(exc),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "agent_persisted": False,
        }

    system_prompt = _extract_system_prompt(agent, messages)
    model = getattr(agent, "model", None) or "claude-opus-4-8"
    hermes_id = getattr(agent, "session_id", None)

    # Lazy session: one ClaudeCliSession per AIAgent / Hermes conversation.
    # Spawned (mapped) on first turn, reused across turns via --resume,
    # closed on hard failure or agent.close (when wired).
    if not hasattr(agent, "_claude_cli_session") or agent._claude_cli_session is None:
        agent._claude_cli_session = ClaudeCliSession(
            oauth_token=token,
            model=model,
            cwd=cwd,
            on_event=make_claude_cli_event_bridge(agent),
            hermes_conversation_id=hermes_id,
        )
    session = agent._claude_cli_session

    try:
        turn = session.run_turn(
            user_input=user_message,
            system_prompt=system_prompt,
            model=model,
            messages=messages,
        )
    except ClaudeCliConcurrencyError:
        # Propagate so conversation_loop can activate the profile fallback
        # (grok/gpt) instead of hanging or returning a dead partial turn.
        # Do NOT retire the multi-turn mapping — saturation is transient.
        raise
    except ClaudeCliError as exc:
        logger.exception("claude_cli turn failed")
        _retire_claude_cli_session(agent, reason=str(exc)[:200])
        # Surface a classifiable error string so the gateway/CLI show it and
        # higher-level fallback (if any) can react. Re-raising would escape
        # the early-return contract of conversation_loop; we return the same
        # shape as codex_app_server crash path instead.
        return {
            "final_response": (
                f"Claude CLI turn failed: {exc}. "
                f"Fall back to default Anthropic HTTP with "
                f"`model.anthropic_runtime: auto` (or unset)."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "agent_persisted": False,
        }
    except Exception as exc:
        logger.exception("claude_cli turn failed (unexpected)")
        _retire_claude_cli_session(agent, reason=str(exc)[:200])
        return {
            "final_response": f"Claude CLI turn failed: {exc}",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "agent_persisted": False,
        }

    # Retire only when the turn signals the mapping is unusable (hard error
    # path already retired above). Success keeps the session for --resume.
    if getattr(turn, "should_retire", False):
        _retire_claude_cli_session(
            agent, reason=getattr(turn, "error", None) or "should_retire"
        )

    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude_cli projected-message flush failed",
                    exc_info=True,
                )

    usage_result = _record_claude_cli_usage(agent, turn)
    api_calls = 1

    # External memory provider sync (mirrors codex path).
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    if (
        turn.final_text
        and not turn.interrupted
        and should_review_memory
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=True,
                review_skills=False,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": not turn.interrupted and turn.error is None and not turn.is_error,
        "partial": turn.interrupted or turn.error is not None or turn.is_error,
        "error": turn.error,
        "agent_persisted": True,
        "claude_cli_session_id": turn.session_id,
        "claude_cli_resumed": getattr(turn, "resumed", False),
        "claude_cli_created_session": getattr(turn, "created_session", False),
        "claude_cli_resume_fallback": getattr(turn, "resume_fallback", False),
        **usage_result,
    }


__all__ = [
    "run_claude_cli_turn",
    "make_claude_cli_event_bridge",
]
