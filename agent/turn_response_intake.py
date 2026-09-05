"""Response intake for the conversation turn loop: normalize the raw provider response into
the assistant message, splice agent-as-provider projections, fire ``post_api_request``, relay
reasoning to the progress callback, and apply the incomplete-scratchpad / Codex-incomplete
continuation guards. Nothing here imports ``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, Optional

from agent.provider_projection import splice_provider_projection
from agent.trajectory import has_incomplete_scratchpad
from agent.turn_truncation import continue_codex_incomplete, normalize_response_for_agent, partial_result

logger = logging.getLogger("agent.conversation_loop")

_REASONING_TAG_RE = re.compile(r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>')


@dataclass
class ResponseIntakeVerdict:
    """``action``: ``"fallthrough"`` (process ``assistant_message``), ``"continue"`` (retry the
    iteration: incomplete scratchpad / Codex continuation) or ``"return"`` (``result`` is the
    turn's result dict). ``assistant_message``/``finish_reason`` are the normalized outputs."""

    action: str
    assistant_message: Any
    finish_reason: Any
    result: Optional[Dict[str, Any]] = None


def _coerce_content_text(raw: Any) -> str:
    """Some OpenAI-compatible servers (llama-server) return content as dict/list, which
    crashes downstream ``.strip()``; normalize to str (multimodal lists → text parts)."""
    if isinstance(raw, dict):
        return raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
    if isinstance(raw, list):
        parts = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return "\n".join(parts)
    return str(raw)


def _fire_post_api_request_hook(
    agent: Any, response: Any, assistant_message: Any, finish_reason: Any, *, api_messages: Any,
    api_call_count: Any, api_duration: Any, api_start_time: Any, api_request_id: Any,
    effective_task_id: Any, turn_id: Any,
) -> None:
    from agent.conversation_loop import _moa_reference_metrics_for_hook

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook as _invoke_hook
        if has_hook("post_api_request"):
            _invoke_hook(
                "post_api_request",
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                session_id=agent.session_id or "",
                platform=agent.platform or "",
                model=agent.model,
                provider=agent.provider,
                base_url=agent.base_url,
                api_mode=agent.api_mode,
                api_call_count=api_call_count,
                api_duration=api_duration,
                started_at=api_start_time,
                ended_at=api_start_time + api_duration,
                # First stream chunk time (epoch s); None if not streamed / no chunk.
                # TTFB = first_chunk_at - started_at.
                first_chunk_at=getattr(agent, "_last_api_first_chunk_at", None),
                finish_reason=finish_reason,
                message_count=len(api_messages),
                response_model=getattr(response, "model", None),
                response=agent._api_response_payload_for_hook(
                    response, assistant_message, finish_reason=finish_reason
                ),
                usage=agent._usage_summary_for_api_request_hook(response),
                assistant_message=assistant_message,
                assistant_content_chars=len(assistant_message.content or ""),
                assistant_tool_call_count=len(getattr(assistant_message, "tool_calls", None) or []),
                moa_references=_moa_reference_metrics_for_hook(agent),
            )
    except Exception:
        pass


def _relay_thinking(agent: Any, content: str) -> None:
    """Relay the model's text to the progress callback: subagents send the first line to
    the parent display; any agent with a structured callback gets ``reasoning.available``."""
    _think_text = _REASONING_TAG_RE.sub('', content.strip()).strip()
    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
    if first_line and getattr(agent, '_delegate_depth', 0) > 0:
        try:
            agent.tool_progress_callback("_thinking", first_line)
        except Exception:
            pass
    elif _think_text:
        try:
            agent.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
        except Exception:
            pass


def normalize_model_response(
    agent: Any, *, response: Any, messages: Any, api_messages: Any, conversation_history: Any,
    api_call_count: Any, api_duration: Any, api_start_time: Any, api_request_id: Any,
    effective_task_id: Any, turn_id: Any,
) -> ResponseIntakeVerdict:
    """Normalize ``response`` into ``assistant_message`` (str content, never dict/list) and run
    the post-response hooks and continuation guards, in the original order."""
    assistant_message = normalize_response_for_agent(agent, response)
    finish_reason = assistant_message.finish_reason

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ResponseIntakeVerdict:
        return ResponseIntakeVerdict(
            action=action, assistant_message=assistant_message, finish_reason=finish_reason,
            result=result,
        )

    if assistant_message.content is not None and not isinstance(assistant_message.content, str):
        assistant_message.content = _coerce_content_text(assistant_message.content)

    # Agent-as-provider projection: splice the provider-agent's own tool work in as
    # call/result rows before this turn's assistant message; no-op for ordinary providers.
    splice_provider_projection(agent, response, messages)

    _fire_post_api_request_hook(
        agent, response, assistant_message, finish_reason, api_messages=api_messages,
        api_call_count=api_call_count, api_duration=api_duration, api_start_time=api_start_time,
        api_request_id=api_request_id, effective_task_id=effective_task_id, turn_id=turn_id,
    )

    content = assistant_message.content
    if content and not agent.quiet_mode:
        if agent.verbose_logging:
            agent._vprint(f"{agent.log_prefix}🤖 Assistant: {content}")
        else:
            agent._vprint(f"{agent.log_prefix}🤖 Assistant: {content[:100]}{'...' if len(content) > 100 else ''}")
    if content and agent.tool_progress_callback:
        _relay_thinking(agent, content)

    # Incomplete <REASONING_SCRATCHPAD> (opened, never closed): the model ran out of
    # output tokens mid-reasoning — retry up to 2 times, then save as partial.
    if has_incomplete_scratchpad(content or ""):
        agent._incomplete_scratchpad_retries += 1
        agent._buffer_vprint("⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
        if agent._incomplete_scratchpad_retries <= 2:
            agent._buffer_vprint(f"🔄 Retrying API call ({agent._incomplete_scratchpad_retries}/2)...")
            return _verdict("continue")  # don't add the broken message
        agent._flush_status_buffer()
        agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
        agent._incomplete_scratchpad_retries = 0
        rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
        agent._cleanup_task_resources(effective_task_id)
        agent._persist_session(messages, conversation_history)
        return _verdict("return", partial_result(
            rolled_back_messages, api_call_count, "Incomplete REASONING_SCRATCHPAD after 2 retries"
        ))
    agent._incomplete_scratchpad_retries = 0

    if agent.api_mode == "codex_responses" and finish_reason == "incomplete":
        _codex_result = continue_codex_incomplete(
            agent, assistant_message, finish_reason, messages=messages,
            conversation_history=conversation_history, api_call_count=api_call_count,
        )
        if _codex_result is not None:
            return _verdict("return", _codex_result)
        return _verdict("continue")
    if hasattr(agent, "_codex_incomplete_retries"):
        agent._codex_incomplete_retries = 0
    return _verdict("fallthrough")
