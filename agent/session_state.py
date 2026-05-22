"""AIAgent session state management — extracted from run_agent.py.

Handles session persistence, database row lifecycle, token counter resets,
and todo-store hydration. Each function takes ``agent`` (AIAgent instance)
as its first argument and preserves the test-patch contract via ``_ra()``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.memory_manager import sanitize_context

logger = logging.getLogger(__name__)


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent
    return run_agent


def reset_session_state(agent) -> None:
    """Reset all session-scoped token counters to 0 for a fresh session.

    Covers token usage counters, cache tokens, API call count, reasoning
    tokens, estimated cost, and the context compressor internal state.
    """
    agent.session_total_tokens = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_api_calls = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    agent._user_turn_count = 0

    if hasattr(agent, "context_compressor") and agent.context_compressor:
        agent.context_compressor.on_session_reset()


def get_session_db_for_recall(agent):
    """Return a SessionDB for recall, lazily creating it if missing.

    Most frontends pass ``session_db`` into ``AIAgent`` explicitly, but
    recall is important enough that a missing constructor argument should
    degrade by opening the default state DB instead of making the advertised
    ``session_search`` tool unusable.
    """
    if agent._session_db is not None:
        return agent._session_db
    try:
        from hermes_state import SessionDB
        agent._session_db = SessionDB()
        return agent._session_db
    except Exception as exc:
        logger.debug("SessionDB unavailable for recall", exc_info=True)
        return None


def ensure_db_session(agent) -> None:
    """Create session DB row on first use. Disables _session_db on failure."""
    if agent._session_db_created or not agent._session_db:
        return
    try:
        agent._session_db.create_session(
            session_id=agent.session_id,
            source=agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
            model=agent.model,
            model_config=agent._session_init_model_config,
            system_prompt=agent._cached_system_prompt,
            user_id=None,
            parent_session_id=agent._parent_session_id,
        )
        agent._session_db_created = True
    except Exception as e:
        logger.warning("Session DB creation failed (will retry next turn): %s", e)


def apply_persist_user_message_override(agent, messages: List[Dict]) -> None:
    """Rewrite the current-turn user message before persistence/return.

    Some call paths need an API-only user-message variant without letting
    that synthetic text leak into persisted transcripts or resumed session
    history. When an override is configured for the active turn, mutate the
    in-memory messages list in place so both persistence and returned
    history stay clean.
    """
    idx = getattr(agent, "_persist_user_message_idx", None)
    override = getattr(agent, "_persist_user_message_override", None)
    if override is None or idx is None:
        return
    if 0 <= idx < len(messages):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            msg["content"] = override


def persist_session(agent, messages: List[Dict], conversation_history: List[Dict] = None) -> None:
    """Persist the current session state — message history & todo store.

    Skips persistence when:
    - ``_skip_session_persist`` is set (ephemeral sessions like background review forks)
    - No session database is available

    Also applies user-message override (persist_user_message) before saving.
    """
    if getattr(agent, "_skip_session_persist", False):
        return
    if not agent._session_db:
        return
    if not agent.session_id:
        return

    apply_persist_user_message_override(agent, messages)
    _drop_trailing_empty_response_scaffolding(agent, messages)

    # Save agent._session_messages for downstream JSON snapshot writers
    agent._session_messages = messages

    # Save to session database
    _flush_messages_to_session_db(agent, messages, conversation_history)

    # Persist todo store alongside messages
    from tools.todo_tool import format_todo_for_session_metadata
    _persist_todo_store(agent, messages)


def _drop_trailing_empty_response_scaffolding(agent, messages: List[Dict]) -> None:
    """Remove private empty-response retry/failure scaffolding from transcript tails.

    Also rewinds past any trailing tool-result / assistant(tool_calls) pair
    that the failed iteration left hanging. Without this, the tail ends at
    a raw ``tool`` message and the next user turn lands as
    ``...tool, user, user`` — a protocol-invalid sequence that most
    providers silently reject (returns empty content), causing the
    empty-retry loop to fire forever.
    """
    if not messages:
        return

    # Pass 1: strip the flagged scaffolding messages themselves.
    dropped_scaffolding = False
    while (
        messages
        and isinstance(messages[-1], dict)
        and (
            messages[-1].get("_empty_recovery_synthetic")
            or messages[-1].get("_empty_terminal_sentinel")
        )
    ):
        messages.pop()
        dropped_scaffolding = True

    # Pass 2: if we stripped scaffolding, rewind through any trailing
    # tool-result messages plus the assistant(tool_calls) message that
    # produced them. This preserves role alternation so the next user
    # message follows a user or assistant message, not an orphan tool
    # result. Only runs when scaffolding was actually present — normal
    # conversation tails (real tool loops mid-progress) are untouched.
    if not dropped_scaffolding:
        return

    # Drop any trailing tool-result messages
    while (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "tool"
    ):
        messages.pop()

    # Drop the assistant message that issued the tool calls, if the tail
    # now ends in an assistant-with-tool_calls
    if (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    ):
        messages.pop()


def _flush_messages_to_session_db(agent, messages: List[Dict], conversation_history: List[Dict] = None) -> None:
    """Incrementally flush new messages to the session database.

    Uses ``_last_flushed_db_idx`` to track which messages have already been
    written, so repeated calls only write truly new messages — preventing
    the duplicate-write bug.

    Also handles multimodal tool results (strips base64 images, keeps text
    summary) and content-part lists (strips image parts, keeps text parts).
    """
    if not agent._session_db:
        return
    _apply_persist_user_message_override(agent, messages) if hasattr(agent, '_apply_persist_user_message_override') else apply_persist_user_message_override(agent, messages)
    try:
        # Retry row creation if the earlier attempt failed transiently.
        if not getattr(agent, "_session_db_created", False):
            ensure_db_session(agent)
        start_idx = len(conversation_history) if conversation_history else 0
        last_flushed = getattr(agent, "_last_flushed_db_idx", 0)
        flush_from = max(start_idx, last_flushed)
        for msg in messages[flush_from:]:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            # Persist multimodal tool results as their text summary only
            try:
                from agent.tool_dispatch_helpers import _is_multimodal_tool_result, _multimodal_text_summary
                if _is_multimodal_tool_result(content):
                    content = _multimodal_text_summary(content)
                elif isinstance(content, list):
                    _txt = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            _txt.append(str(p.get("text", "")))
                        elif isinstance(p, dict) and p.get("type") in {"image", "image_url", "input_image"}:
                            _txt.append("[screenshot]")
                    content = "\n".join(_txt) if _txt else None
            except ImportError:
                pass  # Fallback: store content as-is
            tool_calls_data = None
            # Handle both object-style (SDK) and dict-style tool calls
            if hasattr(msg, "tool_calls") and isinstance(msg.tool_calls, list):
                tool_calls_data = [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in msg.tool_calls
                ]
            elif isinstance(msg.get("tool_calls"), list):
                tool_calls_data = msg["tool_calls"]
            agent._session_db.append_message(
                session_id=agent.session_id,
                role=role,
                content=content,
                tool_name=msg.get("tool_name"),
                tool_calls=tool_calls_data,
                tool_call_id=msg.get("tool_call_id"),
                finish_reason=msg.get("finish_reason"),
                reasoning=msg.get("reasoning") if role == "assistant" else None,
                reasoning_content=msg.get("reasoning_content") if role == "assistant" else None,
                reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                codex_message_items=msg.get("codex_message_items") if role == "assistant" else None,
            )
        agent._last_flushed_db_idx = len(messages)
    except Exception as e:
        logger.warning("Session DB append_message failed: %s", e)


def _get_messages_up_to_last_assistant(agent, messages: List[Dict]) -> List[Dict]:
    """Return messages up to (and including) the last assistant response.

    Strips trailing tool results that have no corresponding assistant message
    ready yet — they'll be saved on the next flush.
    """
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return list(messages)

    # Include the assistant message and everything before it
    return list(messages[: last_assistant_idx + 1])


def _persist_todo_store(agent, messages: List[Dict]) -> None:
    """Persist the todo store state alongside session messages."""
    try:
        from tools.todo_tool import format_todo_for_session_metadata
        todo_json = format_todo_for_session_metadata(agent._todo_store)
        if todo_json and agent._session_db and agent.session_id:
            agent._session_db.update_session_metadata(
                session_id=agent.session_id,
                metadata={"todos": todo_json},
            )
    except Exception as e:
        logger.debug("Todo store persist skipped: %s", e)


def save_session_log(agent, messages: List[Dict[str, Any]] = None) -> None:
    """Save the full message history to the session DB as a compressed log.

    This is the final persist on session close — captures everything,
    including trailing tool results and the closing assistant turn that
    the per-turn flush deliberately omitted.
    """
    if not agent._session_db or not agent.session_id:
        return

    try:
        if messages:
            # Sanitize content before final storage
            sanitized = []
            for msg in messages:
                m = dict(msg)
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = sanitize_context(content)
                sanitized.append(m)

            agent._session_db.save_full_session(
                session_id=agent.session_id,
                messages=sanitized,
                total_tokens=getattr(agent, "session_total_tokens", 0),
                input_tokens=getattr(agent, "session_input_tokens", 0),
                output_tokens=getattr(agent, "session_output_tokens", 0),
                cache_read_tokens=getattr(agent, "session_cache_read_tokens", 0),
                cache_write_tokens=getattr(agent, "session_cache_write_tokens", 0),
                api_calls=getattr(agent, "session_api_calls", 0),
                estimated_cost_usd=getattr(agent, "session_estimated_cost_usd", 0.0),
                reasoning_tokens=getattr(agent, "session_reasoning_tokens", 0),
            )
    except Exception as e:
        logger.debug("Session log save skipped: %s", e)


def hydrate_todo_store(agent, history: List[Dict[str, Any]]) -> None:
    """Recover the todo store from the most recent todo tool response in history.

    Gateway creates a fresh AIAgent per message, so the in-memory todo
    store is empty.  This reconstructs it from the most recent todo-list
    tool call result in the conversation history.
    """
    if not history:
        from tools.interrupt import set_interrupt as _set_interrupt
        _set_interrupt(False)
        return

    # Walk history in reverse to find the most recent todo_list response
    last_todo_response = None
    for msg in reversed(history):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if '"todos"' not in content and '"items"' not in content:
            continue
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            todos = data.get("todos") or data.get("items") or []
            if isinstance(todos, list) and todos:
                last_todo_response = todos
                break

    if last_todo_response:
        agent._todo_store.write(last_todo_response, merge=False)
        if not agent.quiet_mode:
            from agent.display_helpers import vprint
            try:
                vprint(agent, f"{agent.log_prefix}\U0001f4cb Restored {len(last_todo_response)} todo item(s) from history")
            except Exception:
                pass

    # Clear interrupt state — the original run_agent code did this
    from tools.interrupt import set_interrupt as _set_interrupt
    _set_interrupt(False)
