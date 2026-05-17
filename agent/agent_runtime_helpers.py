"""Assorted AIAgent runtime helpers — moved out of run_agent.py for clarity.

Each function takes the parent ``AIAgent`` as its first argument
(``agent``) except for the static helpers (``sanitize_tool_call_arguments``,
``drop_thinking_only_and_merge_users``) which are stateless.  AIAgent
keeps thin forwarders for backward compatibility.

Methods covered:
* ``convert_to_trajectory_format`` — internal -> trajectory-file format
* ``sanitize_tool_call_arguments`` — repair corrupted JSON in tool_calls
* ``repair_message_sequence`` — enforce alternation invariants
* ``strip_think_blocks`` — remove inline reasoning from stored content
* ``recover_with_credential_pool`` — rotate pool entries on 429
* ``try_recover_primary_transport`` — re-create OpenAI client after rate-limit
* ``drop_thinking_only_and_merge_users`` — Anthropic-style cleanup
* ``restore_primary_runtime`` — un-do fallback activation
* ``extract_reasoning`` — pull reasoning fields out of API responses
* ``dump_api_request_debug`` — write request body for post-mortem
* ``anthropic_prompt_cache_policy`` — compute cache_control breakpoints
* ``create_openai_client`` — build the per-agent OpenAI SDK client
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.timeouts import get_provider_request_timeout
from agent.message_sanitization import (
    _repair_tool_call_arguments,
    _sanitize_surrogates,
)
from agent.tool_dispatch_helpers import _trajectory_normalize_msg
from agent.trajectory import convert_scratchpad_to_think
from agent.error_classifier import classify_api_error, FailoverReason
from utils import base_url_host_matches, base_url_hostname, env_var_enabled, atomic_json_write

logger = logging.getLogger(__name__)


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent
    return run_agent



def convert_to_trajectory_format(agent, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
    """
    Convert internal message format to trajectory format for saving.
    
    Args:
        messages (List[Dict]): Internal message history
        user_query (str): Original user query
        completed (bool): Whether the conversation completed successfully
        
    Returns:
        List[Dict]: Messages in trajectory format
    """
    # Normalize multimodal tool results — trajectories are text-only, so
    # replace image-bearing tool messages with their text_summary to avoid
    # embedding ~1MB base64 blobs into every saved trajectory.
    messages = [_trajectory_normalize_msg(m) for m in messages]
    trajectory = []
    
    # Add system message with tool definitions
    system_msg = (
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
        f"<tools>\n{agent._format_tools_for_system_message()}\n</tools>\n"
        "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
    )
    
    trajectory.append({
        "from": "system",
        "value": system_msg
    })
    
    # Add the actual user prompt (from the dataset) as the first human message
    trajectory.append({
        "from": "human",
        "value": user_query
    })
    
    # Skip the first message (the user query) since we already added it above.
    # Prefill messages are injected at API-call time only (not in the messages
    # list), so no offset adjustment is needed here.
    i = 1
    
    while i < len(messages):
        msg = messages[i]
        
        if msg["role"] == "assistant":
            # Check if this message has tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                # Format assistant message with tool calls
                # Add <think> tags around reasoning for trajectory storage
                content = ""
                
                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"
                
                if msg.get("content") and msg["content"].strip():
                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    content += convert_scratchpad_to_think(msg["content"]) + "\n"
                
                # Add tool calls wrapped in XML tags
                for tool_call in msg["tool_calls"]:
                    if not tool_call or not isinstance(tool_call, dict): continue
                    # Parse arguments - should always succeed since we validate during conversation
                    # but keep try-except as safety net
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                    except json.JSONDecodeError:
                        # This shouldn't happen since we validate and retry during conversation,
                        # but if it does, log warning and use empty dict
                        logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                        arguments = {}
                    
                    tool_call_json = {
                        "name": tool_call["function"]["name"],
                        "arguments": arguments
                    }
                    content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                
                # Ensure every gpt turn has a <think> block (empty if no reasoning)
                # so the format is consistent for training data
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content
                
                trajectory.append({
                    "from": "gpt",
                    "value": content.rstrip()
                })
                
                # Collect all subsequent tool responses
                tool_responses = []
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    tool_msg = messages[j]
                    # Format tool response with XML tags
                    tool_response = "<tool_response>\n"
                    
                    # Try to parse tool content as JSON if it looks like JSON
                    tool_content = tool_msg["content"]
                    try:
                        if tool_content.strip().startswith(("{", "[")):
                            tool_content = json.loads(tool_content)
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Keep as string if not valid JSON
                    
                    tool_index = len(tool_responses)
                    tool_name = (
                        msg["tool_calls"][tool_index]["function"]["name"]
                        if tool_index < len(msg["tool_calls"])
                        else "unknown"
                    )
                    tool_response += json.dumps({
                        "tool_call_id": tool_msg.get("tool_call_id", ""),
                        "name": tool_name,
                        "content": tool_content
                    }, ensure_ascii=False)
                    tool_response += "\n</tool_response>"
                    tool_responses.append(tool_response)
                    j += 1
                
                # Add all tool responses as a single message
                if tool_responses:
                    trajectory.append({
                        "from": "tool",
                        "value": "\n".join(tool_responses)
                    })
                    i = j - 1  # Skip the tool messages we just processed
            
            else:
                # Regular assistant message without tool calls
                # Add <think> tags around reasoning for trajectory storage
                content = ""
                
                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"
                
                # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                # (used when native thinking is disabled and model reasons via XML)
                raw_content = msg["content"] or ""
                content += convert_scratchpad_to_think(raw_content)
                
                # Ensure every gpt turn has a <think> block (empty if no reasoning)
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content
                
                trajectory.append({
                    "from": "gpt",
                    "value": content.strip()
                })
        
        elif msg["role"] == "user":
            trajectory.append({
                "from": "human",
                "value": msg["content"]
            })
        
        i += 1
    
    return trajectory



def sanitize_tool_call_arguments(
    messages: list,
    *,
    logger=None,
    session_id: str = None,
) -> int:
    """Repair corrupted assistant tool-call argument JSON in-place."""
    log = logger or logging.getLogger(__name__)
    if not isinstance(messages, list):
        return 0

    repaired = 0
    marker = _ra().AIAgent._TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER

    def _prepend_marker(tool_msg: dict) -> None:
        existing = tool_msg.get("content")
        if isinstance(existing, str):
            if not existing:
                tool_msg["content"] = marker
            elif not existing.startswith(marker):
                tool_msg["content"] = f"{marker}\n{existing}"
            return
        if existing is None:
            tool_msg["content"] = marker
            return
        try:
            existing_text = json.dumps(existing)
        except TypeError:
            existing_text = str(existing)
        tool_msg["content"] = f"{marker}\n{existing_text}"

    message_index = 0
    while message_index < len(messages):
        msg = messages[message_index]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            message_index += 1
            continue

        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            message_index += 1
            continue

        insert_at = message_index + 1
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments")
            if arguments is None or arguments == "":
                function["arguments"] = "{}"
                continue
            if isinstance(arguments, str) and not arguments.strip():
                function["arguments"] = "{}"
                continue
            if not isinstance(arguments, str):
                continue

            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                tool_call_id = tool_call.get("id")
                function_name = function.get("name", "?")
                preview = arguments[:80]
                log.warning(
                    "Corrupted tool_call arguments repaired before request "
                    "(session=%s, message_index=%s, tool_call_id=%s, function=%s, preview=%r)",
                    session_id or "-",
                    message_index,
                    tool_call_id or "-",
                    function_name,
                    preview,
                )
                function["arguments"] = "{}"

                existing_tool_msg = None
                scan_index = message_index + 1
                while scan_index < len(messages):
                    candidate = messages[scan_index]
                    if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                        break
                    if candidate.get("tool_call_id") == tool_call_id:
                        existing_tool_msg = candidate
                        break
                    scan_index += 1

                if existing_tool_msg is None:
                    messages.insert(
                        insert_at,
                        {
                            "role": "tool",
                            "name": function_name if function_name != "?" else "",
                            "tool_call_id": tool_call_id,
                            "content": marker,
                        },
                    )
                    insert_at += 1
                else:
                    _prepend_marker(existing_tool_msg)

                repaired += 1

        message_index += 1

    return repaired



def repair_message_sequence(agent, messages: List[Dict]) -> int:
    """Collapse malformed role-alternation left in the live history.

    Providers (OpenAI, OpenRouter, Anthropic) expect strict alternation:
    after the system message, user/tool alternates with assistant, with
    no two consecutive user messages and no tool-result that doesn't
    follow an assistant-with-tool_calls. Violations cause silent empty
    responses on most providers, which triggers the empty-retry loop.

    This runs right before the API call as a defensive belt — by the
    time it fires, the scaffolding strip should already have prevented
    most shapes, but external callers (gateway multi-queue replay,
    session resume, cron, explicit conversation_history passed in by
    host code) can feed in already-broken histories.

    Repairs applied:
      1. Stray ``tool`` messages whose ``tool_call_id`` doesn't match
         any preceding assistant tool_call — dropped.
      2. Consecutive ``user`` messages — merged with newline separator
         so no user input is lost.

    Deliberately does NOT rewind orphan ``assistant(tool_calls)+tool``
    pairs that precede a user message — that pattern IS valid when the
    previous turn completed normally and the user jumped in to redirect
    before the model got a continuation turn (the ongoing dialog
    pattern). The empty-response scaffolding stripper handles the
    genuinely-broken variant via its flag-gated rewind.

    Returns the number of repairs made (for logging/telemetry).
    """
    if not messages:
        return 0

    repairs = 0

    # Pass 1: drop stray tool messages that don't follow a known
    # assistant tool_call_id. Uses a rolling set of known ids refreshed
    # on each assistant message.
    known_tool_ids: set = set()
    filtered: List[Dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            filtered.append(msg)
            continue
        role = msg.get("role")
        if role == "assistant":
            known_tool_ids = set()
            for tc in (msg.get("tool_calls") or []):
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    known_tool_ids.add(tc_id)
            filtered.append(msg)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in known_tool_ids:
                filtered.append(msg)
            else:
                repairs += 1
        else:
            if role == "user":
                # A user turn closes the tool-result run; subsequent
                # tool messages without a fresh assistant tool_call
                # are orphans.
                known_tool_ids = set()
            filtered.append(msg)

    # Pass 2: merge consecutive user messages. Preserves all user input
    # so nothing the user typed is lost.
    merged: List[Dict] = []
    for msg in filtered:
        if (
            merged
            and isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(merged[-1], dict)
            and merged[-1].get("role") == "user"
        ):
            prev = merged[-1]
            prev_content = prev.get("content", "")
            new_content = msg.get("content", "")
            # Only merge plain-text content; leave multimodal (list)
            # content alone — collapsing image/audio blocks risks
            # mangling the attachment structure.
            if isinstance(prev_content, str) and isinstance(new_content, str):
                prev["content"] = (
                    (prev_content + "\n\n" + new_content)
                    if prev_content and new_content
                    else (prev_content or new_content)
                )
                repairs += 1
                continue
        merged.append(msg)

    if repairs > 0:
        # Rewrite in place so downstream paths (persistence, return
        # value, session DB flush) see the repaired sequence.
        messages[:] = merged

    return repairs



def strip_think_blocks(agent, content: str) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text.

    Handles four cases:
      1. Closed tag pairs (``<think>…</think>``) — the common path when
         the provider emits complete reasoning blocks.
      2. Unterminated open tag at a block boundary (start of text or
         after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
         closing tag is dropped.  Everything from the open tag to end
         of string is stripped.  The block-boundary check mirrors
         ``gateway/stream_consumer.py``'s filter so models that mention
         ``<think>`` in prose aren't over-stripped.
      3. Stray orphan open/close tags that slip through.
      4. Tag variants: ``<think>``, ``<thinking>``, ``<reasoning>``,
         ``<REASONING_SCRATCHPAD>``, ``<thought>`` (Gemma 4), all
         case-insensitive.

    Additionally strips standalone tool-call XML blocks that some open
    models (notably Gemma variants on OpenRouter) emit inside assistant
    content instead of via the structured ``tool_calls`` field:
      * ``<tool_call>…</tool_call>``
      * ``<tool_calls>…</tool_calls>``
      * ``<tool_result>…</tool_result>``
      * ``<function_call>…</function_call>``
      * ``<function_calls>…</function_calls>``
      * ``<function name="…">…</function>`` (Gemma style)
    Ported from openclaw/openclaw#67318. The ``<function>`` variant is
    boundary-gated (only strips when the tag sits at start-of-line or
    after punctuation and carries a ``name="..."`` attribute) so prose
    mentions like "Use <function> in JavaScript" are preserved.
    """
    if not content:
        return ""
    # 1. Closed tag pairs — case-insensitive for all variants so
    #    mixed-case tags (<THINK>, <Thinking>) don't slip through to
    #    the unterminated-tag pass and take trailing content with them.
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
    # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
    #     generic tag names first — they have no attribute gating since
    #     a literal <tool_call> in prose is already vanishingly rare.
    for _tc_name in ("tool_call", "tool_calls", "tool_result",
                      "function_call", "function_calls"):
        content = re.sub(
            rf'<{_tc_name}\b[^>]*>.*?</{_tc_name}>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
    # 1c. <function name="...">...</function> — Gemma-style standalone
    #     tool call. Only strip when the tag sits at a block boundary
    #     (start of text, after a newline, or after sentence-ending
    #     punctuation) AND carries a name="..." attribute. This keeps
    #     prose mentions like "Use <function> to declare" safe.
    content = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
        r'<function\b[^>]*\bname\s*=[^>]*>'
        r'(?:(?:(?!</function>).)*)</function>',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 2. Unterminated reasoning block — open tag at a block boundary
    #    (start of text, or after a newline) with no matching close.
    #    Strip from the tag to end of string.  Fixes #8878 / #9568
    #    (MiniMax M2.7 leaking raw reasoning into assistant content).
    content = re.sub(
        r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 3. Stray orphan open/close tags that slipped through.
    content = re.sub(
        r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )
    # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
    #     unterminated <function name="..."> because a truncated tail
    #     during streaming may still be valuable to the user; matches
    #     OpenClaw's intentional asymmetry.)
    content = re.sub(
        r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )
    return content



def recover_with_credential_pool(
    agent,
    *,
    status_code: Optional[int],
    has_retried_429: bool,
    classified_reason: Optional[FailoverReason] = None,
    error_context: Optional[Dict[str, Any]] = None,
) -> tuple[bool, bool]:
    """Attempt credential recovery via pool rotation.

    Returns (recovered, has_retried_429).
    On rate limits: first occurrence retries same credential (sets flag True).
                    second consecutive failure rotates to next credential.
    On billing exhaustion: immediately rotates.
    On auth failures: attempts token refresh before rotating.

    `classified_reason` lets the recovery path honor the structured error
    classifier instead of relying only on raw HTTP codes. This matters for
    providers that surface billing/rate-limit/auth conditions under a
    different status code, such as Anthropic returning HTTP 400 for
    "out of extra usage".
    """
    pool = agent._credential_pool
    if pool is None:
        return False, has_retried_429

    effective_reason = classified_reason
    if effective_reason is None:
        if status_code == 402:
            effective_reason = FailoverReason.billing
        elif status_code == 429:
            effective_reason = FailoverReason.rate_limit
        elif status_code in {401, 403}:
            effective_reason = FailoverReason.auth

    if effective_reason == FailoverReason.billing:
        rotate_status = status_code if status_code is not None else 402
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            logger.info(
                "Credential %s (billing) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False
        return False, has_retried_429

    if effective_reason == FailoverReason.rate_limit:
        if not has_retried_429:
            return False, True
        rotate_status = status_code if status_code is not None else 429
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            logger.info(
                "Credential %s (rate limit) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False
        return False, True

    if effective_reason == FailoverReason.auth:
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
            agent._swap_credential(refreshed)
            return True, has_retried_429
        # Refresh failed — rotate to next credential instead of giving up.
        # The failed entry is already marked exhausted by try_refresh_current().
        rotate_status = status_code if status_code is not None else 401
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            logger.info(
                "Credential %s (auth refresh failed) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False

    return False, has_retried_429



def try_recover_primary_transport(
    agent, api_error: Exception, *, retry_count: int, max_retries: int,
) -> bool:
    """Attempt one extra primary-provider recovery cycle for transient transport failures.

    After ``max_retries`` exhaust, rebuild the primary client (clearing
    stale connection pools) and give it one more attempt before falling
    back.  This is most useful for direct endpoints (custom, Z.AI,
    Anthropic, OpenAI, local models) where a TCP-level hiccup does not
    mean the provider is down.

    Skipped for proxy/aggregator providers (OpenRouter, Nous) which
    already manage connection pools and retries server-side — if our
    retries through them are exhausted, one more rebuilt client won't help.
    """
    if agent._fallback_activated:
        return False

    # Only for transient transport errors
    error_type = type(api_error).__name__
    if error_type not in _TRANSIENT_TRANSPORT_ERRORS:
        return False

    # Skip for aggregator providers — they manage their own retry infra
    if agent._is_openrouter_url():
        return False
    provider_lower = (agent.provider or "").strip().lower()
    if provider_lower in {"nous", "nous-research"}:
        return False

    try:
        # Close existing client to release stale connections
        if getattr(agent, "client", None) is not None:
            try:
                agent._close_openai_client(
                    agent.client, reason="primary_recovery", shared=True,
                )
            except Exception:
                pass

        # Rebuild from primary snapshot
        rt = agent._primary_runtime
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.base_url = rt["base_url"]
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]

        if agent.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client
            agent._anthropic_api_key = rt["anthropic_api_key"]
            agent._anthropic_base_url = rt["anthropic_base_url"]
            agent._anthropic_client = build_anthropic_client(
                rt["anthropic_api_key"], rt["anthropic_base_url"],
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
            agent.client = None
        else:
            agent.client = agent._create_openai_client(
                dict(rt["client_kwargs"]),
                reason="primary_recovery",
                shared=True,
            )

        wait_time = min(3 + retry_count, 8)
        agent._vprint(
            f"{agent.log_prefix}🔁 Transient {error_type} on {agent.provider} — "
            f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
            force=True,
        )
        time.sleep(wait_time)
        return True
    except Exception as e:
        logging.warning("Primary transport recovery failed: %s", e)
        return False

# ── End provider fallback ──────────────────────────────────────────────



def drop_thinking_only_and_merge_users(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop thinking-only assistant turns; merge any adjacent user messages left behind.

    Runs on the per-call ``api_messages`` copy only. The stored
    conversation history (``agent.messages``) is never mutated, so the
    user still sees the thinking block in the CLI/gateway transcript and
    session persistence keeps the full trace. Only the wire copy sent to
    the provider is cleaned.

    Why drop-and-merge rather than inject stub text:
    - Fabricating ``"."`` / ``"(continued)"`` text lies in the history
      and makes future turns see model output the model didn't emit.
    - Dropping the turn preserves honesty; merging adjacent user messages
      preserves the provider's role-alternation invariant.
    - This is the pattern used by Claude Code's ``normalizeMessagesForAPI``
      (filterOrphanedThinkingOnlyMessages + mergeAdjacentUserMessages).
    """
    if not messages:
        return messages

    # Pass 1: drop thinking-only assistant turns.
    kept = [m for m in messages if not _ra().AIAgent._is_thinking_only_assistant(m)]
    dropped = len(messages) - len(kept)
    if dropped == 0:
        return messages

    # Pass 2: merge any newly-adjacent user messages.
    merged: List[Dict[str, Any]] = []
    merges = 0
    for m in kept:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.get("role") == "user"
            and m.get("role") == "user"
        ):
            prev_content = prev.get("content", "")
            cur_content = m.get("content", "")
            # Work on a copy of ``prev`` so the caller's input dicts are
            # never mutated. ``_sanitize_api_messages`` upstream already
            # hands us per-call copies, but staying pure here means we
            # can be called safely from anywhere (tests, other loops).
            prev_copy = dict(prev)
            # Only string-content merge is meaningful for role-alternation
            # purposes. If either side is a list (multimodal), append as a
            # separate block rather than collapsing.
            if isinstance(prev_content, str) and isinstance(cur_content, str):
                sep = "\n\n" if prev_content and cur_content else ""
                prev_copy["content"] = prev_content + sep + cur_content
            elif isinstance(prev_content, list) and isinstance(cur_content, list):
                prev_copy["content"] = list(prev_content) + list(cur_content)
            elif isinstance(prev_content, list) and isinstance(cur_content, str):
                if cur_content:
                    prev_copy["content"] = list(prev_content) + [
                        {"type": "text", "text": cur_content}
                    ]
                else:
                    prev_copy["content"] = list(prev_content)
            elif isinstance(prev_content, str) and isinstance(cur_content, list):
                new_blocks: List[Dict[str, Any]] = []
                if prev_content:
                    new_blocks.append({"type": "text", "text": prev_content})
                new_blocks.extend(cur_content)
                prev_copy["content"] = new_blocks
            else:
                # Unknown content shape — fall back to appending separately
                # (violates alternation, but safer than raising in a hot path).
                merged.append(m)
                continue
            merged[-1] = prev_copy
            merges += 1
        else:
            merged.append(m)

    logger.debug(
        "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
        "merged %d adjacent user message(s)",
        dropped,
        merges,
    )
    return merged



def restore_primary_runtime(agent) -> bool:
    """Restore the primary runtime at the start of a new turn.

    In long-lived CLI sessions a single AIAgent instance spans multiple
    turns.  Without restoration, one transient failure pins the session
    to the fallback provider for every subsequent turn.  Calling this at
    the top of ``run_conversation()`` makes fallback turn-scoped.

    The gateway caches agents across messages (``_agent_cache`` in
    ``gateway/run.py``), so this restoration IS needed there too.
    """
    if not agent._fallback_activated:
        return False

    if getattr(agent, "_rate_limited_until", 0) > time.monotonic():
        return False  # primary still in rate-limit cooldown, stay on fallback

    rt = agent._primary_runtime
    try:
        # ── Core runtime state ──
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.base_url = rt["base_url"]           # setter updates _base_url_lower
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent._use_prompt_caching = rt["use_prompt_caching"]
        # Default to native layout when the restored snapshot predates the
        # native-vs-proxy split (older sessions saved before this PR).
        agent._use_native_cache_layout = rt.get(
            "use_native_cache_layout",
            agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
        )

        # ── Rebuild client for the primary provider ──
        if agent.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client
            agent._anthropic_api_key = rt["anthropic_api_key"]
            agent._anthropic_base_url = rt["anthropic_base_url"]
            agent._anthropic_client = build_anthropic_client(
                rt["anthropic_api_key"], rt["anthropic_base_url"],
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
            agent.client = None
        else:
            agent.client = agent._create_openai_client(
                dict(rt["client_kwargs"]),
                reason="restore_primary",
                shared=True,
            )

        # ── Restore context engine state ──
        cc = agent.context_compressor
        cc.update_model(
            model=rt["compressor_model"],
            context_length=rt["compressor_context_length"],
            base_url=rt["compressor_base_url"],
            api_key=rt["compressor_api_key"],
            provider=rt["compressor_provider"],
        )

        # ── Reset fallback chain for the new turn ──
        agent._fallback_activated = False
        agent._fallback_index = 0

        logging.info(
            "Primary runtime restored for new turn: %s (%s)",
            agent.model, agent.provider,
        )
        return True
    except Exception as e:
        logging.warning("Failed to restore primary runtime: %s", e)
        return False

# Which error types indicate a transient transport failure worth
# one more attempt with a rebuilt client / connection pool.
_TRANSIENT_TRANSPORT_ERRORS = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "APIConnectionError", "APITimeoutError",
})



def extract_reasoning(agent, assistant_message) -> Optional[str]:
    """
    Extract reasoning/thinking content from an assistant message.
    
    OpenRouter and various providers can return reasoning in multiple formats:
    1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
    2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
    3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)
    
    Args:
        assistant_message: The assistant message object from the API response
        
    Returns:
        Combined reasoning text, or None if no reasoning found
    """
    reasoning_parts = []
    
    # Check direct reasoning field
    if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
        reasoning_parts.append(assistant_message.reasoning)
    
    # Check reasoning_content field (alternative name used by some providers)
    if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
        # Don't duplicate if same as reasoning
        if assistant_message.reasoning_content not in reasoning_parts:
            reasoning_parts.append(assistant_message.reasoning_content)
    
    # Check reasoning_details array (OpenRouter unified format)
    # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        for detail in assistant_message.reasoning_details:
            if isinstance(detail, dict):
                # Extract summary from reasoning detail object
                summary = (
                    detail.get('summary')
                    or detail.get('thinking')
                    or detail.get('content')
                    or detail.get('text')
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary)

    # Some providers embed reasoning directly inside assistant content
    # instead of returning structured reasoning fields.  Only fall back
    # to inline extraction when no structured reasoning was found.
    content = getattr(assistant_message, "content", None)
    if not reasoning_parts and isinstance(content, list):
        # DeepSeek V4 Pro (and compatible providers) return content as a
        # list of typed blocks, e.g.:
        #   [{"type": "thinking", "thinking": "..."}, {"type": "output", ...}]
        # Without this branch the thinking text is silently dropped and the
        # next turn fails with HTTP 400 ("thinking must be passed back").
        # Refs #21944.
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_text = block.get("thinking") or block.get("text") or ""
                thinking_text = thinking_text.strip()
                if thinking_text and thinking_text not in reasoning_parts:
                    reasoning_parts.append(thinking_text)
    if not reasoning_parts and isinstance(content, str) and content:
        inline_patterns = (
            r"<think>(.*?)</think>",
            r"<thinking>(.*?)</thinking>",
            r"<thought>(.*?)</thought>",
            r"<reasoning>(.*?)</reasoning>",
            r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
        )
        for pattern in inline_patterns:
            flags = re.DOTALL | re.IGNORECASE
            for block in re.findall(pattern, content, flags=flags):
                cleaned = block.strip()
                if cleaned and cleaned not in reasoning_parts:
                    reasoning_parts.append(cleaned)
    
    # Combine all reasoning parts
    if reasoning_parts:
        return "\n\n".join(reasoning_parts)
    
    return None



def dump_api_request_debug(
    agent,
    api_kwargs: Dict[str, Any],
    *,
    reason: str,
    error: Optional[Exception] = None,
) -> Optional[Path]:
    """
    Dump a debug-friendly HTTP request record for the active inference API.

    Captures the request body from api_kwargs (excluding transport-only keys
    like timeout). Intended for debugging provider-side 4xx failures where
    retries are not useful.
    """
    try:
        body = copy.deepcopy(api_kwargs)
        body.pop("timeout", None)
        body = {k: v for k, v in body.items() if v is not None}

        api_key = None
        try:
            api_key = getattr(agent.client, "api_key", None)
        except Exception as e:
            logger.debug("Could not extract API key for debug dump: %s", e)

        dump_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": agent.session_id,
            "reason": reason,
            "request": {
                "method": "POST",
                "url": f"{agent.base_url.rstrip('/')}{'/responses' if agent.api_mode == 'codex_responses' else '/chat/completions'}",
                "headers": {
                    "Authorization": f"Bearer {agent._mask_api_key_for_logs(api_key)}",
                    "Content-Type": "application/json",
                },
                "body": body,
            },
        }

        if error is not None:
            error_info: Dict[str, Any] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            for attr_name in ("status_code", "request_id", "code", "param", "type"):
                attr_value = getattr(error, attr_name, None)
                if attr_value is not None:
                    error_info[attr_name] = attr_value

            body_attr = getattr(error, "body", None)
            if body_attr is not None:
                error_info["body"] = body_attr

            response_obj = getattr(error, "response", None)
            if response_obj is not None:
                try:
                    error_info["response_status"] = getattr(response_obj, "status_code", None)
                    error_info["response_text"] = response_obj.text
                except Exception as e:
                    logger.debug("Could not extract error response details: %s", e)

            dump_payload["error"] = error_info

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dump_file = agent.logs_dir / f"request_dump_{agent.session_id}_{timestamp}.json"
        dump_file.write_text(
            json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        agent._vprint(f"{agent.log_prefix}🧾 Request debug dump written to: {dump_file}")

        if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
            print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

        return dump_file
    except Exception as dump_error:
        if agent.verbose_logging:
            logging.warning(f"Failed to dump API request debug payload: {dump_error}")
        return None



def anthropic_prompt_cache_policy(
    agent,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[bool, bool]:
    """Decide whether to apply Anthropic prompt caching and which layout to use.

    Returns ``(should_cache, use_native_layout)``:
      * ``should_cache`` — inject ``cache_control`` breakpoints for this
        request (applies to OpenRouter Claude, native Anthropic, and
        third-party gateways that speak the native Anthropic protocol).
      * ``use_native_layout`` — place markers on the *inner* content
        blocks (native Anthropic accepts and requires this layout);
        when False markers go on the message envelope (OpenRouter and
        OpenAI-wire proxies expect the looser layout).

    Third-party providers using the native Anthropic transport
    (``api_mode == 'anthropic_messages'`` + Claude-named model) get
    caching with the native layout so they benefit from the same
    cost reduction as direct Anthropic callers, provided their
    gateway implements the Anthropic cache_control contract
    (MiniMax, Zhipu GLM, LiteLLM's Anthropic proxy mode all do).

    Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct
    Alibaba (DashScope) also honour Anthropic-style ``cache_control``
    markers on OpenAI-wire chat completions. Upstream pi-mono #3392 /
    pi #3393 documented this for opencode-go Qwen. Without markers
    these providers serve zero cache hits, re-billing the full prompt
    on every turn.
    """
    eff_provider = (provider if provider is not None else agent.provider) or ""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    eff_model = (model if model is not None else agent.model) or ""

    model_lower = eff_model.lower()
    provider_lower = eff_provider.lower()
    is_claude = "claude" in model_lower
    is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
    # Nous Portal proxies to OpenRouter behind the scenes — identical
    # OpenAI-wire envelope cache_control semantics. Treat it as an
    # OpenRouter-equivalent endpoint for caching layout purposes.
    is_nous_portal = "nousresearch" in eff_base_url.lower()
    is_anthropic_wire = eff_api_mode == "anthropic_messages"
    is_native_anthropic = (
        is_anthropic_wire
        and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
    )

    if is_native_anthropic:
        return True, True
    if (is_openrouter or is_nous_portal) and is_claude:
        return True, False
    # Nous Portal Qwen (e.g. qwen3.6-plus) takes the same envelope-layout
    # cache_control path as Portal Claude. Portal proxies to OpenRouter
    # and the upstream Qwen route accepts cache_control markers; without
    # this branch the alibaba-family check below only matches
    # provider=opencode/alibaba and Portal traffic falls through to
    # (False, False), serving 0% cache hits and re-billing the full
    # prompt on every turn.
    if is_nous_portal and "qwen" in model_lower:
        return True, False
    if is_anthropic_wire and is_claude:
        # Third-party Anthropic-compatible gateway.
        return True, True

    # MiniMax on its Anthropic-compatible endpoint serves its own
    # model family (MiniMax-M2.7, M2.5, M2.1, M2) with documented
    # cache_control support (0.1× read pricing, 5-minute TTL).  The
    # blanket is_claude gate above excludes these — opt them in
    # explicitly via provider id or host match so users on
    # provider=minimax / minimax-cn (or custom endpoints pointing at
    # api.minimax.io/anthropic / api.minimaxi.com/anthropic) get the
    # same cost reduction as Claude traffic.
    # Docs: https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache
    if is_anthropic_wire:
        is_minimax_provider = provider_lower in {"minimax", "minimax-cn"}
        is_minimax_host = (
            base_url_host_matches(eff_base_url, "api.minimax.io")
            or base_url_host_matches(eff_base_url, "api.minimaxi.com")
        )
        if is_minimax_provider or is_minimax_host:
            return True, True

    # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire
    # transport that accepts Anthropic-style cache_control markers and
    # rewards them with real cache hits.  Without this branch
    # qwen3.6-plus on opencode-go reports 0% cached tokens and burns
    # through the subscription on every turn.
    model_is_qwen = "qwen" in model_lower
    provider_is_alibaba_family = provider_lower in {
        "opencode", "opencode-zen", "opencode-go", "alibaba",
    }
    if provider_is_alibaba_family and model_is_qwen:
        # Envelope layout (native_anthropic=False): markers on inner
        # content parts, not top-level tool messages.  Matches
        # pi-mono's "alibaba" cacheControlFormat.
        return True, False

    return False, False



def create_openai_client(agent, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    # Treat client_kwargs as read-only. Callers pass agent._client_kwargs (or shallow
    # copies of it) in; any in-place mutation leaks back into the stored dict and is
    # reused on subsequent requests. #10933 hit this by injecting an httpx.Client
    # transport that was torn down after the first request, so the next request
    # wrapped a closed transport and raised "Cannot send a request, as the client
    # has been closed" on every retry. The revert resolved that specific path; this
    # copy locks the contract so future transport/keepalive work can't reintroduce
    # the same class of bug.
    client_kwargs = dict(client_kwargs)
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))
    if agent.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
        from agent.copilot_acp_client import CopilotACPClient

        client = CopilotACPClient(**client_kwargs)
        logger.info(
            "Copilot ACP client created (%s, shared=%s) %s",
            reason,
            shared,
            agent._client_log_context(),
        )
        return client
    if agent.provider == "google-gemini-cli" or str(client_kwargs.get("base_url", "")).startswith("cloudcode-pa://"):
        from agent.gemini_cloudcode_adapter import GeminiCloudCodeClient

        # Strip OpenAI-specific kwargs the Gemini client doesn't accept
        safe_kwargs = {
            k: v for k, v in client_kwargs.items()
            if k in {"api_key", "base_url", "default_headers", "project_id", "timeout"}
        }
        client = GeminiCloudCodeClient(**safe_kwargs)
        logger.info(
            "Gemini Cloud Code Assist client created (%s, shared=%s) %s",
            reason,
            shared,
            agent._client_log_context(),
        )
        return client
    if agent.provider == "gemini":
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url", "") or "")
        if is_native_gemini_base_url(base_url):
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
            }
            if "http_client" not in safe_kwargs:
                keepalive_http = agent._build_keepalive_http_client(base_url)
                if keepalive_http is not None:
                    safe_kwargs["http_client"] = keepalive_http
            client = GeminiNativeClient(**safe_kwargs)
            logger.info(
                "Gemini native client created (%s, shared=%s) %s",
                reason,
                shared,
                agent._client_log_context(),
            )
            return client
    # Inject TCP keepalives so the kernel detects dead provider connections
    # instead of letting them sit silently in CLOSE-WAIT (#10324).  Without
    # this, a peer that drops mid-stream leaves the socket in a state where
    # epoll_wait never fires, ``httpx`` read timeout may not trigger, and
    # the agent hangs until manually killed.  Probes after 30s idle, retry
    # every 10s, give up after 3 → dead peer detected within ~60s.
    #
    # Safety against #10933: the ``client_kwargs = dict(client_kwargs)``
    # above means this injection only lands in the local per-call copy,
    # never back into ``agent._client_kwargs``.  Each ``_create_openai_client``
    # invocation therefore gets its OWN fresh ``httpx.Client`` whose
    # lifetime is tied to the OpenAI client it is passed to.  When the
    # OpenAI client is closed (rebuild, teardown, credential rotation),
    # the paired ``httpx.Client`` closes with it, and the next call
    # constructs a fresh one — no stale closed transport can be reused.
    # Tests in ``tests/run_agent/test_create_openai_client_reuse.py`` and
    # ``tests/run_agent/test_sequential_chats_live.py`` pin this invariant.
    if "http_client" not in client_kwargs:
        keepalive_http = agent._build_keepalive_http_client(client_kwargs.get("base_url", ""))
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http
    # Uses the module-level `OpenAI` name, resolved lazily on first
    # access via __getattr__ below. Tests patch via `run_agent.OpenAI`.
    client = _ra().OpenAI(**client_kwargs)
    logger.info(
        "OpenAI client created (%s, shared=%s) %s",
        reason,
        shared,
        agent._client_log_context(),
    )
    return client



__all__ = [
    "convert_to_trajectory_format",
    "sanitize_tool_call_arguments",
    "repair_message_sequence",
    "strip_think_blocks",
    "recover_with_credential_pool",
    "try_recover_primary_transport",
    "drop_thinking_only_and_merge_users",
    "restore_primary_runtime",
    "extract_reasoning",
    "dump_api_request_debug",
    "anthropic_prompt_cache_policy",
    "create_openai_client",
]
