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
            _ra().logger.info(
                "Credential %s (billing) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False
        return False, has_retried_429

    if effective_reason == FailoverReason.rate_limit:
        usage_limit_reached = False
        if error_context:
            context_reason = str(error_context.get("reason") or "").lower()
            context_message = str(error_context.get("message") or "").lower()
            usage_limit_reached = (
                "usage_limit_reached" in context_reason
                or "usage limit has been reached" in context_message
            )
        if not has_retried_429 and not usage_limit_reached:
            return False, True
        rotate_status = status_code if status_code is not None else 429
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            _ra().logger.info(
                "Credential %s (rate limit) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            agent._swap_credential(next_entry)
            return True, False
        return False, True

    if effective_reason == FailoverReason.auth:
        if agent._is_entitlement_failure(error_context, status_code):
            _ra().logger.info(
                "Credential %s — entitlement-shaped 403 from %s; "
                "skipping pool refresh (account lacks subscription, "
                "not a transient auth failure).",
                status_code if status_code is not None else "auth",
                agent.provider or "provider",
            )
            return False, has_retried_429
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            _ra().logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
            agent._swap_credential(refreshed)
            return True, has_retried_429
        # Refresh failed — rotate to next credential instead of giving up.
        # The failed entry is already marked exhausted by try_refresh_current().
        rotate_status = status_code if status_code is not None else 401
        next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
        if next_entry is not None:
            _ra().logger.info(
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

    _ra().logger.debug(
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
        # Reset the chain index even when no fallback was activated this
        # turn.  Without this, a turn where _try_activate_fallback() was
        # called but returned False (chain exhausted or provider not
        # configured) leaves _fallback_index >= len(_fallback_chain) while
        # _fallback_activated stays False.  The next turn skips this block
        # entirely, stranding the index and silently blocking all future
        # fallback attempts for the session.  Fixes #20465.
        agent._fallback_index = 0
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
            _ra().logger.debug("Could not extract API key for debug dump: %s", e)

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
                    _ra().logger.debug("Could not extract error response details: %s", e)

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
        _ra().logger.info(
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
        _ra().logger.info(
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
            _ra().logger.info(
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
    _ra().logger.info(
        "OpenAI client created (%s, shared=%s) %s",
        reason,
        shared,
        agent._client_log_context(),
    )
    return client


def switch_model(agent, new_model, new_provider, api_key='', base_url='', api_mode=''):
    """Switch the model/provider in-place for a live agent.

    Called by the /model command handlers (CLI and gateway) after
    ``model_switch.switch_model()`` has resolved credentials and
    validated the model.  This method performs the actual runtime
    swap: rebuilding clients, updating caching flags, and refreshing
    the context compressor.

    The implementation mirrors ``_try_activate_fallback()`` for the
    client-swap logic but also updates ``_primary_runtime`` so the
    change persists across turns (unlike fallback which is
    turn-scoped).
    """
    from hermes_cli.providers import determine_api_mode

    # ── Determine api_mode if not provided ──
    if not api_mode:
        api_mode = determine_api_mode(new_provider, base_url)

    # Defense-in-depth: ensure OpenCode base_url doesn't carry a trailing
    # /v1 into the anthropic_messages client, which would cause the SDK to
    # hit /v1/v1/messages.  `model_switch.switch_model()` already strips
    # this, but we guard here so any direct callers (future code paths,
    # tests) can't reintroduce the double-/v1 404 bug.
    if (
        api_mode == "anthropic_messages"
        and new_provider in {"opencode-zen", "opencode-go"}
        and isinstance(base_url, str)
        and base_url
    ):
        base_url = re.sub(r"/v1/?$", "", base_url)

    old_model = agent.model
    old_provider = agent.provider

    # Clear the per-config context_length override so the new model's
    # actual context window is resolved via get_model_context_length()
    # instead of inheriting the stale value from the previous model.
    agent._config_context_length = None

    # ── Swap core runtime fields ──
    agent.model = new_model
    agent.provider = new_provider
    # Use new base_url when provided; only fall back to current when the
    # new provider genuinely has no endpoint (e.g. native SDK providers).
    # Without this guard the old provider's URL (e.g. Ollama's localhost
    # address) would persist silently after switching to a cloud provider
    # that returns an empty base_url string.
    if base_url:
        agent.base_url = base_url
    agent.api_mode = api_mode
    # Invalidate transport cache — new api_mode may need a different transport
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    if api_key:
        agent.api_key = api_key

    # ── Build new client ──
    if api_mode == "anthropic_messages":
        from agent.anthropic_adapter import (
            build_anthropic_client,
            resolve_anthropic_token,
            _is_oauth_token,
        )
        # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own
        # API key — falling back would send Anthropic credentials to third-party endpoints.
        _is_native_anthropic = new_provider == "anthropic"
        effective_key = (api_key or agent.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or agent.api_key or "")
        agent.api_key = effective_key
        agent._anthropic_api_key = effective_key
        agent._anthropic_base_url = base_url or getattr(agent, "_anthropic_base_url", None)
        agent._anthropic_client = build_anthropic_client(
            effective_key, agent._anthropic_base_url,
            timeout=get_provider_request_timeout(agent.provider, agent.model),
        )
        agent._is_anthropic_oauth = _is_oauth_token(effective_key) if _is_native_anthropic else False
        agent.client = None
        agent._client_kwargs = {}
    else:
        effective_key = api_key or agent.api_key
        effective_base = base_url or agent.base_url
        agent._client_kwargs = {
            "api_key": effective_key,
            "base_url": effective_base,
        }
        _sm_timeout = get_provider_request_timeout(agent.provider, agent.model)
        if _sm_timeout is not None:
            agent._client_kwargs["timeout"] = _sm_timeout
        agent.client = agent._create_openai_client(
            dict(agent._client_kwargs),
            reason="switch_model",
            shared=True,
        )

    # ── Re-evaluate prompt caching ──
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        agent._anthropic_prompt_cache_policy(
            provider=new_provider,
            base_url=agent.base_url,
            api_mode=api_mode,
            model=new_model,
        )
    )

    # ── LM Studio: preload before probing context length ──
    agent._ensure_lmstudio_runtime_loaded()

    # ── Update context compressor ──
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        from agent.model_metadata import get_model_context_length
        # Re-read custom_providers from live config so per-model
        # context_length overrides are honored when switching to a
        # custom provider mid-session (closes #15779).
        _sm_custom_providers = None
        try:
            from hermes_cli.config import load_config, get_compatible_custom_providers
            _sm_cfg = load_config()
            _sm_custom_providers = get_compatible_custom_providers(_sm_cfg)
        except Exception:
            _sm_custom_providers = None
        new_context_length = get_model_context_length(
            agent.model,
            base_url=agent.base_url,
            api_key=agent.api_key,
            provider=agent.provider,
            config_context_length=getattr(agent, "_config_context_length", None),
            custom_providers=_sm_custom_providers,
        )
        agent.context_compressor.update_model(
            model=agent.model,
            context_length=new_context_length,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
            provider=agent.provider,
            api_mode=agent.api_mode,
        )

    # ── Invalidate cached system prompt so it rebuilds next turn ──
    agent._cached_system_prompt = None

    # ── Update _primary_runtime so the change persists across turns ──
    _cc = agent.context_compressor if hasattr(agent, "context_compressor") and agent.context_compressor else None
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "compressor_model": getattr(_cc, "model", agent.model) if _cc else agent.model,
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url) if _cc else agent.base_url,
        "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
        "compressor_provider": getattr(_cc, "provider", agent.provider) if _cc else agent.provider,
        "compressor_context_length": _cc.context_length if _cc else 0,
        "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
    }
    if api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })

    # ── Reset fallback state ──
    agent._fallback_activated = False
    agent._fallback_index = 0

    # When the user deliberately swaps primary providers (e.g. openrouter
    # → anthropic), drop any fallback entries that target the OLD primary
    # or the NEW one.  The chain was seeded from config at agent init for
    # the original provider — without pruning, a failed turn on the new
    # primary silently re-activates the provider the user just rejected,
    # which is exactly what was reported during TUI v2 blitz testing
    # ("switched to anthropic, tui keeps trying openrouter").
    old_norm = (old_provider or "").strip().lower()
    new_norm = (new_provider or "").strip().lower()
    fallback_chain = list(getattr(agent, "_fallback_chain", []) or [])
    if old_norm and new_norm and old_norm != new_norm:
        fallback_chain = [
            entry for entry in fallback_chain
            if (entry.get("provider") or "").strip().lower() not in {old_norm, new_norm}
        ]
    agent._fallback_chain = fallback_chain
    agent._fallback_model = fallback_chain[0] if fallback_chain else None

    logging.info(
        "Model switched in-place: %s (%s) -> %s (%s)",
        old_model, old_provider, new_model, new_provider,
    )



def invoke_tool(agent, function_name: str, function_args: dict, effective_task_id: str,
                 tool_call_id: Optional[str] = None, messages: list = None,
                 pre_tool_block_checked: bool = False) -> str:
    """Invoke a single tool and return the result string. No display logic.

    Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
    tools. Used by the concurrent execution path; the sequential path retains
    its own inline invocation for backward-compatible display handling.
    """
    # Check plugin hooks for a block directive before executing anything.
    block_message: Optional[str] = None
    if not pre_tool_block_checked:
        try:
            from hermes_cli.plugins import get_pre_tool_call_block_message
            block_message = get_pre_tool_call_block_message(
                function_name, function_args, task_id=effective_task_id or "",
            )
        except Exception:
            pass
    if block_message is not None:
        return json.dumps({"error": block_message}, ensure_ascii=False)

    if function_name == "todo":
        from tools.todo_tool import todo_tool as _todo_tool
        return _todo_tool(
            todos=function_args.get("todos"),
            merge=function_args.get("merge", False),
            store=agent._todo_store,
        )
    elif function_name == "session_search":
        session_db = agent._get_session_db_for_recall()
        if not session_db:
            from hermes_state import format_session_db_unavailable
            return json.dumps({"success": False, "error": format_session_db_unavailable()})
        from tools.session_search_tool import session_search as _session_search
        return _session_search(
            query=function_args.get("query", ""),
            role_filter=function_args.get("role_filter"),
            limit=function_args.get("limit", 3),
            db=session_db,
            current_session_id=agent.session_id,
        )
    elif function_name == "memory":
        target = function_args.get("target", "memory")
        from tools.memory_tool import memory_tool as _memory_tool
        result = _memory_tool(
            action=function_args.get("action"),
            target=target,
            content=function_args.get("content"),
            old_text=function_args.get("old_text"),
            store=agent._memory_store,
        )
        # Bridge: notify external memory provider of built-in memory writes
        if agent._memory_manager and function_args.get("action") in {"add", "replace"}:
            try:
                agent._memory_manager.on_memory_write(
                    function_args.get("action", ""),
                    target,
                    function_args.get("content", ""),
                    metadata=agent._build_memory_write_metadata(
                        task_id=effective_task_id,
                        tool_call_id=tool_call_id,
                    ),
                )
            except Exception:
                pass
        return result
    elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
        return agent._memory_manager.handle_tool_call(function_name, function_args)
    elif function_name == "clarify":
        from tools.clarify_tool import clarify_tool as _clarify_tool
        return _clarify_tool(
            question=function_args.get("question", ""),
            choices=function_args.get("choices"),
            callback=agent.clarify_callback,
        )
    elif function_name == "delegate_task":
        return agent._dispatch_delegate_task(function_args)
    else:
        return _ra().handle_function_call(
            function_name, function_args, effective_task_id,
            tool_call_id=tool_call_id,
            session_id=agent.session_id or "",
            enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
            skip_pre_tool_call_hook=True,
        )



def repair_tool_call(agent, tool_name: str) -> str | None:
    """Attempt to repair a mismatched tool name before aborting.

    Models sometimes emit variants of a tool name that differ only
    in casing, separators, or class-like suffixes. Normalize
    aggressively before falling back to fuzzy match:

    1. Lowercase direct match.
    2. Lowercase + hyphens/spaces -> underscores.
    3. CamelCase -> snake_case (TodoTool -> todo_tool).
    4. Strip trailing ``_tool`` / ``-tool`` / ``tool`` suffix that
       Claude-style models sometimes tack on (TodoTool_tool ->
       TodoTool -> Todo -> todo). Applied twice so double-tacked
       suffixes like ``TodoTool_tool`` reduce all the way.
    5. Fuzzy match (difflib, cutoff=0.7).

    See #14784 for the original reports (TodoTool_tool, Patch_tool,
    BrowserClick_tool were all returning "Unknown tool" before).

    Returns the repaired name if found in valid_tool_names, else None.
    """
    import re
    from difflib import get_close_matches

    if not tool_name:
        return None

    def _norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "_")

    def _camel_snake(s: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    def _strip_tool_suffix(s: str) -> str | None:
        lc = s.lower()
        for suffix in ("_tool", "-tool", "tool"):
            if lc.endswith(suffix):
                return s[: -len(suffix)].rstrip("_-")
        return None

    # Cheap fast-paths first — these cover the common case.
    lowered = tool_name.lower()
    if lowered in agent.valid_tool_names:
        return lowered
    normalized = _norm(tool_name)
    if normalized in agent.valid_tool_names:
        return normalized

    # Build the full candidate set for class-like emissions.
    cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
    # Strip trailing tool-suffix up to twice — TodoTool_tool needs it.
    for _ in range(2):
        extra: set[str] = set()
        for c in cands:
            stripped = _strip_tool_suffix(c)
            if stripped:
                extra.add(stripped)
                extra.add(_norm(stripped))
                extra.add(_camel_snake(stripped))
        cands |= extra

    for c in cands:
        if c and c in agent.valid_tool_names:
            return c

    # Fuzzy match as last resort.
    matches = get_close_matches(lowered, agent.valid_tool_names, n=1, cutoff=0.7)
    if matches:
        return matches[0]

    return None



def sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs before every LLM call.

    Runs unconditionally — not gated on whether the context compressor
    is present — so orphans from session loading or manual message
    manipulation are always caught.
    """
    # --- Role allowlist: drop messages with roles the API won't accept ---
    filtered = []
    for msg in messages:
        role = msg.get("role")
        if role not in _ra().AIAgent._VALID_API_ROLES:
            _ra().logger.debug(
                "Pre-call sanitizer: dropping message with invalid role %r",
                role,
            )
            continue
        filtered.append(msg)
    messages = filtered

    surviving_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = _ra().AIAgent._get_tool_call_id_static(tc)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # 1. Drop tool results with no matching assistant call
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]
        _ra().logger.debug(
            "Pre-call sanitizer: removed %d orphaned tool result(s)",
            len(orphaned_results),
        )

    # 2. Inject stub results for calls whose result was dropped
    missing_results = surviving_call_ids - result_call_ids
    if missing_results:
        patched: List[Dict[str, Any]] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = _ra().AIAgent._get_tool_call_id_static(tc)
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "name": _ra().AIAgent._get_tool_call_name_static(tc),
                            "content": "[Result unavailable — see context summary above]",
                            "tool_call_id": cid,
                        })
        messages = patched
        _ra().logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s)",
            len(missing_results),
        )
    return messages



def looks_like_codex_intermediate_ack(
    agent,
    user_message: str,
    assistant_content: str,
    messages: List[Dict[str, Any]],
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn."""
    if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
        return False

    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text:
        return False
    if len(assistant_text) > 1200:
        return False

    has_future_ack = bool(
        re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
    )
    if not has_future_ack:
        return False

    action_markers = (
        "look into",
        "look at",
        "inspect",
        "scan",
        "check",
        "analyz",
        "review",
        "explore",
        "read",
        "open",
        "run",
        "test",
        "fix",
        "debug",
        "search",
        "find",
        "walkthrough",
        "report back",
        "summarize",
    )
    workspace_markers = (
        "directory",
        "current directory",
        "current dir",
        "cwd",
        "repo",
        "repository",
        "codebase",
        "project",
        "folder",
        "filesystem",
        "file tree",
        "files",
        "path",
    )

    user_text = (user_message or "").strip().lower()
    user_targets_workspace = (
        any(marker in user_text for marker in workspace_markers)
        or "~/" in user_text
        or "/" in user_text
    )
    assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
    assistant_targets_workspace = any(
        marker in assistant_text for marker in workspace_markers
    )
    return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action




def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:
    """Copy provider-facing reasoning fields onto an API replay message."""
    if source_msg.get("role") != "assistant":
        return

    # 1. Explicit reasoning_content already set — preserve it verbatim
    # (includes DeepSeek/Kimi's own space-placeholder written at creation
    # time, and any valid reasoning content from the same provider).
    #
    # Exception: sessions persisted BEFORE #17341 have empty-string
    # placeholders pinned at creation time. DeepSeek V4 Pro rejects
    # those with HTTP 400. When the active provider enforces the
    # thinking-mode echo, upgrade "" → " " on replay so stale history
    # doesn't 400 the user on the next turn.
    existing = source_msg.get("reasoning_content")
    if isinstance(existing, str):
        if existing == "" and agent._needs_thinking_reasoning_pad():
            api_msg["reasoning_content"] = " "
        else:
            api_msg["reasoning_content"] = existing
        return

    needs_thinking_pad = agent._needs_thinking_reasoning_pad()

    # 2. Cross-provider poisoned history (#15748): on DeepSeek/Kimi,
    # if the source turn has tool_calls AND a 'reasoning' field but no
    # 'reasoning_content' key, the 'reasoning' text was written by a
    # prior provider (e.g. MiniMax) — DeepSeek's own _build_assistant_message
    # pins reasoning_content at creation time for tool-call turns, so the
    # shape (reasoning set, reasoning_content absent, tool_calls present)
    # is unreachable from same-provider DeepSeek history after this fix.
    # Inject a single space to satisfy the API without leaking another
    # provider's chain of thought to DeepSeek/Kimi. Space (not "")
    # because DeepSeek V4 Pro rejects empty-string reasoning_content
    # in thinking mode (refs #17341).
    normalized_reasoning = source_msg.get("reasoning")
    if (
        needs_thinking_pad
        and source_msg.get("tool_calls")
        and isinstance(normalized_reasoning, str)
        and normalized_reasoning
    ):
        api_msg["reasoning_content"] = " "
        return

    # 3. Healthy session: promote 'reasoning' field to 'reasoning_content'
    # for providers that use the internal 'reasoning' key.
    # This must happen before the unconditional empty-string fallback so
    # genuine reasoning content is not overwritten (#15812 regression in
    # PR #15478).
    if isinstance(normalized_reasoning, str) and normalized_reasoning:
        api_msg["reasoning_content"] = normalized_reasoning
        return

    # 4. DeepSeek / Kimi thinking mode: all assistant messages need
    # reasoning_content. Inject a single space to satisfy the provider's
    # requirement when no explicit reasoning content is present. Covers
    # both tool-call turns (already-poisoned history with no reasoning
    # at all) and plain text turns. Space (not "") because DeepSeek V4
    # Pro tightened validation and rejects empty string with HTTP 400
    # ("The reasoning content in the thinking mode must be passed back
    # to the API"). Refs #17341.
    if needs_thinking_pad:
        api_msg["reasoning_content"] = " "
        return

    # 5. reasoning_content was present but not a string (e.g. None after
    # context compaction).  Don't pass null to the API.
    api_msg.pop("reasoning_content", None)



def cleanup_dead_connections(agent) -> bool:
    """Detect and clean up dead TCP connections on the primary client.

    Inspects the httpx connection pool for sockets in unhealthy states
    (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
    and rebuilds the primary client from scratch.

    Returns True if dead connections were found and cleaned up.
    """
    client = getattr(agent, "client", None)
    if client is None:
        return False
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            return False
        transport = getattr(http_client, "_transport", None)
        if transport is None:
            return False
        pool = getattr(transport, "_pool", None)
        if pool is None:
            return False
        connections = (
            getattr(pool, "_connections", None)
            or getattr(pool, "_pool", None)
            or []
        )
        dead_count = 0
        for conn in list(connections):
            # Check for connections that are idle but have closed sockets
            stream = (
                getattr(conn, "_network_stream", None)
                or getattr(conn, "_stream", None)
            )
            if stream is None:
                continue
            sock = getattr(stream, "_sock", None)
            if sock is None:
                sock = getattr(stream, "stream", None)
                if sock is not None:
                    sock = getattr(sock, "_sock", None)
            if sock is None:
                continue
            # Probe socket health with a non-blocking recv peek
            import socket as _socket
            try:
                sock.setblocking(False)
                data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                if data == b"":
                    dead_count += 1
            except BlockingIOError:
                pass  # No data available — socket is healthy
            except OSError:
                dead_count += 1
            finally:
                try:
                    sock.setblocking(True)
                except OSError:
                    pass
        if dead_count > 0:
            _ra().logger.warning(
                "Found %d dead connection(s) in client pool — rebuilding client",
                dead_count,
            )
            agent._replace_primary_openai_client(reason="dead_connection_cleanup")
            return True
    except Exception as exc:
        _ra().logger.debug("Dead connection check error: %s", exc)
    return False



def extract_api_error_context(error: Exception) -> Dict[str, Any]:
    """Extract structured rate-limit details from provider errors."""
    context: Dict[str, Any] = {}

    body = getattr(error, "body", None)
    payload = None
    if isinstance(body, dict):
        payload = body.get("error") if isinstance(body.get("error"), dict) else body
    if isinstance(payload, dict):
        reason = payload.get("code") or payload.get("type") or payload.get("error")
        if isinstance(reason, str) and reason.strip():
            context["reason"] = reason.strip()
        message = payload.get("message") or payload.get("error_description")
        if isinstance(message, str) and message.strip():
            context["message"] = message.strip()
        for key in ("resets_at", "reset_at"):
            value = payload.get(key)
            if value not in {None, ""}:
                context["reset_at"] = value
                break
        retry_after = payload.get("retry_after")
        if retry_after not in {None, ""} and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass
        ratelimit_reset = headers.get("x-ratelimit-reset")
        if ratelimit_reset and "reset_at" not in context:
            context["reset_at"] = ratelimit_reset

    if "message" not in context:
        raw_message = str(error).strip()
        if raw_message:
            context["message"] = raw_message[:500]

    if "reset_at" not in context:
        message = context.get("message") or ""
        if isinstance(message, str):
            delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
            if delay_match:
                value = float(delay_match.group(1))
                seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                context["reset_at"] = time.time() + seconds
            else:
                sec_match = re.search(
                    r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                    message,
                    re.IGNORECASE,
                )
                if sec_match:
                    context["reset_at"] = time.time() + float(sec_match.group(1))

    return context



def apply_pending_steer_to_tool_results(agent, messages: list, num_tool_msgs: int) -> None:
    """Append any pending /steer text to the last tool result in this turn.

    Called at the end of a tool-call batch, before the next API call.
    The steer is appended to the last ``role:"tool"`` message's content
    with a clear marker so the model understands it came from the user
    and NOT from the tool itself. Role alternation is preserved —
    nothing new is inserted, we only modify existing content.

    Args:
        messages: The running messages list.
        num_tool_msgs: Number of tool results appended in this batch;
            used to locate the tail slice safely.
    """
    if num_tool_msgs <= 0 or not messages:
        return
    steer_text = agent._drain_pending_steer()
    if not steer_text:
        return
    # Find the last tool-role message in the recent tail. Skipping
    # non-tool messages defends against future code appending
    # something else at the boundary.
    target_idx = None
    for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
        msg = messages[j]
        if isinstance(msg, dict) and msg.get("role") == "tool":
            target_idx = j
            break
    if target_idx is None:
        # No tool result in this batch (e.g. all skipped by interrupt);
        # put the steer back so the caller's fallback path can deliver
        # it as a normal next-turn user message.
        _lock = getattr(agent, "_pending_steer_lock", None)
        if _lock is not None:
            with _lock:
                if agent._pending_steer:
                    agent._pending_steer = agent._pending_steer + "\n" + steer_text
                else:
                    agent._pending_steer = steer_text
        else:
            existing = getattr(agent, "_pending_steer", None)
            agent._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
        return
    marker = f"\n\nUser guidance: {steer_text}"
    existing_content = messages[target_idx].get("content", "")
    if not isinstance(existing_content, str):
        # Anthropic multimodal content blocks — preserve them and append
        # a text block at the end.
        try:
            blocks = list(existing_content) if existing_content else []
            blocks.append({"type": "text", "text": marker.lstrip()})
            messages[target_idx]["content"] = blocks
        except Exception:
            # Fall back to string replacement if content shape is unexpected.
            messages[target_idx]["content"] = f"{existing_content}{marker}"
    else:
        messages[target_idx]["content"] = existing_content + marker
    _ra().logger.info(
        "Delivered /steer to agent after tool batch (%d chars): %s",
        len(steer_text),
        steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
    )



def force_close_tcp_sockets(client: Any) -> int:
    """Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation.

    When a provider drops a connection mid-stream, httpx's ``client.close()``
    performs a graceful shutdown which leaves sockets in CLOSE-WAIT until the
    OS times them out (often minutes).  This method walks the httpx transport
    pool and issues ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` to
    force an immediate TCP RST, freeing the file descriptors.

    Returns the number of sockets force-closed.
    """
    import socket as _socket

    closed = 0
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            return 0
        transport = getattr(http_client, "_transport", None)
        if transport is None:
            return 0
        pool = getattr(transport, "_pool", None)
        if pool is None:
            return 0
        # httpx uses httpcore connection pools; connections live in
        # _connections (list) or _pool (list) depending on version.
        connections = (
            getattr(pool, "_connections", None)
            or getattr(pool, "_pool", None)
            or []
        )
        for conn in list(connections):
            stream = (
                getattr(conn, "_network_stream", None)
                or getattr(conn, "_stream", None)
            )
            if stream is None:
                continue
            sock = getattr(stream, "_sock", None)
            if sock is None:
                sock = getattr(stream, "stream", None)
                if sock is not None:
                    sock = getattr(sock, "_sock", None)
            if sock is None:
                continue
            try:
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
            closed += 1
    except Exception as exc:
        _ra().logger.debug("Force-close TCP sockets sweep error: %s", exc)
    return closed



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
    "switch_model",
    "invoke_tool",
    "repair_tool_call",
    "sanitize_api_messages",
    "looks_like_codex_intermediate_ack",
    "copy_reasoning_content_for_api",
    "cleanup_dead_connections",
    "extract_api_error_context",
    "apply_pending_steer_to_tool_results",
    "force_close_tcp_sockets",
]
