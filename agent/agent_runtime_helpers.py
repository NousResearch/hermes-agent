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
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.timeouts import get_provider_request_timeout
from agent.message_sanitization import (
    _FULL_ARGS_LOG_BOUND,
    coalesce_tool_call_id,
    tool_call_id_variants,
    tool_result_id_variants,
)
from agent.prompt_builder import format_steer_marker
from agent.tool_dispatch_helpers import _trajectory_normalize_msg, make_tool_result_message
from agent.trajectory import convert_scratchpad_to_think
from agent.credential_pool import (
    STATUS_EXHAUSTED,
    credential_pool_matches_provider,
    resolve_runtime_pool_key,
)
from agent.error_classifier import FailoverReason
from agent.turn_context import drop_stale_api_content
from utils import base_url_host_matches, base_url_hostname, env_var_enabled, atomic_json_write

logger = logging.getLogger(__name__)


# Max consecutive successful credential-pool token refreshes of the SAME entry
# on a persistent auth failure before we give up and let the fallback chain
# activate. A single-entry OAuth pool can re-mint a fresh token indefinitely
# even when the upstream keeps rejecting it, so without this cap the retry loop
# spins forever and never reaches ``_try_activate_fallback``. See #26080.
_MAX_AUTH_REFRESH_ATTEMPTS = 2


_REASONING_TAG_NAMES = ("think", "thinking", "reasoning", "REASONING_SCRATCHPAD", "thought")
_TOOL_CALL_TAG_NAMES = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")

_REASONING_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _REASONING_TAG_NAMES
)

_TOOL_CALL_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
)

# Named <function name=...> blocks — see strip_think_blocks step 1c for the
# full rationale (sentence-boundary lookbehind + tempered-dot body so a plain
# prose mention of "function" is never eaten).
_NAMED_FUNCTION_BLOCK_PATTERN = re.compile(
    r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
    r'<function\b[^>]*\bname\s*=[^>]*>'
    r'(?:(?:(?!</function>).)*)</function>',
    re.DOTALL | re.IGNORECASE,
)

_UNTERMINATED_REASONING_BLOCK_PATTERN = re.compile(
    rf'(?:^|\n)[ \t]*<(?:{"|".join(_REASONING_TAG_NAMES)})\b[^>]*>.*$',
    re.DOTALL | re.IGNORECASE,
)

_ORPHAN_REASONING_TAG_PATTERN = re.compile(
    rf'</?(?:{"|".join(_REASONING_TAG_NAMES)})>\s*',
    re.IGNORECASE,
)

_STRAY_TOOL_CALL_CLOSER_PATTERN = re.compile(
    rf'</(?:{"|".join(_TOOL_CALL_TAG_NAMES)}|function)>\s*',
    re.IGNORECASE,
)


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent
    return run_agent


AGENT_RUNTIME_POST_HOOK_TOOL_NAMES = frozenset(
    {"todo", "session_search", "memory", "clarify", "read_terminal", "desktop_preview", "drive_preview", "annotate_preview", "read_window_below", "setup_mcp", "tour", "delegate_task"}
)


def agent_runtime_owns_post_tool_hook(agent: Any, function_name: str) -> bool:
    """Return True when an agent-level tool path emits its own post hook."""
    if function_name in AGENT_RUNTIME_POST_HOOK_TOOL_NAMES:
        return True
    if getattr(agent, "_context_engine_tool_names", None) and function_name in agent._context_engine_tool_names:
        return True
    memory_manager = getattr(agent, "_memory_manager", None)
    return bool(memory_manager and memory_manager.has_tool(function_name))


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
                        logger.warning("Unexpected invalid JSON in trajectory conversion: %s", tool_call['function']['arguments'][:100])
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
    cursor: Optional[dict] = None,
) -> int:
    """Repair corrupted assistant tool-call argument JSON in-place.

    ``cursor`` (optional) is a caller-owned dict used to skip re-validating
    messages already validated on a previous call.  It stores, under
    ``"prefix"``, the exact message *objects* (strong references) validated
    last time, in order.  On the next call, the longest contiguous prefix of
    ``messages`` whose objects are ``is``-identical to the stored prefix is
    skipped; scanning starts at the first divergence (conservative: any
    reordering, truncation, compression rewrite, or mid-list insertion breaks
    identity at that index and everything from there is re-scanned).

    Safety argument for skipping: a message in the matched prefix was fully
    scanned before — every tool_call argument was either already valid JSON
    or was rewritten to ``"{}"`` (valid).  The only code paths that mutate
    ``function["arguments"]`` on live history dicts between calls are the
    surrogate / non-ASCII sanitizers, which substitute characters *inside*
    JSON string values and cannot invalidate JSON syntax.  Compression,
    repair, undo, and steer paths replace or reorder message dicts, which
    breaks the identity match and forces a re-scan.  Holding strong
    references (the objects themselves, not ``id()``s) makes address reuse
    aliasing (#50372-style) impossible.
    """
    log = logger or logging.getLogger(__name__)
    if not isinstance(messages, list):
        return 0

    start_index = 0
    if cursor is not None:
        prev_prefix = cursor.get("prefix")
        if isinstance(prev_prefix, list):
            limit = min(len(prev_prefix), len(messages))
            while start_index < limit and messages[start_index] is prev_prefix[start_index]:
                start_index += 1

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

    message_index = start_index
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
                # Use the canonical ``call_id || id`` precedence so both the
                # scan for an existing tool result and any inserted stub key
                # on the same id the rest of the pipeline uses. Keying on bare
                # ``id`` here would fail to find a result built with ``call_id``
                # (Codex Responses format) and insert a duplicate stub that
                # itself becomes an orphan (#58168).
                tool_call_id = _ra().AIAgent._get_tool_call_id_static(tool_call) or None
                function_name = function.get("name", "?")
                # Log the FULL original argument string (bounded), not an
                # 80-char preview: this branch is about to overwrite the
                # only copy of these bytes in the transcript with "{}", and
                # for a truncated write_file/patch call the destroyed
                # arguments contain real user content (#80498 — streamed
                # file content survived only as a log preview). A corrupted
                # call is rare, so the oversized WARNING is a fair price for
                # making the data recoverable from agent.log.
                preview = arguments[:_FULL_ARGS_LOG_BOUND]
                log.warning(
                    "Corrupted tool_call arguments repaired before request "
                    "(session=%s, message_index=%s, tool_call_id=%s, function=%s, "
                    "original_arguments=%r)",
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
                    if (
                        tool_result_id_variants(candidate.get("tool_call_id"))
                        & tool_call_id_variants(tool_call)
                    ):
                        existing_tool_msg = candidate
                        break
                    scan_index += 1

                if existing_tool_msg is None:
                    messages.insert(
                        insert_at,
                        make_tool_result_message(
                            function_name if function_name != "?" else "",
                            marker,
                            tool_call_id,
                        ),
                    )
                    insert_at += 1
                else:
                    _prepend_marker(existing_tool_msg)

                repaired += 1

        message_index += 1

    if cursor is not None:
        # Strong references to the exact objects validated this call, in
        # order. Any future divergence (compression, undo, repair, steer)
        # breaks identity at the divergent index and re-scans from there.
        cursor["prefix"] = messages[:]

    return repaired


# Session-scoped in-flight registry backing note_turn_start's cross-agent
# check.  The per-agent marker catches a second turn on the SAME AIAgent
# object, but the gateway caches agents per *routing key* (``_agent_cache``
# in gateway/run.py) while the durable transcript is keyed by *session_id* —
# and the key→id mapping is many-to-one (``switch_session``: /resume from a
# second chat/topic, CLI-continuity rebinding, async-delegation pinning,
# topic-binding tip-walks).  Two routing keys mapped to one session_id run
# concurrent turns on two different agent objects, which per-agent state can
# never see (#64934).  Keyed by session_id so that route produces the same
# named warning.  Process-local by design — same visibility scope as the
# per-agent marker it extends.
_INFLIGHT_TURNS_BY_SESSION: Dict[str, Tuple[str, float]] = {}
_INFLIGHT_TURNS_LOCK = threading.Lock()


def note_turn_start(agent, turn_id: str):
    """Tripwire: detect a turn starting while a previous turn of the same
    agent — or of the same underlying *session* on a different agent object —
    has not completed its turn-end persist.

    Two turns interleaving on one session corrupt the durable transcript:
    their flushes race (user rows can persist out of arrival order), a row
    can be swallowed by the identity-marker dedup over shared history dicts,
    and the second turn runs on a history base that never saw the first
    turn's exchange. This helper does NOT prevent any of that — it names the
    occurrence, with both turn ids, so the dispatch route that let the
    second turn through the busy guard can be identified from logs.

    Returns the previous in-flight turn_id when an overlap is detected,
    else None. Takes ownership of the in-flight slot either way, so a turn
    that crashed before its persist produces at most one warning."""
    prev = getattr(agent, "_inflight_turn_id", None)
    prev_started = getattr(agent, "_inflight_turn_started", 0.0)
    agent._inflight_turn_id = turn_id
    agent._inflight_turn_started = time.time()
    overlap = None
    if prev and prev != turn_id:
        logger.warning(
            "turn %s starting while turn %s (started %.0fs ago) has not "
            "completed its turn-end persist (session=%s) — concurrent turns "
            "on one session; transcript writes may interleave",
            turn_id,
            prev,
            time.time() - prev_started if prev_started else -1.0,
            getattr(agent, "session_id", None) or "-",
        )
        overlap = prev

    # Cross-agent leg: same session_id in flight under a different agent
    # object means two routing keys resolve to one durable session — the
    # busy guard (keyed by routing key) cannot see this overlap at all.
    # Persist-disabled agents (background-review forks) deliberately share
    # the live parent's session_id for prompt-cache warmth but can never
    # write to the transcript — they must not register here (would warn a
    # false overlap against the parent's real turn) nor pop the parent's
    # slot at their persist (note_turn_persisted skips them symmetrically).
    session_id = getattr(agent, "session_id", None)
    if session_id and not getattr(agent, "_persist_disabled", False):
        now = time.time()
        with _INFLIGHT_TURNS_LOCK:
            entry = _INFLIGHT_TURNS_BY_SESSION.get(session_id)
            _INFLIGHT_TURNS_BY_SESSION[session_id] = (turn_id, now)
        # Stamp the session id this turn registered under: compression can
        # rotate agent.session_id mid-turn, and the persist-time clear must
        # pop the slot the turn actually holds, not the rotated id.
        agent._inflight_turn_session_id = session_id
        if entry and entry[0] not in (turn_id, prev):
            logger.warning(
                "turn %s starting while turn %s (started %.0fs ago) is still "
                "in flight on session %s under a different agent object — "
                "two routing keys are mapped to one session_id; concurrent "
                "turns on one session; transcript writes may interleave",
                turn_id,
                entry[0],
                now - entry[1] if entry[1] else -1.0,
                session_id,
            )
            overlap = overlap or entry[0]
    return overlap


def note_turn_persisted(agent):
    """Clear the in-flight marker at turn-end persist (see note_turn_start).

    Called from the single persist funnel; unconditional by design — when two
    turns genuinely overlap, the first persist clears the second turn's slot
    and the tripwire under-reports instead of double-reporting. A diagnostic
    must never be noisier than the defect it hunts."""
    agent._inflight_turn_id = None
    # Symmetric with note_turn_start's cross-agent leg: persist-disabled
    # forks never registered a session slot, and their persist funnel still
    # runs — popping here would steal the live parent turn's slot and make
    # the tripwire under-report the real overlap it exists to catch.
    if not getattr(agent, "_persist_disabled", False):
        session_id = getattr(agent, "_inflight_turn_session_id", None) or getattr(
            agent, "session_id", None
        )
        if session_id:
            with _INFLIGHT_TURNS_LOCK:
                _INFLIGHT_TURNS_BY_SESSION.pop(session_id, None)
    agent._inflight_turn_session_id = None


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
      0. Consecutive ``assistant`` messages with no intervening
         ``tool``/``user`` turn — merged into a single assistant turn
         (union of ``tool_calls``, concatenated ``content``). Strict
         OpenAI-compatible providers (DeepSeek v4, Moonshot/Kimi) reject
         a history where an ``assistant`` message carrying ``tool_calls``
         is immediately followed by another ``assistant`` message instead
         of its ``tool`` results — HTTP 400 "An assistant message with
         'tool_calls' must be followed by tool messages…". The split
         shape is produced by recovery/continuation paths that append an
         interim assistant turn (thinking-prefill, codex
         incomplete-continuation) or by host-fed / legacy-persisted /
         resumed histories. Refs #29148, #49147.
      1. Stray ``tool`` messages whose ``tool_call_id`` doesn't match
         any preceding assistant tool_call — dropped.
      2. ``tool_calls`` on an assistant message that no immediately
         following ``tool`` result answers are pruned — and the turn is
         dropped entirely if that leaves it payload-empty (an empty
         non-final assistant message is itself a 400 on most providers).
      3. Consecutive ``user`` messages — merged with newline separator
         so no user input is lost.

    Deliberately does NOT rewind orphan ``assistant(tool_calls)+tool``
    pairs that precede a user message — that pattern IS valid when the
    previous turn completed normally and the user jumped in to redirect
    before the model got a continuation turn (the ongoing dialog
    pattern). The empty-response scaffolding stripper handles the
    genuinely-broken variant via its flag-gated rewind.

    Pass 2 (prune unanswered ``tool_calls``) answers the complement of
    Pass 1: Pass 1 removes the stray result, Pass 2 removes the orphaned
    call it was displaced from. Context compression can move a tool
    result past a user turn; without this pass the declaring assistant
    message would keep replaying an unanswered ``tool_call`` and strict
    providers (DeepSeek v4) reject that with HTTP 400 "An assistant
    message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'". A call counts as answered when
    the run of ``tool`` messages immediately following its assistant
    message contains a result keyed to ANY of the call's ids (``id`` or
    ``call_id`` — the same superset rule Pass 1 registers). Codex
    interim turns are exempt, as in Pass 0.

    Returns the number of repairs made (for logging/telemetry).
    """
    if not messages:
        return 0

    repairs = 0

    # Pass 0: merge consecutive assistant messages. Runs BEFORE Pass 1 so
    # the merged turn's union of tool_call ids is known when Pass 1
    # validates which tool-result messages are orphans. Two assistant
    # messages are only adjacent here when nothing (no tool result, no
    # user turn) separates them — an intervening ``tool`` message means
    # two distinct, valid tool-call rounds that must NOT be merged.
    #
    # Codex Responses interim turns are exempt: the codex_responses
    # api_mode legitimately keeps multiple consecutive incomplete
    # assistant turns in history, each carrying its own encrypted
    # continuation state (codex_reasoning_items / codex_message_items)
    # that must be replayed verbatim. Collapsing them corrupts the
    # Responses replay chain (the duplicate-detection logic at
    # conversation_loop.py already de-dups identical codex interims).
    def _is_codex_interim(m: Dict) -> bool:
        return bool(
            m.get("codex_reasoning_items")
            or m.get("codex_message_items")
            or m.get("finish_reason") == "incomplete"
        )

    def _is_verification_candidate(m: Dict) -> bool:
        return m.get("finish_reason") in {
            "verification_required",
            "verify_hook_continue",
        }

    collapsed: List[Dict] = []
    for msg in messages:
        if (
            collapsed
            and isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and isinstance(collapsed[-1], dict)
            and collapsed[-1].get("role") == "assistant"
            and not _is_codex_interim(msg)
            and not _is_codex_interim(collapsed[-1])
        ):
            prev = collapsed[-1]
            # Verification candidate collapsing: when the earlier assistant
            # message is a provisional candidate (finish_reason =
            # verification_required / verify_hook_continue), the later
            # response supersedes it for model replay — replace rather than
            # union. Both remain durable in state.db; this only affects the
            # in-memory sequence sent to the model. (#65919 §7)
            if _is_verification_candidate(prev):
                collapsed[-1] = msg
                repairs += 1
                continue
            # Union tool_calls (preserve order, both may carry them).
            prev_calls = list(prev.get("tool_calls") or [])
            new_calls = list(msg.get("tool_calls") or [])
            if new_calls:
                prev["tool_calls"] = prev_calls + new_calls
            elif prev_calls:
                prev["tool_calls"] = prev_calls
            else:
                # Neither turn carries tool calls, but the surviving turn may
                # still carry a stale ``tool_calls: []`` from the earlier
                # message.  An empty array is semantically "no tool calls",
                # yet strict OpenAI-compatible providers (DeepSeek v4,
                # Moonshot/Kimi) reject it with HTTP 400 ("Invalid
                # 'messages[N].tool_calls': empty array...").  Drop the key
                # HERE, at the source: ``sanitize_api_messages`` only fixes
                # the per-call wire copy, so a ``[]`` left on the repaired
                # turn survives in the live/persisted trajectory returned to
                # callers (gateway/WebUI transcripts, session resume,
                # subagents, cron) and is replayed on the next turn — which
                # is how #58755 kept reproducing after the chokepoint fix
                # (#77921).  Popping is non-destructive: an empty array
                # carries no information.
                prev.pop("tool_calls", None)
            # Concatenate plain-text content; leave multimodal (list)
            # content on either side alone to avoid mangling attachment
            # blocks — fall back to keeping the existing content.
            prev_content = prev.get("content")
            new_content = msg.get("content")
            content_rewritten = False
            if isinstance(prev_content, str) and isinstance(new_content, str):
                joined = "\n".join(
                    p for p in (prev_content.strip(), new_content.strip()) if p
                )
                prev["content"] = joined
                # A falsy ``new_content`` (e.g. "") strips to nothing and
                # ``joined`` collapses back to ``prev_content`` unchanged --
                # that must NOT count as a rewrite (wz-heng, #78063 review).
                content_rewritten = joined != prev_content
            elif not prev_content and new_content is not None:
                prev["content"] = new_content
                content_rewritten = new_content != prev_content
            # Carry reasoning_content from the later turn only if the
            # earlier turn lacks it (strict thinking providers require a
            # reasoning_content on the merged tool-call turn; the first
            # non-empty one suffices).
            if not prev.get("reasoning_content") and msg.get("reasoning_content"):
                prev["reasoning_content"] = msg["reasoning_content"]
            # ``prev`` may carry an ``api_content`` sidecar (the exact bytes
            # previously sent to the API, e.g. a sanitize-divergence stamp —
            # see ``_flush_messages_to_session_db``) from BEFORE this merge.
            # The sidecar takes priority over ``content`` at API-build time
            # (``conversation_loop``'s ``api_messages`` build substitutes it
            # back in for role ``assistant``), so leaving it in place while
            # ``prev["content"]`` changes would silently replay the pre-merge
            # bytes and discard everything this merge just concatenated on —
            # the same stale-field-survives-the-merge shape as the
            # ``tool_calls`` gap above, just for a different field. Only drop
            # it when the merge actually changed the resulting value (e.g.
            # the later turn's content is ``None``, or either side is
            # multimodal/list — both branches skip the reassignment and
            # ``prev["content"]`` is untouched; a falsy ``new_content`` that
            # strips to nothing also leaves ``joined`` equal to the original
            # ``prev_content``): in those cases the sidecar is still the
            # exact bytes previously sent for the UNCHANGED content, and
            # dropping it would break the prompt-cache replay invariant for
            # no reason (wz-heng, #78063 review).
            if content_rewritten:
                drop_stale_api_content(prev)
            repairs += 1
            continue
        collapsed.append(msg)

    # Pass 1: drop stray tool messages that don't follow a known assistant
    # tool call. A Responses call can have several equivalent spellings
    # (call_id, id, response_item_id, or a composite ``call|item`` id), so
    # consume the whole alias group when one spelling is matched. Otherwise a
    # duplicate result keyed on the sibling alias would survive and be replayed
    # to strict providers (#66974). Alias expansion lives in
    # ``agent.message_sanitization.tool_call_id_variants`` /
    # ``tool_result_id_variants`` (single policy owner) — which also handles
    # SDK tool_call objects, preserving the #91768 dict-or-object tolerance.
    known_tool_ids: Dict[str, int] = {}
    matched_tool_groups: set = set()
    next_tool_group = 0
    filtered: List[Dict] = []
    for msg in collapsed:
        if not isinstance(msg, dict):
            filtered.append(msg)
            continue
        role = msg.get("role")
        if role == "assistant":
            known_tool_ids = {}
            matched_tool_groups = set()
            for tc in (msg.get("tool_calls") or []):
                variants = tool_call_id_variants(tc)
                if not variants:
                    continue
                group_id = next_tool_group
                next_tool_group += 1
                for tc_id in variants:
                    known_tool_ids.setdefault(tc_id, group_id)
            filtered.append(msg)
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            candidate_groups = {
                known_tool_ids[tc_id]
                for tc_id in result_variants
                if tc_id in known_tool_ids
                and known_tool_ids[tc_id] not in matched_tool_groups
            }
            if not result_variants:
                filtered.append(msg)
            elif candidate_groups:
                # Consume the whole alias group so a SECOND result replaying
                # any sibling spelling falls into the drop branch below —
                # strict providers reject duplicate tool_call_ids with HTTP
                # 400 (#58327, #66974). Credit: #55436.
                group_id = min(candidate_groups)
                filtered.append(msg)
                matched_tool_groups.add(group_id)
            else:
                repairs += 1
        else:
            if role == "user":
                # A user turn closes the tool-result run; subsequent
                # tool messages without a fresh assistant tool_call
                # are orphans.
                known_tool_ids = {}
                matched_tool_groups = set()
            filtered.append(msg)

    # Pass 2: prune tool_calls that were never answered positionally.
    #
    # Pass 1 dropped the stray/displaced tool RESULT — but a tool_call
    # whose result was displaced far beyond the following turn (context
    # compression can move it past a user turn) leaves its declaring
    # assistant message carrying an UNANSWERED tool_call, and strict
    # OpenAI-compatible providers (DeepSeek v4) reject the payload with
    # HTTP 400 "An assistant message with 'tool_calls' must be followed
    # by tool messages responding to each 'tool_call_id' (insufficient
    # tool messages following tool_calls message)". The per-call
    # sanitizer's stub pass is keyed on GLOBAL id presence, which a
    # displaced-but-present result masks (see sanitize_api_messages) —
    # so the durable history must not keep replaying the poisoned turn
    # either. Enforce the positional invariant here: a tool_call is only
    # legitimate when a result for ANY of its id variants (``id`` /
    # ``call_id`` / ``response_item_id`` / composite bridge — the same
    # unified alias policy as Pass 1, via ``tool_call_id_variants`` /
    # ``tool_result_id_variants``) appears in the run of tool messages
    # IMMEDIATELY following the declaring assistant message — before any
    # user turn or further assistant turn. Unanswered calls are pruned;
    # if the message then carries no other payload (no content,
    # reasoning, codex items), the whole turn is dropped — an empty
    # non-final assistant message is itself rejected by providers.
    # Codex interim turns are exempt, as in Pass 0: their calls are
    # replayed through the Responses-items chain, not the tool-result
    # run.
    pruned: List[Dict] = []
    i = 0
    n = len(filtered)
    while i < n:
        msg = filtered[i]
        if not (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and not _is_codex_interim(msg)
        ):
            pruned.append(msg)
            i += 1
            continue
        answered: set = set()
        j = i + 1
        while (
            j < n
            and isinstance(filtered[j], dict)
            and filtered[j].get("role") == "tool"
        ):
            tid = (filtered[j].get("tool_call_id") or "").strip()
            if tid:
                answered.update(tool_result_id_variants(tid))
            j += 1
        kept_calls: List[Dict] = []
        dropped_calls = 0
        for tc in msg.get("tool_calls") or []:
            variants = tool_call_id_variants(tc)
            if variants and (variants & answered):
                kept_calls.append(tc)
            else:
                dropped_calls += 1
        if dropped_calls:
            repairs += 1
            if not kept_calls and not _msg_has_payload(
                {k: v for k, v in msg.items() if k != "tool_calls"}
            ):
                # The pruned call(s) were the message's only payload —
                # dropping the whole turn beats sending an empty
                # assistant message (which most providers 400).
                i += 1
                continue
            if kept_calls:
                msg["tool_calls"] = kept_calls
            else:
                msg.pop("tool_calls", None)
        pruned.append(msg)
        i += 1

    # Pass 3: merge consecutive user messages. Preserves all user input
    # so nothing the user typed is lost.
    merged: List[Dict] = []
    for msg in pruned:
        if (
            merged
            and isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(merged[-1], dict)
            and merged[-1].get("role") == "user"
        ):
            prev = merged[-1]
            # A summary carrier followed by a new user row is a deliberate
            # durable shape after retry/rewind.  Do not absorb the fresh ask
            # into the already-persisted carrier: mutating that dict can make
            # the only in-memory copy diverge from its durable row.  Provider
            # sanitizers merge copies later when strict alternation requires
            # it, without rewriting either durable message.
            from agent.context_compressor import split_user_originated_turn

            handoff, _ = split_user_originated_turn(prev)
            if handoff is not None:
                merged.append(msg)
                continue

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
                # Merged content invalidates the api_content sidecar (exact
                # bytes previously sent for the pre-merge message) — drop it
                # so replay can't substitute stale bytes.
                drop_stale_api_content(prev)
                repairs += 1
                continue
        merged.append(msg)

    if repairs > 0:
        # Rewrite in place so downstream paths (persistence, return
        # value, session DB flush) see the repaired sequence.
        messages[:] = merged

    return repairs


def repair_message_sequence_with_cursor(agent, messages: List[Dict]) -> int:
    """Run :func:`repair_message_sequence` and keep the SessionDB flush
    cursor consistent with the compacted list (#44837).

    ``repair_message_sequence`` merges/drops messages in place, shrinking
    the list. ``_last_flushed_db_idx`` (the DB-write cursor) indexes into
    that list, so after compaction it can point past the new end — the
    turn-end flush would then skip the assistant/tool chain entirely — or
    past unflushed messages shifted to lower indexes.

    Repair preserves object identity for surviving messages, so counting
    the survivors from the previously-flushed prefix gives the exact new
    cursor even when messages are dropped/merged at indexes *before* the
    cursor — a plain ``min()`` clamp would silently skip that many
    unflushed rows. Falls back to the clamp when no prefix snapshot is
    available.

    Returns the number of repairs made (same as ``repair_message_sequence``).
    """
    pre_repair_flushed_ids = None
    flush_cursor = getattr(agent, "_last_flushed_db_idx", None)
    if isinstance(flush_cursor, int) and flush_cursor > 0:
        pre_repair_flushed_ids = {id(m) for m in messages[:flush_cursor]}

    repairs = repair_message_sequence(agent, messages)

    if repairs > 0 and hasattr(agent, "_last_flushed_db_idx"):
        if pre_repair_flushed_ids is not None:
            agent._last_flushed_db_idx = sum(
                1 for m in messages if id(m) in pre_repair_flushed_ids
            )
        else:
            agent._last_flushed_db_idx = min(
                agent._last_flushed_db_idx, len(messages)
            )

    return repairs



def strip_think_blocks(agent, content: str) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text.

    Handles four cases:
      1. Closed tag pairs (`` <think>… ``) — the common path when
         the provider emits complete reasoning blocks.
      2. Unterminated open tag at a block boundary (start of text or
         after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
         closing tag is dropped.  Everything from the open tag to end
         of string is stripped.  The block-boundary check mirrors
         ``gateway/stream_consumer.py``'s filter so models that mention
         `` <think>`` in prose aren't over-stripped.
      3. Stray orphan open/close tags that slip through.
      4. Tag variants: `` <think>``, ``<thinking>``, ``<reasoning>``,
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
    # Coerce non-string content to text before any regex runs.  Providers
    # that return assistant ``content`` as a list of blocks (Anthropic via
    # OpenRouter emits ``[{"type":"text",...}, {"type":"thinking",...}]``) or
    # as a dict flow into this shared helper from several callers — most
    # notably ``_interim_assistant_visible_text`` reading a *stored* history
    # message whose content was persisted as a list.  A raw list/dict reaching
    # ``re.sub`` below raises ``TypeError: expected string or bytes-like
    # object, got 'list'``, which the outer conversation loop swallows and
    # retries forever (observed as an infinite "preparing terminal…" loop on
    # Anthropic models via OpenRouter).  Flatten here so every caller is safe.
    if not isinstance(content, str):
        if isinstance(content, list):
            _parts: list[str] = []
            for _part in content:
                if isinstance(_part, str):
                    _parts.append(_part)
                elif isinstance(_part, dict):
                    _ptype = str(_part.get("type") or "").strip().lower()
                    # Drop reasoning/thinking blocks outright — this function's
                    # whole job is to strip them, and their text lives under
                    # different keys ("thinking", "reasoning") per provider.
                    if _ptype in {"thinking", "reasoning", "redacted_thinking"}:
                        continue
                    _text = _part.get("text")
                    if isinstance(_text, str) and _text:
                        _parts.append(_text)
            content = "".join(_parts)
        elif isinstance(content, dict):
            content = str(content.get("text") or content.get("content") or "")
        else:
            content = str(content)
        if not content:
            return ""
    # 1. Closed tag pairs — case-insensitive for all variants so
    #    mixed-case tags (<THINK>, <Thinking>) don't slip through to
    #    the unterminated-tag pass and take trailing content with them.
    for _pattern in _REASONING_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
    #     generic tag names first — they have no attribute gating since
    #     a literal <tool_call> in prose is already vanishingly rare.
    for _pattern in _TOOL_CALL_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1c. <function name="...">...</function> — Gemma-style standalone
    #     tool call. Only strip when the tag sits at a block boundary
    #     (start of text, after a newline, or after sentence-ending
    #     punctuation) AND carries a name="..." attribute. This keeps
    #     prose mentions like "Use <function> to declare" safe.
    content = _NAMED_FUNCTION_BLOCK_PATTERN.sub('', content)
    # 2. Unterminated reasoning block — open tag at a block boundary
    #    (start of text, or after a newline) with no matching close.
    #    Strip from the tag to end of string.  Fixes #8878 / #9568
    #    (MiniMax M2.7 leaking raw reasoning into assistant content).
    content = _UNTERMINATED_REASONING_BLOCK_PATTERN.sub('', content)
    # 3. Stray orphan open/close tags that slipped through.
    content = _ORPHAN_REASONING_TAG_PATTERN.sub('', content)
    # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
    #     unterminated <function name="..."> because a truncated tail
    #     during streaming may still be valuable to the user; matches
    #     OpenClaw's intentional asymmetry.)
    content = _STRAY_TOOL_CALL_CLOSER_PATTERN.sub('', content)
    return content



def sync_credential_pool_entry_id(agent) -> None:
    """Rebind ``agent._credential_pool_entry_id`` from the current pool + key.

    OAuth refreshes can replace the runtime token before a failed request is
    recovered, so the mutable API-key value alone cannot reliably attribute
    the failure to its source entry.  This resolves the stable pool-entry ID
    for the agent's current ``api_key`` and clears it when no pool is bound.
    """
    pool = getattr(agent, "_credential_pool", None)
    try:
        agent._credential_pool_entry_id = (
            pool.entry_id_for_api_key(getattr(agent, "api_key", None))
            if pool is not None
            else None
        )
    except Exception:
        agent._credential_pool_entry_id = None


def recover_with_credential_pool(
    agent,
    *,
    status_code: Optional[int],
    has_retried_429: bool,
    classified_reason: Optional[FailoverReason] = None,
    error_context: Optional[Dict[str, Any]] = None,
    billing_unverified: bool = False,
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

    `billing_unverified` marks a billing verdict that rests on an ambiguous
    body (``ClassifiedError.billing_unverified``, #82154): the pool persists
    it as ``billing_unverified`` so the exhausted entry gets a short cooldown
    instead of the one-hour billing bench — the same 400 can be a
    content-filter rejection that leaves the credential healthy.
    """
    pool = agent._credential_pool
    if pool is None:
        return False, has_retried_429

    # Defensive guard: if a fallback provider is active and its provider name
    # doesn't match the pool's provider, the pool belongs to the PRIMARY
    # provider.  Mutating it based on fallback errors would corrupt the
    # primary's credential state (see #33088) and, via _swap_credential,
    # overwrite the agent's base_url back to the primary's endpoint — every
    # subsequent request then goes to the wrong host and 404s (see #33163).
    # The pool should only act when the agent is still on the same provider
    # that seeded the pool.
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    pool_provider = (getattr(pool, "provider", "") or "").strip().lower()
    # Guard: skip credential pool recovery when the pool is scoped to a
    # different provider than the agent.  Only guard when the pool has a
    # known provider — an empty pool provider means "unscoped" (applies to
    # any provider).  An empty agent provider is treated as a mismatch
    # because swapping the pool's credentials would set base_url/api_key
    # without fixing the empty provider field, leaving the agent in a
    # corrupted state (provider="" model="").
    if pool_provider:
        # Use the same fail-closed boundary predicate as runtime binding. This
        # recognizes configured named-custom aliases, validates endpoints even
        # for exact custom:* identities, and preserves fallback isolation.
        if not credential_pool_matches_provider(
            pool,
            current_provider,
            base_url=getattr(agent, "base_url", None),
        ):
            _ra().logger.warning(
                "Credential pool provider mismatch: pool=%s, agent=%s — "
                "skipping pool mutation to avoid cross-provider contamination",
                pool_provider, current_provider,
            )
            return False, has_retried_429

    # Attribute the failure to the API key the agent actually dispatched the
    # request with, not to pool.current(). The current() pointer is shared,
    # mutable state — round-robin select() advances it on every call, and
    # concurrent turns or a second process (gateway/dashboard) reloading the
    # pool reset it to None — so by the time recovery runs it routinely points
    # at a DIFFERENT, healthy entry. Marking that entry exhausted copies this
    # request's error/reset time onto it and can take the whole pool offline
    # from a single rate-limited key (#43747). ``_swap_credential`` keeps
    # ``agent.api_key`` in sync with the entry in use, so it identifies the
    # failing entry exactly; fall back to current()'s key only when the agent
    # carries no key at all.
    _api_key_hint = getattr(agent, "api_key", None) or None
    _raw_credential_id = getattr(agent, "_credential_pool_entry_id", None)
    _credential_id = (
        _raw_credential_id
        if isinstance(_raw_credential_id, str) and _raw_credential_id
        else None
    )
    if not _api_key_hint:
        _cur = pool.current()
        if _cur:
            _api_key_hint = getattr(_cur, "runtime_api_key", None)
            if not _credential_id:
                _current_id = getattr(_cur, "id", None)
                if isinstance(_current_id, str) and _current_id:
                    _credential_id = _current_id

    def _rotate_failed_credential(rotate_status: int):
        kwargs = {
            "status_code": rotate_status,
            "error_context": error_context,
            "api_key_hint": _api_key_hint,
        }
        if _credential_id:
            kwargs["credential_id"] = _credential_id
        # Hand the pool the classified semantics, not just the status. A
        # billing 403 (OpenRouter "key limit exceeded", xAI spending limit)
        # and an edge-throttle 403 are the same number but need opposite
        # cooldowns — the pool can only tell them apart if we say which.
        # ``effective_reason`` is resolved below; this closure runs after.
        if effective_reason is not None:
            _failure_reason = effective_reason.value
            if effective_reason == FailoverReason.billing and billing_unverified:
                # Ambiguous billing body (#82154): persist the ambiguity so
                # the cooldown is sized as transient, not a 1-hour bench.
                from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
                _failure_reason = FAILURE_REASON_BILLING_UNVERIFIED
            kwargs["failure_reason"] = _failure_reason
        return pool.mark_exhausted_and_rotate(**kwargs)

    effective_reason = classified_reason
    if effective_reason is None:
        if status_code == 402:
            effective_reason = FailoverReason.billing
        elif status_code == 429:
            effective_reason = FailoverReason.rate_limit
        elif status_code in {401, 403}:
            effective_reason = FailoverReason.auth

    if effective_reason == FailoverReason.upstream_rate_limit:
        # An upstream provider (e.g. DeepSeek behind OpenRouter) is
        # rate-limiting the aggregator's traffic — the user's credential is
        # healthy. Do NOT rotate or mark exhausted; let the caller's fallback
        # path switch to a different model entirely.
        upstream = (error_context or {}).get("upstream_provider") if error_context else None
        if upstream:
            _ra().logger.info(
                "Upstream provider %s rate-limited via aggregator — skipping "
                "credential rotation, deferring to fallback chain",
                upstream,
            )
        else:
            _ra().logger.info(
                "Upstream aggregator 429 (provider unknown) — skipping "
                "credential rotation, deferring to fallback chain"
            )
        return False, has_retried_429

    if effective_reason == FailoverReason.billing:
        rotate_status = status_code if status_code is not None else 402
        # Runtime credentials can be resolved by a separate pool instance,
        # leaving this recovery pool without ``current_id``. Match the key
        # that actually failed instead of quarantining a different account.
        next_entry = _rotate_failed_credential(rotate_status)
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
        # If current credential is already marked exhausted, skip retry and
        # rotate immediately. This prevents the "cancel-between-429s" trap
        # where has_retried_429 (a local var) gets reset on each new prompt,
        # causing the pool to retry the same exhausted credential forever.
        # Prefer the entry matching the failing key over the shared current()
        # pointer, for the same attribution reason as above.
        current_entry = None
        if _credential_id:
            current_entry = next(
                (e for e in pool.entries() if e.id == _credential_id),
                None,
            )
        if _api_key_hint:
            current_entry = current_entry or next(
                (e for e in pool.entries() if e.runtime_api_key == _api_key_hint),
                None,
            )
        if current_entry is None:
            current_entry = pool.current()
        current_last_status = getattr(current_entry, "last_status", None) if current_entry else None
        if current_last_status == STATUS_EXHAUSTED:
            _ra().logger.info(
                "Credential already exhausted (last_status=%s) — rotating immediately instead of retrying",
                current_last_status,
            )
            rotate_status = status_code if status_code is not None else 429
            next_entry = _rotate_failed_credential(rotate_status)
            if next_entry is not None:
                _ra().logger.info(
                    "Credential %s (rate limit, pre-exhausted) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                agent._swap_credential(next_entry)
                return True, False
            return False, True

        usage_limit_reached = False
        if error_context:
            context_reason = str(error_context.get("reason") or "").lower()
            context_message = str(error_context.get("message") or "").lower()
            usage_limit_reached = (
                "usage_limit_reached" in context_reason
                or "gousagelimit" in context_reason
                or "usage limit reached" in context_message
                or "usage limit has been reached" in context_message
            )
        if not has_retried_429 and not usage_limit_reached:
            return False, True
        rotate_status = status_code if status_code is not None else 429
        next_entry = _rotate_failed_credential(rotate_status)
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
        # Subscription/entitlement 403s look like auth failures on the wire
        # but refresh cannot fix them — the OAuth token is already valid,
        # the account simply lacks the entitlement.  Without this guard,
        # the refresh path keeps minting fresh tokens against the
        # same unsubscribed account and the main agent loop spins re-issuing
        # the same 403 until the user Ctrl+C's.
        #
        # Defense-in-depth for #26847: xAI's backend has been seen to 403
        # standard SuperGrok subscribers with bodies that don't match the
        # existing entitlement keyword set in ``_is_entitlement_failure``.
        # Any 403 against ``xai-oauth`` is treated as entitlement here so
        # the refresh loop can't spin in those cases either.
        #
        # Exception (#29344): xAI's ``[WKE=unauthenticated:...]`` suffix and
        # the ``OAuth2 access token could not be validated`` phrasing are
        # xAI's authoritative "this is a stale token, not entitlement"
        # signal.  When either fires we must NOT apply the catch-all
        # override — refresh is the recoverable path for these bodies, and
        # blanket-classifying them as entitlement was the bug that left
        # long-running TUI sessions stuck on stale tokens until the user
        # exited and reopened.
        is_entitlement = agent._is_entitlement_failure(error_context, status_code)
        _auth_haystack = " ".join(
            str(error_context.get(k) or "").lower()
            for k in ("message", "reason", "code", "error")
            if isinstance(error_context, dict)
        )
        if (
            not is_entitlement
            and status_code == 403
            and "oauth authentication is currently not allowed for this organization" in _auth_haystack
        ):
            is_entitlement = True
        if (
            not is_entitlement
            and status_code == 403
            and (agent.provider or "") == "anthropic"
            and getattr(agent, "api_mode", "") == "anthropic_messages"
        ):
            is_entitlement = True
        if not is_entitlement and status_code == 403 and (agent.provider or "") == "xai-oauth":
            _is_xai_auth_failure = (
                "[wke=unauthenticated:" in _auth_haystack
                or "oauth2 access token could not be validated" in _auth_haystack
            )
            if not _is_xai_auth_failure:
                is_entitlement = True
        if is_entitlement:
            _ra().logger.info(
                "Credential %s — entitlement-shaped 403 from %s; "
                "skipping pool refresh (account lacks subscription, "
                "not a transient auth failure).",
                status_code if status_code is not None else "auth",
                agent.provider or "provider",
            )
            return False, has_retried_429
        # Refresh the entry that supplied the failing key, not current():
        # the shared pointer can reference a different, healthy entry, and
        # refreshing it would consume that entry's single-use refresh token
        # (or mark it exhausted on failure) for a failure it never had.
        refresh_kwargs = {"api_key_hint": _api_key_hint}
        if _credential_id:
            refresh_kwargs["credential_id"] = _credential_id
        refreshed = pool.try_refresh_matching(**refresh_kwargs)
        if refreshed is not None:
            # ``try_refresh_matching()`` re-mints a fresh OAuth token and reports
            # success even when the upstream keeps rejecting it — a single-entry
            # pool (common for OAuth/Max subscribers) has nothing to rotate to,
            # so a bare "refreshed → retry" loop spins forever on the same dead
            # token and the configured fallback never activates. Cap consecutive
            # same-entry refreshes and fall through to fallback once exceeded.
            # See #26080.
            refreshed_id = getattr(refreshed, "id", None)
            if refreshed_id is not None:
                refresh_counts = getattr(agent, "_auth_pool_refresh_counts", None)
                if refresh_counts is None:
                    refresh_counts = {}
                    agent._auth_pool_refresh_counts = refresh_counts
                refresh_key = (agent.provider, refreshed_id)
                refresh_counts[refresh_key] = refresh_counts.get(refresh_key, 0) + 1
                if refresh_counts[refresh_key] > _MAX_AUTH_REFRESH_ATTEMPTS:
                    _ra().logger.warning(
                        "Credential auth failure persists after %s refreshes for "
                        "pool entry %s — treating as unrecoverable and allowing "
                        "fallback to activate.",
                        refresh_counts[refresh_key] - 1,
                        refreshed_id,
                    )
                    return False, has_retried_429
            _ra().logger.info("Credential auth failure — refreshed pool entry %s", getattr(refreshed, 'id', '?'))
            agent._swap_credential(refreshed)
            return True, has_retried_429
        # Refresh failed — rotate to next credential instead of giving up.
        # The failed entry is already marked exhausted by the refresh attempt.
        rotate_status = status_code if status_code is not None else 401
        next_entry = _rotate_failed_credential(rotate_status)
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
    # Portal OpenAI-wire traffic still rides aggregator retry infra, so one
    # more rebuilt OpenAI client won't help. Portal Claude on the native
    # Messages route holds a local Anthropic SDK client whose connection
    # pool *does* need the rebuild every other anthropic_messages provider
    # already gets — don't blanket-skip the dual-wire path.
    if (
        provider_lower in {"nous", "nous-portal", "nousresearch"}
        and getattr(agent, "api_mode", None) != "anthropic_messages"
    ):
        return False

    try:
        # Retire the existing client to release stale connections. #70773:
        # never hard-close the shared client here — this runs on the
        # conversation-loop thread while workers from stale-killed streaming
        # attempts may still be unwinding their SSL BIOs on the old pool.
        # ``_retire_shared_openai_client`` shuts the sockets down (FD-safe
        # from any thread) and defers the FD release to GC, which cannot
        # complete until every borrowing thread has unwound.
        if getattr(agent, "client", None) is not None:
            try:
                agent._retire_shared_openai_client(
                    agent.client, reason="primary_recovery",
                )
            except Exception:
                pass

        # Rebuild from primary snapshot
        rt = agent._primary_runtime
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.requested_provider = rt.get("requested_provider", agent.provider)
        agent.base_url = rt["base_url"]
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]
        agent._reasoning_echo_flag = rt.get("reasoning_echo_flag", False)
        agent.request_overrides = dict(rt.get("request_overrides") or {})

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
        elif (agent.provider or "").strip().lower() == "moa":
            # MoA is a virtual provider with empty client_kwargs — rebuilding
            # via _create_openai_client would raise "api_key client option
            # must be set". Recreate the facade through the shared factory so
            # the reference_callback relay survives recovery (#53802).
            from agent.moa_loop import build_moa_facade

            agent.client = build_moa_facade(agent, agent.model)
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
        logger.warning("Primary transport recovery failed: %s", e)
        return False

# ── End provider fallback ──────────────────────────────────────────────



def drop_thinking_only_and_merge_users(
    messages: List[Dict[str, Any]],
    *,
    drop_codex_reasoning_items: bool = True,
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
    kept = [
        m for m in messages
        if not _ra().AIAgent._is_thinking_only_assistant(
            m,
            drop_codex_reasoning_items=drop_codex_reasoning_items,
        )
    ]
    dropped = len(messages) - len(kept)

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

    if dropped == 0 and merges == 0:
        return messages

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

    # ── Reset-aware gate ──
    # The 60s ``_rate_limited_until`` cooldown covers transient rate limits,
    # but subscription-style providers (Claude Pro/Max 5-hour windows, ChatGPT
    # weekly limits) report reset times hours or days away.  The credential
    # pool already stores those timestamps (``last_error_reset_at``); until
    # the earliest one elapses, every restore attempt is a *guaranteed*
    # failure that costs two prompt-cache invalidations per turn (switch to
    # primary, fail, switch back to fallback) and re-marshals the full
    # context each way.  Skip the restore while the pool says nobody can
    # serve, and come back the moment the reset time passes.
    #
    # Fail-open by design: any error (unreadable auth store, legacy pool
    # adapter without ``next_available_at``) falls through to the existing
    # every-turn retry.  A pool with no reset info returns ``None`` and also
    # falls through — this gate only ever *adds* skips for provably
    # limited windows, so recovery can never be later than it is today.
    #
    # When the attached pool belongs to the fallback provider (cross-provider
    # fallback rebinds it), the primary pool is loaded here and handed to the
    # pool-rebind block below via ``prefetched_primary_pool`` so the load
    # happens at most once per restore.
    prefetched_primary_pool = None
    primary_pool_prefetched = False
    try:
        primary_provider = str(
            (agent._primary_runtime or {}).get("provider") or ""
        ).strip().lower()
        primary_runtime_base_url = str(
            (agent._primary_runtime or {}).get("base_url") or ""
        )
        primary_pool_key = resolve_runtime_pool_key(
            primary_provider,
            primary_runtime_base_url,
        )
        pool = getattr(agent, "_credential_pool", None)
        if not credential_pool_matches_provider(
            pool,
            primary_provider,
            base_url=primary_runtime_base_url,
        ):
            from agent.credential_pool import load_pool

            prefetched_primary_pool = (
                load_pool(primary_pool_key) if primary_pool_key else None
            )
            primary_pool_prefetched = True
            if prefetched_primary_pool is not None and credential_pool_matches_provider(
                prefetched_primary_pool,
                primary_provider,
                base_url=primary_runtime_base_url,
            ):
                pool = prefetched_primary_pool
            else:
                prefetched_primary_pool = None
                pool = None
        next_at = getattr(pool, "next_available_at", lambda: None)()
        if next_at is not None and next_at > time.time():
            if not getattr(agent, "_restore_wait_logged", False):
                agent._restore_wait_logged = True
                logger.info(
                    "Primary %s rate-limited until %s; staying on fallback "
                    "%s/%s until the reset elapses",
                    primary_provider or "?",
                    datetime.fromtimestamp(next_at).isoformat(timespec="seconds"),
                    agent.provider,
                    agent.model,
                )
            return False
    except Exception:
        logger.debug(
            "Reset-aware restore gate failed; falling back to per-turn retry",
            exc_info=True,
        )
    agent._restore_wait_logged = False

    rt = agent._primary_runtime
    fallback_route = getattr(agent, "_provider_fallback_route", None)
    if (
        isinstance(fallback_route, (list, tuple))
        and len(fallback_route) == 2
    ):
        previous_model = str(fallback_route[0] or "unknown")
        previous_provider = str(fallback_route[1] or "unknown")
    else:
        previous_model = str(getattr(agent, "model", "") or "unknown")
        previous_provider = str(getattr(agent, "provider", "") or "unknown")
    provider_fallback_active = bool(
        getattr(agent, "_provider_fallback_active", False)
    )
    try:
        # ── Core runtime state ──
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.requested_provider = rt.get("requested_provider", agent.provider)
        agent.base_url = rt["base_url"]           # setter updates _base_url_lower
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]
        if "runtime_capabilities" in rt:
            raw_capabilities = rt["runtime_capabilities"]
            if not isinstance(raw_capabilities, dict):
                logger.warning("Ignoring malformed runtime capabilities snapshot")
            else:
                agent.runtime_capabilities = dict(raw_capabilities)
        elif "capabilities" in rt:
            # Read snapshots written by the initial capability propagation patch.
            raw_capabilities = rt["capabilities"]
            if isinstance(raw_capabilities, dict):
                agent.runtime_capabilities = dict(raw_capabilities)
        agent._reasoning_echo_flag = rt.get("reasoning_echo_flag", False)
        agent.request_overrides = dict(rt.get("request_overrides") or {})
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent._use_prompt_caching = rt["use_prompt_caching"]
        # Default to native layout when the restored snapshot predates the
        # native-vs-proxy split (older sessions saved before this PR).
        agent._use_native_cache_layout = rt.get(
            "use_native_cache_layout",
            agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
        )
        # If the operator has disabled caching via config (cache_ttl is
        # falsy → _cache_disabled flag is set), the disable must survive
        # runtime snapshot restoration (#33555).
        if getattr(agent, "_cache_disabled", False):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False

        # ── Rebuild client for the primary provider ──
        if agent.provider == "moa":
            # MoA is a virtual chat-completions provider.  It never has real
            # OpenAI client kwargs; restoring it after a fallback must recreate
            # the facade, not call OpenAI() with an empty api_key.  Use the
            # shared factory so the restored facade keeps the reference_callback
            # relay wired at init — a bare MoAClient() would silently stop
            # emitting moa.reference/moa.aggregating display events (#53802).
            from agent.moa_loop import build_moa_facade

            agent.client = build_moa_facade(agent, agent.model)
            agent._anthropic_client = None
        elif agent.api_mode == "anthropic_messages":
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
            api_mode=rt.get("compressor_api_mode", ""),
        )

        # ── Rebind and re-select the primary credential pool ──
        # A cross-provider fallback attaches the fallback provider's pool. The
        # runtime fields above restore the primary, but leaving that pool in
        # place makes the next primary 401/429 hit the provider-mismatch guard
        # and disables credential rotation. Reload the primary pool first; if
        # auth storage is temporarily unreadable, clear the mismatched pool.
        primary_provider = str(rt.get("provider") or "").strip().lower()
        primary_runtime_base_url = str(rt.get("base_url") or "")
        primary_pool_key = resolve_runtime_pool_key(
            primary_provider,
            primary_runtime_base_url,
        )
        pool = getattr(agent, "_credential_pool", None)
        pool_provider = str(getattr(pool, "provider", "") or "").strip().lower()
        pool_matches_primary = credential_pool_matches_provider(
            pool,
            primary_provider,
            base_url=primary_runtime_base_url,
        )
        if pool is not None and pool_provider and not pool_matches_primary:
            agent._credential_pool = None
            agent._credential_pool_entry_id = None
            try:
                if primary_pool_prefetched:
                    # Reuse the pool the reset-aware gate already loaded for
                    # this restore — avoids a second disk read of auth.json.
                    if (
                        prefetched_primary_pool is not None
                        and credential_pool_matches_provider(
                            prefetched_primary_pool,
                            primary_provider,
                            base_url=primary_runtime_base_url,
                        )
                    ):
                        agent._credential_pool = prefetched_primary_pool
                else:
                    from agent.credential_pool import load_pool

                    loaded_pool = load_pool(primary_pool_key)
                    if loaded_pool is not None and credential_pool_matches_provider(
                        loaded_pool,
                        primary_provider,
                        base_url=primary_runtime_base_url,
                    ):
                        agent._credential_pool = loaded_pool
            except Exception as exc:
                logger.warning(
                    "Restore could not reload primary credential pool for %s: %s",
                    primary_provider,
                    exc,
                )

        # The snapshot's api_key was captured at construction time.  Across
        # turns the pool may have rotated (token revocation, billing/rate-limit
        # exhaustion, cooldown), leaving the snapshot key stale.  Restoring it
        # blindly re-fails on the first request and burns through the remaining
        # pool entries before cross-provider fallback even gets a chance.  Ask
        # the pool for its current best entry and swap the live credential in.
        # When the pool is absent, empty, or the entry has no usable key, we
        # keep the snapshot key (the existing behavior).  Fixes #25205.
        agent._credential_pool_entry_id = None
        pool = getattr(agent, "_credential_pool", None)
        if pool is not None and pool.has_available():
            entry = pool.select()
            if entry is not None:
                entry_provider = str(getattr(entry, "provider", "") or "").strip().lower()
                entry_matches_primary = credential_pool_matches_provider(
                    entry,
                    primary_provider,
                    base_url=primary_runtime_base_url,
                )

                entry_key = (
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                )
                if entry_key and entry_matches_primary:
                    # ``_swap_credential`` rebuilds the OpenAI/Anthropic client,
                    # reapplies base-url-scoped headers, and carries the
                    # accumulated base_url / OAuth-detection fixes (#33163).
                    agent._swap_credential(entry)
                    logger.info(
                        "Restore re-selected pool entry %s (%s)",
                        getattr(entry, "id", "?"),
                        getattr(entry, "label", "?"),
                    )
                elif entry_key:
                    logger.info(
                        "Restore skipped pool entry %s (%s): provider %s does not match primary provider %s",
                        getattr(entry, "id", "?"),
                        getattr(entry, "label", "?"),
                        entry_provider or "?",
                        primary_provider or "?",
                    )

        # ── Restore reasoning_config if it was saved ──
        # switch_model saves reasoning_config in _primary_runtime. If the
        # snapshot predates that (older sessions), keep the current value.
        saved_reasoning = rt.get("reasoning_config")
        if saved_reasoning is not None:
            agent.reasoning_config = dict(saved_reasoning)

        # ── Reset fallback chain for the new turn ──
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._rate_limit_backoff_count = 0  # reset exponential backoff counter

        # Reset the stale-call circuit breaker (#58962): the streak measured
        # the FALLBACK provider we're leaving; the restored primary deserves
        # a fresh stream attempt before the breaker can trip again.
        from agent.chat_completion_helpers import _reset_stale_streak
        _reset_stale_streak(agent)

        # Undo the fallback's identity rewrite so the prompt is
        # byte-identical to the stored copy again (prefix cache match).
        from agent.chat_completion_helpers import rewrite_prompt_model_identity
        rewrite_prompt_model_identity(agent, rt["model"], rt["provider"])

        logger.info(
            "Primary runtime restored for new turn: %s (%s)",
            agent.model, agent.provider,
        )
        agent._provider_fallback_active = False
        agent._provider_fallback_route = None
        if provider_fallback_active:
            try:
                agent._emit_status(
                    f"✅ Primary model restored: {agent.model} via {agent.provider}; "
                    f"fallback {previous_model} via {previous_provider} is no longer active."
                )
            except Exception:
                # Notification surfaces are best-effort and must never undo a
                # successful runtime restoration.
                pass
        return True
    except Exception as e:
        logger.warning("Failed to restore primary runtime: %s", e)
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
        # Sanitize the session ID into a traversal-free path segment — it can
        # originate from untrusted input (X-Hermes-Session-Id header), and an
        # unsanitized "../"-shaped ID would write the dump outside logs_dir.
        safe_sid = _ra()._safe_session_filename_component(agent.session_id)
        dump_file = agent.logs_dir / f"request_dump_{safe_sid}_{timestamp}.json"

        # Redact secrets before persisting/printing. This dump captures the
        # full request body (system prompt, tool defs, context-embedded
        # values), and this path fires unconditionally on API errors — so it
        # otherwise lands any context-embedded secret in cleartext on disk.
        # Run the serialized dump through the same scrubber used for logs/tool
        # output, then hand the resulting payload back to the shared atomic
        # JSON writer so request dumps keep the same write semantics as before.
        from agent.redact import redact_sensitive_text
        _serialized = json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str)
        _redacted_payload = json.loads(redact_sensitive_text(_serialized, force=True))
        atomic_json_write(dump_file, _redacted_payload, default=str)

        agent._vprint(f"{agent.log_prefix}🧾 Request debug dump written to: {dump_file}")

        if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
            print(json.dumps(_redacted_payload, ensure_ascii=False, indent=2, default=str))

        return dump_file
    except Exception as dump_error:
        if agent.verbose_logging:
            logger.warning("Failed to dump API request debug payload: %s", dump_error)
        return None



def _direct_native_anthropic_tool_cache_capability(
    agent,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> bool:
    """Return whether this resolved destination accepts native tool markers."""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    return (
        eff_api_mode == "anthropic_messages"
        and base_url_hostname(eff_base_url) == "api.anthropic.com"
    )


def cache_ttl_means_disabled(ttl: Any) -> bool:
    """Return True when a ``prompt_caching.cache_ttl`` value means caching off.

    Single source of truth for the disable-synonym detection shared by
    ``agent_init`` (live-agent ``_cache_disabled`` flag) and the stub policy
    paths below. Keeping one predicate prevents the two sites from drifting
    (a synonym added in only one place would recreate #76085).

    Unknown values (e.g. ``"2h"``, integers) are NOT a disable — callers keep
    caching enabled with the default TTL, matching ``agent_init``.
    """
    if ttl in ("5m", "1h"):
        return False
    if ttl is False or ttl is None:
        return True
    return str(ttl).lower() in ("off", "false", "disabled", "no", "none")


# The two cache_ttl tiers accepted by config (anything else is either a
# disable synonym or ignored). Shared by the config readers below and
# mirrored by agent_init's live-agent snapshot.
VALID_CACHE_TTLS = ("5m", "1h")


def _raw_cache_ttl_from_config() -> Any:
    """Read the raw ``prompt_caching.cache_ttl`` config value (may raise)."""
    from hermes_cli.config import load_config_readonly

    pc_cfg = load_config_readonly().get("prompt_caching", {}) or {}
    return pc_cfg.get("cache_ttl", "5m")


def prompt_caching_disabled_from_config() -> bool:
    """Return True when ``prompt_caching.cache_ttl`` is configured as off.

    Same disable detection as ``agent_init`` (via ``cache_ttl_means_disabled``)
    so stub-based policy paths (MoA slot decoration, auxiliary fallback
    replan) honor the same config contract without holding a live
    ``AIAgent`` (#76085 / #33555).
    """
    try:
        ttl = _raw_cache_ttl_from_config()
    except Exception:
        return False
    return cache_ttl_means_disabled(ttl)


def configured_cache_ttl() -> Optional[str]:
    """Return the configured ``prompt_caching.cache_ttl`` tier, if valid.

    Mirrors ``agent_init``'s reading of the same key (``5m``/``1h`` accepted,
    anything else ignored) so stub-based paths without a live ``AIAgent``
    (auxiliary fallback replan) stop regressing a configured ``1h`` to the
    5m default (#84733). Returns ``None`` for unset/disabled/unknown values;
    ``effective_cache_ttl`` resolves ``None`` to ``5m`` downstream.
    """
    try:
        ttl = _raw_cache_ttl_from_config()
    except Exception:
        return None
    return ttl if ttl in VALID_CACHE_TTLS else None


def blank_cache_policy_stub(cache_disabled: Optional[bool] = None):
    """Build the destination-identity-blank stub for ``anthropic_prompt_cache_policy``.

    Single sanctioned constructor for that stub. Callers that resolve cache
    policy against a destination identified out-of-band (not a live
    ``AIAgent``) must go through here so ``_cache_disabled`` is never left
    off a hand-rolled ``SimpleNamespace`` (#76085).

    When ``cache_disabled`` is omitted, falls back to the global config so
    stub paths without an agent snapshot still honor an operator disable.
    """
    from types import SimpleNamespace

    if cache_disabled is None:
        cache_disabled = prompt_caching_disabled_from_config()
    return SimpleNamespace(
        provider="",
        base_url="",
        api_mode="",
        model="",
        _cache_disabled=bool(cache_disabled),
    )


def plan_cache_sections_for_destination(
    messages: list,
    tools: Optional[list],
    *,
    provider: str,
    base_url: str,
    api_mode: str,
    model: str,
    cache_disabled: Optional[bool] = None,
    cache_ttl: Optional[str] = None,
    static_system_prefix: Optional[str] = None,
) -> Tuple[list, list]:
    """Plan request-local cache sections for one resolved destination.

    Shared core of the synchronous acting-aggregator (MoA) and auxiliary
    fallback senders: resolve the cache policy for the destination's real
    provider/base_url/api_mode/model, then either return stripped canonical
    copies (non-caching route) or a :func:`build_prompt_cache_plan` layout
    (caching route, with the direct-native tool marker when the destination
    is api.anthropic.com on the Messages wire).

    Never mutates ``messages`` or ``tools`` — both return values are
    request-local copies.

    ``cache_disabled`` threads the operator's ``prompt_caching.cache_ttl``
    disable into the blank policy stub. When omitted, the live config is
    consulted so MoA/auxiliary paths cannot re-enable markers after the
    user turned caching off (#76085).

    ``cache_ttl`` threads the operator's configured tier (default ``5m``)
    into the destination plan so MoA/auxiliary requests stop regressing to
    the 5m default while the main loop honors ``1h`` (#84733); it is
    clamped per-destination by :func:`effective_cache_ttl` (Qwen → 5m).
    ``static_system_prefix`` threads the builder-declared stable prefix so
    the destination system prompt receives the same early breakpoint the
    main loop applies instead of marking the whole prompt as a breakpoint.
    """
    from agent.prompt_caching import (
        build_prompt_cache_plan,
        effective_cache_ttl,
        envelope_tool_part_cache_markers_supported,
        strip_anthropic_cache_control,
        strip_anthropic_tool_cache_control,
    )

    stub = blank_cache_policy_stub(cache_disabled)
    should_cache, native_layout = anthropic_prompt_cache_policy(
        stub,
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        model=model,
    )
    if not should_cache:
        canonical_messages = copy.deepcopy(messages or [])
        strip_anthropic_cache_control(canonical_messages)
        return canonical_messages, strip_anthropic_tool_cache_control(tools)
    plan = build_prompt_cache_plan(
        messages,
        tools,
        cache_ttl=effective_cache_ttl(
            # effective_cache_ttl resolves None → "5m"; markers are only
            # emitted at all when should_cache passed above, so a
            # cache-disabled agent (_cache_ttl=None) never reaches here
            # with caching active.
            cache_ttl,
            provider=provider,
            model=model,
        ),
        native_anthropic=native_layout,
        static_system_prefix=(
            static_system_prefix if isinstance(static_system_prefix, str) else None
        ),
        direct_native_tool_cache=_direct_native_anthropic_tool_cache_capability(
            stub,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
            model=model,
        ),
        # LiteLLM-style envelope routes forward part-level markers into
        # tool_result.content[] → non-retryable 400 (#89886).
        tool_part_markers=envelope_tool_part_cache_markers_supported(
            provider, base_url
        ),
    )
    return plan.messages, plan.tools


def _is_litellm_route(provider_lower: str, base_url: str) -> bool:
    """True when a route is a LiteLLM proxy, by provider id or host token.

    Provider naming varies per install (``litellm``, ``custom:litellm``, or a
    bare ``custom`` alias pointed at a LiteLLM host), so both signals are
    checked. Both match ``litellm`` as a whole delimited token rather than a
    raw substring: ``base_url_hostname``'s own docstring names substring host
    matching as the false-positive class to avoid, and a plain
    ``"litellm" in ...`` grants Anthropic markers to unrelated routes like
    ``notlitellm.example.com`` or a provider named ``custom:notlitellm``.
    A ``litellm`` *path* segment never qualifies — only the host does.
    """
    if _has_litellm_token(provider_lower, ":-_/"):
        return True
    return _has_litellm_token(base_url_hostname(base_url), ".-")


def _has_litellm_token(value: str, delimiters: str) -> bool:
    """True when ``value`` contains ``litellm`` as a whole delimited token."""
    if not value:
        return False
    for delimiter in delimiters:
        value = value.replace(delimiter, " ")
    return "litellm" in value.split()


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

    If the operator has set ``prompt_caching.cache_ttl`` to a falsy value
    (``false``, ``null``, ``"off"``, etc.) in config.yaml, prompt caching
    is fully disabled — this early return ensures the disable survives
    ``/model`` switches, fallback re-derivation, and runtime snapshot
    restoration (#33555). We check ``"_cache_disabled"`` (set by
    init_agent when the disable is detected) rather than ``_cache_ttl``
    directly, because ``_cache_ttl`` is not yet set when the policy runs
    during the initial ``init_agent`` call.
    """
    if getattr(agent, "_cache_disabled", False):
        return (False, False)

    eff_provider = (provider if provider is not None else agent.provider) or ""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    eff_model = (model if model is not None else agent.model) or ""

    # MoA virtual provider: the agent's model/provider are the preset name and
    # "moa" — neither matches any caching branch, so the ACTING AGGREGATOR
    # (often Claude on OpenRouter) silently lost prompt caching entirely
    # (measured: 85% cache share solo vs 2% on the identical model via MoA —
    # tens of millions of re-billed input tokens per benchmark run). Resolve
    # the policy from the preset's real aggregator slot instead.
    if eff_provider.strip().lower() == "moa":
        try:
            from hermes_cli.config import load_config as _load_moa_cfg
            from hermes_cli.moa_config import resolve_moa_preset
            from hermes_cli.runtime_provider import resolve_runtime_provider

            _preset = resolve_moa_preset(
                _load_moa_cfg().get("moa") or {}, eff_model or None
            )
            _agg = _preset.get("aggregator") or {}
            _agg_provider = str(_agg.get("provider") or "").strip()
            _agg_model = str(_agg.get("model") or "").strip()
            if _agg_provider and _agg_model:
                _agg_base_url = ""
                _agg_api_mode = ""
                try:
                    _rt = resolve_runtime_provider(
                        requested=_agg_provider, target_model=_agg_model
                    )
                    _agg_base_url = _rt.get("base_url") or ""
                    _agg_api_mode = _rt.get("api_mode") or ""
                except Exception:
                    pass
                return anthropic_prompt_cache_policy(
                    agent,
                    provider=_agg_provider,
                    base_url=_agg_base_url,
                    api_mode=_agg_api_mode,
                    model=_agg_model,
                )
        except Exception as _moa_exc:  # pragma: no cover - defensive
            logger.debug("MoA aggregator cache-policy resolution failed: %s", _moa_exc)
        return False, False

    if isinstance(eff_model, dict):
        eff_model = eff_model.get('model') or eff_model.get('default') or ''
    eff_model = eff_model if isinstance(eff_model, str) else str(eff_model or '')
    model_lower = eff_model.lower()
    provider_lower = eff_provider.lower()
    is_claude = "claude" in model_lower
    # Kimi / Moonshot family via OpenRouter: same cache_control wire format
    # as Claude on OpenRouter (envelope layout).  Without this branch
    # moonshotai/kimi-k2.6 falls through to (False, False), serving ~1%
    # cache hits on 64K-token prompts and re-billing the full prompt on
    # every turn.  Observed within-turn progression with cache enabled:
    # 1% → 67% → 84% → 97% (#25970).  Reuses the canonical family matcher
    # (covers bare k1./k2./k25 release slugs the substring check missed).
    from agent.anthropic_adapter import _model_name_is_kimi_family
    is_kimi = (
        _model_name_is_kimi_family(eff_model) or "moonshot" in model_lower
    )
    is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
    # Nous Portal proxies to OpenRouter behind the scenes — identical
    # OpenAI-wire envelope cache_control semantics. Treat it as an
    # OpenRouter-equivalent endpoint for caching layout purposes.
    is_nous_portal = base_url_host_matches(eff_base_url, "nousresearch.com")
    is_anthropic_wire = eff_api_mode == "anthropic_messages"
    is_native_anthropic = (
        is_anthropic_wire
        and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
    )

    # A configured route may use an arbitrary provider name and model alias
    # that are canonicalized only after Hermes sends the request. Honor its
    # existing per-model ``prompt_caching`` capability instead of guessing
    # support from either spelling. Explicit false is authoritative too.
    #
    # The declaration only controls the two transports handled by this marker
    # planner. Responses and Bedrock use separate caching protocols and must
    # not receive Anthropic-style cache_control fields.
    custom_prompt_caching = None
    _supports_anthropic_cache_markers = eff_api_mode in {
        "anthropic_messages",
        "chat_completions",
    }
    _litellm_openai_wire = (
        eff_api_mode == "chat_completions"
        and is_claude
        and _is_litellm_route(provider_lower, eff_base_url)
    )
    _custom_providers = getattr(agent, "_custom_providers", None)
    _route_may_be_custom = False
    if not _supports_anthropic_cache_markers:
        # Responses/Bedrock never consume the declaration — skip the
        # identity probe entirely for those transports.
        pass
    elif _custom_providers:
        # The normalized list is already attached after agent initialization.
        # Use cheap runtime identity signals before calling the capability
        # helper so an unrelated configured provider does not put every
        # built-in chat-completions request on the route-normalization path.
        #
        # Identity must match the authoritative helper's semantics:
        # get_custom_provider_model_capability compares base URLs via
        # normalize_route_base_url, and runtime provider ids go through
        # custom_provider_aliases (space→hyphen, custom: prefix variants).
        # A raw-string gate here would silently drop declarations whose
        # config spelling differs only in host case / trailing slash.
        from hermes_cli.providers import custom_provider_aliases
        from hermes_cli.route_identity import normalize_route_base_url

        _provider_ids = {provider_lower}
        if provider_lower.startswith("custom:"):
            _provider_ids.add(provider_lower.removeprefix("custom:"))
        _eff_url_normalized = normalize_route_base_url(eff_base_url)
        for _entry in _custom_providers:
            if not isinstance(_entry, dict):
                continue
            _entry_ids = custom_provider_aliases(
                str(_entry.get("name") or ""),
                str(_entry.get("provider_key") or ""),
            )
            if _provider_ids & _entry_ids or (
                _eff_url_normalized
                and normalize_route_base_url(_entry.get("base_url"))
                == _eff_url_normalized
            ):
                _route_may_be_custom = True
                break
    elif _custom_providers is None:
        # None = the list is not attached yet (early agent initialization or
        # a blank_cache_policy_stub destination); an attached empty list means
        # the agent initialized with no custom providers and correctly never
        # matches. Avoid rebuilding the list for ordinary built-in routes,
        # while still recognizing arbitrary config keys and built-in-name
        # overrides that point at a different endpoint.
        try:
            from hermes_cli.providers import get_provider

            # allow_network=False: this runs per request destination; a cold
            # models.dev cache must not trigger a foreground registry fetch
            # from the send path. A catalog miss (None) degrades to the
            # conservative side (route may be custom → capability lookup).
            _provider_def = get_provider(eff_provider, allow_network=False)
            _route_may_be_custom = _provider_def is None or (
                bool(_provider_def.base_url)
                and base_url_hostname(_provider_def.base_url)
                != base_url_hostname(eff_base_url)
            )
        except Exception as _pd_exc:
            logger.debug(
                "provider lookup failed during cache-policy pre-gate: %s",
                _pd_exc,
            )
            _route_may_be_custom = provider_lower.startswith("custom:")

    if _supports_anthropic_cache_markers and (
        is_anthropic_wire or _litellm_openai_wire or _route_may_be_custom
    ):
        try:
            from hermes_cli.config import get_custom_provider_model_capability

            custom_prompt_caching = get_custom_provider_model_capability(
                model=eff_model,
                base_url=eff_base_url,
                capability="prompt_caching",
                custom_providers=_custom_providers,
            )
        except Exception as _cap_exc:
            logger.debug(
                "custom-provider prompt_caching capability lookup failed: %s",
                _cap_exc,
            )
    if custom_prompt_caching is not None:
        # Layout follows the transport, not the declaration: native Messages
        # uses inner-block markers; OpenAI-compatible chat uses the envelope
        # layout already emitted for OpenRouter and LiteLLM.
        return custom_prompt_caching, custom_prompt_caching and is_anthropic_wire

    # MiniMax-M3 rides MiniMax's server-side automatic prefix cache on the
    # Anthropic wire (content-keyed, no marker needed); explicit cache_control
    # is documented for M2.7/M2.5/M2.1/M2 only, so markers on M3 are dead
    # weight — never observable (cache_creation always 0) nor billable.
    # Checked BEFORE the native-Anthropic return: provider="anthropic"
    # pointed at a MiniMax /anthropic proxy is a supported override
    # (_anthropic_base_url_override_ok) that would otherwise return
    # (True, True) above this exclusion.
    # Docs: https://platform.minimax.io/docs/api-reference/text-prompt-caching
    is_minimax_provider = provider_lower in {"minimax", "minimax-cn"}
    is_minimax_host = (
        base_url_host_matches(eff_base_url, "api.minimax.io")
        or base_url_host_matches(eff_base_url, "api.minimaxi.com")
    )
    is_minimax_route = is_minimax_provider or is_minimax_host
    if is_anthropic_wire and is_minimax_route:
        from agent.model_metadata import _model_name_suggests_minimax_m3

        if _model_name_suggests_minimax_m3(eff_model):
            return False, False

    if is_native_anthropic:
        return True, True
    # Envelope layout is an OpenAI-wire construct. Portal Claude on the native
    # Messages route must fall through to the third-party anthropic_messages
    # branch below, which emits inner-block cache_control breakpoints; the
    # envelope form would be dropped and serve 0% cache hits.
    if (
        (is_openrouter or is_nous_portal)
        and (is_claude or is_kimi)
        and not is_anthropic_wire
    ):
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

    # LiteLLM fronting a Claude model on the OpenAI-compatible wire.
    # The branch above only matches LiteLLM in Anthropic proxy mode
    # (api_mode == "anthropic_messages"). A LiteLLM deployment that
    # exposes /v1/chat/completions instead matched no grant branch above
    # and fell through to (False, False): no cache_control is injected, the
    # system prompt goes on the wire as a plain string, and the provider
    # serves zero cache hits — the entire prompt is re-billed at full price
    # every turn. Same failure class already documented above for
    # Qwen/DashScope. The endpoint supports Anthropic-style cache_control
    # fine; only the provider detection missed it (#84506).
    #
    # Gated on the Claude family only: a Gemini/GPT/Qwen route through the
    # same proxy must not receive markers — some strict OpenAI-wire relays
    # reject the cache_control block format outright (cf. the DeepSeek /
    # OpenCode exclusion below, #77217).
    #
    # Envelope layout (native_anthropic=False), matching every other
    # OpenAI-wire grant in this function. The native inner-block layout
    # writes a TOP-LEVEL msg["cache_control"] on role:tool and
    # empty-content messages and relies on the Anthropic adapter to
    # relocate it — but that adapter only runs for api_mode ==
    # "anthropic_messages" (agent/transports/anthropic.py), and the
    # chat_completions transport performs no relocation. On this wire the
    # native layout therefore (a) silently loses those breakpoints, spending
    # 2 of the 4 available on markers the provider never sees, and (b) when
    # LiteLLM relocates a top-level marker itself for an OpenRouter-backed
    # Claude route, lands it on an empty text block — the HTTP 400
    # "text content blocks must contain" shape handled in
    # agent/anthropic_adapter.py (#69512).
    #
    # Gated on chat_completions explicitly rather than `not
    # is_anthropic_wire`: codex_responses / bedrock_converse are separate
    # transports with their own marker handling and must not be swept in.
    if _litellm_openai_wire:
        return True, False

    # MiniMax on its Anthropic-compatible endpoint serves its own
    # model family (MiniMax-M2.7, M2.5, M2.1, M2) with documented
    # cache_control support (0.1× read pricing, 5-minute TTL).  The
    # blanket is_claude gate above excludes these — opt them in
    # explicitly via provider id or host match so users on
    # provider=minimax / minimax-cn (or custom endpoints pointing at
    # api.minimax.io/anthropic / api.minimaxi.com/anthropic) get the
    # same cost reduction as Claude traffic.  MiniMax-M3 never reaches
    # here — it is excluded before the native-Anthropic return above.
    # Docs: https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache
    if is_anthropic_wire and is_minimax_route:
        return True, True

    # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire
    # transport that accepts Anthropic-style cache_control markers and
    # rewards them with real cache hits.  Without this branch
    # qwen3.6-plus on opencode-go reports 0% cached tokens and burns
    # through the subscription on every turn.
    #
    # NOTE: DeepSeek models on OpenCode are intentionally excluded.
    # OpenCode Zen's relay rejects the Anthropic-style content block
    # format that cache markers produce (content becomes a block array
    # instead of a plain string), causing HTTP 400 (#77217).
    # Single source of truth for the family set and the qwen-model
    # predicate — shared with the effective_cache_ttl clamp so the
    # opt-in and the TTL clamp can never desync (#84733).
    from agent.prompt_caching import ALIBABA_FAMILY_PROVIDERS, is_qwen_model

    model_is_qwen = is_qwen_model(model_lower)
    provider_is_alibaba_family = provider_lower in ALIBABA_FAMILY_PROVIDERS
    if provider_is_alibaba_family and model_is_qwen:
        # Envelope layout (native_anthropic=False): markers on inner
        # content parts, not top-level tool messages.  Matches
        # pi-mono's "alibaba" cacheControlFormat.
        return True, False

    return False, False



def create_openai_client(agent, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    from agent.ssl_verify import resolve_httpx_verify
    # Treat client_kwargs as read-only. Callers pass agent._client_kwargs (or shallow
    # copies of it) in; any in-place mutation leaks back into the stored dict and is
    # reused on subsequent requests. #10933 hit this by injecting an httpx.Client
    # transport that was torn down after the first request, so the next request
    # wrapped a closed transport and raised "Cannot send a request, as the client
    # has been closed" on every retry. The revert resolved that specific path; this
    # copy locks the contract so future transport/keepalive work can't reintroduce
    # the same class of bug.
    client_kwargs = dict(client_kwargs)
    # The MoA virtual provider has no real OpenAI wire endpoint - the facade
    # *is* the client. Rebuilding a native OpenAI client while
    # agent.provider == "moa" (client replacement, stream-retry pool cleanup,
    # credential rotation, fallback+restore) drops the facade: the next primary
    # call either raises a `_moa_prepared_request` TypeError (#78382) or, when
    # _client_kwargs carry an unrelated relay base_url, leaks the request to a
    # foreign gateway. Rebuild the facade instead (build_moa_facade also
    # re-wires the reference relay, see #53802).
    if (getattr(agent, "provider", "") or "").strip().lower() == "moa":
        from agent.moa_loop import build_moa_facade
        return build_moa_facade(agent, getattr(agent, "model", None) or "default")
    ssl_ca_cert = client_kwargs.pop("ssl_ca_cert", None)
    ssl_verify_cfg = client_kwargs.pop("ssl_verify", None)
    httpx_verify = resolve_httpx_verify(ca_bundle=ssl_ca_cert, ssl_verify=ssl_verify_cfg)
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
    if agent.provider == "gemini":
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url", "") or "")
        if is_native_gemini_base_url(base_url):
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
            }
            if "http_client" not in safe_kwargs:
                keepalive_http = agent._build_keepalive_http_client(
                    base_url, verify=httpx_verify,
                )
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
        keepalive_http = agent._build_keepalive_http_client(
            client_kwargs.get("base_url", ""), verify=httpx_verify,
        )
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http
    # Delegate all rate-limit / 5xx retry to hermes's outer conversation loop,
    # which honors Retry-After and applies adaptive/jittered backoff. The OpenAI
    # SDK default (max_retries=2) uses its own 1-2s backoff that ignores
    # Retry-After and double-retries inside our loop — the same deadlock the
    # Anthropic clients hit (#26293). This is the single chokepoint every primary
    # OpenAI/aggregator client passes through (init, switch_model, recovery,
    # restore, request-scoped); auxiliary_client builds its own clients and keeps
    # SDK retries because it is NOT wrapped by the conversation loop.
    client_kwargs.setdefault("max_retries", 0)
    # Defense-in-depth: guarantee Copilot requests carry the integration
    # headers regardless of which build path we came through. The primary
    # header wiring lives in `_apply_client_headers_for_base_url`, but two
    # rebuild paths (`primary_recovery`, `restore_primary` in this module)
    # reconstruct the client purely from a `_primary_runtime` snapshot and do
    # NOT re-run that wiring. If the snapshot's client_kwargs ever lacks
    # `default_headers` (older snapshot, header-less resolver result), the
    # client goes out WITHOUT `Copilot-Integration-Id: vscode-chat`; the
    # Copilot server then routes it to the "copilot-language-server" integrator
    # whose model allowlist omits enterprise-only models (claude-opus-4.8) →
    # HTTP 400 model_not_available_for_integrator on every turn. This chokepoint
    # is the single place every primary OpenAI client passes through, so filling
    # missing Copilot headers here closes the whole class. We only ADD missing
    # keys — never override headers a caller deliberately set.
    try:
        if base_url_host_matches(str(client_kwargs.get("base_url", "")), "githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers
            existing = dict(client_kwargs.get("default_headers") or {})
            existing_lower = {k.lower() for k in existing}
            for hk, hv in copilot_default_headers().items():
                if hk.lower() not in existing_lower:
                    existing[hk] = hv
            client_kwargs["default_headers"] = existing
    except Exception:
        _ra().logger.debug("Copilot default-header guard skipped", exc_info=True)
    # OpenCode Free: the tier is served ANONYMOUSLY — any bearer the relay
    # doesn't recognize (including placeholders) is a 401. Route every
    # opencode-free client through the shared keyless header policy: an
    # empty Authorization default_header overrides the SDK's
    # "Bearer <api_key>" so no credential ever reaches the wire.
    if agent.provider == "opencode-free":
        from hermes_cli.models import opencode_zen_free_headers

        _existing = dict(client_kwargs.get("default_headers") or {})
        _existing.update(opencode_zen_free_headers())
        client_kwargs["default_headers"] = _existing

    # All primary construction and recovery paths must identify Hermes to the
    # official Codex endpoint, including snapshots with custom header overrides.
    from agent.auxiliary_client import _apply_required_codex_headers

    _apply_required_codex_headers(
        client_kwargs,
        access_token=client_kwargs.get("api_key", ""),
        base_url=str(client_kwargs.get("base_url", "")),
    )
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


def _apply_switched_provider_request_overrides(agent, new_provider):
    """Re-derive the switched-to provider's ``request_overrides`` onto a live agent.

    A ``custom_providers`` entry can carry an ``extra_body`` (e.g.
    ``chat_template_kwargs`` to toggle a local model's thinking). The gateway
    rebuild path carries this via ``request_overrides``; an *in-place* swap
    (CLI / TUI ``/model``) must re-derive it for the switched-to provider,
    otherwise the previous provider's ``extra_body`` lingers.

    The switched-to entry is matched by **provider key, base_url, and model** —
    the same condition ``agent_init._merge_custom_provider_extra_body`` applies
    at build time — via the shared ``_custom_provider_extra_body_for_agent``
    matcher. Matching by name alone would let a *different* model selected at the
    same named endpoint inherit an ``extra_body`` configured for another model.
    A stale ``extra_body`` is always cleared when the switched-to provider/model
    resolves none; non-provider overrides (``service_tier`` / ``speed`` from
    ``/fast``) are preserved.
    """
    from agent.agent_init import _custom_provider_extra_body_for_agent

    # Prefer the init-time cache (agent_init stores ``agent._custom_providers``
    # right where it runs its own _merge_custom_provider_extra_body); fall back
    # to a fresh load only if a caller built the agent without it.
    custom_providers = getattr(agent, "_custom_providers", None)
    if custom_providers is None:
        try:
            from hermes_cli.config import load_config, get_compatible_custom_providers
            custom_providers = get_compatible_custom_providers(load_config())
        except Exception:
            custom_providers = []

    new_extra_body = _custom_provider_extra_body_for_agent(
        provider=new_provider,
        model=getattr(agent, "model", "") or "",
        base_url=getattr(agent, "base_url", "") or "",
        custom_providers=custom_providers or [],
    )

    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    overrides.pop("extra_body", None)  # always drop the previous provider's extra_body
    if new_extra_body:
        overrides["extra_body"] = dict(new_extra_body)
    agent.request_overrides = overrides


def switch_model(
    agent,
    new_model,
    new_provider,
    api_key='',
    base_url='',
    api_mode='',
    capabilities=None,
):
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
    from agent.native_compaction import resolve_native_compaction_capabilities

    old_model = agent.model
    old_provider = agent.provider

    # ── Determine api_mode if not provided ──
    # Pass model so dual-wire providers (Nous Portal anthropic/* → Messages)
    # resolve correctly; without it determine_api_mode falls back to the
    # openai_chat overlay default.
    if not api_mode:
        api_mode = determine_api_mode(new_provider, base_url, model=new_model)

    normalized_new_provider = (new_provider or "").strip().lower()
    if not base_url and normalized_new_provider == "openai":
        # An omitted URL means the provider's canonical direct endpoint.
        base_url = "https://api.openai.com/v1"

    # Same-provider switches may omit base_url intentionally (for example, a
    # direct caller refreshing credentials). Resolve capabilities from the
    # endpoint that the normalization below will retain, not from the empty
    # raw argument.
    effective_base_url = base_url
    if not effective_base_url and (old_provider or "").strip().lower() == (
        new_provider or ""
    ).strip().lower():
        effective_base_url = getattr(agent, "base_url", "")

    destination_capabilities = (
        dict(capabilities)
        if isinstance(capabilities, dict)
        else resolve_native_compaction_capabilities(
            model=new_model,
            base_url=effective_base_url,
            provider=new_provider,
            is_codex_backend=(new_provider or '').strip().lower() == 'openai-codex',
        )
    )

    # Defense-in-depth: ensure OpenCode base_url doesn't carry a trailing
    # /v1 into the anthropic_messages client, which would cause the SDK to
    # hit /v1/v1/messages.  `model_switch.switch_model()` already strips
    # this, but we guard here so any direct callers (future code paths,
    # tests) can't reintroduce the double-/v1 404 bug.
    from hermes_cli.models import opencode_provider_family

    if (
        api_mode == "anthropic_messages"
        and opencode_provider_family(new_provider) is not None
        and isinstance(base_url, str)
        and base_url
    ):
        base_url = re.sub(r"/v1/?$", "", base_url)

    # ── Snapshot all fields the swap+rebuild can mutate ──
    # If the rebuild raises (bad API key, network error, build_anthropic_client
    # failure, etc.) we restore these atomically so the agent isn't left with a
    # new model/provider name paired with the OLD client — that mismatch causes
    # HTTP 400s like "claude-sonnet-4-6 is not supported on openai-codex" on the
    # next turn.  Callers in cli.py / gateway/run.py / tui_gateway/server.py
    # catch the re-raised exception and show the user a warning; without this
    # rollback the warning is misleading because the swap partially succeeded.
    # Use a sentinel so we can distinguish "attribute was unset" from
    # "attribute was None" and skip the restore for genuinely-missing
    # attributes (tests construct bare agents via __new__ without all fields).
    _MISSING = object()
    _snapshot = {
        name: getattr(agent, name, _MISSING)
        for name in (
            "model",
            "provider",
            "requested_provider",
            "base_url",
            "api_mode",
            "api_key",
            "client",
            "_anthropic_client",
            "_anthropic_api_key",
            "_anthropic_base_url",
            "_is_anthropic_oauth",
            "_config_context_length",
            "_reasoning_echo_flag",
            "runtime_capabilities",
        )
    }
    # _client_kwargs is a dict — snapshot a shallow copy so mutating the
    # live dict doesn't poison the rollback target.
    _snapshot["_client_kwargs"] = dict(getattr(agent, "_client_kwargs", {}) or {})
    # Snapshot the credential pool reference so a failed client rebuild can
    # restore the original pool (issue #52727: pool reload is part of this
    # switch and must be reversible on rollback).
    _snapshot["_credential_pool"] = getattr(agent, "_credential_pool", _MISSING)
    _snapshot["_credential_pool_entry_id"] = getattr(
        agent, "_credential_pool_entry_id", _MISSING
    )

    def _restore_snapshot() -> None:
        for _name, _value in _snapshot.items():
            if _value is _MISSING:
                # Attribute did not exist before the swap — don't fabricate it.
                continue
            try:
                setattr(agent, _name, _value)
            except Exception:  # noqa: BLE001
                pass

    try:
        # Clear the per-config context_length override so the new model's
        # actual context window is resolved via get_model_context_length()
        # instead of inheriting the stale value from the previous model.
        agent._config_context_length = None

        # ── Swap core runtime fields ──
        agent.model = new_model
        agent.provider = new_provider
        agent.requested_provider = new_provider
        # Re-read reasoning_echo from config so the flag reflects the new
        # primary model's setting (see _reasoning_echo_opt_in).
        agent._reasoning_echo_flag = agent._read_reasoning_echo_from_config()
        # Use the new base_url when provided. When it's empty AND the
        # provider is actually changing, do NOT fall back to the current
        # (old provider's) URL — that silently pairs the new provider label
        # with the previous provider's endpoint (e.g. new_provider=minimax
        # paired with the leftover api.githubcopilot.com URL), and every
        # request after the switch 400s at the wrong host. This mismatched
        # pair also gets snapshotted into _primary_runtime below, so it
        # keeps re-applying on every subsequent turn until a full restart.
        # Fail loud instead: the caller (model_switch.switch_model())
        # already resolves base_url for every real provider, so an empty
        # value here means resolution failed upstream, not that the
        # provider genuinely has none. Re-selecting the SAME provider with
        # an empty base_url (e.g. a credential-only refresh) is still fine
        # to keep the current URL. See #47828.
        old_norm_provider = (old_provider or "").strip().lower()
        new_norm_provider = (new_provider or "").strip().lower()
        if base_url:
            agent.base_url = base_url
        elif old_norm_provider != new_norm_provider:
            raise ValueError(
                f"switch_model: no base_url resolved for provider "
                f"'{new_provider}' (switching from '{old_provider}'); "
                "refusing to keep the previous provider's endpoint"
            )
        agent.api_mode = api_mode
        # Invalidate transport cache — new api_mode may need a different transport
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        if api_key:
            agent.api_key = api_key

        # ── Reload credential pool for the new provider (issue #52727) ──
        # Without this, ``recover_with_credential_pool`` sees a
        # ``pool.provider != agent.provider`` mismatch and short-circuits,
        # leaving the new provider with no rotation/recovery on 401/429 and
        # burning the original pool's entries. Only reload when the provider
        # actually changed (or the pool was missing) — re-selecting the same
        # provider must not churn the pool reference. A reload failure is
        # logged + swallowed: the switch itself must still complete.
        old_norm = (old_provider or "").strip().lower()
        new_norm = (new_provider or "").strip().lower()
        if old_norm != new_norm or getattr(agent, "_credential_pool", None) is None:
            # A pool bound to the old provider is worse than no pool: the
            # recovery guard rejects it and every later 401/429 skips rotation.
            agent._credential_pool = None
            agent._credential_pool_entry_id = None
            try:
                from agent.credential_pool import load_pool
                agent._credential_pool = load_pool(new_provider)
            except Exception as _pool_exc:  # noqa: BLE001
                logger.warning(
                    "switch_model: credential pool reload failed for %s (%s); "
                    "continuing without pool rotation this turn",
                    new_provider, _pool_exc,
                )
        # ── Build new client ──
        if (new_provider or "").strip().lower() == "moa":
            from agent.moa_loop import build_moa_facade

            # The MoA virtual provider speaks only chat.completions via the
            # MoAClient facade — the aggregator's real transport
            # (codex_responses / anthropic_messages) is resolved and applied
            # *inside* the reference/aggregator fan-out, never on the outer
            # primary call. determine_api_mode("moa", ...) above may have left
            # api_mode set to the aggregator's transport; if the conversation
            # loop sees that, it dispatches client.responses.create (which the
            # facade has no .responses for) and the call falls through to the
            # moa://local placeholder → HTTP 404 → fallback to a reference
            # model. Pin chat_completions here so the primary call always goes
            # through MoAClient.chat.completions, matching agent_init.py.
            agent.api_mode = "chat_completions"
            agent.api_key = api_key or "moa-virtual-provider"
            agent.base_url = "moa://local"
            agent._client_kwargs = {}
            agent.client = build_moa_facade(agent, agent.model)
        elif api_mode == "anthropic_messages":
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

            # MiniMax OAuth: swap static string for a per-request callable token
            # provider so the rebuilt client survives 15-min token expiry. See
            # the matching block in agent_init.py for the full rationale.
            if new_provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
                try:
                    from hermes_cli.auth import build_minimax_oauth_token_provider
                    effective_key = build_minimax_oauth_token_provider()
                except Exception as _mm_exc:  # noqa: BLE001
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "MiniMax OAuth: failed to install per-request token provider "
                        "on switch (%s); using static bearer.",
                        _mm_exc,
                    )

            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = base_url or getattr(agent, "_anthropic_base_url", None)
            agent._anthropic_client = build_anthropic_client(
                effective_key, agent._anthropic_base_url,
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = _is_oauth_token(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False
            agent.client = None
            agent._client_kwargs = {}
        else:
            effective_key = api_key or agent.api_key
            effective_base = base_url or agent.base_url
            agent._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            try:
                from hermes_cli.config import (
                    apply_custom_provider_tls_to_client_kwargs,
                    get_compatible_custom_providers,
                    load_config_readonly,
                )

                # Read custom_providers from live config (not the init-time
                # snapshot on ``agent._custom_providers``) so ssl_ca_cert /
                # ssl_verify edits are honored when switching mid-session,
                # matching the context-length reload below (#15779).
                apply_custom_provider_tls_to_client_kwargs(
                    agent._client_kwargs,
                    str(effective_base or ""),
                    get_compatible_custom_providers(load_config_readonly()),
                )
            except Exception:
                logger.debug("custom-provider TLS resolution skipped on switch_model", exc_info=True)
            _sm_timeout = get_provider_request_timeout(agent.provider, agent.model)
            if _sm_timeout is not None:
                agent._client_kwargs["timeout"] = _sm_timeout
            # Reapply provider-specific headers (e.g. OpenRouter HTTP-Referer,
            # X-Title) that were lost when _client_kwargs was rebuilt from
            # scratch.  Without this, model switches clear attribution headers
            # and OpenRouter logs show "Unknown" for subsequent requests.
            agent._apply_client_headers_for_base_url(effective_base)
            agent.client = agent._create_openai_client(
                dict(agent._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        sync_credential_pool_entry_id(agent)
    except Exception:
        # Rollback every mutated field to the pre-swap snapshot so the agent
        # is left consistent (old model + old provider + old client) and the
        # caller's exception handler can surface a meaningful warning.  The
        # exception is re-raised; cli.py / gateway/run.py / tui_gateway catch
        # it and print "Agent swap failed; change applied to next session".
        _restore_snapshot()
        raise

    # ── LM Studio: preload before probing context length ──
    _sm_custom_providers = None
    try:
        from hermes_cli.config import (
            get_compatible_custom_providers,
            get_custom_provider_context_length,
            load_config,
        )

        _sm_cfg = load_config()
        _sm_custom_providers = get_compatible_custom_providers(_sm_cfg)
        _destination_context_intent = get_custom_provider_context_length(
            model=agent.model,
            base_url=agent.base_url,
            custom_providers=_sm_custom_providers,
        )
    except Exception:
        _destination_context_intent = None
    agent._config_context_length = _destination_context_intent
    if hasattr(agent, "_ensure_lmstudio_runtime_loaded"):
        try:
            _runtime_context_length = agent._ensure_lmstudio_runtime_loaded(
                _destination_context_intent
            )
        except Exception:
            _restore_snapshot()
            raise
    else:
        _runtime_context_length = None
    if (
        hasattr(agent, "_lmstudio_load_was_unverified")
        and agent._lmstudio_load_was_unverified(_runtime_context_length)
    ):
        logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length during model switch; continuing "
            "with configured context"
        )
    if hasattr(agent, "_effective_lmstudio_context_length"):
        _effective_context_length = agent._effective_lmstudio_context_length(
            _destination_context_intent,
            _runtime_context_length,
        )
    else:
        _effective_context_length = _destination_context_intent

    # ── Re-evaluate prompt caching ──
    # Refresh the custom-provider snapshot from the config just loaded above
    # so the per-model ``prompt_caching`` capability lookup sees the same
    # live list the context-length resolution used — without this, a flag
    # added to config.yaml after session start is invisible to a /model
    # switch (the policy would read the stale init-time snapshot).
    if _sm_custom_providers is not None:
        agent._custom_providers = _sm_custom_providers
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        agent._anthropic_prompt_cache_policy(
            provider=new_provider,
            base_url=agent.base_url,
            api_mode=api_mode,
            model=new_model,
        )
    )

    # ── Update context compressor ──
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        from agent.model_metadata import get_model_context_length
        if _sm_custom_providers is None:
            try:
                from hermes_cli.config import get_compatible_custom_providers, load_config
                _sm_custom_providers = get_compatible_custom_providers(load_config())
            except Exception:
                _sm_custom_providers = None
        # ``agent.api_key`` may be a callable (Azure Foundry Entra ID
        # token provider). ``get_model_context_length`` expects a
        # string for its live-probe paths; for Foundry the context
        # length normally resolves via config or static catalogs and
        # never hits a probe, but coerce to empty string defensively.
        _ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
        try:
            new_context_length = get_model_context_length(
                agent.model,
                base_url=agent.base_url,
                api_key=_ctx_api_key,
                provider=agent.provider,
                config_context_length=_effective_context_length,
                custom_providers=_sm_custom_providers,
            )
            agent.context_compressor.update_model(
                model=agent.model,
                context_length=new_context_length,
                base_url=agent.base_url,
                api_key=agent.api_key,  # context_compressor forwards to call_llm; callable preserved
                provider=agent.provider,
                api_mode=agent.api_mode,
            )
        except Exception:
            _restore_snapshot()
            raise

    # ── Re-resolve reasoning_config from per-model override ──
    # The new model may have a different reasoning_effort override. Re-read
    # config so the override takes effect immediately on /model switch —
    # resolved through the shared chokepoint (per-model > global; YAML
    # boolean False = disabled).
    try:
        from hermes_constants import resolve_reasoning_config
        from hermes_cli.config import load_config as _sm_load_config

        _reasoning_cfg = _sm_load_config() or {}
        agent.reasoning_config = resolve_reasoning_config(_reasoning_cfg, agent.model)
        logger.info(
            "switch_model: reasoning_config resolved for %s: %s",
            agent.model, agent.reasoning_config,
        )
    except Exception as _reasoning_err:
        logger.debug("switch_model: could not re-resolve reasoning_config: %s", _reasoning_err)

    # ── Invalidate cached system prompt so it rebuilds next turn ──
    agent._cached_system_prompt = None

    # Publish the destination capability map only after every runtime setup
    # above has succeeded. Failed switches must leave the old map intact.
    agent.runtime_capabilities = destination_capabilities

    # ── Reset the cross-turn stale-call circuit breaker (#58962) ──
    # The breaker's error text tells the user to "switch models ... then
    # retry"; without this reset the streak stays latched and the freshly
    # selected (healthy) provider would keep short-circuiting before any
    # stream is even attempted.
    from agent.chat_completion_helpers import _reset_stale_streak
    _reset_stale_streak(agent)

    # ── Update _primary_runtime so the change persists across turns ──
    _cc = agent.context_compressor if hasattr(agent, "context_compressor") and agent.context_compressor else None
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "reasoning_config": dict(agent.reasoning_config) if getattr(agent, "reasoning_config", None) else None,
        "reasoning_echo_flag": getattr(agent, "_reasoning_echo_flag", False),
        # Request-level overrides (extra_body etc.) must travel with the
        # switched-to identity; without this, a post-switch transport
        # recovery or fallback restore would resurrect the PRE-switch
        # overrides via the stale init-time snapshot (#75091 seam).
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "runtime_capabilities": dict(getattr(agent, "runtime_capabilities", {}) or {}),
        "compressor_model": getattr(_cc, "model", agent.model) if _cc else agent.model,
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url) if _cc else agent.base_url,
        "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
        "compressor_provider": getattr(_cc, "provider", agent.provider) if _cc else agent.provider,
        "compressor_context_length": _cc.context_length if _cc else 0,
        "compressor_api_mode": getattr(_cc, "api_mode", agent.api_mode) if _cc else agent.api_mode,
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
    agent._provider_fallback_active = False
    agent._provider_fallback_route = None
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

    # Apply the switched-to provider's request_overrides (custom_providers
    # extra_body, e.g. chat_template_kwargs). See helper for rationale.
    try:
        _apply_switched_provider_request_overrides(agent, new_provider)
    except Exception:
        logger.debug("switch_model: request_overrides re-derivation failed", exc_info=True)

    logger.info(
        "Model switched in-place: %s (%s) -> %s (%s)",
        old_model, old_provider, new_model, new_provider,
    )

    # ── Persist billing route to session DB ──
    # The agent's _session_db / session_id may not be set in all contexts
    # (tests, bare agents without a session DB, etc.).  This ensures the
    # dashboard Model cards show the actual provider after a mid-session
    # /model switch instead of the stale session-creation provider.
    # See #48248 for the full bug description.
    _session_db = getattr(agent, "_session_db", None)
    _session_id = getattr(agent, "session_id", None)
    if _session_db is not None and _session_id:
        try:
            _session_db.update_session_billing_route(
                _session_id,
                provider=agent.provider,
                base_url=agent.base_url,
                billing_mode=getattr(agent, "api_mode", None),
            )
        except Exception:
            logger.warning(
                "Failed to persist billing route after model switch",
                exc_info=True,
            )


def invoke_tool(agent, function_name: str, function_args: dict, effective_task_id: str,
                 tool_call_id: Optional[str] = None, messages: list = None,
                 pre_tool_block_checked: bool = False,
                 skip_tool_request_middleware: bool = False,
                 tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
                 skip_tool_execution_middleware: bool = False) -> str:
    """Invoke a single tool and return the result string. No display logic.

    Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
    tools. Used by the concurrent execution path; the sequential path retains
    its own inline invocation for backward-compatible display handling.
    """
    if not isinstance(function_args, dict):
        function_args = {}

    _tool_middleware_trace = list(tool_request_middleware_trace or [])
    try:
        from hermes_cli.middleware import apply_tool_request_middleware

        if not skip_tool_request_middleware:
            _tool_request_mw = apply_tool_request_middleware(
                function_name,
                function_args,
                task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            )
            function_args = _tool_request_mw.payload
            _tool_middleware_trace = _tool_request_mw.trace
    except Exception as _mw_err:
        logger.debug("tool_request middleware error: %s", _mw_err)

    # Check plugin hooks for a block or approval directive before executing.
    block_message: Optional[str] = None
    if not pre_tool_block_checked:
        try:
            from hermes_cli.plugins import _dispatch_pre_tool_call_hooks
            block_message, modified_args = _dispatch_pre_tool_call_hooks(
                function_name, function_args, task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                middleware_trace=list(_tool_middleware_trace),
            )
            if modified_args is not None:
                function_args = modified_args
        except Exception:
            block_message = None
    if block_message is not None:
        result = json.dumps({"error": block_message}, ensure_ascii=False)
        try:
            from model_tools import _emit_post_tool_call_hook
            _emit_post_tool_call_hook(
                function_name=function_name,
                function_args=function_args,
                result=result,
                task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                status="blocked",
                error_type="plugin_block",
                error_message=block_message,
                middleware_trace=list(_tool_middleware_trace),
            )
        except Exception:
            pass
        return result

    tool_start_time = time.monotonic()

    def _finish_agent_tool(result: Any, observed_args: Optional[dict] = None) -> Any:
        hook_args = observed_args if isinstance(observed_args, dict) else function_args
        try:
            from model_tools import _emit_post_tool_call_hook
            _emit_post_tool_call_hook(
                function_name=function_name,
                function_args=hook_args,
                result=result,
                task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                duration_ms=int((time.monotonic() - tool_start_time) * 1000),
                middleware_trace=list(_tool_middleware_trace),
            )
        except Exception:
            pass
        return result

    if function_name == "todo":
        def _execute(next_args: dict) -> Any:
            from tools.todo_tool import todo_tool as _todo_tool
            return _finish_agent_tool(
                _todo_tool(
                    todos=next_args.get("todos"),
                    merge=next_args.get("merge", False),
                    store=agent._todo_store,
                ),
                next_args,
            )
    elif function_name == "session_search":
        def _execute(next_args: dict) -> Any:
            session_db = agent._get_session_db_for_recall()
            if not session_db:
                from hermes_state import format_session_db_unavailable
                return _finish_agent_tool(json.dumps({"success": False, "error": format_session_db_unavailable()}), next_args)
            from tools.session_search_tool import session_search as _session_search
            return _finish_agent_tool(
                _session_search(
                    query=next_args.get("query", ""),
                    role_filter=next_args.get("role_filter"),
                    limit=next_args.get("limit", 3),
                    session_id=next_args.get("session_id"),
                    around_message_id=next_args.get("around_message_id"),
                    window=next_args.get("window", 5),
                    sort=next_args.get("sort"),
                    detail=next_args.get("detail", "adaptive"),
                    db=session_db,
                    current_session_id=agent.session_id,
                ),
                next_args,
            )
    elif function_name == "memory":
        def _execute(next_args: dict) -> Any:
            target = next_args.get("target", "memory")
            operations = next_args.get("operations")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=next_args.get("action"),
                target=target,
                content=next_args.get("content"),
                old_text=next_args.get("old_text"),
                operations=operations,
                store=agent._memory_store,
            )
            # Mirror successful built-in memory writes to external providers.
            # All gating/op-expansion lives behind the manager interface
            # (MemoryManager.notify_memory_tool_write).
            if agent._memory_manager:
                agent._memory_manager.notify_memory_tool_write(
                    result,
                    next_args,
                    build_metadata=lambda: agent._build_memory_write_metadata(
                        task_id=effective_task_id,
                        tool_call_id=tool_call_id,
                    ),
                )
            return _finish_agent_tool(result, next_args)
    elif agent._memory_manager and agent._memory_manager.has_tool(function_name):
        def _execute(next_args: dict) -> Any:
            return _finish_agent_tool(agent._memory_manager.handle_tool_call(function_name, next_args), next_args)
    elif function_name == "clarify":
        def _execute(next_args: dict) -> Any:
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _finish_agent_tool(
                _clarify_tool(
                    question=next_args.get("question", ""),
                    choices=next_args.get("choices"),
                    multi_select=next_args.get("multi_select", False),
                    questions=next_args.get("questions"),
                    callback=agent.clarify_callback,
                ),
                next_args,
            )
    elif function_name == "read_terminal":
        def _execute(next_args: dict) -> Any:
            from tools.read_terminal_tool import read_terminal_tool as _read_terminal_tool
            return _finish_agent_tool(
                _read_terminal_tool(
                    start_line=next_args.get("start_line"),
                    count=next_args.get("count"),
                    callback=getattr(agent, "read_terminal_callback", None),
                ),
                next_args,
            )
    elif function_name == "desktop_preview":
        def _execute(next_args: dict) -> Any:
            # action=read needs the GUI callback (agent-level); open/close go
            # through the registry handler like any other tool.
            if (next_args.get("action") or "").strip() == "read":
                from tools.read_preview_tool import read_preview_tool as _read_preview_tool
                return _finish_agent_tool(
                    _read_preview_tool(
                        start=next_args.get("start"),
                        count=next_args.get("count"),
                        callback=getattr(agent, "read_preview_callback", None),
                    ),
                    next_args,
                )
            from tools.preview_tool import _handle_preview
            return _finish_agent_tool(_handle_preview(next_args), next_args)
    elif function_name == "drive_preview":
        def _execute(next_args: dict) -> Any:
            from tools.drive_preview_tool import drive_preview_tool as _drive_preview_tool
            return _finish_agent_tool(
                _drive_preview_tool(
                    action=next_args.get("action", ""),
                    ref=next_args.get("ref"),
                    selector=next_args.get("selector"),
                    text=next_args.get("text"),
                    key=next_args.get("key"),
                    submit=next_args.get("submit"),
                    amount=next_args.get("amount"),
                    to=next_args.get("to"),
                    limit=next_args.get("max"),
                    callback=getattr(agent, "drive_preview_callback", None),
                ),
                next_args,
            )
    elif function_name == "annotate_preview":
        def _execute(next_args: dict) -> Any:
            from tools.annotate_preview_tool import annotate_preview_tool as _annotate_preview_tool
            return _finish_agent_tool(
                _annotate_preview_tool(
                    action=next_args.get("action", "add"),
                    ref=next_args.get("ref"),
                    selector=next_args.get("selector"),
                    label=next_args.get("label"),
                    callback=getattr(agent, "drive_preview_callback", None),
                ),
                next_args,
            )
    elif function_name == "read_window_below":
        def _execute(next_args: dict) -> Any:
            from tools.read_window_tool import read_window_below_tool as _read_window_below_tool
            return _finish_agent_tool(
                _read_window_below_tool(
                    callback=getattr(agent, "read_window_below_callback", None),
                ),
                next_args,
            )
    elif function_name == "tour":
        def _execute(next_args: dict) -> Any:
            from tools.tour_tool import tour_tool as _tour_tool
            return _finish_agent_tool(
                _tour_tool(
                    action=next_args.get("action", ""),
                    surface=next_args.get("surface"),
                    selector=next_args.get("selector"),
                    title=next_args.get("title"),
                    text=next_args.get("text"),
                    side=next_args.get("side"),
                    steps=next_args.get("steps"),
                    step_index=next_args.get("step_index"),
                    callback=getattr(agent, "tour_callback", None),
                ),
                next_args,
            )
    elif function_name == "setup_mcp":
        def _execute(next_args: dict) -> Any:
            from tools.setup_mcp_tool import setup_mcp_tool as _setup_mcp_tool
            return _finish_agent_tool(
                _setup_mcp_tool(
                    server=next_args.get("server", ""),
                    action=next_args.get("action", "install"),
                    reason=next_args.get("reason", ""),
                    callback=getattr(agent, "setup_mcp_callback", None),
                ),
                next_args,
            )
    elif function_name == "delegate_task":
        def _execute(next_args: dict) -> Any:
            return _finish_agent_tool(agent._dispatch_delegate_task(next_args), next_args)
    else:
        def _execute(next_args: dict) -> Any:
            dispatch_kwargs = dict(
                tool_call_id=tool_call_id,
                session_id=agent.session_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                tool_request_middleware_trace=list(_tool_middleware_trace),
            )
            if skip_tool_execution_middleware:
                dispatch_kwargs["skip_tool_execution_middleware"] = True
            return _ra().handle_function_call(
                function_name,
                next_args,
                effective_task_id,
                **dispatch_kwargs,
            )

    if skip_tool_execution_middleware:
        return _execute(function_args)

    from hermes_cli.middleware import run_tool_execution_middleware

    return run_tool_execution_middleware(
        function_name,
        function_args,
        lambda next_args: _execute(next_args if isinstance(next_args, dict) else function_args),
        original_args=function_args,
        task_id=effective_task_id or "",
        session_id=getattr(agent, "session_id", "") or "",
        tool_call_id=tool_call_id or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
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

    # VolcEngine api/plan workaround (issue #33007): the endpoint's
    # protocol-translation layer occasionally leaks raw XML attribute
    # fragments into tool_use.name, e.g.
    #   `terminal" parameter="command" string="true`
    #   `execute_code" parameter="code" string="true`
    #   `session_search" parameter="session_id" string="true`
    # We trim at the first unambiguous XML/quote character so the rest
    # of the repair pipeline (lowercase / snake_case / fuzzy match)
    # can resolve the cleaned name to a real tool.
    #
    # Crucially we DO NOT split on whitespace: legitimate inputs like
    # "write file" must keep flowing through ``_norm`` -> ``write_file``
    # (covered by test_space_to_underscore in
    # tests/run_agent/test_repair_tool_call_name.py).
    for _xml_sep in ('"', "'", "<", ">"):
        _idx = tool_name.find(_xml_sep)
        if _idx > 0:
            tool_name = tool_name[:_idx]
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


def _tool_call_id_variants(tc: Any) -> set:
    """Return every id a tool result might legitimately match this tool_call on.

    Thin forwarder — the policy owner is
    ``agent.message_sanitization.tool_call_id_variants`` (handles ``id``,
    ``call_id``, ``response_item_id``, and composite ``call|item`` spellings).
    Kept for backward compatibility with existing importers.
    """
    return set(tool_call_id_variants(tc))


# Placeholder substituted for an empty non-final message that would otherwise
# make the provider reject the whole request. Kept identical to the stub-
# creation placeholder in chat_completion_helpers so a healed transcript reads
# consistently whether the empty turn was caught at write time or send time.
_INTERRUPTED_PLACEHOLDER = "[response interrupted]"


def _msg_has_payload(msg: Dict[str, Any]) -> bool:
    """True if ``msg`` carries anything the API treats as non-empty content.

    Covers string content, non-empty multimodal content lists, tool_calls,
    tool_call_id linkage (tool results), and reasoning payloads. Mirrors the
    emptiness checks used by ``AIAgent._is_thinking_only_assistant`` but is
    role-agnostic so it can vet user/assistant/tool turns uniformly.
    """
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            return True
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                # any typed block (text/image/tool_use/document/...) counts,
                # as long as a text block is not itself blank
                if block.get("type") == "text":
                    if isinstance(block.get("text"), str) and block["text"].strip():
                        return True
                    continue
                return True
            elif block:
                return True
    elif content not in (None, ""):
        return True
    # Structural payloads that make an "empty-content" message still valid.
    if msg.get("tool_calls"):
        return True
    if isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"].strip():
        return True
    if msg.get("reasoning") or msg.get("reasoning_details"):
        return True
    # Codex Responses item carriers: a commentary-phase assistant turn
    # persists with content:"" by DESIGN — its text lives in
    # ``codex_message_items`` (delivered via the interim callback) and the
    # structured items are replayed for prefix-cache hits.  Same for
    # ``codex_reasoning_items``.  These turns are never wire-empty on any
    # api_mode: the codex transport replays the items, and the
    # chat-completions transport strips the carriers only after this repair
    # pass has already run.  Treat them as payload so the repair never
    # rewrites a designed-empty codex turn (July 2026: a write-time pad that
    # ignored this broke codex commentary replay in CI).
    if msg.get("codex_message_items") or msg.get("codex_reasoning_items"):
        return True
    return False


def repair_empty_non_final_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Heal empty-content non-final messages before they reach the provider.

    Root-cause context: a stream that dies with 0 recovered characters (peer
    reset, stall-kill) could persist an assistant turn with ``content=None``
    and no tool_calls. The Anthropic message schema — and the litellm/Bedrock
    proxies in front of it — reject ANY request whose transcript contains an
    empty non-final message:

        "all messages must have non-empty content except for the optional
         final assistant message"  (HTTP 400 INVALID_REQUEST_BODY)

    Once such a message lands mid-transcript it poisons EVERY subsequent turn
    of that session until it scrolls out of context. The write-time guard in
    ``chat_completion_helpers`` stops NEW stubs, but sessions already carrying
    one (persisted before the guard, or fed in from a host history) stay stuck
    and previously needed a manual DB edit + gateway restart to recover.

    This pass is the self-healing counterpart: it runs unconditionally on the
    per-call ``api_messages`` copy, so a poisoned transcript repairs itself
    IN MEMORY on the very next send — no restart, no DB surgery. The final
    message is left untouched (an empty final assistant turn is legal). The
    stored conversation history is never mutated; only the wire copy is
    repaired, so the UI/session trace stays faithful.

    Repair strategy is substitution, not deletion: dropping a mid-transcript
    turn can break role alternation and tool-call pairing, whereas an honest
    minimal placeholder keeps the sequence intact and reads correctly as an
    interrupted turn on replay.
    """
    if not messages or len(messages) < 2:
        return messages

    repaired: List[Dict[str, Any]] = []
    healed = 0
    last_idx = len(messages) - 1
    for idx, msg in enumerate(messages):
        if (
            idx != last_idx
            and isinstance(msg, dict)
            # tool results are validated by their own orphan/pairing pass; an
            # empty tool result is a separate (and rarer) concern.
            and msg.get("role") in ("assistant", "user")
            and not _msg_has_payload(msg)
        ):
            # Shallow-copy so stored history / prompt caching stays byte-stable.
            fixed = dict(msg)
            fixed["content"] = _INTERRUPTED_PLACEHOLDER
            repaired.append(fixed)
            healed += 1
        else:
            repaired.append(msg)

    if healed:
        _ra().logger.warning(
            "Pre-call sanitizer: healed %d empty non-final message(s) by "
            "substituting placeholder content — an empty-content turn was in "
            "the transcript and would 400 the request ('messages must have "
            "non-empty content' / INVALID_REQUEST_BODY). Self-recovering the "
            "poisoned transcript in memory; no restart needed.",
            healed,
        )
        return repaired
    return messages


def _classify_tool_call_orphans(messages: List[Dict[str, Any]]):
    """Classify orphaned tool-call / tool-result pairs in *messages*.

    Returns a 4-tuple ``(surviving_call_ids, result_call_ids,
    orphaned_results, missing_tool_calls)``:

    - ``surviving_call_ids``: every id variant carried by any assistant
      ``tool_calls`` entry. A tool_call may carry SEVERAL equivalent id
      spellings (``id`` fc_..., ``call_id`` call_..., ``response_item_id``,
      or a composite ``call|item`` bridge id) — register EVERY variant so a
      result matching any of them survives (#55626, #63000).
    - ``result_call_ids``: every id variant referenced by a ``tool`` result.
    - ``orphaned_results``: the actual ``tool`` message dicts whose complete
      alias set matches no assistant call (compare by identity — ``id(msg)``
      — when filtering, since dicts are unhashable).
    - ``missing_tool_calls``: the actual tool_call entries with no matching
      result on ANY alias (after orphaned results are excluded).

    This is the single source of truth for GLOBAL orphan *detection* (does
    a matching id exist anywhere in the transcript). Its remaining consumer
    is the context compressor's ``_sanitize_tool_pairs`` (strip orphans from
    the durable history); the id-resolution and variant-expansion rules live
    here so they can never drift between call sites again (#58357).

    ``sanitize_api_messages`` no longer uses the global check: strict
    positional providers (DeepSeek v4, Kimi) reject a call whose result is
    not in the IMMEDIATELY-following tool run even when a matching id exists
    elsewhere, so its pairing pass walks the transcript positionally instead
    (#94704) — a strictly stronger invariant that subsumes the global one at
    that site. Both share the same ``tool_call_id_variants`` /
    ``tool_result_id_variants`` alias policy, which is the part that must
    not fork.
    """
    assistant_call_variants: List[tuple[Any, frozenset[str]]] = []
    surviving_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            variants = tool_call_id_variants(tc)
            if variants:
                assistant_call_variants.append((tc, variants))
                surviving_call_ids.update(variants)

    result_entries = [
        (msg, tool_result_id_variants(msg.get("tool_call_id")))
        for msg in messages
        if msg.get("role") == "tool"
    ]
    result_call_ids: set[str] = set()
    for _, variants in result_entries:
        result_call_ids.update(variants)

    orphaned_results = [
        msg
        for msg, variants in result_entries
        if variants and not (variants & surviving_call_ids)
    ]
    orphaned_ids = {id(msg) for msg in orphaned_results}
    surviving_result_variants = [
        variants
        for msg, variants in result_entries
        if variants and id(msg) not in orphaned_ids
    ]
    missing_tool_calls = [
        tc
        for tc, variants in assistant_call_variants
        if not any(variants & rv for rv in surviving_result_variants)
    ]
    return surviving_call_ids, result_call_ids, orphaned_results, missing_tool_calls


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

    # --- Heal empty-content non-final messages (self-recovery) ---
    # A dead stream can leave an empty assistant stub (or an empty user turn)
    # mid-transcript; the provider then 400s EVERY subsequent request until it
    # scrolls out. Repair it here, on the per-call copy, so a poisoned session
    # recovers itself in memory on the next send — no restart, no DB edit.
    # Done first so a substituted turn participates normally in the tool-pair
    # and dedup passes below.
    messages = repair_empty_non_final_messages(messages)

    # --- Drop empty / malformed tool_calls arrays on assistant messages ---
    # An assistant message carrying ``tool_calls: []`` (an empty array) — or a
    # non-list value under the key — is semantically identical to an assistant
    # message with no tool calls, but strict OpenAI-compatible providers reject
    # the empty array outright: DeepSeek v4 returns HTTP 400 "Invalid
    # 'messages[N].tool_calls': empty array. Expected an array with minimum
    # length 1, but got an empty array instead." (#58755, follow-up to #56980).
    # Empty arrays reach here from session resume, host-fed histories, or the
    # consecutive-assistant merge in ``repair_message_sequence`` (which
    # preserves a pre-existing ``[]`` on the surviving turn). This is the final
    # pre-API chokepoint, so normalize defensively — and, per the #56980
    # review, do it HERE on the per-call copy rather than in
    # ``repair_message_sequence``, which would destructively rewrite the
    # persisted trajectory. Shallow-copy the message before dropping the key so
    # stored history (and prompt caching) stays byte-stable.
    normalized: List[Dict[str, Any]] = []
    dropped_empty_tool_calls = 0
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and "tool_calls" in msg
            and not (isinstance(msg["tool_calls"], list) and msg["tool_calls"])
        ):
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            dropped_empty_tool_calls += 1
        normalized.append(msg)
    if dropped_empty_tool_calls:
        messages = normalized
        _ra().logger.debug(
            "Pre-call sanitizer: dropped empty/invalid tool_calls on %d "
            "assistant message(s)",
            dropped_empty_tool_calls,
        )

    # --- Repair tool_calls whose function.name is empty/missing ---
    # Some providers (and partially-streamed responses) emit a tool_call with
    # id="call_xxx" but function.name="". Downstream Responses-API adapters
    # silently DROP such function_call items while still emitting the matching
    # function_call_output, producing the gateway's HTTP 400
    # "No tool call found for function call output with call_id ...".
    #
    # We do NOT drop the call: hermes' own dispatch loop intentionally keeps an
    # empty-name call paired with a synthesized anti-priming tool result
    # ("tool name was empty", see #47967) so weak models self-correct instead of
    # being fed the full tool catalog. Dropping the call here would (a) orphan
    # that result and strip the anti-priming signal, and (b) still leave any
    # provider-side orphan. Instead, rename the blank name to a non-empty
    # sentinel so the call and its result stay PAIRED — the adapter no longer
    # drops the function_call, so there is no orphaned output and no 400, while
    # the result content the model needs is preserved.
    _EMPTY_NAME_SENTINEL = "invalid_tool_call"
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        if not tcs:
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function")
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
            if isinstance(name, str) and name.strip():
                continue
            _ra().logger.warning(
                "Pre-call sanitizer: repairing tool_call with empty "
                "function.name -> %r (id=%s)",
                _EMPTY_NAME_SENTINEL,
                _ra().AIAgent._get_tool_call_id_static(tc),
            )
            if isinstance(fn, dict):
                fn["name"] = _EMPTY_NAME_SENTINEL
            elif fn is not None and hasattr(fn, "name"):
                try:
                    fn.name = _EMPTY_NAME_SENTINEL
                except Exception:
                    pass
            elif isinstance(tc, dict):
                tc["function"] = {"name": _EMPTY_NAME_SENTINEL, "arguments": "{}"}

    # --- Drop tool results with a missing/empty tool_call_id ---
    # The positional pairing walk below also catches this shape (an id-less
    # result expands to zero variants, matches no declared call, and is
    # dropped as a positional orphan), but keep the explicit early filter so
    # the distinct failure mode keeps its own log line and the guarantee
    # doesn't silently depend on the walk's internals: a result with no
    # ``tool_call_id`` at all is a schema violation strict OpenAI-compatible
    # providers reject outright. ``repair_message_sequence``'s
    # Pass 1 already drops this shape (`if tc_id and tc_id in
    # known_tool_ids`) when it runs first on the same list, but any caller
    # that reaches this function without going through
    # ``repair_message_sequence`` first has no such guard. Drop explicitly
    # here so this "final chokepoint" claim (see module docstring) actually
    # holds regardless of caller (#78071).
    _pre_id_filter_count = len(messages)
    messages = [
        m for m in messages
        if not (m.get("role") == "tool" and not (m.get("tool_call_id") or "").strip())
    ]
    if len(messages) != _pre_id_filter_count:
        _ra().logger.debug(
            "Pre-call sanitizer: dropped %d tool result(s) with missing/empty tool_call_id",
            _pre_id_filter_count - len(messages),
        )

    # --- Positional tool_call <-> tool_result pairing ---
    # Strict OpenAI-compatible providers (DeepSeek v4, Kimi) enforce the
    # POSITIONAL invariant: an assistant message carrying tool_calls must
    # be IMMEDIATELY followed by tool messages covering every
    # tool_call_id. The previous implementation compared global id sets,
    # which misses the failure mode where a result exists somewhere in
    # the transcript but not in the run right after its call — an
    # interrupted turn or a compression window can displace a result
    # past a user turn. The id then survives in the global result set,
    # so the call looks answered, no stub is injected, and the provider
    # rejects the payload with HTTP 400 "An assistant message with
    # 'tool_calls' must be followed by tool messages responding to each
    # 'tool_call_id' (insufficient tool messages following tool_calls
    # message)". Rewritten as a single rolling walk on the per-call
    # copy (#94704):
    #   (a) tool results that do not immediately follow an assistant
    #       message declaring their id are dropped (positional orphans —
    #       includes results appearing BEFORE their call, which strict
    #       providers also reject);
    #   (b) declared ids not covered by the immediately-following tool
    #       run get a stub result injected at the end of that run, even
    #       when a mispositioned result exists elsewhere.
    # Matching is variant-aware (``tool_call_id_variants`` /
    # ``tool_result_id_variants``): a result keyed on ANY alias spelling
    # (``id`` / ``call_id`` / ``response_item_id`` / composite bridge)
    # answers the call, preserving the unified alias policy from
    # #55626/#63000/#93251.
    paired: List[Dict[str, Any]] = []
    declared_calls: Dict[str, tuple] = {}
    dropped_positional_orphans = 0
    added_stubs = 0

    def _flush_unanswered_stubs() -> None:
        nonlocal added_stubs
        for key in sorted(declared_calls):
            tc, _variants = declared_calls[key]
            cid = coalesce_tool_call_id(tc) or key
            paired.append({
                "role": "tool",
                "name": _ra().AIAgent._get_tool_call_name_static(tc),
                "content": "[Result unavailable — see context summary above]",
                "tool_call_id": cid,
            })
            added_stubs += 1
        declared_calls.clear()

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            # A new assistant turn closes the previous tool-result run:
            # anything still pending was never answered positionally.
            _flush_unanswered_stubs()
            declared_calls = {}
            for tc in msg.get("tool_calls") or []:
                variants = tool_call_id_variants(tc)
                if variants:
                    # Key on a stable representative of the alias group so
                    # a result matching ANY spelling can consume the call.
                    declared_calls[sorted(variants)[0]] = (tc, variants)
            paired.append(msg)
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            matched = next(
                (
                    key
                    for key, (_tc, variants) in declared_calls.items()
                    if variants & result_variants
                ),
                None,
            )
            if matched is not None:
                paired.append(msg)
                # Consume so a duplicate result reusing the id falls into
                # the drop branch (same semantics as the old global
                # dedup; strict providers reject duplicate tool_call_id).
                declared_calls.pop(matched, None)
            else:
                dropped_positional_orphans += 1
        else:
            if role == "user":
                # A user turn closes the tool-result run; subsequent
                # tool messages without a fresh declaring assistant
                # turn are orphans.
                _flush_unanswered_stubs()
            paired.append(msg)
    # The transcript may end right after an unanswered assistant turn.
    _flush_unanswered_stubs()
    if dropped_positional_orphans or added_stubs:
        messages = paired
    if dropped_positional_orphans:
        _ra().logger.debug(
            "Pre-call sanitizer: removed %d positionally orphaned tool result(s)",
            dropped_positional_orphans,
        )
    if added_stubs:
        _ra().logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s) for "
            "positionally unanswered tool call(s)",
            added_stubs,
        )

    # 3. Deduplicate tool_call_ids. Strict providers (DeepSeek) reject a
    # payload where the same tool_call_id appears more than once with HTTP 400
    # "Duplicate value for 'tool_call_id'" (#58327). Duplicates can arise from
    # retries, crash/resume glitches, or a compression window that re-emits a
    # tool result. This is the final pre-API chokepoint, so dedup defensively
    # here even though repair_message_sequence also consumes matched ids.
    #   (a) collapse duplicate tool_calls WITHIN an assistant message
    #   (b) drop tool results that answer no OUTSTANDING tool call
    #
    # (b) tracks outstanding calls rather than every id ever seen, because
    # ``tool_call_id`` is NOT globally unique in practice: llama.cpp emits a
    # single constant id for every tool call it ever returns (verified: three
    # separate completions from one server all carry the same id). A
    # seen-once-drop-forever rule reads the SECOND legitimate tool result of
    # such a session as a duplicate and deletes it, so from the second tool
    # call onward the model never sees any result — it announces its next
    # action and the turn dies with the work unfinished. Outstanding-call
    # semantics keep both protections intact: a re-emitted result still
    # answers no pending call and is still dropped, while a genuine new call
    # that reuses the id re-arms that id first.
    # Variant-group tracking: answering or deduping one spelling consumes
    # its siblings too. A Codex/Responses tool_call registers ``id``
    # (fc_...), ``call_id`` (call_...), ``response_item_id``, and composite
    # spellings (#55626/#58168/#63000); tracking only the coalesced id here
    # made a result keyed on any OTHER variant look like it answered no
    # outstanding call, so this pass deleted the very result step 2's
    # variant-aware matching had just preserved (issue #93251 — whole
    # parallel batches vanished).
    seen_assistant_call_ids: set = set()
    outstanding_call_ids: set = set()
    outstanding_groups: Dict[int, frozenset] = {}
    variant_to_group: Dict[str, int] = {}
    next_group_id = 0
    deduped: List[Dict[str, Any]] = []
    removed_dupes = 0
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            kept_tcs = []
            for tc in msg.get("tool_calls") or []:
                variants = tool_call_id_variants(tc)
                if variants and variants & seen_assistant_call_ids:
                    removed_dupes += 1
                    continue
                if variants:
                    group_id = next_group_id
                    next_group_id += 1
                    outstanding_groups[group_id] = variants
                    for variant in variants:
                        seen_assistant_call_ids.add(variant)
                        outstanding_call_ids.add(variant)
                        variant_to_group.setdefault(variant, group_id)
                kept_tcs.append(tc)
            if kept_tcs:
                msg = {**msg, "tool_calls": kept_tcs}
            elif len(kept_tcs) != len(msg.get("tool_calls") or []):
                msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            deduped.append(msg)
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            candidate_groups = {
                variant_to_group[variant]
                for variant in result_variants
                if variant in variant_to_group
                and variant in outstanding_call_ids
            }
            if result_variants and not candidate_groups:
                removed_dupes += 1
                continue
            if candidate_groups:
                # Answered: consume EVERY variant of the matched call so a
                # second result replaying any sibling spelling is still
                # caught above, and the ids are re-armable by the next
                # assistant call that reuses them.
                group_id = min(candidate_groups)
                group_variants = outstanding_groups.pop(group_id, frozenset())
                for variant in group_variants:
                    outstanding_call_ids.discard(variant)
                    seen_assistant_call_ids.discard(variant)
                    if variant_to_group.get(variant) == group_id:
                        variant_to_group.pop(variant, None)
            deduped.append(msg)
        else:
            deduped.append(msg)
    if removed_dupes:
        messages = deduped
        _ra().logger.debug(
            "Pre-call sanitizer: removed %d duplicate tool_call_id reference(s)",
            removed_dupes,
        )
    return messages



def looks_like_codex_intermediate_ack(
    agent,
    user_message: Any,
    assistant_content: str,
    messages: List[Dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn.

    ``require_workspace`` (default True) keeps the original codex-coding scope:
    the ack must reference a filesystem/repo workspace. The conversation loop
    passes ``require_workspace=False`` when the user has explicitly opted into
    intent-ack continuation for all api_modes (``agent.intent_ack_continuation``
    is ``true`` or a model-list), so general autonomous workflows ("I'll run a
    health check on the server", "I'll start the deployment") — which carry a
    future-ack and an action verb but no filesystem reference — are caught too.
    The future-ack + short-content + no-prior-tools + action-verb requirements
    always apply, which is what keeps conversational "I'll help you brainstorm"
    replies from tripping it.
    """
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

    assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
    if not assistant_mentions_action:
        return False

    # Opted-in (all-api_mode) path: a future-ack + action verb + no prior tool
    # call is enough — the user asked us to keep going when the model only
    # announces intent, regardless of whether a filesystem is involved.
    if not require_workspace:
        return True

    # ``user_message`` is typed ``str`` but can arrive as an OpenAI-style
    # multi-part content list (``[{type:"text",...}, {type:"image_url",...}]``)
    # for vision requests routed through the OpenAI-compat API server. A
    # truthy list survives ``(user_message or "")`` and then ``.strip()``
    # raises ``AttributeError`` — flatten to text first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log

    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    user_targets_workspace = (
        any(marker in user_text for marker in workspace_markers)
        or "~/" in user_text
        or "/" in user_text
    )
    assistant_targets_workspace = any(
        marker in assistant_text for marker in workspace_markers
    )
    return user_targets_workspace or assistant_targets_workspace


# Conservative "trailing continue-intent" detector for the said-continue-but-
# stopped stall guard (agent.stall_guards). Matches only when the message TAIL
# announces an immediate next action ("Let me now…", "I will now…",
# "Next, I…"), which is the observed stall shape: the model narrates the next
# step and then ends the turn with no tool call. Kept deliberately narrow so
# ordinary answers that merely contain "I will" mid-sentence never trip it.
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$",
    re.IGNORECASE,
)

# Content longer than this is a substantive reply, not a dangling ack.
_TRAILING_CONTINUE_INTENT_MAX_CHARS = 400


def trailing_continue_intent(text: str) -> bool:
    """Whether ``text`` is a short reply ENDING on an announced next action.

    Used by the stall-guard extension of the intent-ack continuation path in
    ``agent.conversation_loop``: when a turn is about to end with this shape
    (no tool calls, short content, trailing intent), the loop re-prompts via
    the existing bounded continuation mechanism instead of stopping.
    """
    t = (text or "").strip()
    if not t or len(t) > _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        return False
    return bool(_TRAILING_CONTINUE_INTENT_RE.search(t[-160:]))


def intent_ack_continuation_mode(agent) -> str:
    """Classify the resolved intent-ack continuation mode for this turn.

    Returns one of:
      * ``"off"``        — never continue.
      * ``"codex_only"`` — historical scope: continue only on the
        ``codex_responses`` api_mode, and only for codebase/workspace acks
        (``require_workspace=True``).
      * ``"all"``        — user opted in for every api_mode; continue on any
        future-ack + action verb (``require_workspace=False``).

    Mirrors the four-mode shape of ``agent.tool_use_enforcement``: ``"auto"``
    (default) → codex_only; ``True``/"true"/"always"/"yes"/"on" → all;
    ``False``/"false"/"never"/"no"/"off" → off; ``list`` → all when a substring
    matches the active model name, else off.
    """
    mode = getattr(agent, "_intent_ack_continuation", "auto")

    if mode is True or (isinstance(mode, str) and mode.lower() in {"true", "always", "yes", "on"}):
        return "all"
    if mode is False or (isinstance(mode, str) and mode.lower() in {"false", "never", "no", "off"}):
        return "off"
    if isinstance(mode, list):
        model_lower = (agent.model or "").lower()
        return "all" if any(p.lower() in model_lower for p in mode if isinstance(p, str)) else "off"
    # "auto" or any unrecognised value — historical codex-only behavior.
    return "codex_only" if agent.api_mode == "codex_responses" else "off"


def intent_ack_continuation_enabled(agent) -> bool:
    """Whether intent-ack continuation should fire at all for this turn.

    The ``codex_ack_continuations < 2`` per-turn cap and the
    ``looks_like_codex_intermediate_ack`` detector are applied by the caller;
    this only decides the on/off gate. Callers that also need to know whether
    the workspace requirement applies should use ``intent_ack_continuation_mode``
    directly (``"codex_only"`` ⇒ require_workspace=True, ``"all"`` ⇒ False).
    """
    return intent_ack_continuation_mode(agent) != "off"




def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:
    """Copy provider-facing reasoning fields onto an API replay message.

    Forwarder — the strip-vs-repad POLICY is owned by
    ``agent.message_sanitization.apply_reasoning_content_policy`` (audit F4);
    this only supplies the agent's cached provider-direction flag.
    """
    from agent.message_sanitization import apply_reasoning_content_policy

    apply_reasoning_content_policy(
        source_msg, api_msg, agent._needs_thinking_reasoning_pad()
    )


def reapply_reasoning_echo_for_provider(agent, api_messages: list) -> int:
    """Re-pad (or strip) assistant turns' reasoning_content for the active provider.

    ``api_messages`` is built once, before the retry loop, while the *primary*
    provider is active.  A mid-conversation fallback can then switch providers,
    so the reasoning fields baked into ``api_messages`` are shaped for the
    *prior* provider and must be reconciled against the *current* one:

    * Switching TO a require-side provider (DeepSeek / Kimi / MiMo thinking
      mode): assistant turns built when the prior provider did NOT need the
      echo-back go out without ``reasoning_content`` and the new provider
      rejects them with HTTP 400 ("The reasoning_content in the thinking mode
      must be passed back").  Re-apply the pad.

    * Switching TO a strict provider that rejects the field (Mistral,
      Cerebras, Groq, SambaNova, …): assistant turns built under a reasoning
      primary carry a ``reasoning_content`` pad (often a single space ``" "``),
      and the strict provider rejects it with HTTP 400/422 ("Extra inputs are
      not permitted").  Strip the field.  This is the exact cross-provider
      fallback bug from #45655 — a DeepSeek primary pads history with ``" "``,
      the request falls back to Mistral, and Mistral 422s on the stale pad.

    Calling this immediately before building the request kwargs reconciles the
    fields against the *current* provider.  It is idempotent and safe to call
    every iteration; it covers every fallback path.

    Returns the number of assistant turns whose reasoning_content was added or
    removed.
    """
    from agent.message_sanitization import reapply_reasoning_echo

    return reapply_reasoning_echo(
        api_messages, agent._needs_thinking_reasoning_pad()
    )


def _iter_httpx_pool_objects(http_client: Any):
    """Yield httpcore pool objects reachable from an httpx client.

    Hermes' keepalive client (#10324 / ``_build_keepalive_http_client``) and
    any ``HTTP(S)_PROXY`` configuration put live connections on *mounted*
    transports (``client._mounts``), not only on the default
    ``client._transport``. Walking the default transport alone makes
    ``force_close_tcp_sockets`` return 0 while a stream is still mid-recv —
    the interrupt logs success and the provider keeps burning the slot
    (#72975).
    """
    seen_pools: set[int] = set()

    def _emit(pool: Any):
        if pool is None:
            return
        marker = id(pool)
        if marker in seen_pools:
            return
        seen_pools.add(marker)
        yield pool

    def _pools_for_transport(transport: Any):
        if transport is None:
            return
        # Normal httpx.HTTPTransport / HTTPProxy-as-transport: connections
        # live under ``_pool``. HTTPProxy itself *is* a ConnectionPool and
        # may be mounted directly — then ``_connections`` is on the
        # transport.
        pool = getattr(transport, "_pool", None)
        if pool is not None:
            yield from _emit(pool)
            return
        if getattr(transport, "_connections", None) is not None:
            yield from _emit(transport)

    try:
        yield from _pools_for_transport(getattr(http_client, "_transport", None))
        mounts = getattr(http_client, "_mounts", None) or {}
        for _pattern, mounted in list(mounts.items()):
            yield from _pools_for_transport(mounted)
    except Exception:
        return


def _connection_candidates(conn: Any):
    """Walk nested ``_connection`` wrappers (proxy tunnel → HTTP11/2)."""
    seen: set[int] = set()
    stack = [conn]
    while stack:
        candidate = stack.pop()
        if candidate is None:
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        yield candidate
        inner = getattr(candidate, "_connection", None)
        if inner is not None and id(inner) not in seen:
            stack.append(inner)


def _iter_pool_sockets(client: Any):
    """Yield raw sockets reachable from an OpenAI/httpx client pool.

    httpcore 1.x stores the concrete HTTP11/HTTP2 connection under
    ``conn._connection``; older versions exposed stream attributes directly
    on the pool entry. Proxy tunnels wrap another layer
    (``TunnelHTTPConnection`` / ``ForwardHTTPConnection``). Keep the
    traversal defensive because these are private transport internals and
    vary across httpx/httpcore releases.

    Also walks ``httpx`` mount transports — see ``_iter_httpx_pool_objects``
    — and in-flight httpcore ``PoolRequest.connection`` objects, which stay
    reachable even when ``_connections`` is empty during checkout (#85252).
    """
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            # Some SDK wrappers *are* the httpx client (or expose the pool
            # directly). Fall through so mount-aware discovery still runs.
            http_client = client
        pools = list(_iter_httpx_pool_objects(http_client))
    except Exception:
        return

    if not pools:
        return

    seen: set[int] = set()
    for pool in pools:
        # Empty-list is falsy: use ``is None`` so an empty ``_connections``
        # still lets us walk in-flight ``_requests`` rather than skipping
        # the pool entirely.
        raw_conns = getattr(pool, "_connections", None)
        if raw_conns is None:
            raw_conns = getattr(pool, "_pool", None)
        connections = list(raw_conns or [])
        for pool_req in list(getattr(pool, "_requests", None) or []):
            conn = getattr(pool_req, "connection", None)
            if conn is not None:
                connections.append(conn)
        for conn in connections:
            for candidate in _connection_candidates(conn):
                stream = (
                    getattr(candidate, "_network_stream", None)
                    or getattr(candidate, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    get_extra_info = getattr(stream, "get_extra_info", None)
                    if callable(get_extra_info):
                        try:
                            sock = get_extra_info("socket")
                        except Exception:
                            sock = None
                if sock is None:
                    wrapped = getattr(stream, "stream", None)
                    if wrapped is not None:
                        sock = getattr(wrapped, "_sock", None)
                if sock is None:
                    # anyio-backed streams expose the raw socket through
                    # SocketAttribute.raw_socket when available.
                    wrapped = getattr(stream, "_stream", None)
                    extra = getattr(wrapped, "extra", None)
                    if callable(extra):
                        try:
                            from anyio.abc import SocketAttribute
                            sock = extra(SocketAttribute.raw_socket)
                        except Exception:
                            sock = None
                if sock is None:
                    continue
                marker = id(sock)
                if marker in seen:
                    continue
                seen.add(marker)
                yield sock


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
        dead_count = 0
        for sock in _iter_pool_sockets(client):
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
        if not message and isinstance(payload.get("error"), str):
            # xAI uses a top-level string ``error`` beside a structured
            # ``code`` (for example personal-team-blocked:spending-limit).
            message = payload.get("error")
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
            delay_match = re.search(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", message, re.IGNORECASE)
            if delay_match:
                value = float(delay_match.group(1))
                seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                context["reset_at"] = time.time() + seconds
            else:
                resets_in_match = re.search(
                    r"resets?\s+in\s+"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?",
                    message,
                    re.IGNORECASE,
                )
                if resets_in_match and any(resets_in_match.groups()):
                    hours = float(resets_in_match.group(1) or 0)
                    minutes = float(resets_in_match.group(2) or 0)
                    seconds = float(resets_in_match.group(3) or 0)
                    context["reset_at"] = time.time() + (hours * 3600) + (minutes * 60) + seconds
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
    marker = format_steer_marker(steer_text)
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
    """Abort in-flight TCP I/O by shutting down sockets WITHOUT closing FDs.

    When a provider drops a connection mid-stream — or the user issues an
    interrupt — we want to unblock httpx's reader/writer immediately rather
    than waiting for the kernel's per-connection timeout. ``shutdown(SHUT_RDWR)``
    achieves that: it sends FIN, breaks any pending ``recv``/``send`` with EOF
    or ``EPIPE``, but does NOT release the file descriptor.

    Historically this helper also called ``socket.close()`` so the FD got
    released immediately, but that's unsafe when (as is the case for both the
    interrupt-abort path and stale-call kill path) the helper runs on a
    different thread than the one driving the request:

      * The Python ``socket.socket`` we close here is the SAME object held by
        httpx's pool, so closing it via Python sets its ``_fd`` to -1 and
        future operations on that Python object fail safely.
      * BUT the SSL wrapper (``ssl.SSLSocket``'s underlying OpenSSL ``BIO``)
        caches the raw integer FD. Once ``os.close(fd)`` runs, the kernel may
        immediately recycle that integer to the next ``open()`` call — e.g.
        the kanban dispatcher opening ``kanban.db``.
      * The owning worker thread then unwinds httpx, the SSL layer flushes a
        pending TLS record, and the encrypted bytes get written into the
        wrong file (issue #29507: 24-byte TLS application-data record
        clobbering SQLite header bytes 5..28).

    The fix is to let the owning thread own the close. ``shutdown()`` from any
    thread is FD-safe; ``close()`` is not. The httpx connection's own close
    path — which runs from the worker thread when it unwinds — will release
    the FD via the same ``socket.socket`` object, and because Python's socket
    close atomically swaps ``_fd`` to -1 *before* issuing ``os.close``, there
    is no FD-aliasing window when only one thread closes.

    Returns the number of sockets shut down. (Field kept as
    ``tcp_force_closed=N`` in the log line for backwards-compatible parsing.)
    """
    import socket as _socket

    shutdown_count = 0
    try:
        for sock in _iter_pool_sockets(client):
            try:
                # Clear a blocking timeout first so a hung SSL_read on the
                # owner thread notices the shutdown. Some stacks ignore
                # SHUT_RDWR alone while recv is blocked with timeout=None
                # (#85252). Still no close() — that is the #29507 race.
                settimeout = getattr(sock, "settimeout", None)
                if callable(settimeout):
                    try:
                        settimeout(0)
                    except OSError:
                        pass
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                # Already shut down / not connected / FD invalid — all benign.
                pass
            # IMPORTANT (#29507): do NOT call sock.close() here. See docstring.
            shutdown_count += 1
    except Exception as exc:
        _ra().logger.debug("Force-close TCP sockets sweep error: %s", exc)
    return shutdown_count



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
    "prompt_caching_disabled_from_config",
    "blank_cache_policy_stub",
    "plan_cache_sections_for_destination",
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
    "_iter_pool_sockets",
    "force_close_tcp_sockets",
]
