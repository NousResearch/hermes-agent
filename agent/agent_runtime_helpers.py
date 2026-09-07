"""Assorted AIAgent runtime helpers (message repair/sanitization, credential recovery, primary
runtime restore, prompt-cache policy, client construction, model switching, tool invocation).
Each function takes the parent ``AIAgent`` as ``agent`` except the stateless message helpers.
``_ra()`` resolves ``run_agent`` lazily so tests patching ``run_agent.X`` keep intercepting.
"""

from __future__ import annotations
import contextlib
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
    _FULL_ARGS_LOG_BOUND, coalesce_tool_call_id, tool_call_id_variants, tool_result_id_variants
)
from agent.prompt_builder import format_steer_marker
from agent.tool_dispatch_helpers import _trajectory_normalize_msg, make_tool_result_message
from agent.trajectory import convert_scratchpad_to_think
from agent.credential_pool import (
    STATUS_EXHAUSTED, credential_pool_matches_provider, resolve_runtime_pool_key
)
from agent.error_classifier import FailoverReason
from agent.turn_context import drop_stale_api_content
from utils import base_url_host_matches, base_url_hostname, env_var_enabled, atomic_json_write
logger = logging.getLogger(__name__)

# Cap same-entry OAuth refreshes on a persistent auth failure, else a single-entry pool re-mints forever.
_MAX_AUTH_REFRESH_ATTEMPTS = 2
_REASONING_TAG_NAMES = ("think", "thinking", "reasoning", "REASONING_SCRATCHPAD", "thought")
_TOOL_CALL_TAG_NAMES = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")
_REASONING_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE) for name in _REASONING_TAG_NAMES
)
_TOOL_CALL_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
)

# Named <function name=...> blocks; boundary- and name-gated (see _THINK_STRIP_PATTERNS note).
_NAMED_FUNCTION_BLOCK_PATTERN = re.compile(
    r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
    r'<function\b[^>]*\bname\s*=[^>]*>'
    r'(?:(?:(?!</function>).)*)</function>', re.DOTALL | re.IGNORECASE,
)
_UNTERMINATED_REASONING_BLOCK_PATTERN = re.compile(
    rf'(?:^|\n)[ \t]*<(?:{"|".join(_REASONING_TAG_NAMES)})\b[^>]*>.*$', re.DOTALL | re.IGNORECASE
)
_ORPHAN_REASONING_TAG_PATTERN = re.compile(
    rf'</?(?:{"|".join(_REASONING_TAG_NAMES)})>\s*', re.IGNORECASE
)
_STRAY_TOOL_CALL_CLOSER_PATTERN = re.compile(
    rf'</(?:{"|".join(_TOOL_CALL_TAG_NAMES)}|function)>\s*', re.IGNORECASE
)

# A tool-call opener with no closer, or GLM-style argument markup
# (<arg_key>/<arg_value>) outside any closed block, means the stream was
# cut mid-serialization of a text-channel tool call (#101899). The call
# can't be recovered; strip from the block-boundary opener (or the line
# holding the first stray argument tag) to the end of the text.
_UNTERMINATED_TOOL_CALL_PATTERN = re.compile(
    rf'(?:^|\n)[ \t]*<(?:{"|".join(_TOOL_CALL_TAG_NAMES)})\b[^>]*>.*$'
    r'|(?:^|\n)[^\n<]*</?arg_(?:key|value)\b.*$',
    re.DOTALL | re.IGNORECASE,
)


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent
    return run_agent


AGENT_RUNTIME_POST_HOOK_TOOL_NAMES = frozenset({
    "todo_list", "session_search", "memory", "clarify", "read_terminal", "desktop_preview",
    "drive_preview", "annotate_preview", "read_window_below", "setup_mcp", "gui_tour", "delegate_task",
})

_TRAJECTORY_SYSTEM_PROMPT = (
    "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
    "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
    "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
    "into functions. After calling & executing the functions, you will be provided with function results within "
    "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
    "<tools>\n{tools}\n</tools>\n"
    "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
    "{{'title': 'FunctionCall', 'type': 'object', 'properties': {{'name': {{'title': 'Name', 'type': 'string'}}, "
    "'arguments': {{'title': 'Arguments', 'type': 'object'}}}}, 'required': ['name', 'arguments']}}\n"
    "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
    "Example:\n<tool_call>\n{{'name': <function-name>,'arguments': <args-dict>}}\n</tool_call>"
)


def _trajectory_gpt_prefix(msg: Dict[str, Any]) -> str:
    """Leading ``<think>`` block from native reasoning tokens, if any."""
    if msg.get("reasoning") and msg["reasoning"].strip():
        return f"<think>\n{msg['reasoning']}\n</think>\n"
    return ""


def _with_think_block(content: str) -> str:
    """Every gpt turn gets a <think> block (empty if none) for a consistent training format."""
    return content if "<think>" in content else "<think>\n</think>\n" + content


def _trajectory_tool_call_turn(msg: Dict[str, Any]) -> str:
    content = _trajectory_gpt_prefix(msg)
    if msg.get("content") and msg["content"].strip():
        # <REASONING_SCRATCHPAD> -> <think> (model reasons via XML when native thinking is off)
        content += convert_scratchpad_to_think(msg["content"]) + "\n"
    for tool_call in msg["tool_calls"]:
        if not tool_call or not isinstance(tool_call, dict):
            continue
        raw_args = tool_call["function"]["arguments"]
        # Arguments were validated during conversation; degrade to {} rather than abort.
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            logger.warning("Unexpected invalid JSON in trajectory conversion: %s", raw_args[:100])
            arguments = {}
        tool_call_json = {"name": tool_call["function"]["name"], "arguments": arguments}
        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
    return _with_think_block(content).rstrip()


def _trajectory_tool_responses(msg: Dict[str, Any], messages: List[Dict[str, Any]], start: int) -> Tuple[List[str], int]:
    """Collect the ``<tool_response>`` blocks for the tool run starting at ``start``; returns ``(blocks, next_index)``."""
    tool_responses = []
    j = start
    while j < len(messages) and messages[j]["role"] == "tool":
        tool_msg = messages[j]
        tool_content = tool_msg["content"]
        try:  # pretty-print tool content if it looks like JSON
            if tool_content.strip().startswith(("{", "[")):
                tool_content = json.loads(tool_content)
        except (json.JSONDecodeError, AttributeError):
            pass
        tool_index = len(tool_responses)
        tool_name = (
            msg["tool_calls"][tool_index]["function"]["name"]
            if tool_index < len(msg["tool_calls"])
            else "unknown"
        )
        payload = json.dumps(
            {"tool_call_id": tool_msg.get("tool_call_id", ""), "name": tool_name, "content": tool_content},
            ensure_ascii=False,
        )
        tool_responses.append(f"<tool_response>\n{payload}\n</tool_response>")
        j += 1
    return tool_responses, j


def convert_to_trajectory_format(agent, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
    """Convert internal message history to trajectory format for saving."""
    # Trajectories are text-only: swap image-bearing tool messages for their text_summary so ~1MB
    # base64 blobs are not embedded.
    messages = [_trajectory_normalize_msg(m) for m in messages]
    trajectory = [
        {"from": "system", "value": _TRAJECTORY_SYSTEM_PROMPT.format(tools=agent._format_tools_for_system_message())},
        {"from": "human", "value": user_query},
    ]
    # Skip messages[0] (already added). Prefill is injected at API-call time only, so no offset adjustment is needed.
    i = 1
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            if msg.get("tool_calls"):
                trajectory.append({"from": "gpt", "value": _trajectory_tool_call_turn(msg)})
                tool_responses, j = _trajectory_tool_responses(msg, messages, i + 1)
                if tool_responses:
                    trajectory.append({"from": "tool", "value": "\n".join(tool_responses)})
                    i = j - 1  # skip the tool messages just processed
            else:
                content = _trajectory_gpt_prefix(msg) + convert_scratchpad_to_think(msg["content"] or "")
                trajectory.append({"from": "gpt", "value": _with_think_block(content).strip()})
        elif msg["role"] == "user":
            trajectory.append({"from": "human", "value": msg["content"]})
        i += 1
    return trajectory


def _prepend_corruption_marker(tool_msg: dict, marker: str) -> None:
    existing = tool_msg.get("content")
    if isinstance(existing, str) and existing.startswith(marker):
        return
    if not isinstance(existing, (str, type(None))):
        try:
            existing = json.dumps(existing)
        except TypeError:
            existing = str(existing)
    tool_msg["content"] = f"{marker}\n{existing}" if existing else marker


def _find_tool_result(messages: list, start: int, tool_call: dict) -> Optional[dict]:
    """The tool result answering ``tool_call`` in the run starting at ``start``, if any."""
    for candidate in messages[start:]:
        if not isinstance(candidate, dict) or candidate.get("role") != "tool":
            return None
        if tool_result_id_variants(candidate.get("tool_call_id")) & tool_call_id_variants(tool_call):
            return candidate
    return None


def _cursor_skip_prefix(messages: list, cursor: Optional[dict]) -> int:
    """Length of the ``is``-identical prefix already validated on the previous call."""
    prev_prefix = cursor.get("prefix") if cursor is not None else None
    start = 0
    if isinstance(prev_prefix, list):
        while start < min(len(prev_prefix), len(messages)) and messages[start] is prev_prefix[start]:
            start += 1
    return start


def sanitize_tool_call_arguments(
    messages: list, *, logger=None, session_id: str = None, cursor: Optional[dict] = None
) -> int:
    """Repair corrupted assistant tool-call argument JSON in-place.
    ``cursor["prefix"]`` holds strong refs (not ``id()``: address reuse aliases) to the
    messages validated last call; the ``is``-identical prefix is skipped. Safe because only
    the surrogate sanitizers mutate live dicts; every other path replaces dicts, breaking identity.

    Safety argument for skipping: a message in the matched prefix was fully scanned before — every tool_call
    argument was either already valid JSON or was rewritten to ``"{}"`` (valid). The only code paths that
    mutate ``function["arguments"]`` on live history dicts between calls are the surrogate / non-ASCII
    sanitizers, which substitute characters *inside* JSON string values and cannot invalidate JSON syntax.
    Compression, repair, undo, and steer paths replace or reorder message dicts, which breaks the identity
    match and forces a re-scan. Holding strong references (the objects themselves, not ``id()``s) makes
    address reuse aliasing (#50372-style) impossible.
    """
    log = logger or logging.getLogger(__name__)
    if not isinstance(messages, list):
        return 0
    repaired = 0
    marker = _ra().AIAgent._TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER
    message_index = _cursor_skip_prefix(messages, cursor)
    while message_index < len(messages):
        msg = messages[message_index]
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) and msg.get("role") == "assistant" else None
        if not isinstance(tool_calls, list) or not tool_calls:
            message_index += 1
            continue
        insert_at = message_index + 1
        for tool_call in tool_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                function["arguments"] = "{}"
                continue
            if not isinstance(arguments, str):
                continue
            with contextlib.suppress(json.JSONDecodeError):
                json.loads(arguments)
                continue
            # Canonical ``call_id || id`` precedence so scan and stub share the id the pipeline
            # uses; bare ``id`` misses Codex call_id results and orphans a stub.
            # Keying on bare ``id`` here would fail to find a result built with ``call_id`` (Codex Responses
            # format) and insert a duplicate stub that itself becomes an orphan (#58168).
            tool_call_id = _ra().AIAgent._get_tool_call_id_static(tool_call) or None
            function_name = function.get("name", "?")
            # Log the FULL (bounded) argument string: we are about to overwrite the only copy, which
            # may hold real user content from a truncated write_file/patch.
            log.warning(
                "Corrupted tool_call arguments repaired before request "
                "(session=%s, message_index=%s, tool_call_id=%s, function=%s, "
                "original_arguments=%r)", session_id or "-", message_index, tool_call_id or "-",
                function_name, arguments[:_FULL_ARGS_LOG_BOUND],
            )
            function["arguments"] = "{}"
            existing_tool_msg = _find_tool_result(messages, message_index + 1, tool_call)
            if existing_tool_msg is None:
                messages.insert(
                    insert_at,
                    make_tool_result_message(function_name if function_name != "?" else "", marker, tool_call_id),
                )
                insert_at += 1
            else:
                _prepend_corruption_marker(existing_tool_msg, marker)
            repaired += 1
        message_index += 1
    if cursor is not None:
        # Strong refs to the objects validated this call; any divergence (compression, undo, repair,
        # steer) forces a re-scan from that index.
        cursor["prefix"] = messages[:]
    return repaired


# Session-scoped in-flight registry for note_turn_start: the gateway caches agents per routing key
# while the transcript is keyed by session_id, so two agent objects can run concurrent turns on one
# session unseen by per-agent state.
_INFLIGHT_TURNS_BY_SESSION: Dict[str, Tuple[str, float]] = {}
_INFLIGHT_TURNS_LOCK = threading.Lock()


def note_turn_start(agent, turn_id: str):
    """Tripwire: warn when a turn starts while a previous turn of the same agent or session
    (on another agent object) has not finished its persist. Does not prevent the overlap; it
    names both turn ids so the dispatch route that bypassed the busy guard is findable in logs.
    Returns the previous in-flight turn_id on overlap, else None; takes the slot either way."""
    prev = getattr(agent, "_inflight_turn_id", None)
    prev_started = getattr(agent, "_inflight_turn_started", 0.0)
    agent._inflight_turn_id = turn_id
    agent._inflight_turn_started = time.time()
    overlap = None
    if prev and prev != turn_id:
        logger.warning(
            "turn %s starting while turn %s (started %.0fs ago) has not "
            "completed its turn-end persist (session=%s) — concurrent turns "
            "on one session; transcript writes may interleave", turn_id, prev,
            time.time() - prev_started if prev_started else -1.0, getattr(agent, "session_id", None) or "-",
        )
        overlap = prev
    # Cross-agent leg: same session_id in flight under another agent object (busy guard is keyed by
    # routing key and cannot see it). Persist-disabled forks share the parent's session_id but never
    # write, so they must not register or pop here (note_turn_persisted skips them symmetrically).
    session_id = getattr(agent, "session_id", None)
    if session_id and not getattr(agent, "_persist_disabled", False):
        now = time.time()
        with _INFLIGHT_TURNS_LOCK:
            entry = _INFLIGHT_TURNS_BY_SESSION.get(session_id)
            _INFLIGHT_TURNS_BY_SESSION[session_id] = (turn_id, now)
        # Record the session id registered under: compression can rotate agent.session_id mid-turn
        # and persist must pop the slot actually held.
        agent._inflight_turn_session_id = session_id
        if entry and entry[0] not in (turn_id, prev):
            logger.warning(
                "turn %s starting while turn %s (started %.0fs ago) is still "
                "in flight on session %s under a different agent object — "
                "two routing keys are mapped to one session_id; concurrent "
                "turns on one session; transcript writes may interleave", turn_id, entry[0],
                now - entry[1] if entry[1] else -1.0, session_id,
            )
            overlap = overlap or entry[0]
    return overlap


def note_turn_persisted(agent):
    """Clear the in-flight marker at turn-end persist (see note_turn_start). Unconditional by
    design: on a real overlap the first persist clears the second slot, so the tripwire
    under-reports rather than double-reports."""
    agent._inflight_turn_id = None
    # Persist-disabled forks never registered a slot; popping here would steal the live parent
    # turn's slot (symmetric with note_turn_start).
    if not getattr(agent, "_persist_disabled", False):
        session_id = getattr(agent, "_inflight_turn_session_id", None) or getattr(agent, "session_id", None)
        if session_id:
            with _INFLIGHT_TURNS_LOCK:
                _INFLIGHT_TURNS_BY_SESSION.pop(session_id, None)
    agent._inflight_turn_session_id = None


def _is_codex_interim(m: Dict) -> bool:
    """Codex Responses interim turn: carries its own continuation state, replayed verbatim."""
    return bool(
        m.get("codex_reasoning_items")
        or m.get("codex_message_items")
        or m.get("finish_reason") == "incomplete"
    )


def _merge_assistant_into(prev: Dict, msg: Dict) -> None:
    """Fold a consecutive assistant ``msg`` into ``prev`` (union tool_calls, concat text)."""
    prev_calls = list(prev.get("tool_calls") or [])
    new_calls = list(msg.get("tool_calls") or [])
    if new_calls:
        prev["tool_calls"] = prev_calls + new_calls
    elif prev_calls:
        prev["tool_calls"] = prev_calls
    else:
        # Drop a stale ``tool_calls: []`` at the source: strict providers (DeepSeek v4, Kimi) 400 on
        # it and it persists into replayed history.
        # Neither turn carries tool calls, but the surviving turn may still carry a stale ``tool_calls: []``
        # from the earlier message. An empty array is semantically "no tool calls", yet strict
        # OpenAI-compatible providers (DeepSeek v4, Moonshot/Kimi) reject it with HTTP 400 ("Invalid
        # 'messages[N].tool_calls': empty array..."). Drop the key HERE, at the source:
        # ``sanitize_api_messages`` only fixes the per-call wire copy, so a ``[]`` left on the repaired turn
        # survives in the live/persisted trajectory returned to callers (gateway/WebUI transcripts, session
        # resume, subagents, cron) and is replayed on the next turn — which is how #58755 kept reproducing
        # after the chokepoint fix (#77921). Popping is non-destructive: an empty array carries no
        # information.
        prev.pop("tool_calls", None)
    # Concatenate plain-text content only; leave multimodal (list) content alone.
    prev_content = prev.get("content")
    new_content = msg.get("content")
    content_rewritten = False
    if isinstance(prev_content, str) and isinstance(new_content, str):
        joined = "\n".join(p for p in (prev_content.strip(), new_content.strip()) if p)
        prev["content"] = joined
        # A falsy new_content leaves ``joined`` == prev_content; that is not a rewrite.
        # "") strips to nothing and ``joined`` collapses back to ``prev_content`` unchanged -- that must NOT
        # count as a rewrite (wz-heng, #78063 review).
        content_rewritten = joined != prev_content
    elif not prev_content and new_content is not None:
        prev["content"] = new_content
        content_rewritten = new_content != prev_content
    # Carry reasoning_content from the later turn only if the earlier lacks it (strict thinking
    # providers need one on the merged tool-call turn).
    if not prev.get("reasoning_content") and msg.get("reasoning_content"):
        prev["reasoning_content"] = msg["reasoning_content"]
    # A stale ``api_content`` sidecar overrides ``content`` at API-build time and would replay
    # pre-merge bytes; drop it only when content actually changed.
    # ``prev`` may carry an ``api_content`` sidecar (the exact bytes previously sent to the API, e.g. a
    # sanitize-divergence stamp — see ``_flush_messages_to_session_db``) from BEFORE this merge. The sidecar
    # takes priority over ``content`` at API-build time (``conversation_loop``'s ``api_messages`` build
    # substitutes it back in for role ``assistant``), so leaving it in place while ``prev["content"]``
    # changes would silently replay the pre-merge bytes and discard everything this merge just concatenated
    # on — the same stale-field-survives-the-merge shape as the ``tool_calls`` gap above, just for a
    # different field. Only drop it when the merge actually changed the resulting value (e.g. the later
    # turn's content is ``None``, or either side is multimodal/list — both branches skip the reassignment
    # and ``prev["content"]`` is untouched; a falsy ``new_content`` that strips to nothing also leaves
    # ``joined`` equal to the original ``prev_content``): in those cases the sidecar is still the exact
    # bytes previously sent for the UNCHANGED content, and dropping it would break the prompt-cache replay
    # invariant for no reason (wz-heng, #78063 review).
    if content_rewritten:
        drop_stale_api_content(prev)


def _merge_consecutive_assistants(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 0: merge consecutive assistant turns (codex interims exempt)."""
    repairs = 0
    collapsed: List[Dict] = []
    for msg in messages:
        prev = collapsed[-1] if collapsed and isinstance(collapsed[-1], dict) else None
        if (
            prev is not None and prev.get("role") == "assistant"
            and isinstance(msg, dict) and msg.get("role") == "assistant"
            and not _is_codex_interim(msg) and not _is_codex_interim(prev)
        ):
            # A provisional verification candidate is superseded, not unioned.
            if prev.get("finish_reason") in {"verification_required", "verify_hook_continue"}:
                collapsed[-1] = msg
            else:
                _merge_assistant_into(prev, msg)
            repairs += 1
            continue
        collapsed.append(msg)
    return collapsed, repairs


def _drop_stray_tool_results(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 1: drop tool results not following a known assistant tool call. Consumes the whole
    alias group (call_id/id/response_item_id/composite) so a duplicate keyed on a sibling
    alias is not replayed to strict providers."""
    repairs = 0
    known_tool_ids: Dict[str, int] = {}  # alias -> group id; reset by assistant/user turns
    # Pass 1: drop stray tool messages that don't follow a known assistant tool call. A Responses call can
    # have several equivalent spellings (call_id, id, response_item_id, or a composite ``call|item`` id), so
    # consume the whole alias group when one spelling is matched. Alias expansion lives in
    # ``agent.message_sanitization.tool_call_id_variants`` / ``tool_result_id_variants`` (single policy
    # owner) — which also handles SDK tool_call objects, preserving the #91768 dict-or-object tolerance.
    matched_tool_groups: set = set()
    next_tool_group = 0
    filtered: List[Dict] = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role in ("assistant", "user"):
            # An assistant turn starts a new tool-result run; a user turn closes it (later tool
            # messages are orphans).
            known_tool_ids = {}
            matched_tool_groups = set()
            for tc in (msg.get("tool_calls") or []) if role == "assistant" else ():
                variants = tool_call_id_variants(tc)
                if variants:
                    for tc_id in variants:
                        known_tool_ids.setdefault(tc_id, next_tool_group)
                    next_tool_group += 1
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            candidate_groups = {
                known_tool_ids[tc_id] for tc_id in result_variants
                if tc_id in known_tool_ids and known_tool_ids[tc_id] not in matched_tool_groups
            }
            if result_variants and not candidate_groups:
                repairs += 1
                continue
            if candidate_groups:
                matched_tool_groups.add(min(candidate_groups))
        filtered.append(msg)
    return filtered, repairs


def _prune_unanswered_tool_calls(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 2: prune tool_calls not answered in the IMMEDIATELY following tool run (a displaced
    result masks the per-call stub pass and strict providers 400). Payload-empty turns are
    dropped; codex interims exempt."""
    repairs = 0
    pruned: List[Dict] = []
    for i, msg in enumerate(messages):
        if not (
            isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls")
            and not _is_codex_interim(msg)
        ):
            pruned.append(msg)
            continue
        answered: set = set()
        for follower in messages[i + 1:]:
            if not (isinstance(follower, dict) and follower.get("role") == "tool"):
                break
            tid = (follower.get("tool_call_id") or "").strip()
            if tid:
                answered.update(tool_result_id_variants(tid))
        kept_calls = [tc for tc in msg["tool_calls"] if tool_call_id_variants(tc) & answered]
        if len(kept_calls) != len(msg["tool_calls"]):
            repairs += 1
            if not kept_calls and not _msg_has_payload({k: v for k, v in msg.items() if k != "tool_calls"}):
                # Pruned calls were the only payload; drop the turn (empty assistant messages 400).
                continue
            if kept_calls:
                msg["tool_calls"] = kept_calls
            else:
                msg.pop("tool_calls", None)
        pruned.append(msg)
    return pruned, repairs


def _merge_consecutive_users(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 3: merge consecutive plain-text user messages (no user input lost)."""
    from agent.context_compressor import split_user_originated_turn

    repairs = 0
    merged: List[Dict] = []
    for msg in messages:
        prev = merged[-1] if merged and isinstance(merged[-1], dict) else None
        if (
            prev is not None and prev.get("role") == "user"
            and isinstance(msg, dict) and msg.get("role") == "user"
            # A summary carrier followed by a new user row is a deliberate durable shape after
            # retry/rewind; never mutate the persisted carrier (sanitizers merge copies later).
            and split_user_originated_turn(prev)[0] is None
            # Only merge plain-text content; leave multimodal (list) content alone.
            and isinstance(prev.get("content", ""), str) and isinstance(msg.get("content", ""), str)
        ):
            prev_content, new_content = prev.get("content", ""), msg.get("content", "")
            prev["content"] = (
                (prev_content + "\n\n" + new_content) if prev_content and new_content else (prev_content or new_content)
            )
            # Merged content invalidates the api_content sidecar; drop it so replay cannot use stale bytes.
            drop_stale_api_content(prev)
            repairs += 1
            continue
        merged.append(msg)
    return merged, repairs


_SEQUENCE_REPAIR_PASSES = (
    _merge_consecutive_assistants, _drop_stray_tool_results, _prune_unanswered_tool_calls,
    _merge_consecutive_users,
)


def repair_message_sequence(agent, messages: List[Dict]) -> int:
    """Collapse malformed role-alternation left in the live history; returns repair count.
    Providers require strict alternation after the system message (violations: silent empty
    responses or 400s); this is the pre-call belt for host-fed, resumed or replayed histories.
    Passes in order: merge consecutive assistant turns (BEFORE orphan detection so the merged
    tool_call-id union is known); drop stray tool results; prune unanswered tool_calls; merge
    consecutive user turns. A user turn directly after an assistant turn is valid and left alone.
    """
    if not messages:
        return 0
    repairs = 0
    current = messages
    for repair_pass in _SEQUENCE_REPAIR_PASSES:
        current, made = repair_pass(current)
        repairs += made
    if repairs > 0:
        # Rewrite in place so persistence/return value/DB flush see the repaired sequence.
        messages[:] = current
    return repairs


def repair_message_sequence_with_cursor(agent, messages: List[Dict]) -> int:
    """Run :func:`repair_message_sequence` and keep ``_last_flushed_db_idx`` consistent. Repair
    shrinks the list in place; counting identity-preserved survivors of the flushed prefix gives
    the exact new cursor, whereas a ``min()`` clamp would skip unflushed rows (used only without a snapshot)."""
    flush_cursor = getattr(agent, "_last_flushed_db_idx", None)
    flushed_ids = {id(m) for m in messages[:flush_cursor]} if isinstance(flush_cursor, int) and flush_cursor > 0 else None
    repairs = repair_message_sequence(agent, messages)
    if repairs > 0 and hasattr(agent, "_last_flushed_db_idx"):
        if flushed_ids is not None:
            agent._last_flushed_db_idx = sum(1 for m in messages if id(m) in flushed_ids)
        else:
            agent._last_flushed_db_idx = min(agent._last_flushed_db_idx, len(messages))
    return repairs


def _flatten_content_text(content: Any) -> str:
    """Flatten list/dict content (e.g. Anthropic-via-OpenRouter block lists) to text: a raw list
    hitting ``re.sub`` raises TypeError and the loop retries forever. Thinking/reasoning blocks
    are dropped outright; their text key varies per provider."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text")
            for part in content
            if isinstance(part, str) or (
                isinstance(part, dict)
                and str(part.get("type") or "").strip().lower() not in {"thinking", "reasoning", "redacted_thinking"}
                and isinstance(part.get("text"), str) and part.get("text")
            )
        )
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


# Order matters: closed pairs first (case-insensitive so mixed-case tags don't fall through to the
# unterminated pass and eat trailing content), then tool-call XML blocks, the boundary+name-gated
# <function> block, the unterminated reasoning block, stray orphan reasoning tags, and finally stray
# tool-call CLOSERS only (bare/unterminated <function> is kept: a truncated streaming tail may still
# be valuable, matching OpenClaw's asymmetry).
_THINK_STRIP_PATTERNS = (
    *_REASONING_BLOCK_PATTERNS, *_TOOL_CALL_BLOCK_PATTERNS, _NAMED_FUNCTION_BLOCK_PATTERN,
    _UNTERMINATED_REASONING_BLOCK_PATTERN, _ORPHAN_REASONING_TAG_PATTERN,
    _STRAY_TOOL_CALL_CLOSER_PATTERN, _UNTERMINATED_TOOL_CALL_PATTERN,
)


def strip_think_blocks(agent, content: str) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text: closed tag
    pairs, unterminated open tags at a block boundary (mirrors ``gateway/stream_consumer.py``),
    stray orphan tags (all case-insensitive variants), and standalone tool-call XML blocks some
    open models emit; ``<function>`` is boundary- and ``name=``-gated so prose mentions survive."""
    content = _flatten_content_text(content) if content else ""
    for pattern in _THINK_STRIP_PATTERNS if content else ():
        content = pattern.sub('', content)
    return content


def sync_credential_pool_entry_id(agent) -> None:
    """Rebind ``agent._credential_pool_entry_id`` from the current pool + key. OAuth refreshes
    can replace the token before recovery runs, so the key alone cannot attribute a failure;
    the stable entry ID can. Cleared when no pool is bound."""
    pool = getattr(agent, "_credential_pool", None)
    try:
        agent._credential_pool_entry_id = (
            pool.entry_id_for_api_key(getattr(agent, "api_key", None)) if pool is not None else None
        )
    except Exception:
        agent._credential_pool_entry_id = None


_STATUS_TO_FAILOVER_REASON = {
    402: FailoverReason.billing, 429: FailoverReason.rate_limit, 401: FailoverReason.auth,
    403: FailoverReason.auth,
}
_USAGE_LIMIT_REASON_TOKENS = ("usage_limit_reached", "gousagelimit")
_USAGE_LIMIT_MESSAGE_TOKENS = ("usage limit reached", "usage limit has been reached")


def _failed_credential_identity(agent, pool) -> Tuple[Optional[str], Optional[str]]:
    """``(api_key_hint, credential_id)`` of the key actually dispatched, not ``pool.current()``:
    the shared pointer often points at a different healthy entry, and marking it exhausted
    can take the whole pool offline from one 429."""
    api_key_hint = getattr(agent, "api_key", None) or None
    raw_id = getattr(agent, "_credential_pool_entry_id", None)
    credential_id = raw_id if isinstance(raw_id, str) and raw_id else None
    if not api_key_hint:
        cur = pool.current()
        if cur:
            api_key_hint = getattr(cur, "runtime_api_key", None)
            if not credential_id:
                current_id = getattr(cur, "id", None)
                if isinstance(current_id, str) and current_id:
                    credential_id = current_id
    return api_key_hint, credential_id


def _is_entitlement_403(agent, status_code, error_context) -> bool:
    """Entitlement 403s look like auth failures but refresh cannot fix them. Any xai-oauth 403
    is entitlement EXCEPT xAI's stale-token signals (``[WKE=unauthenticated:...]``,
    "could not be validated"), which must stay refreshable."""
    if agent._is_entitlement_failure(error_context, status_code):
        return True
    if status_code != 403:
        return False
    haystack = " ".join(
        # Subscription/entitlement 403s look like auth failures on the wire but refresh cannot fix them —
        # the OAuth token is already valid, the account simply lacks the entitlement. Without this guard,
        # the refresh path keeps minting fresh tokens against the same unsubscribed account and the main
        # agent loop spins re-issuing the same 403 until the user Ctrl+C's. Defense-in-depth for #26847:
        # xAI's backend has been seen to 403 standard SuperGrok subscribers with bodies that don't match the
        # existing entitlement keyword set in ``_is_entitlement_failure``. Any 403 against ``xai-oauth`` is
        # treated as entitlement here so the refresh loop can't spin in those cases either. Exception
        # (#29344): xAI's ``[WKE=unauthenticated:...]`` suffix and the ``OAuth2 access token could not be
        # validated`` phrasing are xAI's authoritative "this is a stale token, not entitlement" signal. When
        # either fires we must NOT apply the catch-all override — refresh is the recoverable path for these
        # bodies, and blanket-classifying them as entitlement was the bug that left long-running TUI
        # sessions stuck on stale tokens until the user exited and reopened.
        str(error_context.get(k) or "").lower()
        for k in ("message", "reason", "code", "error")
        if isinstance(error_context, dict)
    )
    if "oauth authentication is currently not allowed for this organization" in haystack:
        return True
    provider = agent.provider or ""
    if provider == "anthropic" and getattr(agent, "api_mode", "") == "anthropic_messages":
        return True
    if provider == "xai-oauth":
        return not (
            "[wke=unauthenticated:" in haystack
            or "oauth2 access token could not be validated" in haystack
        )
    return False


def _recover_auth_failure(agent, pool, *, status_code, has_retried_429, error_context, api_key_hint, credential_id, rotate_and_swap):
    if _is_entitlement_403(agent, status_code, error_context):
        _ra().logger.info(
            "Credential %s — entitlement-shaped 403 from %s; "
            "skipping pool refresh (account lacks subscription, "
            "not a transient auth failure).", status_code if status_code is not None else "auth",
            agent.provider or "provider",
        )
        return False, has_retried_429
    # Refresh the entry that supplied the failing key, not current(): refreshing a healthy entry
    # burns its single-use refresh token for a failure it never had.
    refresh_kwargs = {"api_key_hint": api_key_hint}
    if credential_id:
        refresh_kwargs["credential_id"] = credential_id
    refreshed = pool.try_refresh_matching(**refresh_kwargs)
    if refreshed is None:
        # Refresh failed; rotate (the failed entry is already marked exhausted).
        return (True, False) if rotate_and_swap(401, "auth refresh failed") else (False, has_retried_429)
    # try_refresh_matching() reports success even when upstream keeps rejecting; cap same-entry
    # refreshes so a single-entry pool falls through to fallback.
    refreshed_id = getattr(refreshed, "id", None)
    if refreshed_id is not None:
        if getattr(agent, "_auth_pool_refresh_counts", None) is None:
            agent._auth_pool_refresh_counts = {}
        refresh_counts = agent._auth_pool_refresh_counts
        refresh_key = (agent.provider, refreshed_id)
        refresh_counts[refresh_key] = refresh_counts.get(refresh_key, 0) + 1
        if refresh_counts[refresh_key] > _MAX_AUTH_REFRESH_ATTEMPTS:
            _ra().logger.warning(
                "Credential auth failure persists after %s refreshes for "
                "pool entry %s — treating as unrecoverable and allowing "
                "fallback to activate.", refresh_counts[refresh_key] - 1, refreshed_id,
            )
            return False, has_retried_429
    _ra().logger.info("Credential auth failure — refreshed pool entry %s", getattr(refreshed, 'id', '?'))
    agent._swap_credential(refreshed)
    return True, has_retried_429


def _recover_rate_limit(pool, *, has_retried_429, error_context, api_key_hint, credential_id, rotate_and_swap):
    # Already-exhausted credential: rotate immediately. Avoids the "cancel-between-429s" trap where
    # the local has_retried_429 resets per prompt and retries forever.
    current_entry = None
    if credential_id:
        current_entry = next((e for e in pool.entries() if e.id == credential_id), None)
    if api_key_hint:
        current_entry = current_entry or next(
            (e for e in pool.entries() if e.runtime_api_key == api_key_hint), None
        )
    if current_entry is None:
        current_entry = pool.current()
    current_last_status = getattr(current_entry, "last_status", None) if current_entry else None
    if current_last_status == STATUS_EXHAUSTED:
        _ra().logger.info(
            "Credential already exhausted (last_status=%s) — rotating immediately instead of retrying",
            current_last_status,
        )
        return (True, False) if rotate_and_swap(429, "rate limit, pre-exhausted") else (False, True)
    usage_limit_reached = False
    if error_context:
        context_reason = str(error_context.get("reason") or "").lower()
        context_message = str(error_context.get("message") or "").lower()
        usage_limit_reached = any(t in context_reason for t in _USAGE_LIMIT_REASON_TOKENS) or any(
            t in context_message for t in _USAGE_LIMIT_MESSAGE_TOKENS
        )
    if not has_retried_429 and not usage_limit_reached:
        return False, True
    return (True, False) if rotate_and_swap(429, "rate limit") else (False, True)


def recover_with_credential_pool(
    agent, *, status_code: Optional[int], has_retried_429: bool,
    classified_reason: Optional[FailoverReason] = None,
    error_context: Optional[Dict[str, Any]] = None, billing_unverified: bool = False,
) -> tuple[bool, bool]:
    """Attempt credential recovery via pool rotation; returns (recovered, has_retried_429).
    Rate limits: retry once, then rotate. Billing: rotate immediately. Auth: refresh before
    rotating. ``classified_reason`` beats raw HTTP codes (e.g. Anthropic 400 "out of extra
    usage"); ``billing_unverified`` gives the entry a short cooldown, not the one-hour bench."""
    pool = agent._credential_pool
    if pool is None:
        return False, has_retried_429
    # The pool belongs to the PRIMARY provider: acting on fallback errors would corrupt its state
    # and reset base_url to the primary endpoint. Empty pool provider means unscoped; empty agent
    # provider is a mismatch (swap would leave provider="" model="").
    # Defensive guard: if a fallback provider is active and its provider name doesn't match the pool's
    # provider, the pool belongs to the PRIMARY provider. Mutating it based on fallback errors would corrupt
    # the primary's credential state (see #33088) and, via _swap_credential, overwrite the agent's base_url
    # back to the primary's endpoint — every subsequent request then goes to the wrong host and 404s (see
    # #33163). The pool should only act when the agent is still on the same provider that seeded the pool.
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    pool_provider = (getattr(pool, "provider", "") or "").strip().lower()
    if pool_provider and not credential_pool_matches_provider(
        pool, current_provider, base_url=getattr(agent, "base_url", None)
    ):
        # Same fail-closed boundary predicate as runtime binding.
        _ra().logger.warning(
            "Credential pool provider mismatch: pool=%s, agent=%s — "
            "skipping pool mutation to avoid cross-provider contamination",
            pool_provider, current_provider,
        )
        return False, has_retried_429
    api_key_hint, credential_id = _failed_credential_identity(agent, pool)
    effective_reason = classified_reason
    if effective_reason is None:
        effective_reason = _STATUS_TO_FAILOVER_REASON.get(status_code)

    def _rotate_and_swap(default_status: int, label: str) -> bool:
        """Rotate away from the failed credential; True when a new entry was swapped in."""
        rotate_status = status_code if status_code is not None else default_status
        kwargs = {
            "status_code": rotate_status,
            "error_context": error_context,
            "api_key_hint": api_key_hint,
        }
        if credential_id:
            kwargs["credential_id"] = credential_id
        # Pass classified semantics, not just the status: a billing 403 and an edge-throttle 403
        # need opposite cooldowns.
        if effective_reason is not None:
            failure_reason = effective_reason.value
            if effective_reason == FailoverReason.billing and billing_unverified:
                # Ambiguous billing body: size the cooldown as transient, not a 1-hour bench.
                from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
                failure_reason = FAILURE_REASON_BILLING_UNVERIFIED
            kwargs["failure_reason"] = failure_reason
        next_entry = pool.mark_exhausted_and_rotate(**kwargs)
        if next_entry is None:
            return False
        _ra().logger.info(
            "Credential %s (%s) — rotated to pool entry %s",
            rotate_status, label, getattr(next_entry, "id", "?"),
        )
        agent._swap_credential(next_entry)
        return True
    if effective_reason == FailoverReason.upstream_rate_limit:
        # Upstream (e.g. DeepSeek behind OpenRouter) is throttling the aggregator; the credential is
        # healthy. Do not rotate/exhaust; let fallback switch models.
        upstream = (error_context or {}).get("upstream_provider") if error_context else None
        if upstream:
            _ra().logger.info(
                "Upstream provider %s rate-limited via aggregator — skipping "
                "credential rotation, deferring to fallback chain", upstream,
            )
        else:
            _ra().logger.info(
                "Upstream aggregator 429 (provider unknown) — skipping "
                "credential rotation, deferring to fallback chain"
            )
        return False, has_retried_429
    if effective_reason == FailoverReason.billing:
        # A separate pool instance may have resolved runtime credentials, leaving no ``current_id``;
        # match the key that failed, not a different account.
        return (True, False) if _rotate_and_swap(402, "billing") else (False, has_retried_429)
    if effective_reason == FailoverReason.rate_limit:
        return _recover_rate_limit(
            pool, has_retried_429=has_retried_429, error_context=error_context,
            api_key_hint=api_key_hint, credential_id=credential_id, rotate_and_swap=_rotate_and_swap,
        )
    if effective_reason == FailoverReason.auth:
        return _recover_auth_failure(
            agent, pool, status_code=status_code, has_retried_429=has_retried_429,
            error_context=error_context, api_key_hint=api_key_hint, credential_id=credential_id,
            rotate_and_swap=_rotate_and_swap,
        )
    return False, has_retried_429


def _apply_primary_runtime_fields(agent, rt: Dict[str, Any]) -> None:
    """Copy the identity/transport fields of a ``_primary_runtime`` snapshot onto ``agent``
    (shared by transport recovery and turn-start restore; the caller rebuilds the client)."""
    agent.model = rt["model"]
    agent.provider = rt["provider"]
    agent.requested_provider = rt.get("requested_provider", agent.provider)
    agent.base_url = rt["base_url"]           # setter updates _base_url_lower
    agent.api_mode = rt["api_mode"]
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    agent.api_key = rt["api_key"]
    agent._reasoning_echo_flag = rt.get("reasoning_echo_flag", False)
    agent.request_overrides = dict(rt.get("request_overrides") or {})
    agent._client_kwargs = dict(rt["client_kwargs"])


def _build_anthropic_client_from_runtime(agent, rt: Dict[str, Any]) -> None:
    """Rebuild the native Anthropic client from a ``_primary_runtime`` snapshot."""
    from agent.anthropic_adapter import build_anthropic_client
    agent._anthropic_api_key = rt["anthropic_api_key"]
    agent._anthropic_base_url = rt["anthropic_base_url"]
    agent._anthropic_client = build_anthropic_client(
        rt["anthropic_api_key"], rt["anthropic_base_url"],
        timeout=get_provider_request_timeout(agent.provider, agent.model),
    )
    agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
    agent.client = None


def _rebuild_primary_client(agent, rt: Dict[str, Any], *, reason: str) -> None:
    """Rebuild the primary client from a ``_primary_runtime`` snapshot (MoA facade / native Anthropic / OpenAI wire)."""
    if (agent.provider or "").strip().lower() == "moa":
        # MoA has empty client_kwargs; rebuild via the shared facade factory so the
        # reference_callback relay survives recovery.
        from agent.moa_loop import build_moa_facade
        agent.client = build_moa_facade(agent, agent.model)
        # MoA is a virtual chat-completions provider. It never has real OpenAI client kwargs; restoring it
        # after a fallback must recreate the facade, not call OpenAI() with an empty api_key. Use the shared
        # factory so the restored facade keeps the reference_callback relay wired at init — a bare
        # MoAClient() would silently stop emitting moa.reference/moa.aggregating display events (#53802).
        agent._anthropic_client = None
    elif agent.api_mode == "anthropic_messages":
        _build_anthropic_client_from_runtime(agent, rt)
    else:
        agent.client = agent._create_openai_client(dict(rt["client_kwargs"]), reason=reason, shared=True)


def try_recover_primary_transport(
    agent, api_error: Exception, *, retry_count: int, max_retries: int,
) -> bool:
    """Rebuild the primary client once and retry after ``max_retries`` exhaust on a transient
    transport error. Skipped for aggregators (OpenRouter, Nous) that manage retries server-side."""
    error_type = type(api_error).__name__
    if agent._fallback_activated or error_type not in _TRANSIENT_TRANSPORT_ERRORS or agent._is_openrouter_url():
        return False
    # Portal OpenAI-wire traffic rides aggregator retry infra (skip), but Portal Claude on native
    # Messages holds a local Anthropic client that needs the rebuild.
    if (
        (agent.provider or "").strip().lower() in {"nous", "nous-portal", "nousresearch"}
        and getattr(agent, "api_mode", None) != "anthropic_messages"
    ):
        return False
    try:
        # Never hard-close the shared client here: stale streaming workers may still be unwinding on
        # the old pool; _retire_shared_openai_client defers FD release to GC.
        # Retire the existing client to release stale connections. #70773: never hard-close the shared
        # client here — this runs on the conversation-loop thread while workers from stale-killed streaming
        # attempts may still be unwinding their SSL BIOs on the old pool. ``_retire_shared_openai_client``
        # shuts the sockets down (FD-safe from any thread) and defers the FD release to GC, which cannot
        # complete until every borrowing thread has unwound.
        if getattr(agent, "client", None) is not None:
            with contextlib.suppress(Exception):
                agent._retire_shared_openai_client(agent.client, reason="primary_recovery")
        rt = agent._primary_runtime
        _apply_primary_runtime_fields(agent, rt)
        if agent.api_mode == "anthropic_messages":
            _build_anthropic_client_from_runtime(agent, rt)
        elif (agent.provider or "").strip().lower() == "moa":
            # MoA is a virtual provider with empty client_kwargs — rebuilding via _create_openai_client
            # would raise "api_key client option must be set". Recreate the facade through the shared
            # factory so the reference_callback relay survives recovery (#53802).
            from agent.moa_loop import build_moa_facade
            agent.client = build_moa_facade(agent, agent.model)
        else:
            agent.client = agent._create_openai_client(dict(rt["client_kwargs"]), reason="primary_recovery", shared=True)
        wait_time = min(3 + retry_count, 8)
        agent._vprint(
            f"{agent.log_prefix}🔁 Transient {error_type} on {agent.provider} — "
            f"rebuilt client, waiting {wait_time}s before one last primary attempt.", force=True,
        )
        time.sleep(wait_time)
        return True
    except Exception as e:
        logger.warning("Primary transport recovery failed: %s", e)
        return False


def _merge_user_content(prev_content: Any, cur_content: Any) -> Any:
    """Merged content for two adjacent user messages (``_UNMERGEABLE`` for unknown shapes):
    string+string joins with a blank line; list sides append as separate blocks."""
    if isinstance(prev_content, str) and isinstance(cur_content, str):
        return prev_content + ("\n\n" if prev_content and cur_content else "") + cur_content
    if isinstance(prev_content, list) and isinstance(cur_content, list):
        return list(prev_content) + list(cur_content)
    if isinstance(prev_content, list) and isinstance(cur_content, str):
        return list(prev_content) + ([{"type": "text", "text": cur_content}] if cur_content else [])
    if isinstance(prev_content, str) and isinstance(cur_content, list):
        return ([{"type": "text", "text": prev_content}] if prev_content else []) + list(cur_content)
    return _UNMERGEABLE


_UNMERGEABLE = object()


def drop_thinking_only_and_merge_users(
    messages: List[Dict[str, Any]], *, drop_codex_reasoning_items: bool = True
) -> List[Dict[str, Any]]:
    """Drop thinking-only assistant turns and merge adjacent user messages left behind, on the
    per-call ``api_messages`` copy only (``agent.messages`` is never mutated). Drop-and-merge
    (not stub text) keeps history honest and preserves role alternation."""
    if not messages:
        return messages
    kept = [
        m for m in messages
        if not _ra().AIAgent._is_thinking_only_assistant(m, drop_codex_reasoning_items=drop_codex_reasoning_items)
    ]
    dropped = len(messages) - len(kept)
    merged: List[Dict[str, Any]] = []
    merges = 0
    for m in kept:
        prev = merged[-1] if merged else None
        content = _UNMERGEABLE
        if prev is not None and prev.get("role") == "user" and m.get("role") == "user":
            content = _merge_user_content(prev.get("content", ""), m.get("content", ""))
        if content is _UNMERGEABLE:
            # Not a user pair, or an unknown content shape: append separately (the latter violates
            # alternation, but is safer than raising in a hot path).
            merged.append(m)
        else:
            merged[-1] = {**prev, "content": content}  # copy so caller dicts are never mutated
            merges += 1
    if dropped == 0 and merges == 0:
        return messages
    _ra().logger.debug(
        "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
        "merged %d adjacent user message(s)", dropped, merges,
    )
    return merged


def _primary_reset_gate_blocks(agent, rt, primary_provider, primary_runtime_base_url, matches_primary, load_primary_pool):
    """Reset-aware gate: skip a guaranteed-to-fail restore while the primary pool reports a
    future reset; fails open on any error/None. Returns ``(blocked, prefetched_pool, prefetched)``
    so the rebind step reuses the loaded pool (one auth.json read at most)."""
    prefetched_pool, prefetched = None, False
    try:
        pool = getattr(agent, "_credential_pool", None)
        if not matches_primary(pool):
            prefetched_pool = pool = load_primary_pool()
            prefetched = True
        next_at = getattr(pool, "next_available_at", lambda: None)()
        if next_at is not None and next_at > time.time():
            if not getattr(agent, "_restore_wait_logged", False):
                agent._restore_wait_logged = True
                logger.info(
                    "Primary %s rate-limited until %s; staying on fallback "
                    "%s/%s until the reset elapses", primary_provider or "?",
                    datetime.fromtimestamp(next_at).isoformat(timespec="seconds"), agent.provider,
                    agent.model,
                )
            return True, prefetched_pool, prefetched
    except Exception:
        logger.debug("Reset-aware restore gate failed; falling back to per-turn retry", exc_info=True)
    return False, prefetched_pool, prefetched


def _restore_runtime_capabilities(agent, rt: Dict[str, Any]) -> None:
    # ``capabilities`` is the legacy key from the initial capability propagation patch.
    raw = rt["runtime_capabilities"] if "runtime_capabilities" in rt else rt.get("capabilities")
    if isinstance(raw, dict):
        agent.runtime_capabilities = dict(raw)
    elif "runtime_capabilities" in rt:
        logger.warning("Ignoring malformed runtime capabilities snapshot")


def _rebind_primary_credential_pool(agent, primary_provider, matches_primary, load_primary_pool, prefetched_pool, prefetched) -> None:
    """Rebind and re-select the primary credential pool after a fallback turn. A cross-provider
    fallback attaches its own pool, which would trip the provider-mismatch guard on the next
    401/429: reload the primary pool, else clear it. The snapshot api_key may be stale after
    rotation; re-select the pool's best entry, keeping the snapshot key when none is usable."""
    pool = getattr(agent, "_credential_pool", None)
    pool_provider = str(getattr(pool, "provider", "") or "").strip().lower()
    if pool is not None and pool_provider and not matches_primary(pool):
        agent._credential_pool = None
        agent._credential_pool_entry_id = None
        try:
            # Reuse the pool the reset-aware gate already loaded (avoids a second auth.json read).
            agent._credential_pool = prefetched_pool if prefetched else load_primary_pool()
        except Exception as exc:
            logger.warning(
                "Restore could not reload primary credential pool for %s: %s", primary_provider, exc
            )
    agent._credential_pool_entry_id = None
    pool = getattr(agent, "_credential_pool", None)
    entry = pool.select() if pool is not None and pool.has_available() else None
    if entry is None or not (getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")):
        return
    if matches_primary(entry):
        # _swap_credential rebuilds the client and reapplies base-url-scoped headers.
        # ``_swap_credential`` rebuilds the OpenAI/Anthropic client, reapplies base-url-scoped headers, and
        # carries the accumulated base_url / OAuth-detection fixes (#33163).
        agent._swap_credential(entry)
        logger.info(
            "Restore re-selected pool entry %s (%s)",
            getattr(entry, "id", "?"), getattr(entry, "label", "?"),
        )
    else:
        logger.info(
            "Restore skipped pool entry %s (%s): provider %s does not match primary provider %s",
            getattr(entry, "id", "?"), getattr(entry, "label", "?"),
            str(getattr(entry, "provider", "") or "").strip().lower() or "?",
            primary_provider or "?",
        )


def restore_primary_runtime(agent) -> bool:
    """Restore the primary runtime at the start of a new turn so fallback stays turn-scoped
    (long-lived CLI agents and the gateway's cached agents)."""
    if not agent._fallback_activated:
        # Reset the index even without activation: a failed _try_activate_fallback() can strand
        # _fallback_index past the chain end and silently block future fallbacks.
        agent._fallback_index = 0
        return False
    # Reset the chain index even when no fallback was activated this turn. Without this, a turn where
    # _try_activate_fallback() was called but returned False (chain exhausted or provider not configured)
    # leaves _fallback_index >= len(_fallback_chain) while _fallback_activated stays False. The next turn
    # skips this block entirely, stranding the index and silently blocking all future fallback attempts for
    # the session. Fixes #20465.
    if getattr(agent, "_rate_limited_until", 0) > time.monotonic():
        return False  # primary still in rate-limit cooldown, stay on fallback
    rt = agent._primary_runtime
    primary_provider = str((rt or {}).get("provider") or "").strip().lower()
    primary_runtime_base_url = str((rt or {}).get("base_url") or "")

    def _matches_primary(candidate) -> bool:
        return credential_pool_matches_provider(candidate, primary_provider, base_url=primary_runtime_base_url)

    def _load_primary_pool():
        """Load the primary provider's pool; None when absent or provider-mismatched."""
        from agent.credential_pool import load_pool
        key = resolve_runtime_pool_key(primary_provider, primary_runtime_base_url)
        loaded = load_pool(key) if key else None
        return loaded if loaded is not None and _matches_primary(loaded) else None
    blocked, prefetched_pool, prefetched = _primary_reset_gate_blocks(
        agent, rt, primary_provider, primary_runtime_base_url, _matches_primary, _load_primary_pool
    )
    if blocked:
        return False
    agent._restore_wait_logged = False
    fallback_route = getattr(agent, "_provider_fallback_route", None)
    if not (isinstance(fallback_route, (list, tuple)) and len(fallback_route) == 2):
        fallback_route = (getattr(agent, "model", ""), getattr(agent, "provider", ""))
    previous_model, previous_provider = (str(v or "unknown") for v in fallback_route)
    provider_fallback_active = bool(getattr(agent, "_provider_fallback_active", False))
    try:
        _apply_primary_runtime_fields(agent, rt)
        _restore_runtime_capabilities(agent, rt)
        agent._use_prompt_caching = rt["use_prompt_caching"]
        # Default to native layout for snapshots predating the native-vs-proxy split.
        agent._use_native_cache_layout = rt.get(
            "use_native_cache_layout",
            agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
        )
        # An operator cache disable (_cache_disabled) must survive snapshot restoration.
        if getattr(agent, "_cache_disabled", False):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False
        _rebuild_primary_client(agent, rt, reason="restore_primary")
        agent.context_compressor.update_model(
            model=rt["compressor_model"], context_length=rt["compressor_context_length"],
            base_url=rt["compressor_base_url"], api_key=rt["compressor_api_key"],
            provider=rt["compressor_provider"], api_mode=rt.get("compressor_api_mode", ""),
        )
        _rebind_primary_credential_pool(
            agent, primary_provider, _matches_primary, _load_primary_pool, prefetched_pool, prefetched
        )
        # Older snapshots have no reasoning_config; keep the current value.
        saved_reasoning = rt.get("reasoning_config")
        if saved_reasoning is not None:
            agent.reasoning_config = dict(saved_reasoning)
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._rate_limit_backoff_count = 0
        # Reset the stale-call circuit breaker: its streak measured the fallback provider.
        from agent.chat_completion_helpers import _reset_stale_streak, rewrite_prompt_model_identity
        _reset_stale_streak(agent)
        # Undo the fallback's identity rewrite so the prompt is byte-identical to the stored copy
        # again (prefix cache match).
        rewrite_prompt_model_identity(agent, rt["model"], rt["provider"])
        logger.info("Primary runtime restored for new turn: %s (%s)", agent.model, agent.provider)
        agent._provider_fallback_active = False
        agent._provider_fallback_route = None
        if provider_fallback_active:
            # Notification surfaces are best-effort and must never undo a successful restore.
            with contextlib.suppress(Exception):
                agent._emit_status(
                    f"✅ Primary model restored: {agent.model} via {agent.provider}; "
                    f"fallback {previous_model} via {previous_provider} is no longer active."
                )
        return True
    except Exception as e:
        logger.warning("Failed to restore primary runtime: %s", e)
        return False


# Transient transport failures worth one more attempt with a rebuilt client / connection pool.
_TRANSIENT_TRANSPORT_ERRORS = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout", "ConnectError", "RemoteProtocolError",
    "APIConnectionError", "APITimeoutError",
})
_INLINE_REASONING_PATTERNS = tuple(
    re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in ("think", "thinking", "thought", "reasoning", "REASONING_SCRATCHPAD")
)


def extract_reasoning(agent, assistant_message) -> Optional[str]:
    """Reasoning text from ``reasoning`` / ``reasoning_content`` / ``reasoning_details``
    (OpenRouter unified), else inline thinking blocks in the content; None when absent."""
    parts: List[str] = []

    def _add(text) -> None:
        if text and text not in parts:
            parts.append(text)
    _add(getattr(assistant_message, "reasoning", None))
    _add(getattr(assistant_message, "reasoning_content", None))
    # reasoning_details: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
    for detail in getattr(assistant_message, "reasoning_details", None) or []:
        if isinstance(detail, dict):
            _add(detail.get('summary') or detail.get('thinking') or detail.get('content') or detail.get('text'))
    # Fall back to reasoning embedded in content only when no structured field was found.
    content = getattr(assistant_message, "content", None)
    if not parts and isinstance(content, list):
        # DeepSeek V4 Pro returns typed content blocks ({"type": "thinking", ...}); dropping them
        # makes the next turn fail with HTTP 400 "thinking must be passed back".
        # Refs #21944.
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                _add((block.get("thinking") or block.get("text") or "").strip())
    if not parts and isinstance(content, str) and content:
        for pattern in _INLINE_REASONING_PATTERNS:
            for block in pattern.findall(content):
                _add(block.strip())
    return "\n\n".join(parts) if parts else None


def _api_error_debug_info(error: Exception) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": type(error).__name__, "message": str(error)}
    info.update({
        k: v for k in ("status_code", "request_id", "code", "param", "type", "body")
        if (v := getattr(error, k, None)) is not None
    })
    response_obj = getattr(error, "response", None)
    if response_obj is not None:
        try:
            info["response_status"] = getattr(response_obj, "status_code", None)
            info["response_text"] = response_obj.text
        except Exception as e:
            _ra().logger.debug("Could not extract error response details: %s", e)
    return info


def dump_api_request_debug(
    agent, api_kwargs: Dict[str, Any], *, reason: str, error: Optional[Exception] = None
) -> Optional[Path]:
    """Dump the request body from api_kwargs (minus transport keys) for debugging provider 4xx failures."""
    try:
        body = {k: v for k, v in copy.deepcopy(api_kwargs).items() if v is not None and k != "timeout"}
        api_key = None
        try:
            api_key = getattr(agent.client, "api_key", None)
        except Exception as e:
            _ra().logger.debug("Could not extract API key for debug dump: %s", e)
        endpoint = "/responses" if agent.api_mode == "codex_responses" else "/chat/completions"
        dump_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(), "session_id": agent.session_id, "reason": reason,
            "request": {
                "method": "POST", "url": f"{agent.base_url.rstrip('/')}{endpoint}",
                "headers": {
                    "Authorization": f"Bearer {agent._mask_api_key_for_logs(api_key)}",
                    "Content-Type": "application/json",
                },
                "body": body,
            },
        }
        if error is not None:
            dump_payload["error"] = _api_error_debug_info(error)
        # Sanitize the session ID (may come from an untrusted X-Hermes-Session-Id header) so a
        # "../"-shaped ID cannot write outside logs_dir.
        from agent.session_persistence import _safe_session_filename_component
        safe_sid = _safe_session_filename_component(agent.session_id)
        dump_file = agent.logs_dir / f"request_dump_{safe_sid}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        # Redact secrets first: this fires unconditionally on API errors and captures the full
        # request body, so context-embedded secrets would otherwise land in cleartext on disk.
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
    agent, *, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_mode: Optional[str] = None, model: Optional[str] = None,
) -> bool:
    """Return whether this resolved destination accepts native tool markers."""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    return eff_api_mode == "anthropic_messages" and base_url_hostname(eff_base_url) == "api.anthropic.com"


# The cache_ttl tiers accepted by config; mirrored by agent_init's live-agent snapshot.
VALID_CACHE_TTLS = ("5m", "1h")


def cache_ttl_means_disabled(ttl: Any) -> bool:
    """True when a ``prompt_caching.cache_ttl`` value means caching off (single predicate shared
    by ``agent_init`` and the stub policy paths). Unknown values (``"2h"``, ints) are NOT a disable."""
    if ttl in VALID_CACHE_TTLS:
        return False
    return ttl is False or ttl is None or str(ttl).lower() in ("off", "false", "disabled", "no", "none")


def _raw_cache_ttl_from_config(default: Any) -> Any:
    """Raw ``prompt_caching.cache_ttl`` config value, or ``default`` when config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly
        return (load_config_readonly().get("prompt_caching", {}) or {}).get("cache_ttl", "5m")
    except Exception:
        return default


def prompt_caching_disabled_from_config() -> bool:
    """True when ``prompt_caching.cache_ttl`` is configured as off (same detection as ``agent_init``).

    Same disable detection as ``agent_init`` (via ``cache_ttl_means_disabled``) so stub-based policy paths
    (MoA slot decoration, auxiliary fallback replan) honor the same config contract without holding a live
    ``AIAgent`` (#76085 / #33555).
    """
    return cache_ttl_means_disabled(_raw_cache_ttl_from_config("5m"))


def configured_cache_ttl() -> Optional[str]:
    """Configured ``prompt_caching.cache_ttl`` tier (``5m``/``1h``), else None; mirrors
    ``agent_init`` so stub paths don't regress a configured ``1h`` to 5m."""
    ttl = _raw_cache_ttl_from_config(None)
    return ttl if ttl in VALID_CACHE_TTLS else None


def blank_cache_policy_stub(cache_disabled: Optional[bool] = None):
    """Destination-identity-blank stub for ``anthropic_prompt_cache_policy``; the sole sanctioned
    constructor so ``_cache_disabled`` is never omitted (None consults the global config)."""
    from types import SimpleNamespace
    if cache_disabled is None:
        cache_disabled = prompt_caching_disabled_from_config()
    return SimpleNamespace(provider="", base_url="", api_mode="", model="", _cache_disabled=bool(cache_disabled))


def plan_cache_sections_for_destination(
    messages: list, tools: Optional[list], *, provider: str, base_url: str, api_mode: str,
    model: str, cache_disabled: Optional[bool] = None, cache_ttl: Optional[str] = None,
    static_system_prefix: Optional[str] = None,
) -> Tuple[list, list]:
    """Plan request-local cache sections for one resolved destination (MoA / auxiliary senders):
    stripped copies (non-caching route) or a ``build_prompt_cache_plan`` layout; never mutates
    inputs. ``cache_disabled``/``cache_ttl`` default to live config so the operator's disable and
    tier are honored; ``static_system_prefix`` gives the system prompt the main loop's early breakpoint.

    ``cache_disabled`` threads the operator's ``prompt_caching.cache_ttl`` disable into the blank policy
    stub. When omitted, the live config is consulted so MoA/auxiliary paths cannot re-enable markers after
    the user turned caching off (#76085).
    """
    from agent.prompt_caching import (
        build_prompt_cache_plan, effective_cache_ttl, envelope_tool_part_cache_markers_supported,
        strip_anthropic_cache_control, strip_anthropic_tool_cache_control,
    )
    # The policy function reads agent.* only as fallbacks for kwargs we don't pass; blank_cache_policy_stub
    # is the only sanctioned stub so _cache_disabled cannot be left off again (#76085).
    stub = blank_cache_policy_stub(cache_disabled)
    dest = dict(provider=provider, base_url=base_url, api_mode=api_mode, model=model)
    should_cache, native_layout = anthropic_prompt_cache_policy(stub, **dest)
    if not should_cache:
        canonical_messages = copy.deepcopy(messages or [])
        strip_anthropic_cache_control(canonical_messages)
        return canonical_messages, strip_anthropic_tool_cache_control(tools)
    plan = build_prompt_cache_plan(
        messages, tools,
        # effective_cache_ttl resolves None → "5m"; cache-disabled agents never reach here.
        cache_ttl=effective_cache_ttl(cache_ttl, provider=provider, model=model),
        native_anthropic=native_layout,
        static_system_prefix=static_system_prefix if isinstance(static_system_prefix, str) else None,
        direct_native_tool_cache=_direct_native_anthropic_tool_cache_capability(stub, **dest),
        # LiteLLM-style envelope routes forward part-level markers into tool_result.content[] →
        # non-retryable 400.
        tool_part_markers=envelope_tool_part_cache_markers_supported(provider, base_url),
    )
    return plan.messages, plan.tools


def _is_litellm_route(provider_lower: str, base_url: str) -> bool:
    """True when a route is a LiteLLM proxy: ``litellm`` as a whole delimited token (not
    substring) in the provider id or host; a path segment never qualifies."""
    return _has_litellm_token(provider_lower, ":-_/") or _has_litellm_token(base_url_hostname(base_url), ".-")


def _has_litellm_token(value: str, delimiters: str) -> bool:
    """True when ``value`` contains ``litellm`` as a whole delimited token."""
    if not value:
        return False
    return "litellm" in value.translate(str.maketrans(delimiters, " " * len(delimiters))).split()


def _moa_aggregator_cache_policy(agent, eff_model: str) -> tuple[bool, bool]:
    """MoA virtual provider: resolve the policy from the preset's real aggregator slot (the
    virtual provider matches no caching branch and would silently lose caching)."""
    try:
        from hermes_cli.config import load_config as _load_moa_cfg
        from hermes_cli.moa_config import resolve_moa_preset
        from hermes_cli.runtime_provider import resolve_runtime_provider
        agg = resolve_moa_preset(_load_moa_cfg().get("moa") or {}, eff_model or None).get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if agg_provider and agg_model:
            agg_base_url = agg_api_mode = ""
            with contextlib.suppress(Exception):
                rt = resolve_runtime_provider(requested=agg_provider, target_model=agg_model)
                agg_base_url = rt.get("base_url") or ""
                agg_api_mode = rt.get("api_mode") or ""
            return anthropic_prompt_cache_policy(
                agent, provider=agg_provider, base_url=agg_base_url, api_mode=agg_api_mode, model=agg_model
            )
    except Exception as _moa_exc:  # pragma: no cover - defensive
        logger.debug("MoA aggregator cache-policy resolution failed: %s", _moa_exc)
    return False, False


def _route_may_be_custom(agent, eff_provider: str, provider_lower: str, eff_base_url: str) -> bool:
    """Cheap identity gate deciding whether a custom-provider capability lookup is worth running."""
    custom_providers = getattr(agent, "_custom_providers", None)
    if custom_providers:
        # Same semantics as the capability helper (normalize_route_base_url +
        # custom_provider_aliases) so spelling differences don't drop declarations.
        from hermes_cli.providers import custom_provider_aliases
        from hermes_cli.route_identity import normalize_route_base_url
        provider_ids = {provider_lower, provider_lower.removeprefix("custom:")}
        eff_url_normalized = normalize_route_base_url(eff_base_url)
        return any(
            provider_ids & custom_provider_aliases(str(entry.get("name") or ""), str(entry.get("provider_key") or ""))
            or (eff_url_normalized and normalize_route_base_url(entry.get("base_url")) == eff_url_normalized)
            for entry in custom_providers if isinstance(entry, dict)
        )
    if custom_providers is not None:
        return False  # attached empty list never matches
    # None = list not attached yet (early init or blank stub). Avoid rebuilding the list for
    # ordinary built-in routes.
    try:
        from hermes_cli.providers import get_provider
        # allow_network=False: never trigger a registry fetch from the send path; a catalog miss
        # degrades to the conservative capability lookup.
        provider_def = get_provider(eff_provider, allow_network=False)
        return provider_def is None or (
            bool(provider_def.base_url)
            and base_url_hostname(provider_def.base_url) != base_url_hostname(eff_base_url)
        )
    except Exception as _pd_exc:
        logger.debug("provider lookup failed during cache-policy pre-gate: %s", _pd_exc)
        return provider_lower.startswith("custom:")


def anthropic_prompt_cache_policy(
    agent, *, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_mode: Optional[str] = None, model: Optional[str] = None,
) -> tuple[bool, bool]:
    """Decide whether to apply Anthropic prompt caching; returns ``(should_cache, use_native_layout)``.
    Native layout puts markers on inner content blocks (Anthropic wire), else on the message
    envelope (OpenRouter / OpenAI-wire proxies; Qwen/Alibaba too). The operator disable is read
    from ``_cache_disabled`` (not ``_cache_ttl``, unset during init) so it survives switches
    and restores. Branch ORDER is load-bearing (see inline notes).

    Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct Alibaba (DashScope) also honour
    Anthropic-style ``cache_control`` markers on OpenAI-wire chat completions. Upstream pi-mono #3392 / pi
    #3393 documented this for opencode-go Qwen. Without markers these providers serve zero cache hits,
    re-billing the full prompt on every turn.
    """
    if getattr(agent, "_cache_disabled", False):
        return (False, False)
    eff_provider = (provider if provider is not None else agent.provider) or ""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    eff_model = (model if model is not None else agent.model) or ""
    if eff_provider.strip().lower() == "moa":
        return _moa_aggregator_cache_policy(agent, eff_model)
    if isinstance(eff_model, dict):
        eff_model = eff_model.get('model') or eff_model.get('default') or ''
    eff_model = eff_model if isinstance(eff_model, str) else str(eff_model or '')
    model_lower = eff_model.lower()
    provider_lower = eff_provider.lower()
    is_claude = "claude" in model_lower
    # Kimi/Moonshot via OpenRouter uses the same envelope cache_control as Claude; without this it
    # serves ~1% cache hits. Family matcher covers bare k1./k2. slugs.
    # Without this branch moonshotai/kimi-k2.6 falls through to (False, False), serving ~1% cache hits on
    # 64K-token prompts and re-billing the full prompt on every turn. Observed within-turn progression with
    # cache enabled: 1% → 67% → 84% → 97% (#25970). Reuses the canonical family matcher (covers bare
    # k1./k2./k25 release slugs the substring check missed).
    from agent.anthropic_endpoints import _model_name_is_kimi_family
    is_kimi = _model_name_is_kimi_family(eff_model) or "moonshot" in model_lower
    is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
    # Nous Portal proxies to OpenRouter; treat as OpenRouter-equivalent for cache layout.
    is_nous_portal = base_url_host_matches(eff_base_url, "nousresearch.com")
    is_anthropic_wire = eff_api_mode == "anthropic_messages"
    is_native_anthropic = is_anthropic_wire and (
        eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com"
    )
    # Honor a configured route's per-model ``prompt_caching`` capability (explicit false too); only
    # for the two transports this planner handles, not Responses/Bedrock.
    supports_cache_markers = eff_api_mode in {"anthropic_messages", "chat_completions"}
    litellm_openai_wire = (
        eff_api_mode == "chat_completions" and is_claude and _is_litellm_route(provider_lower, eff_base_url)
    )
    if supports_cache_markers and (
        is_anthropic_wire
        or litellm_openai_wire
        or _route_may_be_custom(agent, eff_provider, provider_lower, eff_base_url)
    ):
        try:
            from hermes_cli.config import get_custom_provider_model_capability
            custom_prompt_caching = get_custom_provider_model_capability(
                model=eff_model, base_url=eff_base_url, capability="prompt_caching",
                custom_providers=getattr(agent, "_custom_providers", None),
            )
            if custom_prompt_caching is not None:
                # Layout follows the transport: native Messages → inner blocks; OpenAI wire → envelope.
                return custom_prompt_caching, custom_prompt_caching and is_anthropic_wire
        except Exception as _cap_exc:
            logger.debug("custom-provider prompt_caching capability lookup failed: %s", _cap_exc)
    # MiniMax-M3 uses server-side automatic prefix caching; explicit markers are dead weight.
    # Checked BEFORE the native-Anthropic return since provider="anthropic" may point at a MiniMax
    # proxy.
    is_minimax_route = (
        provider_lower in {"minimax", "minimax-cn"}
        or base_url_host_matches(eff_base_url, "api.minimax.io")
        or base_url_host_matches(eff_base_url, "api.minimaxi.com")
    )
    if is_anthropic_wire and is_minimax_route:
        from agent.model_metadata import _model_name_suggests_minimax_m3
        if _model_name_suggests_minimax_m3(eff_model):
            return False, False
    if is_native_anthropic:
        return True, True
    # Envelope layout is OpenAI-wire only; Portal Claude on native Messages must fall through to the
    # anthropic_messages branch (inner-block markers) or it serves 0% cache hits.
    if (is_openrouter or is_nous_portal) and (is_claude or is_kimi) and not is_anthropic_wire:
        return True, False
    # Nous Portal Qwen takes the envelope path too; the alibaba-family check below only matches
    # provider=opencode/alibaba and would leave Portal traffic uncached.
    if is_nous_portal and "qwen" in model_lower:
        return True, False
    if is_anthropic_wire and is_claude:
        return True, True  # third-party Anthropic-compatible gateway
    # LiteLLM fronting Claude on the OpenAI wire supports cache_control but matched no grant above.
    # Claude-only: strict relays reject the block format for other models. Envelope layout: native
    # top-level markers are only relocated by the anthropic_messages adapter and 400 via LiteLLM.
    # Gated on chat_completions; codex_responses/bedrock_converse have their own handling.
    if litellm_openai_wire:
        return True, False
    # MiniMax's own models (M2.x) on its Anthropic-compatible endpoint support cache_control; opt
    # them in past the is_claude gate. M3 is excluded above.
    if is_anthropic_wire and is_minimax_route:
        return True, True
    # Qwen/Alibaba on OpenCode and DashScope accept envelope cache_control on the OpenAI wire
    # (pi-mono's "alibaba" cacheControlFormat). DeepSeek on OpenCode is excluded: its relay 400s on
    # block-array content. Family set/predicate shared with the effective_cache_ttl clamp.
    # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire transport that accepts
    # Anthropic-style cache_control markers and rewards them with real cache hits. Without this branch
    # qwen3.6-plus on opencode-go reports 0% cached tokens and burns through the subscription on every turn.
    # OpenCode Zen's relay rejects the Anthropic-style content block format that cache markers produce
    # (content becomes a block array instead of a plain string), causing HTTP 400 (#77217).
    from agent.prompt_caching import ALIBABA_FAMILY_PROVIDERS, is_qwen_model
    if provider_lower in ALIBABA_FAMILY_PROVIDERS and is_qwen_model(model_lower):
        return True, False
    return False, False


def _provider_supplied_client(agent, client_kwargs: dict) -> Any | None:
    """Ask the registered ProviderProfile for a custom client, if any. Resolves by provider name,
    then by ``base_url`` prefix so a URL-only runtime (``acp://…``) still reaches its profile.
    A profile that raises is logged and skipped: a third-party plugin must not be able to take
    the turn down, it can only fail to provide a client."""
    try:
        from providers import get_provider_profile
    except Exception:
        return None
    profile = None
    provider_name = (getattr(agent, "provider", "") or "").strip()
    if provider_name:
        try:
            profile = get_provider_profile(provider_name)
        except Exception:
            profile = None
    if profile is None:
        base_url = str(client_kwargs.get("base_url", "") or "").strip()
        if base_url:
            profile = _profile_for_base_url(base_url)
    if profile is None:
        return None
    try:
        return profile.create_client(**client_kwargs)
    except Exception:
        _ra().logger.warning(
            "Provider profile %r failed to create a client; falling back to the standard client path",
            getattr(profile, "name", provider_name) or "?", exc_info=True,
        )
        return None


def _profile_for_base_url(base_url: str) -> Any | None:
    """Registered profile whose own base_url is a prefix of ``base_url`` (provider name did not
    resolve). Prefix, not equality: the replaced copilot-acp branch keyed on
    ``startswith("acp://copilot")``, so a path or user override under the same root must resolve."""
    try:
        from providers import list_providers
        candidates = list_providers()
    except Exception:
        return None
    target = base_url.rstrip("/").lower()
    for candidate in candidates or []:
        own = str(getattr(candidate, "base_url", "") or "").rstrip("/").lower()
        if own and (target == own or target.startswith(own + "/")):
            return candidate
    return None


def _ensure_copilot_headers(client_kwargs: dict) -> None:
    """Defense-in-depth: recovery/restore rebuild from a snapshot without re-running header
    wiring; a missing Copilot-Integration-Id causes model_not_available_for_integrator 400s.
    Only ADD missing keys, never override."""
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


def _gemini_native_client(agent, client_kwargs: dict, httpx_verify, *, reason: str, shared: bool):
    """Native Gemini client when the base_url is the Gemini API, else None."""
    from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url
    base_url = str(client_kwargs.get("base_url", "") or "")
    if not is_native_gemini_base_url(base_url):
        return None
    safe_kwargs = {
        k: v for k, v in client_kwargs.items()
        if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
    }
    if "http_client" not in safe_kwargs:
        keepalive_http = agent._build_keepalive_http_client(base_url, verify=httpx_verify)
        if keepalive_http is not None:
            safe_kwargs["http_client"] = keepalive_http
    client = GeminiNativeClient(**safe_kwargs)
    _ra().logger.info(
        "Gemini native client created (%s, shared=%s) %s", reason, shared, agent._client_log_context()
    )
    return client


def create_openai_client(agent, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    from agent.ssl_verify import resolve_httpx_verify
    # Treat client_kwargs as read-only: callers pass agent._client_kwargs, and in-place mutation
    # leaks into later requests (a torn-down httpx transport got reused).
    # Callers pass agent._client_kwargs (or shallow copies of it) in; any in-place mutation leaks back into
    # the stored dict and is reused on subsequent requests. #10933 hit this by injecting an httpx.Client
    # transport that was torn down after the first request, so the next request wrapped a closed transport
    # and raised "Cannot send a request, as the client has been closed" on every retry. The revert resolved
    # that specific path; this copy locks the contract so future transport/keepalive work can't reintroduce
    # the same class of bug.
    client_kwargs = dict(client_kwargs)
    # The MoA virtual provider has no OpenAI wire endpoint; the facade *is* the client. Rebuild the
    # facade, never a native client (TypeError; relay re-wire).
    # Rebuilding a native OpenAI client while agent.provider == "moa" (client replacement, stream-retry pool
    # cleanup, credential rotation, fallback+restore) drops the facade: the next primary call either raises
    # a `_moa_prepared_request` TypeError (#78382) or, when _client_kwargs carry an unrelated relay
    # base_url, leaks the request to a foreign gateway. Rebuild the facade instead (build_moa_facade also
    # re-wires the reference relay, see #53802).
    if (getattr(agent, "provider", "") or "").strip().lower() == "moa":
        from agent.moa_loop import build_moa_facade
        return build_moa_facade(agent, getattr(agent, "model", None) or "default")
    ssl_ca_cert = client_kwargs.pop("ssl_ca_cert", None)
    ssl_verify_cfg = client_kwargs.pop("ssl_verify", None)
    httpx_verify = resolve_httpx_verify(ca_bundle=ssl_ca_cert, ssl_verify=ssl_verify_cfg)
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))
    # Provider-supplied client (registration seam): a provider whose wire protocol is not
    # OpenAI-over-HTTP supplies its own client from ProviderProfile.create_client(). Consulted
    # before the built-in ladder so a profile registered from ~/.hermes/plugins/ or a pip entry
    # point can ship a transport without editing this function (what makes an out-of-tree ACP
    # provider possible). None (the default) falls through, so existing providers are unaffected.
    provider_client = _provider_supplied_client(agent, client_kwargs)
    if provider_client is not None:
        _ra().logger.info(
            "%s client created from provider profile (%s, shared=%s) %s",
            agent.provider, reason, shared, agent._client_log_context(),
        )
        return provider_client
    if agent.provider == "gemini":
        client = _gemini_native_client(agent, client_kwargs, httpx_verify, reason=reason, shared=shared)
        if client is not None:
            return client
    # TCP keepalives so dead provider connections are detected (~60s) instead of hanging in
    # CLOSE-WAIT. Injected into the local copy only, so each client gets its own httpx.Client;
    # pinned by tests/run_agent/test_create_openai_client_reuse.py and
    # test_sequential_chats_live.py. What IS shared across those per-client wrappers is the
    # connection pool: ``build_keepalive_http_client`` mounts a process-shared ``HTTPTransport``
    # behind a per-client view whose ``close()`` is a no-op for the pool, so a closed wrapper
    # never takes a sibling's (or the successor's) connections with it
    # (tests/agent/test_shared_http_transport.py).
    # Without this, a peer that drops mid-stream leaves the socket in a state where epoll_wait never fires,
    # ``httpx`` read timeout may not trigger, and the agent hangs until manually killed. Probes after 30s
    # idle, retry every 10s, give up after 3 → dead peer detected within ~60s. Safety against #10933: the
    # ``client_kwargs = dict(client_kwargs)`` above means this injection only lands in the local per-call
    # copy, never back into ``agent._client_kwargs``. Each ``_create_openai_client`` invocation therefore
    # gets its OWN fresh ``httpx.Client`` whose lifetime is tied to the OpenAI client it is passed to. When
    # the OpenAI client is closed (rebuild, teardown, credential rotation), the paired ``httpx.Client``
    # closes with it, and the next call constructs a fresh one — no stale closed transport can be reused.
    if "http_client" not in client_kwargs:
        keepalive_http = agent._build_keepalive_http_client(client_kwargs.get("base_url", ""), verify=httpx_verify)
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http
    # Retries belong to the outer conversation loop (honors Retry-After); SDK retries would
    # double-retry inside it. auxiliary_client keeps SDK retries as it isn't wrapped.
    # Delegate all rate-limit / 5xx retry to hermes's outer conversation loop, which honors Retry-After and
    # applies adaptive/jittered backoff. The OpenAI SDK default (max_retries=2) uses its own 1-2s backoff
    # that ignores Retry-After and double-retries inside our loop — the same deadlock the Anthropic clients
    # hit (#26293). This is the single chokepoint every primary OpenAI/aggregator client passes through
    # (init, switch_model, recovery, restore, request-scoped); auxiliary_client builds its own clients and
    # keeps SDK retries because it is NOT wrapped by the conversation loop.
    client_kwargs.setdefault("max_retries", 0)
    _ensure_copilot_headers(client_kwargs)
    # OpenCode Free is served anonymously: any unrecognized bearer is a 401, so an empty
    # Authorization default_header overrides the SDK's "Bearer <api_key>".
    if agent.provider == "opencode-free":
        from hermes_cli.models import opencode_zen_free_headers
        client_kwargs["default_headers"] = {**(client_kwargs.get("default_headers") or {}), **opencode_zen_free_headers()}
    # All primary construction and recovery paths must identify Hermes to the official Codex
    # endpoint, including snapshots with custom header overrides.
    from agent.codex_headers import apply_required_codex_headers
    apply_required_codex_headers(
        client_kwargs, access_token=client_kwargs.get("api_key", ""),
        base_url=str(client_kwargs.get("base_url", "")),
    )
    # ``process_bootstrap.OpenAI`` is a lazy SDK proxy; resolved at call time so tests can patch it.
    from agent import process_bootstrap
    client = process_bootstrap.OpenAI(**client_kwargs)
    _ra().logger.info("OpenAI client created (%s, shared=%s) %s", reason, shared, agent._client_log_context())
    return client


def _apply_switched_provider_request_overrides(agent, new_provider):
    """Re-derive the switched-to provider's ``request_overrides`` (custom_providers ``extra_body``).
    Matches by provider key, base_url AND model (same rule as
    ``agent_init._merge_custom_provider_extra_body``) so a different model at the same endpoint
    never inherits another's ``extra_body``. Stale ``extra_body`` cleared; ``service_tier``/``speed`` kept."""
    from agent.agent_init import _custom_provider_extra_body_for_agent
    # Prefer the init-time cache (agent._custom_providers); reload only if absent.
    custom_providers = getattr(agent, "_custom_providers", None)
    if custom_providers is None:
        try:
            from hermes_cli.config import load_config, get_compatible_custom_providers
            custom_providers = get_compatible_custom_providers(load_config())
        except Exception:
            custom_providers = []
    new_extra_body = _custom_provider_extra_body_for_agent(
        provider=new_provider, model=getattr(agent, "model", "") or "",
        base_url=getattr(agent, "base_url", "") or "", custom_providers=custom_providers or [],
    )
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    overrides.pop("extra_body", None)  # always drop the previous provider's extra_body
    if new_extra_body:
        overrides["extra_body"] = dict(new_extra_body)
    agent.request_overrides = overrides


# Pool reload is part of the switch and must be reversible on rollback, hence the pool fields.
_SWITCH_SNAPSHOT_FIELDS = (
    "model", "provider", "requested_provider", "base_url", "api_mode", "api_key", "client",
    "_anthropic_client", "_anthropic_api_key", "_anthropic_base_url", "_is_anthropic_oauth",
    "_config_context_length", "_reasoning_echo_flag", "runtime_capabilities",
    "_credential_pool", "_credential_pool_entry_id",
)
_MISSING = object()


def _snapshot_switch_state(agent) -> Dict[str, Any]:
    """Snapshot every field the swap+rebuild mutates so a failed rebuild rolls back atomically
    (else a new model name + OLD client 400s next turn). The sentinel distinguishes unset from
    None: tests build bare agents via ``__new__`` without all fields."""
    snapshot = {name: getattr(agent, name, _MISSING) for name in _SWITCH_SNAPSHOT_FIELDS}
    # Shallow-copy the dict so mutating the live one doesn't poison the rollback target.
    snapshot["_client_kwargs"] = dict(getattr(agent, "_client_kwargs", {}) or {})
    return snapshot


def _restore_switch_snapshot(agent, snapshot: Dict[str, Any]) -> None:
    for name, value in snapshot.items():
        if value is _MISSING:
            continue  # attribute did not exist before the swap; don't fabricate it
        with contextlib.suppress(Exception):
            setattr(agent, name, value)


def _resolve_switch_destination(agent, new_model, new_provider, base_url, api_mode, capabilities, old_norm, new_norm):
    """Resolve ``(api_mode, base_url, destination_capabilities)`` for the switch target."""
    from hermes_cli.providers import determine_api_mode
    from agent.native_compaction import resolve_native_compaction_capabilities
    from hermes_cli.models import opencode_provider_family
    # Pass model so dual-wire providers (Nous Portal anthropic/* -> Messages) resolve correctly.
    if not api_mode:
        api_mode = determine_api_mode(new_provider, base_url, model=new_model)
    if not base_url and new_norm == "openai":
        # An omitted URL means the provider's canonical direct endpoint.
        base_url = "https://api.openai.com/v1"
    # Same-provider switches may omit base_url (e.g. credential refresh); resolve capabilities from
    # the endpoint the normalization below retains.
    effective_base_url = base_url
    if not effective_base_url and old_norm == new_norm:
        effective_base_url = getattr(agent, "base_url", "")
    destination_capabilities = (
        dict(capabilities)
        if isinstance(capabilities, dict)
        else resolve_native_compaction_capabilities(
            model=new_model, base_url=effective_base_url, provider=new_provider,
            is_codex_backend=new_norm == "openai-codex",
        )
    )
    # Guard against a trailing /v1 on OpenCode base_url reaching the anthropic_messages client
    # (double-/v1 404); model_switch already strips it, direct callers may not.
    if (
        api_mode == "anthropic_messages"
        and opencode_provider_family(new_provider) is not None
        and isinstance(base_url, str)
        and base_url
    ):
        base_url = re.sub(r"/v1/?$", "", base_url)
    return api_mode, base_url, destination_capabilities


def _build_switched_client(agent, new_provider, api_key, base_url, api_mode, new_norm) -> None:
    """Build the client for the switched-to destination (MoA facade / native Anthropic / OpenAI wire)."""
    if new_norm == "moa":
        from agent.moa_loop import build_moa_facade
        # MoA speaks only chat.completions via the MoAClient facade; the aggregator's real transport
        # is applied inside the fan-out. Pin api_mode so the loop never dispatches
        # client.responses.create against the facade (matches agent_init.py).
        agent.api_mode = "chat_completions"
        agent.api_key = api_key or "moa-virtual-provider"
        agent.base_url = "moa://local"
        agent._client_kwargs = {}
        agent.client = build_moa_facade(agent, agent.model)
        return
    if api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_client
        from agent.anthropic_credentials import resolve_anthropic_token, _is_oauth_token
        # Only fall back to ANTHROPIC_TOKEN for native Anthropic; other anthropic_messages providers
        # must never receive Anthropic credentials.
        is_native_anthropic = new_provider == "anthropic"
        effective_key = api_key or agent.api_key or (resolve_anthropic_token() if is_native_anthropic else "") or ""
        # MiniMax OAuth: per-request callable token provider survives 15-min expiry (rationale in
        # agent_init.py).
        if new_provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
            try:
                from hermes_cli.auth import build_minimax_oauth_token_provider
                effective_key = build_minimax_oauth_token_provider()
            except Exception as _mm_exc:  # noqa: BLE001
                logger.warning(
                    "MiniMax OAuth: failed to install per-request token provider "
                    "on switch (%s); using static bearer.", _mm_exc,
                )
        agent.api_key = agent._anthropic_api_key = effective_key
        agent._anthropic_base_url = base_url or getattr(agent, "_anthropic_base_url", None)
        agent._anthropic_client = build_anthropic_client(
            effective_key, agent._anthropic_base_url,
            timeout=get_provider_request_timeout(agent.provider, agent.model),
        )
        agent._is_anthropic_oauth = bool(is_native_anthropic and isinstance(effective_key, str) and _is_oauth_token(effective_key))
        agent.client = None
        agent._client_kwargs = {}
        return
    effective_base = base_url or agent.base_url
    agent._client_kwargs = {"api_key": api_key or agent.api_key, "base_url": effective_base}
    try:
        from hermes_cli.config import (
            apply_custom_provider_tls_to_client_kwargs, get_compatible_custom_providers,
            load_config_readonly,
        )
        # Read live config, not agent._custom_providers, so mid-session ssl_ca_cert / ssl_verify
        # edits are honored.
        # Read custom_providers from live config (not the init-time snapshot on ``agent._custom_providers``)
        # so ssl_ca_cert / ssl_verify edits are honored when switching mid-session, matching the
        # context-length reload below (#15779).
        apply_custom_provider_tls_to_client_kwargs(
            agent._client_kwargs, str(effective_base or ""),
            get_compatible_custom_providers(load_config_readonly()),
        )
    except Exception:
        logger.debug("custom-provider TLS resolution skipped on switch_model", exc_info=True)
    timeout = get_provider_request_timeout(agent.provider, agent.model)
    if timeout is not None:
        agent._client_kwargs["timeout"] = timeout
    # Reapply provider headers (OpenRouter HTTP-Referer/X-Title) lost when _client_kwargs was
    # rebuilt; otherwise attribution shows "Unknown".
    agent._apply_client_headers_for_base_url(effective_base)
    agent.client = agent._create_openai_client(dict(agent._client_kwargs), reason="switch_model", shared=True)


def _swap_switch_runtime(agent, new_model, new_provider, api_key, base_url, api_mode, old_provider, old_norm, new_norm) -> None:
    """Swap identity/transport fields, reload the pool, rebuild the client (rolled back by the caller on error)."""
    # Clear the per-config override so the new model's context window is re-resolved.
    agent._config_context_length = None
    agent.model = new_model
    agent.provider = agent.requested_provider = new_provider
    # Re-read reasoning_echo so the flag reflects the new primary model (see _reasoning_echo_opt_in).
    agent._reasoning_echo_flag = agent._read_reasoning_echo_from_config()
    # Empty base_url while the provider changes means upstream resolution failed; falling back to
    # the old provider's URL pairs the wrong host and persists via _primary_runtime. Fail loud.
    # Same-provider re-select (credential refresh) may keep the URL.
    if base_url:
        agent.base_url = base_url
    elif old_norm != new_norm:
        raise ValueError(
            f"switch_model: no base_url resolved for provider "
            f"'{new_provider}' (switching from '{old_provider}'); "
            "refusing to keep the previous provider's endpoint"
        )
    agent.api_mode = api_mode
    # New api_mode may need a different transport.
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    if api_key:
        agent.api_key = api_key
    # Reload the credential pool on provider change: a pool with a mismatched provider makes
    # recover_with_credential_pool short-circuit. Reload failure is non-fatal.
    if old_norm != new_norm or getattr(agent, "_credential_pool", None) is None:
        # A pool bound to the old provider is worse than none: the recovery guard rejects it.
        agent._credential_pool = None
        agent._credential_pool_entry_id = None
        try:
            from agent.credential_pool import load_pool
            agent._credential_pool = load_pool(new_provider)
        except Exception as _pool_exc:  # noqa: BLE001
            logger.warning(
                "switch_model: credential pool reload failed for %s (%s); "
                "continuing without pool rotation this turn", new_provider, _pool_exc,
            )
    _build_switched_client(agent, new_provider, api_key, base_url, api_mode, new_norm)
    sync_credential_pool_entry_id(agent)


def _resolve_switch_context_length(agent, snapshot):
    """Resolve the destination context length (LM Studio preload first); returns ``(custom_providers, effective_len)``."""
    custom_providers = None
    try:
        from hermes_cli.config import (
            get_compatible_custom_providers, get_custom_provider_context_length, load_config
        )
        custom_providers = get_compatible_custom_providers(load_config())
        intent = get_custom_provider_context_length(
            model=agent.model, base_url=agent.base_url, custom_providers=custom_providers
        )
    except Exception:
        intent = None
    agent._config_context_length = intent
    runtime_len = None
    if hasattr(agent, "_ensure_lmstudio_runtime_loaded"):
        try:
            runtime_len = agent._ensure_lmstudio_runtime_loaded(intent)
        except Exception:
            _restore_switch_snapshot(agent, snapshot)
            raise
    if hasattr(agent, "_lmstudio_load_was_unverified") and agent._lmstudio_load_was_unverified(runtime_len):
        logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length during model switch; continuing "
            "with configured context"
        )
    effective = intent
    if hasattr(agent, "_effective_lmstudio_context_length"):
        effective = agent._effective_lmstudio_context_length(intent, runtime_len)
    return custom_providers, effective


def _update_switch_compressor(agent, custom_providers, effective_context_length, snapshot) -> None:
    """Point the context compressor at the new model (rolls back the switch on failure)."""
    from agent.model_metadata import get_model_context_length
    if custom_providers is None:
        try:
            from hermes_cli.config import get_compatible_custom_providers, load_config
            custom_providers = get_compatible_custom_providers(load_config())
        except Exception:
            custom_providers = None
    # agent.api_key may be a callable (Azure Foundry Entra ID); get_model_context_length expects a
    # string for live probes, so coerce defensively.
    ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
    try:
        new_context_length = get_model_context_length(
            agent.model, base_url=agent.base_url, api_key=ctx_api_key, provider=agent.provider,
            config_context_length=effective_context_length, custom_providers=custom_providers,
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
        _restore_switch_snapshot(agent, snapshot)
        raise


def _build_primary_runtime_snapshot(agent, api_mode) -> Dict[str, Any]:
    """The ``_primary_runtime`` record that persists a switch across turns."""
    cc = getattr(agent, "context_compressor", None) or None
    rt = {
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
        # Overrides must travel with the switched-to identity or a later recovery/restore resurrects
        # PRE-switch overrides from the stale init snapshot.
        # See #75091.
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "runtime_capabilities": dict(getattr(agent, "runtime_capabilities", {}) or {}),
        "compressor_model": getattr(cc, "model", agent.model),
        "compressor_base_url": getattr(cc, "base_url", agent.base_url),
        "compressor_api_key": getattr(cc, "api_key", ""),
        "compressor_provider": getattr(cc, "provider", agent.provider),
        "compressor_context_length": cc.context_length if cc else 0,
        "compressor_api_mode": getattr(cc, "api_mode", agent.api_mode),
        "compressor_threshold_tokens": cc.threshold_tokens if cc else 0,
    }
    if api_mode == "anthropic_messages":
        rt.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })
    return rt


def _finish_switch(agent, new_provider, old_norm, new_norm) -> None:
    """Post-switch bookkeeping: fallback reset/prune, request_overrides, billing route."""
    agent._fallback_activated = False
    agent._provider_fallback_active = False
    agent._provider_fallback_route = None
    agent._fallback_index = 0
    # On a deliberate provider swap, prune fallback entries targeting the OLD or NEW primary;
    # otherwise a failed turn silently re-activates the provider the user just rejected.
    fallback_chain = list(getattr(agent, "_fallback_chain", []) or [])
    if old_norm and new_norm and old_norm != new_norm:
        fallback_chain = [
            entry for entry in fallback_chain
            if (entry.get("provider") or "").strip().lower() not in {old_norm, new_norm}
        ]
    agent._fallback_chain = fallback_chain
    agent._fallback_model = fallback_chain[0] if fallback_chain else None
    # Apply the switched-to provider's request_overrides (custom_providers extra_body).
    try:
        _apply_switched_provider_request_overrides(agent, new_provider)
    except Exception:
        logger.debug("switch_model: request_overrides re-derivation failed", exc_info=True)


def _persist_switch_billing_route(agent) -> None:
    """Persist the billing route so dashboard Model cards show the post-switch provider."""
    # _session_db / session_id may be unset (tests, bare agents).
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db is None or not session_id:
        return
    try:
        session_db.update_session_billing_route(
            session_id, provider=agent.provider, base_url=agent.base_url,
            billing_mode=getattr(agent, "api_mode", None),
        )
    except Exception:
        logger.warning("Failed to persist billing route after model switch", exc_info=True)


def switch_model(
    agent, new_model, new_provider, api_key='', base_url='', api_mode='', capabilities=None
):
    """Switch the model/provider in-place for a live agent (rebuild clients, caching flags,
    compressor). Mirrors ``_try_activate_fallback()`` but also updates ``_primary_runtime`` so
    the change persists across turns. A failed swap/rebuild rolls back to the pre-switch
    snapshot and re-raises (callers catch)."""
    old_model = agent.model
    old_provider = agent.provider
    # ── Reload credential pool for the new provider (issue #52727) ── Without this,
    # ``recover_with_credential_pool`` sees a ``pool.provider != agent.provider`` mismatch and
    # short-circuits, leaving the new provider with no rotation/recovery on 401/429 and burning the original
    # pool's entries. Only reload when the provider actually changed (or the pool was missing) —
    # re-selecting the same provider must not churn the pool reference. A reload failure is logged +
    # swallowed: the switch itself must still complete.
    old_norm = (old_provider or "").strip().lower()
    new_norm = (new_provider or "").strip().lower()
    api_mode, base_url, destination_capabilities = _resolve_switch_destination(
        agent, new_model, new_provider, base_url, api_mode, capabilities, old_norm, new_norm
    )
    snapshot = _snapshot_switch_state(agent)
    try:
        _swap_switch_runtime(
            agent, new_model, new_provider, api_key, base_url, api_mode, old_provider, old_norm, new_norm
        )
    except Exception:
        _restore_switch_snapshot(agent, snapshot)
        raise
    custom_providers, effective_context_length = _resolve_switch_context_length(agent, snapshot)
    # Refresh the custom-provider snapshot from the config just loaded so the prompt_caching lookup
    # sees flags added to config.yaml after session start.
    if custom_providers is not None:
        agent._custom_providers = custom_providers
    agent._use_prompt_caching, agent._use_native_cache_layout = agent._anthropic_prompt_cache_policy(
        provider=new_provider, base_url=agent.base_url, api_mode=api_mode, model=new_model
    )
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        _update_switch_compressor(agent, custom_providers, effective_context_length, snapshot)
    # Re-read the per-model reasoning_effort override so it applies immediately (per-model > global;
    # YAML False = disabled).
    try:
        from hermes_constants import resolve_reasoning_config
        from hermes_cli.config import load_config as _sm_load_config
        agent.reasoning_config = resolve_reasoning_config(_sm_load_config() or {}, agent.model)
        logger.info(
            "switch_model: reasoning_config resolved for %s: %s", agent.model, agent.reasoning_config
        )
    except Exception as _reasoning_err:
        logger.debug("switch_model: could not re-resolve reasoning_config: %s", _reasoning_err)
    # Invalidate the cached system prompt so it rebuilds next turn.
    agent._cached_system_prompt = None
    # Publish the destination capability map only after every runtime setup above has succeeded.
    # Failed switches must leave the old map intact.
    agent.runtime_capabilities = destination_capabilities
    # Reset the cross-turn stale-call circuit breaker; otherwise the latched streak keeps
    # short-circuiting the freshly selected healthy provider.
    from agent.chat_completion_helpers import _reset_stale_streak
    _reset_stale_streak(agent)
    agent._primary_runtime = _build_primary_runtime_snapshot(agent, api_mode)
    _finish_switch(agent, new_provider, old_norm, new_norm)
    logger.info(
        "Model switched in-place: %s (%s) -> %s (%s)",
        old_model, old_provider, new_model, new_provider,
    )
    _persist_switch_billing_route(agent)


def _pre_tool_block_message(agent, function_name, function_args, effective_task_id, tool_call_id, middleware_trace):
    """Plugin pre-tool-call hook verdict: ``(block_message, function_args)``; failures never block."""
    try:
        from hermes_cli.plugins import _dispatch_pre_tool_call_hooks
        block_message, modified_args = _dispatch_pre_tool_call_hooks(
            function_name, function_args, task_id=effective_task_id or "",
            session_id=getattr(agent, "session_id", "") or "", tool_call_id=tool_call_id or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            api_request_id=getattr(agent, "_current_api_request_id", "") or "",
            middleware_trace=list(middleware_trace),
        )
        return block_message, (modified_args if modified_args is not None else function_args)
    except Exception:
        return None, function_args


def invoke_tool(agent, function_name: str, function_args: dict, effective_task_id: str,
                 tool_call_id: Optional[str] = None, messages: list = None,
                 pre_tool_block_checked: bool = False,
                 skip_tool_request_middleware: bool = False,
                 tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
                 skip_tool_execution_middleware: bool = False) -> str:
    """Invoke a single tool (agent-level or registry-dispatched) and return the result string;
    no display logic. Used by the concurrent path; the sequential path keeps its own inline
    invocation for display."""
    from agent.inline_tool_executors import (
        InlineToolContext, emit_terminal_post_tool_call, resolve_invoke_tool_executor, tool_hook_ids
    )
    if not isinstance(function_args, dict):
        function_args = {}
    hook_ids = tool_hook_ids(agent, effective_task_id, tool_call_id)
    _tool_middleware_trace = list(tool_request_middleware_trace or [])
    try:
        from hermes_cli.middleware import apply_tool_request_middleware
        if not skip_tool_request_middleware:
            _tool_request_mw = apply_tool_request_middleware(function_name, function_args, **hook_ids)
            function_args = _tool_request_mw.payload
            _tool_middleware_trace = _tool_request_mw.trace
    except Exception as _mw_err:
        logger.debug("tool_request middleware error: %s", _mw_err)
    block_message: Optional[str] = None
    if not pre_tool_block_checked:
        block_message, function_args = _pre_tool_block_message(
            agent, function_name, function_args, effective_task_id, tool_call_id, _tool_middleware_trace
        )
    if block_message is not None:
        result = json.dumps({"error": block_message}, ensure_ascii=False)
        emit_terminal_post_tool_call(
            agent, function_name=function_name, function_args=function_args, result=result,
            effective_task_id=effective_task_id, tool_call_id=tool_call_id, status="blocked",
            error_type="plugin_block", error_message=block_message,
            middleware_trace=_tool_middleware_trace,
        )
        return result
    tool_start_time = time.monotonic()
    inline_executor = resolve_invoke_tool_executor(agent, function_name)
    if inline_executor is not None:
        inline_ctx = InlineToolContext(
            effective_task_id=effective_task_id, tool_call_id=tool_call_id, messages=messages
        )

        def _execute(next_args: dict) -> Any:
            result = inline_executor(agent, next_args, inline_ctx)
            emit_terminal_post_tool_call(
                agent, function_name=function_name,
                function_args=next_args if isinstance(next_args, dict) else function_args,
                result=result, effective_task_id=effective_task_id, tool_call_id=tool_call_id,
                duration_ms=int((time.monotonic() - tool_start_time) * 1000),
                middleware_trace=_tool_middleware_trace,
            )
            return result
    else:
        def _execute(next_args: dict) -> Any:
            dispatch_kwargs = dict(
                tool_call_id=tool_call_id, session_id=agent.session_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                tool_request_middleware_trace=list(_tool_middleware_trace),
            )
            if skip_tool_execution_middleware:
                dispatch_kwargs["skip_tool_execution_middleware"] = True
            import model_tools
            if function_name in {"request_user_input", "check_user_input"}:
                dispatch_kwargs["session_db"] = getattr(agent, "_session_db", None)
            return model_tools.handle_function_call(function_name, next_args, effective_task_id, **dispatch_kwargs)
    if skip_tool_execution_middleware:
        return _execute(function_args)
    from hermes_cli.middleware import run_tool_execution_middleware
    return run_tool_execution_middleware(
        function_name, function_args,
        lambda next_args: _execute(next_args if isinstance(next_args, dict) else function_args),
        original_args=function_args, **hook_ids,
    )


def repair_tool_call(agent, tool_name: str) -> str | None:
    """Repair a mismatched tool name (case, separators, CamelCase, ``_tool`` suffixes twice so
    ``TodoTool_tool`` reduces fully, then fuzzy match) before aborting. Returns the repaired
    name if in valid_tool_names, else None."""
    from difflib import get_close_matches
    if not tool_name:
        return None
    # VolcEngine api/plan leaks XML attribute fragments into tool_use.name (`terminal"
    # parameter="command" ...`); trim at the first quote/angle bracket. Do NOT split on whitespace:
    # "write file" must reach ``_norm`` -> ``write_file``.
    # `terminal" parameter="command" string="true` `execute_code" parameter="code" string="true`
    # `session_search" parameter="session_id" string="true` We trim at the first unambiguous XML/quote
    # character so the rest of the repair pipeline (lowercase / snake_case / fuzzy match) can resolve the
    # cleaned name to a real tool. Crucially we DO NOT split on whitespace: legitimate inputs like "write
    # file" must keep flowing through ``_norm`` -> ``write_file`` (covered by test_space_to_underscore in
    # tests/run_agent/test_repair_tool_call_name.py). See #33007.
    for _xml_sep in ('"', "'", "<", ">"):
        _idx = tool_name.find(_xml_sep)
        if _idx > 0:
            tool_name = tool_name[:_idx]
    if not tool_name:
        return None
    _norm = lambda s: s.lower().replace("-", "_").replace(" ", "_")  # noqa: E731
    _camel_snake = lambda s: re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()  # noqa: E731

    def _strip_tool_suffix(s: str) -> str | None:
        lc = s.lower()
        return next((s[: -len(sfx)].rstrip("_-") for sfx in ("_tool", "-tool", "tool") if lc.endswith(sfx)), None)
    # Cheap fast-paths first.
    lowered = tool_name.lower()
    if lowered in agent.valid_tool_names:
        return lowered
    normalized = _norm(tool_name)
    if normalized in agent.valid_tool_names:
        return normalized
    cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
    for _ in range(2):  # strip trailing tool-suffix up to twice (TodoTool_tool needs it)
        extra: set[str] = set()
        for c in cands:
            stripped = _strip_tool_suffix(c)
            if stripped:
                extra.update((stripped, _norm(stripped), _camel_snake(stripped)))
        cands |= extra
    for c in cands:
        if c and c in agent.valid_tool_names:
            return c
    matches = get_close_matches(lowered, agent.valid_tool_names, n=1, cutoff=0.7)
    return matches[0] if matches else None


# Placeholder for an empty non-final message the provider would reject. Kept identical to the stub
# placeholder in chat_completion_helpers so healed transcripts read consistently.
_INTERRUPTED_PLACEHOLDER = "[response interrupted]"

# Escalate repeated heals once per session window, then stay quiet. Default threshold; tunable via
# ``agent.sanitizer_heal_escalation_threshold`` (<= 0 disables).
# Repeated heals of the same poisoned transcript used to WARNING on every send (#96870).
# ``_EMPTY_HEAL_ESCALATE_AFTER`` is the built-in default; deployments tune it via
# ``agent.sanitizer_heal_escalation_threshold`` in config.yaml (<= 0 disables escalation entirely — WARNINGs
# still fire per window).
_EMPTY_HEAL_ESCALATE_AFTER = 3
_EMPTY_HEAL_WINDOW_S = 600.0
_empty_heal_log_state: Dict[str, Dict[str, Any]] = {}
_empty_heal_log_lock = threading.Lock()
# Sessions already told ONCE (out-of-band, never in conversation context); kept apart from the
# windowed log state so a new window never re-arms the notice.
# Session keys that already received the one-time user notice. Separate from the windowed log state so a new
# 10-minute window never re-notifies: the user is told ONCE per session, ever (#96870 — out-of-band,
# delivery channel only, never injected into conversation context).
_empty_heal_user_notified: set = set()
# One-shot pending notices keyed by session, drained via ``consume_pending_sanitizer_heal_notice``
# and delivered via the status/warning callback.
_empty_heal_pending_notice: Dict[str, str] = {}


def _content_has_payload(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return content not in (None, "")
    # Any typed block counts, as long as a text block is not itself blank.
    return any(
        (block.get("type") != "text" or (isinstance(block.get("text"), str) and block["text"].strip()))
        if isinstance(block, dict) else bool(block)
        for block in content
    )


def _msg_has_payload(msg: Dict[str, Any]) -> bool:
    """True if ``msg`` carries anything the API treats as non-empty content (text, multimodal
    blocks, tool_calls, reasoning). Role-agnostic counterpart of ``AIAgent._is_thinking_only_assistant``.
    Codex Responses item carriers persist with content:"" by design (text lives in codex_*_items
    and is replayed); treating them as payload keeps the repair from rewriting a designed-empty turn."""
    return _content_has_payload(msg.get("content")) or bool(
        msg.get("tool_calls")
        or (isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"].strip())
        or msg.get("reasoning")
        or msg.get("reasoning_details")
        or msg.get("codex_message_items")
        or msg.get("codex_reasoning_items")
    )


def fill_empty_non_final_wire_payload(msg: Dict[str, Any], *, is_final: bool) -> bool:
    """Write the interrupted placeholder onto an empty non-final wire copy; True when filled.
    Pass the per-call copy only; durable history must not be mutated."""
    if is_final or not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
        return False
    if _msg_has_payload(msg):
        return False
    msg["content"] = _INTERRUPTED_PLACEHOLDER
    return True


def _session_id_for_heal_log() -> str:
    try:
        from hermes_logging import _session_context
        return str(getattr(_session_context, "session_id", None) or "")
    except Exception:
        return ""


def _heal_escalation_threshold() -> int:
    """Escalation threshold from ``agent.sanitizer_heal_escalation_threshold``, else the module default (fail-safe on any read error)."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly().get("agent", {}) or {}).get("sanitizer_heal_escalation_threshold")
        if raw is not None:
            return int(raw)
    return _EMPTY_HEAL_ESCALATE_AFTER


def consume_pending_sanitizer_heal_notice() -> Optional[str]:
    """Drain the one-time user notice for the current session (at most one per session lifetime).
    Delivered through the status/warning callback, NEVER appended to the conversation context."""
    key = _session_id_for_heal_log() or "-"
    with _empty_heal_log_lock:
        return _empty_heal_pending_notice.pop(key, None)


def get_sanitizer_heal_stats() -> Dict[str, Dict[str, Any]]:
    """Read-only per-session sanitiser heal counters (``heal_events``, ``messages_healed``, ``escalated``) for diagnostics."""
    with _empty_heal_log_lock:
        return {
            k: {
                "heal_events": v.get("total_events", v.get("count", 0)),
                "messages_healed": v.get("total_healed", 0),
                "escalated": k in _empty_heal_user_notified,
            }
            for k, v in _empty_heal_log_state.items()
        }


def _log_empty_non_final_heal(healed: int) -> None:
    """WARNING on the first heals in a window, one ERROR at the threshold, then silent. The
    threshold also queues a ONE-TIME out-of-band user notice (drained by
    ``consume_pending_sanitizer_heal_notice``); never re-armed by a new window."""
    key = _session_id_for_heal_log() or "-"
    threshold = _heal_escalation_threshold()
    now = time.monotonic()
    with _empty_heal_log_lock:
        state = _empty_heal_log_state.get(key)
        if state is None or (now - state["window_start"]) > _EMPTY_HEAL_WINDOW_S:
            prior = state or {}
            state = _empty_heal_log_state[key] = {
                "count": 0, "window_start": now, "escalated": False,
                "total_events": prior.get("total_events", 0), "total_healed": prior.get("total_healed", 0),
            }
        state["count"] += 1
        state["total_events"] = state.get("total_events", 0) + 1
        state["total_healed"] = state.get("total_healed", 0) + healed
        count, total_events, total_healed = state["count"], state["total_events"], state["total_healed"]
        if state["escalated"]:
            return
        escalate = threshold > 0 and count >= threshold
        if escalate:
            state["escalated"] = True
            if key not in _empty_heal_user_notified:
                _empty_heal_user_notified.add(key)
                _empty_heal_pending_notice[key] = (
                    "⚠️ Your session transcript required repeated repair "
                    f"({total_events} heal passes so far). Replies keep "
                    "working, but a corrupted turn is stuck in this "
                    "session's history — run /debug share or `hermes "
                    "doctor` to capture diagnostics, or /new to start a clean session."
                )
    if escalate:
        _ra().logger.error(
            "Pre-call sanitizer: repeated-heal escalation for session %s — "
            "healed %d empty non-final message(s) this send; heal pattern: "
            "%d heal events / %d messages healed this session "
            "(%d in the current session window, threshold %d). The transcript "
            "is being repaired on every send; /new drops the poisoned turns.", key, healed,
            total_events, total_healed, count, threshold,
        )
        return
    _ra().logger.warning(
        "Pre-call sanitizer: healed %d empty non-final message(s) by "
        "substituting placeholder content — an empty-content turn was in "
        "the transcript and would 400 the request ('messages must have "
        "non-empty content' / INVALID_REQUEST_BODY). Self-recovering the "
        "poisoned transcript in memory; no restart needed.", healed,
    )


def repair_empty_non_final_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Substitute a placeholder for empty-content non-final messages on the per-call copy.
    Anthropic/litellm/Bedrock 400 on any empty non-final message and a persisted stub poisons
    every later turn; repairing the wire copy heals the session in memory. Substitution (not
    deletion) keeps role alternation and tool-call pairing intact. The final message is untouched."""
    if not messages or len(messages) < 2:
        return messages
    repaired: List[Dict[str, Any]] = []
    healed = 0
    last_idx = len(messages) - 1
    for idx, msg in enumerate(messages):
        # Tool results are checked by their own pairing pass; empty ones are a separate concern.
        if idx != last_idx and isinstance(msg, dict) and msg.get("role") in ("assistant", "user") and not _msg_has_payload(msg):
            # Shallow-copy so stored history / prompt caching stays byte-stable.
            repaired.append({**msg, "content": _INTERRUPTED_PLACEHOLDER})
            healed += 1
        else:
            repaired.append(msg)
    if healed:
        _log_empty_non_final_heal(healed)
        return repaired
    return messages


def _classify_tool_call_orphans(messages: List[Dict[str, Any]]):
    """Classify orphaned tool-call / tool-result pairs; single source of truth for GLOBAL orphan
    detection. Returns ``(surviving_call_ids, result_call_ids, orphaned_results, missing_tool_calls)``;
    every id variant of a tool_call is registered so a result matching any alias survives, and
    ``orphaned_results`` are the actual dicts (filter by ``id(msg)``). ``sanitize_api_messages``
    pairs positionally instead but shares the ``*_id_variants`` alias policy."""
    assistant_call_variants = [
        (tc, variants)
        for msg in messages if msg.get("role") == "assistant"
        for tc in msg.get("tool_calls") or []
        if (variants := tool_call_id_variants(tc))
    ]
    surviving_call_ids: set[str] = set().union(*(v for _, v in assistant_call_variants))
    result_entries = [
        (msg, tool_result_id_variants(msg.get("tool_call_id"))) for msg in messages if msg.get("role") == "tool"
    ]
    result_call_ids: set[str] = set().union(*(v for _, v in result_entries))
    orphaned_results = [msg for msg, v in result_entries if v and not (v & surviving_call_ids)]
    orphaned_ids = {id(msg) for msg in orphaned_results}
    surviving_result_variants = [v for msg, v in result_entries if v and id(msg) not in orphaned_ids]
    missing_tool_calls = [
        tc for tc, v in assistant_call_variants if not any(v & rv for rv in surviving_result_variants)
    ]
    return surviving_call_ids, result_call_ids, orphaned_results, missing_tool_calls


def _drop_invalid_roles(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop messages whose role the API won't accept."""
    valid = _ra().AIAgent._VALID_API_ROLES
    for msg in messages:
        if msg.get("role") not in valid:
            _ra().logger.debug("Pre-call sanitizer: dropping message with invalid role %r", msg.get("role"))
    return [m for m in messages if m.get("role") in valid]


def _drop_empty_tool_calls_arrays(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strict providers 400 on ``tool_calls: []``; normalize on shallow copies so history stays byte-stable."""
    # --- Drop empty / malformed tool_calls arrays on assistant messages --- An assistant message carrying
    # ``tool_calls: []`` (an empty array) — or a non-list value under the key — is semantically identical to
    # an assistant message with no tool calls, but strict OpenAI-compatible providers reject the empty array
    # outright: DeepSeek v4 returns HTTP 400 "Invalid 'messages[N].tool_calls': empty array. Expected an
    # array with minimum length 1, but got an empty array instead." (#58755, follow-up to #56980). Empty
    # arrays reach here from session resume, host-fed histories, or the consecutive-assistant merge in
    # ``repair_message_sequence`` (which preserves a pre-existing ``[]`` on the surviving turn). This is the
    # final pre-API chokepoint, so normalize defensively — and, per the #56980 review, do it HERE on the
    # per-call copy rather than in ``repair_message_sequence``, which would destructively rewrite the
    # persisted trajectory. Shallow-copy the message before dropping the key so stored history (and prompt
    # caching) stays byte-stable.
    normalized: List[Dict[str, Any]] = []
    dropped = 0
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            # Defense-in-depth: a strict OpenAI-compatible provider (e.g. onerouter / Qwen, DeepSeek v4)
            # rejects an assistant message carrying ``tool_calls: []`` (empty array) with HTTP 400 "Empty
            # tool_calls is not supported in message." The pre-API sanitizer in agent_runtime_helpers drops
            # these, but only on the conversation_loop path — other routes can reach the wire without it.
            # For every request that serializes through this transport (conversation loop and any caller
            # using it), this is the last boundary, so normalize here. Requests built by fully separate
            # payload paths (e.g. some auxiliary clients) never pass through this layer and are out of scope
            # for it. (#58755 follow-up)
            and "tool_calls" in msg
            and not (isinstance(msg["tool_calls"], list) and msg["tool_calls"])
        ):
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            dropped += 1
        normalized.append(msg)
    if not dropped:
        return messages
    _ra().logger.debug(
        "Pre-call sanitizer: dropped empty/invalid tool_calls on %d assistant message(s)", dropped
    )
    return normalized


def _repair_nameless_tool_calls(messages: List[Dict[str, Any]]) -> None:
    """Rename empty/missing ``function.name`` to a sentinel (in place): dropping would unpair the
    anti-priming result the dispatch loop keeps for empty-name calls, and Responses adapters
    400 on nameless calls."""
    sentinel = "invalid_tool_call"
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function")
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
            if isinstance(name, str) and name.strip():
                continue
            _ra().logger.warning(
                "Pre-call sanitizer: repairing tool_call with empty function.name -> %r (id=%s)",
                sentinel, _ra().AIAgent._get_tool_call_id_static(tc),
            )
            if isinstance(fn, dict):
                fn["name"] = sentinel
            elif fn is not None and hasattr(fn, "name"):
                with contextlib.suppress(Exception):
                    fn.name = sentinel
            elif isinstance(tc, dict):
                tc["function"] = {"name": sentinel, "arguments": "{}"}


def _drop_results_without_ids(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop tool results with a missing/empty tool_call_id. Kept explicit (not left to the
    positional walk) for its own log line and so the final-chokepoint guarantee holds for
    callers skipping ``repair_message_sequence``."""
    kept = [
        m for m in messages
        if not (m.get("role") == "tool" and not (m.get("tool_call_id") or "").strip())
    ]
    if len(kept) != len(messages):
        _ra().logger.debug(
            "Pre-call sanitizer: dropped %d tool result(s) with missing/empty tool_call_id",
            len(messages) - len(kept),
        )
    return kept


def _pair_tool_calls_positionally(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Positional tool_call <-> tool_result pairing: strict providers (DeepSeek v4, Kimi) require
    results IMMEDIATELY after their call. Drops positional orphans, stubs unanswered declared
    ids; matching is alias-aware."""
    # --- Positional tool_call <-> tool_result pairing --- Strict OpenAI-compatible providers (DeepSeek v4,
    # Kimi) enforce the POSITIONAL invariant: an assistant message carrying tool_calls must be IMMEDIATELY
    # followed by tool messages covering every tool_call_id. The previous implementation compared global id
    # sets, which misses the failure mode where a result exists somewhere in the transcript but not in the
    # run right after its call — an interrupted turn or a compression window can displace a result past a
    # user turn. The id then survives in the global result set, so the call looks answered, no stub is
    # injected, and the provider rejects the payload with HTTP 400 "An assistant message with 'tool_calls'
    # must be followed by tool messages responding to each 'tool_call_id' (insufficient tool messages
    # following tool_calls message)". Rewritten as a single rolling walk on the per-call copy (#94704): (a)
    # tool results that do not immediately follow an assistant message declaring their id are dropped
    # (positional orphans — includes results appearing BEFORE their call, which strict providers also
    # reject); (b) declared ids not covered by the immediately-following tool run get a stub result injected
    # at the end of that run, even when a mispositioned result exists elsewhere. Matching is variant-aware
    # (``tool_call_id_variants`` / ``tool_result_id_variants``): a result keyed on ANY alias spelling
    # (``id`` / ``call_id`` / ``response_item_id`` / composite bridge) answers the call, preserving the
    # unified alias policy from #55626/#63000/#93251.
    paired: List[Dict[str, Any]] = []
    declared_calls: Dict[str, tuple] = {}
    dropped = 0
    stubs = 0

    def _flush_unanswered_stubs() -> None:
        nonlocal stubs
        for key in sorted(declared_calls):
            tc, _variants = declared_calls[key]
            paired.append({
                "role": "tool", "name": _ra().AIAgent._get_tool_call_name_static(tc),
                "content": "[Result unavailable — see context summary above]",
                "tool_call_id": coalesce_tool_call_id(tc) or key,
            })
            stubs += 1
        declared_calls.clear()

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            # A new assistant turn closes the previous tool-result run: anything still pending was
            # never answered positionally.
            _flush_unanswered_stubs()
            declared_calls = {}
            for tc in msg.get("tool_calls") or []:
                variants = tool_call_id_variants(tc)
                if variants:
                    # Key on a stable representative of the alias group so a result matching ANY
                    # spelling can consume the call.
                    declared_calls[sorted(variants)[0]] = (tc, variants)
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            matched = next((k for k, (_tc, v) in declared_calls.items() if v & result_variants), None)
            if matched is None:
                dropped += 1
                continue
            # Consume so a duplicate result reusing the id is dropped (strict providers reject duplicates).
            declared_calls.pop(matched, None)
        elif role == "user":
            # A user turn closes the tool-result run; later tool messages are orphans.
            _flush_unanswered_stubs()
        paired.append(msg)
    # The transcript may end right after an unanswered assistant turn.
    _flush_unanswered_stubs()
    if dropped:
        _ra().logger.debug("Pre-call sanitizer: removed %d positionally orphaned tool result(s)", dropped)
    if stubs:
        _ra().logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s) for "
            "positionally unanswered tool call(s)", stubs,
        )
    return paired if (dropped or stubs) else messages


def _dedupe_tool_call_ids(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate tool_call_ids (strict providers 400 on duplicates): collapse duplicates within
    an assistant message, drop results answering no OUTSTANDING call. Tracks outstanding calls
    (not ids ever seen) because llama.cpp reuses one constant id, and whole variant groups so
    alias-keyed results are not deleted."""
    outstanding: Dict[str, int] = {}  # every alias of an unanswered call -> its group id
    # 3. Deduplicate tool_call_ids. Strict providers (DeepSeek) reject a payload where the same tool_call_id
    #   appears more than once with HTTP 400 "Duplicate value for 'tool_call_id'" (#58327). Duplicates can
    #   arise from retries, crash/resume glitches, or a compression window that re-emits a tool result. This
    #   is the final pre-API chokepoint, so dedup defensively here even though repair_message_sequence also
    #   consumes matched ids. (a) collapse duplicate tool_calls WITHIN an assistant message (b) drop tool
    #   results that answer no OUTSTANDING tool call (b) tracks outstanding calls rather than every id ever
    #   seen, because ``tool_call_id`` is NOT globally unique in practice: llama.cpp emits a single constant
    #   id for every tool call it ever returns (verified: three separate completions from one server all
    #   carry the same id). A seen-once-drop-forever rule reads the SECOND legitimate tool result of such a
    #   session as a duplicate and deletes it, so from the second tool call onward the model never sees any
    #   result — it announces its next action and the turn dies with the work unfinished. Outstanding-call
    #   semantics keep both protections intact: a re-emitted result still answers no pending call and is
    #   still dropped, while a genuine new call that reuses the id re-arms that id first. Variant-group
    #   tracking: answering or deduping one spelling consumes its siblings too. A Codex/Responses tool_call
    #   registers ``id`` (fc_...), ``call_id`` (call_...), ``response_item_id``, and composite spellings
    #   (#55626/#58168/#63000); tracking only the coalesced id here made a result keyed on any OTHER variant
    #   look like it answered no outstanding call, so this pass deleted the very result step 2's
    #   variant-aware matching had just preserved (issue #93251 — whole parallel batches vanished).
    outstanding_groups: Dict[int, frozenset] = {}
    next_group_id = 0
    deduped: List[Dict[str, Any]] = []
    removed = 0
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            kept_tcs = []
            for tc in msg.get("tool_calls") or []:
                variants = tool_call_id_variants(tc)
                if variants and variants & outstanding.keys():
                    removed += 1
                    continue
                if variants:
                    group_id = next_group_id
                    next_group_id += 1
                    outstanding_groups[group_id] = variants
                    for variant in variants:
                        outstanding.setdefault(variant, group_id)
                kept_tcs.append(tc)
            if kept_tcs:
                msg = {**msg, "tool_calls": kept_tcs}
            elif len(kept_tcs) != len(msg.get("tool_calls") or []):
                msg = {k: v for k, v in msg.items() if k != "tool_calls"}
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            candidate_groups = {outstanding[v] for v in result_variants if v in outstanding}
            if result_variants and not candidate_groups:
                removed += 1
                continue
            if candidate_groups:
                # Consume EVERY variant of the matched call; ids are re-armed by the next call reusing them.
                # Consume the whole alias group so a SECOND result replaying any sibling spelling falls into
                # the drop branch below — strict providers reject duplicate tool_call_ids with HTTP 400
                # (#58327, #66974). Credit: #55436.
                group_id = min(candidate_groups)
                for variant in outstanding_groups.pop(group_id, frozenset()):
                    if outstanding.get(variant) == group_id:
                        del outstanding[variant]
        deduped.append(msg)
    if not removed:
        return messages
    _ra().logger.debug(
        "Pre-call sanitizer: removed %d duplicate tool_call_id reference(s)", removed
    )
    return deduped


def _realign_tool_result_names(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Align each tool result's wire ``name`` with its call's function name (per-call copy only):
    Google 400s on a mismatch, routine when tool_search bridges via ``tool_call``."""
    # 4. Google matches functionResponse.name against functionCall.name and rejects a mismatch with HTTP 400
    #   "Request contains an invalid argument" (INVALID_ARGUMENT); behind an OpenAI-compatible gateway that
    #   surfaces only as a generic "Provider returned error". When tool_search defers MCP/plugin tools the
    #   model calls the bridge tool ``tool_call``, while ``make_tool_result_message()`` labels the result
    #   with the unwrapped internal tool name (``mcp__github__create_issue``) that dispatch, hooks, logging,
    #   and guardrails need. #72089 fixed exactly this for the native Gemini adapter, which now prefers
    #   ``tool_name_by_call_id`` over the result name; requests that reach Gemini through the
    #   OpenAI-compatible path (OpenRouter, Vertex/LiteLLM proxies, any OpenAI-shaped gateway) skip that
    #   translation entirely and still send the internal name on the wire. Normalizing here rather than in
    #   the OpenAI-compat serializer keeps it provider-agnostic: Gemini reaches Hermes under many model
    #   strings and base URLs, so sniffing for "is this really Google?" is unreliable, and every other
    #   provider either ignores the field or agrees with the call name. Runs on the per-call copy, so the
    #   stored trajectory keeps the real tool name for the session DB and the UI — only the wire payload
    #   changes. A no-op for the native Gemini path, which already resolves the same name. A result whose
    #   assistant call frame is missing entirely never reaches here — pass 1 above drops it as an orphan —
    #   so the only results this pass sees are ones whose call name is knowable.
    call_names: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                # Strip on insert to match the lookup below so padded ids still pair.
                cid = (_ra().AIAgent._get_tool_call_id_static(tc) or "").strip()
                nm = _ra().AIAgent._get_tool_call_name_static(tc)
                if cid and nm:
                    call_names[cid] = nm
    realigned: List[Tuple[str, str]] = []
    aligned: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            expected = call_names.get((msg.get("tool_call_id") or "").strip())
            current = msg.get("name")
            # Only rewrite a present, disagreeing name; clean transcripts must stay byte-identical for prompt caching.
            if expected and current and current != expected:
                msg = {**msg, "name": expected}
                realigned.append((current, expected))
        aligned.append(msg)
    if not realigned:
        return messages
    _ra().logger.debug(
        "Pre-call sanitizer: realigned %d tool result name(s) with their "
        "tool_call function name (%s)", len(realigned),
        ", ".join(f"{was} -> {now}" for was, now in realigned),
    )
    return aligned


def sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs before every LLM call; runs unconditionally (not
    gated on the compressor). Order matters: empty non-final messages are healed first so the
    substituted turn participates in the pairing and dedup passes."""
    messages = _drop_invalid_roles(messages)
    messages = repair_empty_non_final_messages(messages)
    messages = _drop_empty_tool_calls_arrays(messages)
    _repair_nameless_tool_calls(messages)
    messages = _drop_results_without_ids(messages)
    messages = _pair_tool_calls_positionally(messages)
    messages = _dedupe_tool_call_ids(messages)
    return _realign_tool_result_names(messages)


_ACK_FUTURE_RE = re.compile(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b")
_ACK_ACTION_MARKERS = (
    "look into", "look at", "inspect", "scan", "check", "analyz", "review", "explore", "read", "open",
    "run", "test", "fix", "debug", "search", "find", "walkthrough", "report back", "summarize",
)
_ACK_WORKSPACE_MARKERS = (
    "directory", "current directory", "current dir", "cwd", "repo", "repository", "codebase",
    "project", "folder", "filesystem", "file tree", "files", "path",
)


def looks_like_codex_intermediate_ack(
    agent, user_message: Any, assistant_content: str, messages: List[Dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn.
    ``require_workspace=False`` (opt-in for all api_modes) drops the filesystem/repo reference
    requirement; future-ack + short-content + no-prior-tools + action-verb checks always apply."""
    if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
        return False
    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False
    if not _ACK_FUTURE_RE.search(assistant_text):
        return False
    if not any(marker in assistant_text for marker in _ACK_ACTION_MARKERS):
        return False
    # Opted-in (all-api_mode) path: future-ack + action verb + no prior tool call suffices.
    if not require_workspace:
        return True
    # ``user_message`` may be a multi-part content list (vision via the OpenAI-compat server); a
    # list survives ``or ""`` and ``.strip()`` raises, so flatten first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    return (
        any(marker in user_text for marker in _ACK_WORKSPACE_MARKERS)
        or "~/" in user_text
        or "/" in user_text
        or any(marker in assistant_text for marker in _ACK_WORKSPACE_MARKERS)
    )


# Narrow "trailing continue-intent" detector for the stall guard (agent.stall_guards): only the
# message TAIL announcing a next action, so mid-sentence "I will" never trips it.
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$", re.IGNORECASE,
)

# Content longer than this is a substantive reply, not a dangling ack.
_TRAILING_CONTINUE_INTENT_MAX_CHARS = 400


def trailing_continue_intent(text: str) -> bool:
    """Whether ``text`` is a short reply ENDING on an announced next action (stall-guard re-prompt trigger)."""
    t = (text or "").strip()
    if not t or len(t) > _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        return False
    return bool(_TRAILING_CONTINUE_INTENT_RE.search(t[-160:]))


_INTENT_ACK_ON = {"true", "always", "yes", "on"}
_INTENT_ACK_OFF = {"false", "never", "no", "off"}


def intent_ack_continuation_mode(agent) -> str:
    """Intent-ack continuation mode: ``"off"``, ``"codex_only"`` (workspace acks on codex_responses)
    or ``"all"``. Mirrors ``agent.tool_use_enforcement``: ``"auto"`` -> codex_only; true-ish -> all;
    false-ish -> off; ``list`` -> all when a substring matches the active model name, else off."""
    mode = getattr(agent, "_intent_ack_continuation", "auto")
    if mode is True or (isinstance(mode, str) and mode.lower() in _INTENT_ACK_ON):
        return "all"
    if mode is False or (isinstance(mode, str) and mode.lower() in _INTENT_ACK_OFF):
        return "off"
    if isinstance(mode, list):
        model_lower = (agent.model or "").lower()
        return "all" if any(p.lower() in model_lower for p in mode if isinstance(p, str)) else "off"
    # "auto" or any unrecognised value: historical codex-only behavior.
    return "codex_only" if agent.api_mode == "codex_responses" else "off"


def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:
    """Forward reasoning fields onto an API replay message; policy lives in ``agent.message_sanitization.apply_reasoning_content_policy``."""
    from agent.message_sanitization import apply_reasoning_content_policy
    apply_reasoning_content_policy(source_msg, api_msg, agent._needs_thinking_reasoning_pad())


def reapply_reasoning_echo_for_provider(agent, api_messages: list) -> int:
    """Re-pad or strip assistant turns' reasoning_content for the CURRENT provider after a
    fallback switch: ``api_messages`` is shaped for the primary; require-side providers
    (DeepSeek/Kimi/MiMo) 400 without the pad, strict ones (Mistral, Cerebras, Groq) 400/422
    with it. Idempotent; returns the number of assistant turns changed.

    * Switching TO a strict provider that rejects the field (Mistral, Cerebras, Groq, SambaNova, …):
    assistant turns built under a reasoning primary carry a ``reasoning_content`` pad (often a single space
    ``" "``), and the strict provider rejects it with HTTP 400/422 ("Extra inputs are not permitted"). This
    is the exact cross-provider fallback bug from #45655 — a DeepSeek primary pads history with ``" "``, the
    request falls back to Mistral, and Mistral 422s on the stale pad.
    """
    from agent.message_sanitization import reapply_reasoning_echo
    return reapply_reasoning_echo(api_messages, agent._needs_thinking_reasoning_pad())


def _iter_httpx_pools_with_owner(http_client: Any):
    """Yield ``(pool, owner)`` pairs reachable from an httpx client, including mounted transports:
    keepalive and proxy configs put live connections on ``client._mounts``, which a
    ``_transport``-only walk misses.

    ``owner`` is ``None`` for a pool this client owns outright, or the ``_SharedTransport`` view
    id when the pool is process-shared with other clients
    (``process_bootstrap.build_keepalive_http_client``). Callers must then touch only the
    in-flight requests stamped with that owner.

    Walking the default transport alone makes ``force_close_tcp_sockets`` return 0 while a stream is still
    mid-recv — the interrupt logs success and the provider keeps burning the slot (#72975).
    """
    seen_pools: set[int] = set()
    try:
        transports = [getattr(http_client, "_transport", None)]
        transports += list((getattr(http_client, "_mounts", None) or {}).values())
        for transport in transports:
            if transport is None:
                continue
            # Connections live under ``_pool``; a directly mounted HTTPProxy *is* a ConnectionPool,
            # so ``_connections`` may sit on the transport itself.
            pool = getattr(transport, "_pool", None)
            if pool is None and getattr(transport, "_connections", None) is not None:
                pool = transport
            if pool is not None and id(pool) not in seen_pools:
                seen_pools.add(id(pool))
                owner = id(transport) if type(transport).__name__ == "_SharedTransport" else None
                yield pool, owner
    except Exception:
        return


def _iter_httpx_pool_objects(http_client: Any):
    """Yield httpcore pool objects reachable from an httpx client."""
    for pool, _owner in _iter_httpx_pools_with_owner(http_client):
        yield pool


def _connection_candidates(conn: Any):
    """Walk nested ``_connection`` wrappers (proxy tunnel → HTTP11/2)."""
    seen: set[int] = set()
    while conn is not None and id(conn) not in seen:
        seen.add(id(conn))
        yield conn
        conn = getattr(conn, "_connection", None)


def _socket_from_stream(stream: Any):
    """Raw socket behind an httpcore network stream (several backends), or None."""
    sock = getattr(stream, "_sock", None)
    if sock is None and callable(getattr(stream, "get_extra_info", None)):
        with contextlib.suppress(Exception):
            sock = stream.get_extra_info("socket")
    if sock is None:
        sock = getattr(getattr(stream, "stream", None), "_sock", None)
    if sock is None and callable(getattr(getattr(stream, "_stream", None), "extra", None)):
        # anyio-backed streams expose the raw socket through SocketAttribute.raw_socket.
        with contextlib.suppress(Exception):
            from anyio.abc import SocketAttribute
            sock = stream._stream.extra(SocketAttribute.raw_socket)
    return sock


def _iter_pool_sockets(client: Any):
    """Yield raw sockets reachable from an OpenAI/httpx client pool. Defensive over private
    httpcore internals (``conn._connection``, proxy tunnel wrappers) that vary by release; also
    walks mount transports and in-flight ``PoolRequest.connection`` objects (``_connections``
    is empty during checkout)."""
    try:
        # Some SDK wrappers *are* the httpx client; fall through so mount-aware discovery runs.
        http_client = getattr(client, "_client", None)
        pools = list(_iter_httpx_pools_with_owner(client if http_client is None else http_client))
    except Exception:
        return
    if not pools:
        return
    from agent.process_bootstrap import HERMES_TRANSPORT_OWNER_EXT
    seen: set[int] = set()
    for pool, owner in pools:
        # ``is None``, not falsiness: an empty ``_connections`` must still let us walk in-flight ``_requests``.
        raw_conns = getattr(pool, "_connections", None)
        if raw_conns is None:
            raw_conns = getattr(pool, "_pool", None)
        # A process-shared pool carries other clients' idle + in-flight connections: only this
        # client's own in-flight requests (stamped by ``_SharedTransport.handle_request``) may be
        # shut down.
        connections = [] if owner is not None else list(raw_conns or [])
        for pool_req in list(getattr(pool, "_requests", None) or []):
            if owner is not None:
                exts = getattr(getattr(pool_req, "request", None), "extensions", None) or {}
                if exts.get(HERMES_TRANSPORT_OWNER_EXT) != owner:
                    continue
            conn = getattr(pool_req, "connection", None)
            if conn is not None:
                connections.append(conn)
        for conn in connections:
            for candidate in _connection_candidates(conn):
                stream = getattr(candidate, "_network_stream", None) or getattr(candidate, "_stream", None)
                sock = _socket_from_stream(stream) if stream is not None else None
                if sock is not None and id(sock) not in seen:
                    seen.add(id(sock))
                    yield sock


def _socket_is_dead(sock) -> bool:
    """Probe socket health with a non-blocking recv peek."""
    import socket as _socket
    try:
        sock.setblocking(False)
        return sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False  # no data available: socket is healthy
    except OSError:
        return True
    finally:
        with contextlib.suppress(OSError):
            sock.setblocking(True)


def cleanup_dead_connections(agent) -> bool:
    """Force-close and rebuild the primary client if its pool has dead sockets (CLOSE-WAIT, errors); returns True if cleaned."""
    client = getattr(agent, "client", None)
    if client is None:
        return False
    try:
        dead_count = sum(1 for sock in _iter_pool_sockets(client) if _socket_is_dead(sock))
        if dead_count > 0:
            _ra().logger.warning("Found %d dead connection(s) in client pool — rebuilding client", dead_count)
            agent._replace_primary_openai_client(reason="dead_connection_cleanup")
            return True
    except Exception as exc:
        _ra().logger.debug("Dead connection check error: %s", exc)
    return False


_QUOTA_RESET_DELAY_RE = re.compile(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", re.IGNORECASE)
_RESETS_IN_RE = re.compile(
    r"resets?\s+in\s+"
    r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?", re.IGNORECASE,
)
_RETRY_AFTER_SECONDS_RE = re.compile(r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)", re.IGNORECASE)


def _reset_delay_from_message(message: str) -> Optional[float]:
    """Seconds-until-reset parsed from free-text provider messages, or None."""
    m = _QUOTA_RESET_DELAY_RE.search(message)
    if m:
        value = float(m.group(1))
        return value / 1000.0 if m.group(2).lower() == "ms" else value
    m = _RESETS_IN_RE.search(message)
    if m and any(m.groups()):
        return float(m.group(1) or 0) * 3600 + float(m.group(2) or 0) * 60 + float(m.group(3) or 0)
    m = _RETRY_AFTER_SECONDS_RE.search(message)
    return float(m.group(1)) if m else None


def _set_reset_from_retry_after(context: Dict[str, Any], retry_after: Any) -> None:
    if retry_after in {None, ""} or "reset_at" in context:
        return
    with contextlib.suppress(TypeError, ValueError):
        context["reset_at"] = time.time() + float(retry_after)


def extract_api_error_context(error: Exception) -> Dict[str, Any]:
    """Extract structured rate-limit details from provider errors."""
    context: Dict[str, Any] = {}
    body = getattr(error, "body", None)
    payload = (body.get("error") if isinstance(body.get("error"), dict) else body) if isinstance(body, dict) else None
    if isinstance(payload, dict):
        reason = payload.get("code") or payload.get("type") or payload.get("error")
        if isinstance(reason, str) and reason.strip():
            context["reason"] = reason.strip()
        message = payload.get("message") or payload.get("error_description")
        if not message and isinstance(payload.get("error"), str):
            # xAI uses a top-level string ``error`` beside a structured ``code``.
            message = payload.get("error")
        if isinstance(message, str) and message.strip():
            context["message"] = message.strip()
        reset = next((payload.get(k) for k in ("resets_at", "reset_at") if payload.get(k) not in {None, ""}), None)
        if reset is not None:
            context["reset_at"] = reset
        _set_reset_from_retry_after(context, payload.get("retry_after"))
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers:
        _set_reset_from_retry_after(context, headers.get("retry-after") or headers.get("Retry-After") or None)
        ratelimit_reset = headers.get("x-ratelimit-reset")
        if ratelimit_reset and "reset_at" not in context:
            context["reset_at"] = ratelimit_reset
    if "message" not in context and str(error).strip():
        context["message"] = str(error).strip()[:500]
    if "reset_at" not in context and isinstance(context.get("message") or "", str):
        delay = _reset_delay_from_message(context.get("message") or "")
        if delay is not None:
            context["reset_at"] = time.time() + delay
    return context


def _requeue_pending_steer(agent, steer_text: str) -> None:
    """Put drained steer text back so the caller's fallback delivers it as a next-turn user message."""
    # Under the lock the slot is read directly: an initialized agent always has both attributes, so a
    # missing ``_pending_steer`` there is a real bug and must fail loud. The lock-less branch only
    # exists for test stubs built via ``object.__new__`` that skipped ``__init__``.
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


def apply_pending_steer_to_tool_results(agent, messages: list, num_tool_msgs: int) -> None:
    """Append pending /steer text to the last ``role:"tool"`` message of this batch (bounded by
    ``num_tool_msgs``), marked as user-origin. Modifies existing content only, so role
    alternation is preserved."""
    if num_tool_msgs <= 0 or not messages:
        return
    steer_text = agent._drain_pending_steer()
    if not steer_text:
        return
    # Skip non-tool messages in the tail in case something else is appended at the boundary.
    tail = range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1)
    target = next((messages[j] for j in tail if isinstance(messages[j], dict) and messages[j].get("role") == "tool"), None)
    if target is None:
        # No tool result in this batch (e.g. all skipped by interrupt).
        _requeue_pending_steer(agent, steer_text)
        return
    marker = format_steer_marker(steer_text)
    existing_content = target.get("content", "")
    if isinstance(existing_content, str):
        target["content"] = existing_content + marker
    else:
        # Anthropic multimodal content blocks: preserve them and append a text block.
        try:
            target["content"] = [*(existing_content or []), {"type": "text", "text": marker.lstrip()}]
        except Exception:
            # Fall back to string replacement if content shape is unexpected.
            target["content"] = f"{existing_content}{marker}"
    _ra().logger.info(
        "Delivered /steer to agent after tool batch (%d chars): %s", len(steer_text),
        steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
    )


def force_close_tcp_sockets(client: Any) -> int:
    """Abort in-flight TCP I/O via ``shutdown(SHUT_RDWR)`` WITHOUT closing FDs. ``close()`` from
    a non-owner thread is unsafe: the SSL BIO caches the raw FD, the kernel recycles it, and a
    flushed TLS record lands in the wrong file (once clobbered a SQLite header). ``shutdown()``
    is FD-safe from any thread. Returns the count (logged as ``tcp_force_closed=N``)."""
    import socket as _socket
    shutdown_count = 0
    try:
        for sock in _iter_pool_sockets(client):
            try:
                # Clear a blocking timeout so a hung SSL_read notices the shutdown. Still no close().
                settimeout = getattr(sock, "settimeout", None)
                if callable(settimeout):
                    with contextlib.suppress(OSError):
                        settimeout(0)
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                pass  # already shut down / not connected / FD invalid: all benign
            shutdown_count += 1
    except Exception as exc:
        _ra().logger.debug("Force-close TCP sockets sweep error: %s", exc)
    return shutdown_count


__all__ = [
    "convert_to_trajectory_format", "sanitize_tool_call_arguments", "repair_message_sequence",
    "strip_think_blocks", "recover_with_credential_pool", "try_recover_primary_transport",
    "drop_thinking_only_and_merge_users", "restore_primary_runtime", "extract_reasoning",
    "dump_api_request_debug", "prompt_caching_disabled_from_config", "blank_cache_policy_stub",
    "plan_cache_sections_for_destination", "anthropic_prompt_cache_policy", "create_openai_client",
    "switch_model", "invoke_tool", "repair_tool_call", "sanitize_api_messages",
    "looks_like_codex_intermediate_ack", "copy_reasoning_content_for_api", "cleanup_dead_connections",
    "extract_api_error_context", "apply_pending_steer_to_tool_results", "_iter_pool_sockets",
    "force_close_tcp_sockets",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def agent_runtime_owns_post_tool_hook(agent: Any, function_name: str) -> bool:
    """Return True when an agent-level tool path emits its own post hook."""
    if function_name in AGENT_RUNTIME_POST_HOOK_TOOL_NAMES:
        return True
    if getattr(agent, "_context_engine_tool_names", None) and function_name in agent._context_engine_tool_names:
        return True
    memory_manager = getattr(agent, "_memory_manager", None)
    return bool(memory_manager and memory_manager.has_tool(function_name))

def intent_ack_continuation_enabled(agent) -> bool:
    """Whether intent-ack continuation should fire at all for this turn.

    The ``codex_ack_continuations < 2`` per-turn cap and the
    ``looks_like_codex_intermediate_ack`` detector are applied by the caller;
    this only decides the on/off gate. Callers that also need to know whether
    the workspace requirement applies should use ``intent_ack_continuation_mode``
    directly (``"codex_only"`` ⇒ require_workspace=True, ``"all"`` ⇒ False).
    """
    return intent_ack_continuation_mode(agent) != "off"
# ---- END PLUGIN-COMPAT ----
