"""Codex Responses API adapter.

Pure format-conversion and normalization logic for the OpenAI Responses API
(used by OpenAI Codex, xAI, GitHub Models, and other Responses-compatible endpoints).

Extracted from run_agent.py to isolate Responses API-specific logic from the
core agent loop. All functions are stateless — they operate on the data passed
in and return transformed results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.message_content import get_message_field, to_plain_data
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

logger = logging.getLogger(__name__)


def _classify_responses_issuer(
    *,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
    is_codex_backend: bool = False,
    base_url: Optional[str] = None,
) -> str:
    """Stable identifier for the Responses endpoint that mints encrypted_content.

    ``reasoning.encrypted_content`` is sealed to the endpoint that issued it:
    replaying a Codex-minted blob against xAI (or vice versa) deterministically
    returns HTTP 400 ``invalid_encrypted_content``. Stamping the issuer on
    persisted reasoning items and filtering at replay time lets a single
    conversation switch models without poisoning history with un-decryptable
    reasoning blocks.
    """
    if is_xai_responses:
        return "xai_responses"
    if is_github_responses:
        return "github_responses"
    if is_codex_backend:
        return "codex_backend"
    if base_url:
        return f"other:{base_url}"
    return "other"


# Throttle the per-process cross-issuer skip warning so we don't flood logs
# when a long history contains many stale-issuer reasoning blocks.
_CROSS_ISSUER_WARN_EMITTED = False

_HOSTED_MULTI_AGENT_ITEM_TYPES = {
    "multi_agent_call",
    "multi_agent_call_output",
    "agent_message",
}
_MULTI_AGENT_ACTIONS = {
    "spawn_agent",
    "interrupt_agent",
    "list_agents",
    "send_message",
    "followup_task",
    "wait_agent",
}


# Matches Codex/Harmony tool-call serialization that occasionally leaks into
# assistant-message content when the model fails to emit a structured
# ``function_call`` item.  Accepts the common forms:
#
#   to=functions.exec_command
#   assistant to=functions.exec_command
#   <|channel|>commentary to=functions.exec_command
#
# ``to=functions.<name>`` is the stable marker — the optional ``assistant`` or
# Harmony channel prefix varies by degeneration mode.  Case-insensitive to
# cover lowercase/uppercase ``assistant`` variants.
_TOOL_CALL_LEAK_PATTERN = re.compile(
    r"(?:^|[\s>|])to=functions\.[A-Za-z_][\w.]*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Multimodal content helpers
# ---------------------------------------------------------------------------

def _chat_content_to_responses_parts(content: Any, *, role: str = "user") -> List[Dict[str, Any]]:
    """Convert chat-style multimodal content to Responses API input parts.

    Input:  ``[{"type":"text"|"image_url", ...}]`` (native OpenAI Chat format)
    Output: ``[{"type":"input_text"|"output_text"|"input_image", ...}]`` (Responses format)

    The ``role`` parameter controls the text content type:
    - ``"user"`` (default) → ``"input_text"``
    - ``"assistant"`` → ``"output_text"``

    The Responses API rejects ``input_text`` inside assistant messages and
    ``output_text`` inside user messages, so callers MUST pass the correct
    role for the message being converted.

    Returns an empty list when ``content`` is not a list or contains no
    recognized parts — callers fall back to the string path.
    """
    text_type = "output_text" if role == "assistant" else "input_text"
    if not isinstance(content, list):
        return []
    converted: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                converted.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip().lower()
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                converted.append({"type": text_type, "text": text})
            continue
        if ptype in {"image_url", "input_image"}:
            image_ref = part.get("image_url")
            detail = part.get("detail")
            if isinstance(image_ref, dict):
                url = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url = image_ref
            if not isinstance(url, str) or not url:
                continue
            image_part: Dict[str, Any] = {"type": "input_image", "image_url": url}
            if isinstance(detail, str) and detail.strip():
                image_part["detail"] = detail.strip()
            converted.append(image_part)
    return converted


def _summarize_user_message_for_log(content: Any, *, sep: str = " ") -> str:
    """Flatten message content to a plain-text summary.

    Multimodal messages arrive as a list of ``{type:"text"|"image_url", ...}``
    parts from the API server.  Several consumers want a plain string:

    - Logging, spinner previews, and trajectory files (the default ``sep=" "``).
    - External memory providers, which feed the text to regexes
      (``sanitize_context``) and text APIs — a raw list crashes the sync with
      ``expected string or bytes-like object, got 'list'`` (use ``sep="\\n"``).

    Text parts are joined with ``sep``; images become a ``[N image(s)]`` marker
    so the turn isn't recorded as if the attachment never existed.  Returns an
    empty string for empty lists and ``str(content)`` for unexpected scalar
    types.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_bits: List[str] = []
        image_count = 0
        for part in content:
            if isinstance(part, str):
                if part:
                    text_bits.append(part)
                continue
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "").strip().lower()
            if ptype in {"text", "input_text", "output_text"}:
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_bits.append(text)
            elif ptype in {"image_url", "input_image"}:
                image_count += 1
        summary = sep.join(text_bits).strip()
        if image_count:
            note = f"[{image_count} image{'s' if image_count != 1 else ''}]"
            summary = f"{note} {summary}" if summary else note
        return summary
    try:
        return str(content)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Generate a deterministic call_id from tool call content.

    Used as a fallback when the API doesn't provide a call_id.
    Deterministic IDs prevent cache invalidation — random UUIDs would
    make every API call's prefix unique, breaking OpenAI's prompt cache.
    """
    seed = f"{fn_name}:{arguments}:{index}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"call_{digest}"


def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
    """Split a stored tool id into (call_id, response_item_id)."""
    if not isinstance(raw_id, str):
        return None, None
    value = raw_id.strip()
    if not value:
        return None, None
    if "|" in value:
        call_id, response_item_id = value.split("|", 1)
        call_id = call_id.strip() or None
        response_item_id = response_item_id.strip() or None
        return call_id, response_item_id
    if value.startswith("fc_"):
        return None, value
    return value, None


def _derive_responses_function_call_id(
    call_id: str,
    response_item_id: Optional[str] = None,
) -> str:
    """Build a valid Responses `function_call.id` (must start with `fc_`)."""
    if isinstance(response_item_id, str):
        candidate = response_item_id.strip()
        if candidate.startswith("fc_"):
            return candidate

    source = (call_id or "").strip()
    if source.startswith("fc_"):
        return source
    if source.startswith("call_") and len(source) > len("call_"):
        return f"fc_{source[len('call_'):]}"

    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
    if sanitized.startswith("fc_"):
        return sanitized
    if sanitized.startswith("call_") and len(sanitized) > len("call_"):
        return f"fc_{sanitized[len('call_'):]}"
    if sanitized:
        return f"fc_{sanitized[:48]}"

    seed = source or str(response_item_id or "") or uuid.uuid4().hex
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
    return f"fc_{digest}"


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

def _responses_tools(tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
    """Convert chat-completions tool schemas to Responses function-tool schemas."""
    if not tools:
        return None

    converted: List[Dict[str, Any]] = []
    for item in tools:
        fn = item.get("function", {}) if isinstance(item, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "strict": False,
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted or None


# Provider-executed built-in tool *declaration* types accepted on the
# Responses ``tools`` array.  These are declared by ``type`` alone (no
# client-side name/parameters schema) and run server-side — the provider
# owns the implementation and reports progress via the matching ``*_call``
# output items.  Hermes injects xAI's native ``web_search`` for the xAI
# transport (see agent/transports/codex.py); the rest are listed so the
# preflight validator passes them through rather than rejecting them as
# "unsupported type".  Mirrors the ``*_call`` item-type set used in
# _normalize_codex_response.
_RESPONSES_BUILTIN_TOOL_TYPES = {
    "web_search",
    "web_search_preview",
    "file_search",
    "code_interpreter",
    "image_generation",
    "computer_use_preview",
    "local_shell",
}


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------

_RESPONSE_MESSAGE_STATUSES = {"completed", "incomplete", "in_progress"}


def _normalize_responses_message_status(value: Any, *, default: str = "completed") -> str:
    """Normalize a Responses assistant message status for replay.

    The API accepts completed/incomplete/in_progress on replayed assistant
    output messages.  Preserve those exactly (modulo case/hyphen spelling) so
    incomplete Codex continuation turns don't get falsely marked completed.
    """
    if isinstance(value, str):
        status = value.strip().lower().replace("-", "_").replace(" ", "_")
        if status in _RESPONSE_MESSAGE_STATUSES:
            return status
    return default


def _chat_messages_to_responses_input(
    messages: List[Dict[str, Any]],
    *,
    is_xai_responses: bool = False,
    replay_encrypted_reasoning: bool = True,
    current_issuer_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert internal chat-style messages to Responses input items.

    ``is_xai_responses`` is kept for transport signature compatibility but
    no longer suppresses encrypted reasoning replay.  Earlier (PR #26644,
    May 2026) we believed xAI's OAuth/SuperGrok ``/v1/responses`` surface
    rejected replayed ``encrypted_content`` reasoning items minted by
    prior turns, and we stripped them.  That decision was wrong — xAI
    explicitly relies on Hermes threading encrypted reasoning back across
    turns for cross-turn coherence (the whole point of their partnership
    integration).  We now replay encrypted reasoning on every Responses
    transport (xAI, native Codex, custom relays) and let xAI tell us
    explicitly if a specific surface ever rejects a payload.

    ``replay_encrypted_reasoning`` is the per-session kill switch.  Some
    OpenAI-compatible relays accept the request but later reject the
    replayed encrypted blob with HTTP 400 ``invalid_encrypted_content``;
    when that happens the retry loop calls
    ``AIAgent._disable_codex_reasoning_replay`` which both strips cached
    items from the conversation history and threads ``replay_enabled=False``
    through this converter so subsequent turns send no reasoning items.

    ``current_issuer_kind`` enables a per-item cross-issuer guard. The
    Responses API's ``encrypted_content`` blob is decryptable only by the
    endpoint that minted it — replaying a Codex-issued blob against xAI
    (or vice versa) always yields HTTP 400 ``invalid_encrypted_content``
    and breaks every subsequent turn in the same session.  When this
    argument is provided and a reasoning item carries an ``_issuer_kind``
    stamp from a different endpoint, the item is dropped from the replayed
    input.  Legacy items without a stamp are still replayed
    (backwards-compatible).  The two guards compose:
    ``replay_encrypted_reasoning=False`` is the session-wide kill switch
    (drops ALL replay); ``current_issuer_kind`` is the per-item filter
    that runs only when replay is still enabled.
    """
    items: List[Dict[str, Any]] = []
    seen_item_ids: set = set()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue

        if role in {"user", "assistant"}:
            content = msg.get("content", "")
            if isinstance(content, list):
                content_parts = _chat_content_to_responses_parts(content, role=role)
                text_type = "output_text" if role == "assistant" else "input_text"
                content_text = "".join(
                    p.get("text", "") for p in content_parts if p.get("type") == text_type
                )
            else:
                content_parts = []
                content_text = str(content) if content is not None else ""

            if role == "assistant":
                codex_message_items = msg.get("codex_message_items")
                multi_agent_replay = bool(
                    isinstance(codex_message_items, list)
                    and any(
                        isinstance(item, dict)
                        and (
                            item.get("type") in _HOSTED_MULTI_AGENT_ITEM_TYPES
                            or isinstance(item.get("agent"), dict)
                        )
                        for item in codex_message_items
                    )
                )

                # Replay encrypted reasoning items from previous turns
                # so the API can maintain coherent reasoning chains.
                # This applies to every Responses transport including
                # xAI — see _chat_messages_to_responses_input docstring
                # for the May 2026 reversal of the earlier xAI gate.
                codex_reasoning = (
                    msg.get("codex_reasoning_items")
                    if replay_encrypted_reasoning and not multi_agent_replay
                    else None
                )
                has_codex_reasoning = False
                if isinstance(codex_reasoning, list):
                    for ri in codex_reasoning:
                        if isinstance(ri, dict) and ri.get("encrypted_content"):
                            item_id = ri.get("id")
                            if item_id and item_id in seen_item_ids:
                                continue
                            # Cross-issuer guard: drop reasoning blocks that
                            # were minted by a different Responses endpoint.
                            # The current endpoint cannot decrypt foreign
                            # encrypted_content and would reject the whole
                            # request with HTTP 400 invalid_encrypted_content.
                            # Unstamped (legacy) items pass through.
                            item_issuer = ri.get("_issuer_kind")
                            if (
                                current_issuer_kind is not None
                                and item_issuer is not None
                                and item_issuer != current_issuer_kind
                            ):
                                global _CROSS_ISSUER_WARN_EMITTED
                                if not _CROSS_ISSUER_WARN_EMITTED:
                                    logger.warning(
                                        "Dropping reasoning item minted by %s while "
                                        "calling %s — encrypted_content is sealed to "
                                        "its issuer. This happens when a session "
                                        "switches model providers mid-conversation.",
                                        item_issuer, current_issuer_kind,
                                    )
                                    _CROSS_ISSUER_WARN_EMITTED = True
                                continue
                            # Strip the "id" field — with store=False the
                            # Responses API cannot look up items by ID and
                            # returns 404.  The encrypted_content blob is
                            # self-contained for reasoning chain continuity.
                            # Also strip the internal "_issuer_kind" stamp;
                            # it is a Hermes-side metadata key and not part
                            # of the Responses API schema.
                            replay_item = {
                                k: v for k, v in ri.items()
                                if k not in ("id", "_issuer_kind")
                            }
                            items.append(replay_item)
                            if item_id:
                                seen_item_ids.add(item_id)
                            has_codex_reasoning = True

                # Normal Codex turns preserve assistant messages for phase/cache
                # continuity. Multi-agent turns use the same persisted carrier
                # for the complete ordered output list because the beta requires
                # every hosted item and agent-attributed function call on resume.
                replayed_message_items = 0
                if multi_agent_replay:
                    for raw_item in codex_message_items:
                        if not isinstance(raw_item, dict):
                            continue
                        replay_item = {
                            key: value
                            for key, value in raw_item.items()
                            if not str(key).startswith("_")
                        }
                        if replay_item:
                            items.append(replay_item)
                            replayed_message_items += 1
                elif isinstance(codex_message_items, list):
                    for raw_item in codex_message_items:
                        if not isinstance(raw_item, dict):
                            continue
                        if raw_item.get("type") != "message" or raw_item.get("role") != "assistant":
                            continue
                        raw_content_parts = raw_item.get("content")
                        if not isinstance(raw_content_parts, list):
                            continue

                        normalized_content_parts = []
                        for part in raw_content_parts:
                            if not isinstance(part, dict):
                                continue
                            part_type = str(part.get("type") or "").strip()
                            if part_type not in {"output_text", "text"}:
                                continue
                            text = part.get("text", "")
                            if text is None:
                                text = ""
                            if not isinstance(text, str):
                                text = str(text)
                            normalized_content_parts.append({"type": "output_text", "text": text})

                        if not normalized_content_parts:
                            continue

                        replay_item = {
                            "type": "message",
                            "role": "assistant",
                            "status": _normalize_responses_message_status(raw_item.get("status")),
                            "content": normalized_content_parts,
                        }
                        item_id = raw_item.get("id")
                        if isinstance(item_id, str) and item_id.strip():
                            replay_item["id"] = item_id.strip()
                        phase = raw_item.get("phase")
                        if isinstance(phase, str) and phase.strip():
                            replay_item["phase"] = phase.strip()
                        items.append(replay_item)
                        replayed_message_items += 1

                if replayed_message_items > 0:
                    pass
                elif content_parts:
                    items.append({"role": "assistant", "content": content_parts})
                elif content_text.strip():
                    items.append({"role": "assistant", "content": content_text})
                elif has_codex_reasoning:
                    # The Responses API requires a following item after each
                    # reasoning item (otherwise: missing_following_item error).
                    # When the assistant produced only reasoning with no visible
                    # content, emit an empty assistant message as the required
                    # following item.
                    items.append({"role": "assistant", "content": ""})

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list) and not multi_agent_replay:
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        fn_name = fn.get("name")
                        if not isinstance(fn_name, str) or not fn_name.strip():
                            continue

                        embedded_call_id, embedded_response_item_id = _split_responses_tool_id(
                            tc.get("id")
                        )
                        call_id = tc.get("call_id")
                        if not isinstance(call_id, str) or not call_id.strip():
                            call_id = embedded_call_id
                        if not isinstance(call_id, str) or not call_id.strip():
                            if (
                                isinstance(embedded_response_item_id, str)
                                and embedded_response_item_id.startswith("fc_")
                                and len(embedded_response_item_id) > len("fc_")
                            ):
                                call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                            else:
                                _raw_args = str(fn.get("arguments", "{}"))
                                call_id = _deterministic_call_id(fn_name, _raw_args, len(items))
                        call_id = call_id.strip()

                        arguments = fn.get("arguments", "{}")
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        elif not isinstance(arguments, str):
                            arguments = str(arguments)
                        arguments = arguments.strip() or "{}"

                        items.append({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": fn_name,
                            "arguments": arguments,
                        })
                continue

            # Non-assistant (user) role: emit multimodal parts when present,
            # otherwise fall back to the text payload.
            if content_parts:
                items.append({"role": role, "content": content_parts})
            else:
                items.append({"role": role, "content": content_text})
            continue

        if role == "tool":
            raw_tool_call_id = msg.get("tool_call_id")
            call_id, _ = _split_responses_tool_id(raw_tool_call_id)
            if not isinstance(call_id, str) or not call_id.strip():
                if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                    call_id = raw_tool_call_id.strip()
            if not isinstance(call_id, str) or not call_id.strip():
                continue

            # Multimodal tool result: convert OpenAI-style content list into
            # Responses ``function_call_output.output`` array. The Responses
            # API accepts ``output`` as either a string or an array of
            # ``input_text``/``input_image`` items. See
            # https://developers.openai.com/api/reference/python/resources/responses/.
            tool_content = msg.get("content")
            output_value: Any
            if isinstance(tool_content, list):
                converted = _chat_content_to_responses_parts(
                    tool_content, role="user",
                )
                if converted:
                    output_value = converted
                else:
                    output_value = ""
            else:
                output_value = str(tool_content or "")

            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_value,
            })

    return items


# ---------------------------------------------------------------------------
# Input preflight / validation
# ---------------------------------------------------------------------------

def _normalize_multi_agent_agent(value: Any, *, context: str) -> Optional[Dict[str, str]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{context} agent must be an object.")
    agent_name = value.get("agent_name")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError(f"{context} agent.agent_name must be a non-empty string.")
    return {"agent_name": agent_name.strip()}


def _normalize_hosted_multi_agent_item(
    item: Dict[str, Any], *, index: int
) -> Dict[str, Any]:
    item_type = item.get("type")
    context = f"Codex Responses input[{index}] {item_type}"

    def required_string(name: str) -> str:
        value = item.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} is missing {name}.")
        return value.strip()

    normalized: Dict[str, Any] = {
        "type": item_type,
        "id": required_string("id"),
    }
    agent = _normalize_multi_agent_agent(item.get("agent"), context=context)
    if agent is not None:
        normalized["agent"] = agent

    if item_type in {"multi_agent_call", "multi_agent_call_output"}:
        action = required_string("action")
        if action not in _MULTI_AGENT_ACTIONS:
            raise ValueError(f"{context} has unsupported action {action!r}.")
        normalized["action"] = action
        normalized["call_id"] = required_string("call_id")
        if item_type == "multi_agent_call":
            normalized["arguments"] = required_string("arguments")
        else:
            output = item.get("output")
            if not isinstance(output, list) or not all(
                isinstance(part, dict) for part in output
            ):
                raise ValueError(f"{context} output must be a list of objects.")
            normalized["output"] = [dict(part) for part in output]
        return normalized

    normalized["author"] = required_string("author")
    normalized["recipient"] = required_string("recipient")
    content = item.get("content")
    if not isinstance(content, list) or not all(
        isinstance(part, dict) for part in content
    ):
        raise ValueError(f"{context} content must be a list of objects.")
    normalized["content"] = [dict(part) for part in content]
    return normalized


def _preflight_codex_input_items(
    raw_items: Any,
    *,
    allow_multi_agent: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError("Codex Responses input must be a list of input items.")

    normalized: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"Codex Responses input[{idx}] must be an object.")

        item_type = item.get("type")
        if item_type in _HOSTED_MULTI_AGENT_ITEM_TYPES:
            if not allow_multi_agent:
                raise ValueError(
                    f"Codex Responses input[{idx}] has unsupported item type {item_type!r}."
                )
            normalized.append(
                _normalize_hosted_multi_agent_item(item, index=idx)
            )
            continue

        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

            arguments = item.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = str(arguments)
            arguments = arguments.strip() or "{}"

            function_call: Dict[str, Any] = {
                "type": "function_call",
                "call_id": call_id.strip(),
                "name": name.strip(),
                "arguments": arguments,
            }
            if allow_multi_agent:
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id.strip():
                    function_call["id"] = item_id.strip()
                agent = _normalize_multi_agent_agent(
                    item.get("agent"),
                    context=f"Codex Responses input[{idx}] function_call",
                )
                if agent is not None:
                    function_call["agent"] = agent
                for field in ("status", "caller", "namespace"):
                    value = item.get(field)
                    if value is not None:
                        function_call[field] = value
            normalized.append(function_call)
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
            output = item.get("output", "")
            if output is None:
                output = ""
            # Output may be a string OR an array of structured content
            # items (input_text / input_image) for multimodal tool results.
            # Both shapes are accepted by the Responses API. We preserve
            # the array form when present.
            if isinstance(output, list):
                # Validate each item is a recognised content shape; drop
                # anything else to avoid 4xx from the API.
                cleaned: List[Dict[str, Any]] = []
                for part in output:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "input_text":
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            cleaned.append({"type": "input_text", "text": text})
                    elif ptype == "input_image":
                        url = part.get("image_url")
                        if isinstance(url, str) and url:
                            entry: Dict[str, Any] = {"type": "input_image", "image_url": url}
                            detail = part.get("detail")
                            if isinstance(detail, str) and detail.strip():
                                entry["detail"] = detail.strip()
                            cleaned.append(entry)
                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": cleaned if cleaned else "",
                    }
                )
                continue
            if not isinstance(output, str):
                output = str(output)

            normalized.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id.strip(),
                    "output": output,
                }
            )
            continue

        if item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id:
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)
                reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
                # Do NOT include the "id" in the outgoing item — with
                # store=False (our default) the API tries to resolve the
                # id server-side and returns 404.  The id is still used
                # above for local deduplication via seen_ids.
                summary = item.get("summary")
                if isinstance(summary, list):
                    reasoning_item["summary"] = summary
                else:
                    reasoning_item["summary"] = []
                if allow_multi_agent:
                    if isinstance(item_id, str) and item_id.strip():
                        reasoning_item["id"] = item_id.strip()
                    agent = _normalize_multi_agent_agent(
                        item.get("agent"),
                        context=f"Codex Responses input[{idx}] reasoning",
                    )
                    if agent is not None:
                        reasoning_item["agent"] = agent
                    for field in ("content", "status"):
                        value = item.get(field)
                        if value is not None:
                            reasoning_item[field] = value
                normalized.append(reasoning_item)
            continue

        if item_type == "message":
            role = item.get("role")
            if role != "assistant":
                raise ValueError(f"Codex Responses input[{idx}] message items must have role='assistant'.")
            content = item.get("content")
            if not isinstance(content, list):
                raise ValueError(f"Codex Responses input[{idx}] message item must have content list.")
            normalized_content = []
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    raise ValueError(
                        f"Codex Responses input[{idx}] message content[{part_idx}] must be an object."
                    )
                part_type = part.get("type")
                if part_type not in {"output_text", "text"}:
                    raise ValueError(
                        f"Codex Responses input[{idx}] message content[{part_idx}] has unsupported type {part_type!r}."
                    )
                text = part.get("text", "")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    text = str(text)
                normalized_content.append({"type": "output_text", "text": text})
            if not normalized_content:
                raise ValueError(f"Codex Responses input[{idx}] message item must contain at least one text part.")
            normalized_item: Dict[str, Any] = {
                "type": "message",
                "role": "assistant",
                "status": _normalize_responses_message_status(item.get("status")),
                "content": normalized_content,
            }
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id.strip():
                normalized_item["id"] = item_id.strip()
            phase = item.get("phase")
            if isinstance(phase, str) and phase.strip():
                normalized_item["phase"] = phase.strip()
            if allow_multi_agent:
                agent = _normalize_multi_agent_agent(
                    item.get("agent"),
                    context=f"Codex Responses input[{idx}] message",
                )
                if agent is not None:
                    normalized_item["agent"] = agent
            normalized.append(normalized_item)
            continue

        role = item.get("role")
        if role in {"user", "assistant"}:
            content = item.get("content", "")
            if content is None:
                content = ""
            if isinstance(content, list):
                # Multimodal content from ``_chat_messages_to_responses_input``
                # is already in Responses format (``input_text`` / ``output_text``
                # / ``input_image``).  Validate each part and pass through.
                # Use the correct text type for the role — ``output_text`` for
                # assistant messages, ``input_text`` for user messages.
                text_type = "output_text" if role == "assistant" else "input_text"
                validated: List[Dict[str, Any]] = []
                for part_idx, part in enumerate(content):
                    if isinstance(part, str):
                        if part:
                            validated.append({"type": text_type, "text": part})
                        continue
                    if not isinstance(part, dict):
                        raise ValueError(
                            f"Codex Responses input[{idx}].content[{part_idx}] must be an object or string."
                        )
                    ptype = str(part.get("type") or "").strip().lower()
                    if ptype in {"input_text", "text", "output_text"}:
                        text = part.get("text", "")
                        if not isinstance(text, str):
                            text = str(text or "")
                        validated.append({"type": text_type, "text": text})
                    elif ptype in {"input_image", "image_url"}:
                        image_ref = part.get("image_url", "")
                        detail = part.get("detail")
                        if isinstance(image_ref, dict):
                            url = image_ref.get("url", "")
                            detail = image_ref.get("detail", detail)
                        else:
                            url = image_ref
                        if not isinstance(url, str):
                            url = str(url or "")
                        image_part: Dict[str, Any] = {"type": "input_image", "image_url": url}
                        if isinstance(detail, str) and detail.strip():
                            image_part["detail"] = detail.strip()
                        validated.append(image_part)
                    else:
                        raise ValueError(
                            f"Codex Responses input[{idx}].content[{part_idx}] has unsupported type {part.get('type')!r}."
                        )
                normalized.append({"role": role, "content": validated})
                continue
            if not isinstance(content, str):
                content = str(content)

            normalized.append({"role": role, "content": content})
            continue

        raise ValueError(
            f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
        )

    return normalized


def _preflight_codex_api_kwargs(
    api_kwargs: Any,
    *,
    allow_stream: bool = False,
) -> Dict[str, Any]:
    if not isinstance(api_kwargs, dict):
        raise ValueError("Codex Responses request must be a dict.")

    required = {"model", "instructions", "input"}
    missing = [key for key in required if key not in api_kwargs]
    if missing:
        raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

    model = api_kwargs.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Codex Responses request 'model' must be a non-empty string.")
    model = model.strip()

    instructions = api_kwargs.get("instructions")
    if instructions is None:
        instructions = ""
    if not isinstance(instructions, str):
        instructions = str(instructions)
    instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

    extra_body_for_mode = api_kwargs.get("extra_body")
    multi_agent_config = (
        extra_body_for_mode.get("multi_agent")
        if isinstance(extra_body_for_mode, dict)
        else None
    )
    allow_multi_agent = bool(
        isinstance(multi_agent_config, dict)
        and multi_agent_config.get("enabled") is True
    )
    normalized_input = _preflight_codex_input_items(
        api_kwargs.get("input"),
        allow_multi_agent=allow_multi_agent,
    )

    tools = api_kwargs.get("tools")
    normalized_tools = None
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("Codex Responses request 'tools' must be a list when provided.")
        normalized_tools = []
        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"Codex Responses tools[{idx}] must be an object.")

            tool_type = tool.get("type")

            # Provider-executed built-in tools (xAI native web_search, code
            # interpreter, etc.) are declared by ``type`` alone and carry no
            # ``name``/``parameters`` schema — the provider owns the
            # implementation.  Pass them through verbatim instead of forcing
            # them through the function-tool validation below (which would
            # otherwise reject them with "unsupported type").  See
            # agent/transports/codex.py for where xAI's native web_search is
            # injected.
            if tool_type in _RESPONSES_BUILTIN_TOOL_TYPES:
                normalized_tools.append(dict(tool))
                continue

            if tool_type != "function":
                raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

            name = tool.get("name")
            parameters = tool.get("parameters")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
            if not isinstance(parameters, dict):
                raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

            description = tool.get("description", "")
            if description is None:
                description = ""
            if not isinstance(description, str):
                description = str(description)

            strict = tool.get("strict", False)
            if not isinstance(strict, bool):
                strict = bool(strict)

            normalized_tools.append(
                {
                    "type": "function",
                    "name": name.strip(),
                    "description": description,
                    "strict": strict,
                    "parameters": parameters,
                }
            )

    store = api_kwargs.get("store", False)
    if store is not False:
        raise ValueError("Codex Responses contract requires 'store' to be false.")

    allowed_keys = {
        "model", "instructions", "input", "tools", "store",
        "reasoning", "include", "max_output_tokens", "temperature",
        "tool_choice", "parallel_tool_calls", "prompt_cache_key", "service_tier",
        "extra_headers", "extra_query", "extra_body", "timeout",
    }
    normalized: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": normalized_input,
        "store": False,
    }
    if normalized_tools is not None:
        normalized["tools"] = normalized_tools

    # Pass through reasoning config
    reasoning = api_kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        normalized["reasoning"] = reasoning
    include = api_kwargs.get("include")
    if isinstance(include, list):
        normalized["include"] = include
    service_tier = api_kwargs.get("service_tier")
    if isinstance(service_tier, str) and service_tier.strip():
        normalized["service_tier"] = service_tier.strip()

    # Pass through max_output_tokens and temperature
    max_output_tokens = api_kwargs.get("max_output_tokens")
    if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
        normalized["max_output_tokens"] = int(max_output_tokens)
    timeout = api_kwargs.get("timeout")
    if (
        isinstance(timeout, (int, float))
        and not isinstance(timeout, bool)
        and 0 < float(timeout) < float("inf")
    ):
        normalized["timeout"] = float(timeout)
    temperature = api_kwargs.get("temperature")
    if isinstance(temperature, (int, float)):
        normalized["temperature"] = float(temperature)

    # Pass through tool_choice, parallel_tool_calls, prompt_cache_key
    for passthrough_key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key"):
        val = api_kwargs.get(passthrough_key)
        if val is not None:
            normalized[passthrough_key] = val

    extra_headers = api_kwargs.get("extra_headers")
    if extra_headers is not None:
        if not isinstance(extra_headers, dict):
            raise ValueError("Codex Responses request 'extra_headers' must be an object.")
        normalized_headers: Dict[str, str] = {}
        for key, value in extra_headers.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Codex Responses request 'extra_headers' keys must be non-empty strings.")
            if value is None:
                continue
            normalized_headers[key.strip()] = str(value)
        if normalized_headers:
            normalized["extra_headers"] = normalized_headers

    extra_query = api_kwargs.get("extra_query")
    if extra_query is not None:
        if not isinstance(extra_query, dict):
            raise ValueError("Codex Responses request 'extra_query' must be an object.")
        normalized_query: Dict[str, Any] = {}
        for key, value in extra_query.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "Codex Responses request 'extra_query' keys must be non-empty strings."
                )
            if value is not None:
                normalized_query[key.strip()] = value
        if normalized_query:
            normalized["extra_query"] = normalized_query

    extra_body = api_kwargs.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("Codex Responses request 'extra_body' must be an object.")
        # Pass extra_body through verbatim — used by xAI Responses to
        # carry `prompt_cache_key` as a body-level field (the documented
        # cache-routing surface on /v1/responses). The openai SDK
        # serializes extra_body into the JSON body without per-field
        # type checks, so it survives Responses.stream() kwarg-signature
        # changes that would otherwise raise TypeError before the wire.
        if extra_body:
            normalized["extra_body"] = dict(extra_body)

    if allow_stream:
        stream = api_kwargs.get("stream")
        if stream is not None and stream is not True:
            raise ValueError("Codex Responses 'stream' must be true when set.")
        if stream is True:
            normalized["stream"] = True
        allowed_keys.add("stream")
    elif "stream" in api_kwargs:
        raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

    # Safety-net sanitization for xAI Responses (#28490): defense-in-depth
    # for the same slash-enum strip that ``chat_completion_helpers`` and
    # ``auxiliary_client`` apply at request-build time.  If a future code
    # path forgets to sanitize before calling us, this catches the bypass
    # so xAI doesn't 400 with ``Invalid arguments passed to the model``
    # (HuggingFace IDs like ``Qwen/Qwen3.5-0.8B`` from MCP tool schemas).
    #
    # Gated on the model name pattern because native Codex (OpenAI) DOES
    # accept slash-containing enum values — stripping them there would
    # silently degrade tool-schema constraints.  xAI is the only
    # Responses-API surface that rejects the shape.
    model_name_for_provider_check = str(api_kwargs.get("model") or "").lower()
    is_xai_model = model_name_for_provider_check.startswith(("grok-", "x-ai/grok-"))
    if is_xai_model and normalized.get("tools"):
        try:
            from tools.schema_sanitizer import strip_slash_enum
            normalized["tools"], _ = strip_slash_enum(normalized["tools"])
        except Exception:
            pass  # Best-effort — the caller-level sanitization should have handled it

    unexpected = sorted(key for key in api_kwargs if key not in allowed_keys)
    if unexpected:
        raise ValueError(
            f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
        )

    return normalized


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------

def _extract_responses_message_text(item: Any) -> str:
    """Extract assistant text from a Responses message output item."""
    content = _responses_field(item, "content")
    if not isinstance(content, list):
        return ""

    chunks: List[str] = []
    for part in content:
        ptype = _responses_field(part, "type")
        if ptype not in {"output_text", "text"}:
            continue
        text = _responses_field(part, "text")
        if isinstance(text, str) and text:
            chunks.append(text)
    return "".join(chunks).strip()


def _extract_responses_reasoning_text(item: Any) -> str:
    """Extract a compact reasoning text from a Responses reasoning item."""
    summary = _responses_field(item, "summary")
    if isinstance(summary, list):
        chunks: List[str] = []
        for part in summary:
            text = _responses_field(part, "text")
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()
    text = _responses_field(item, "text")
    if isinstance(text, str) and text:
        return text.strip()
    return ""


def _format_responses_error(error_obj: Any, response_status: str) -> str:
    """Build a human-readable error string from a Responses ``response.error`` payload.

    The OpenAI Responses API carries failure details under ``response.error``
    on terminal ``response.failed`` events, in the shape
    ``{"code": "rate_limit_exceeded", "message": "Slow down", "param": ...}``.
    Earlier code only surfaced ``message``, which left users staring at bare
    strings like ``"Slow down"`` while the failure mode (rate limit vs
    context-length vs internal_error vs model-overloaded) was hidden in
    ``code``. We now prefix ``code`` when both are present so consumers can
    distinguish failure modes without parsing the bare message.

    Falls back to ``code`` alone when ``message`` is empty, and to a stable
    default referencing the response status when no error payload is
    available at all. Adapted from anomalyco/opencode#28757.
    """
    # Pull code and message from either dict or attribute-style payloads.
    code: Any = None
    message: Any = None
    if isinstance(error_obj, dict):
        code = error_obj.get("code")
        message = error_obj.get("message")
    elif error_obj is not None:
        code = getattr(error_obj, "code", None)
        message = getattr(error_obj, "message", None)

    code_str = str(code).strip() if isinstance(code, str) else (str(code).strip() if code else "")
    message_str = str(message).strip() if isinstance(message, str) else (str(message).strip() if message else "")

    if code_str and message_str:
        return f"{code_str}: {message_str}"
    if message_str:
        return message_str
    if code_str:
        return code_str
    if error_obj:
        # Last-resort: stringify whatever the provider sent so it's at least
        # visible in logs/UI rather than silently swallowed.
        return str(error_obj)
    return f"Responses API returned status '{response_status}'"


# ---------------------------------------------------------------------------
# Full response normalization
# ---------------------------------------------------------------------------


def _responses_field(value: Any, name: str, default: Any = None) -> Any:
    return get_message_field(value, name, default)


def _responses_agent_name(value: Any) -> Optional[str]:
    agent = _responses_field(value, "agent")
    agent_name = _responses_field(agent, "agent_name") if agent is not None else None
    if isinstance(agent_name, str) and agent_name.strip():
        return agent_name.strip()
    return None


def _responses_value_to_plain_data(value: Any) -> Any:
    """Convert SDK/namespace response values into JSON-safe plain data."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            value = model_dump(mode="json", exclude_none=True)
        except TypeError:
            value = model_dump()
    plain = to_plain_data(value)
    if isinstance(plain, dict):
        return {
            str(key): _responses_value_to_plain_data(item)
            for key, item in plain.items()
            if not str(key).startswith("_") and item is not None
        }
    if isinstance(plain, list):
        return [_responses_value_to_plain_data(item) for item in plain]
    if plain is None or isinstance(plain, (str, int, float, bool)):
        return plain
    return str(plain)


def _responses_output_item_to_dict(item: Any, item_type: str) -> Dict[str, Any]:
    raw = _responses_value_to_plain_data(item)
    if not isinstance(raw, dict):
        raw = {}
    raw["type"] = item_type
    return raw


def _normalize_codex_response(
    response: Any,
    *,
    issuer_kind: Optional[str] = None,
) -> tuple[Any, str]:
    """Normalize a Responses API object to an assistant_message-like object.

    ``issuer_kind`` (when provided) is stamped onto each reasoning item the
    response yields, so future replays can detect when the active endpoint
    differs from the one that minted the encrypted_content blob and drop
    the item instead of triggering HTTP 400 invalid_encrypted_content.
    """
    output = getattr(response, "output", None)
    if not isinstance(output, list) or not output:
        # The Codex backend can return empty output when the answer was
        # delivered entirely via stream events. Check output_text as a
        # last-resort fallback before raising.
        out_text = getattr(response, "output_text", None)
        if isinstance(out_text, str) and out_text.strip():
            logger.debug(
                "Codex response has empty output but output_text is present (%d chars); "
                "synthesizing output item.", len(out_text.strip()),
            )
            output = [SimpleNamespace(
                type="message", role="assistant", status="completed",
                content=[SimpleNamespace(type="output_text", text=out_text.strip())],
            )]
            response.output = output
        else:
            raise RuntimeError("Responses API returned no output items")

    response_status = getattr(response, "status", None)
    if isinstance(response_status, str):
        response_status = response_status.strip().lower()
    else:
        response_status = None

    if response_status in {"failed", "cancelled"}:
        error_obj = getattr(response, "error", None)
        error_msg = _format_responses_error(error_obj, response_status)
        raise RuntimeError(error_msg)

    multi_agent_response = any(
        str(_responses_field(item, "type", "") or "")
        in _HOSTED_MULTI_AGENT_ITEM_TYPES
        or _responses_agent_name(item) is not None
        for item in output
    )

    content_parts: List[str] = []
    root_final_parts: List[str] = []
    root_fallback_parts: List[str] = []
    reasoning_parts: List[str] = []
    reasoning_items_raw: List[Dict[str, Any]] = []
    message_items_raw: List[Dict[str, Any]] = (
        [
            _responses_output_item_to_dict(
                item,
                str(_responses_field(item, "type", "") or ""),
            )
            for item in output
        ]
        if multi_agent_response
        else []
    )
    tool_calls: List[Any] = []
    has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
    saw_streaming_or_item_incomplete = response_status in {"queued", "in_progress"}
    saw_commentary_phase = False
    saw_final_answer_phase = False
    saw_reasoning_item = False

    # Server-side built-in tool calls (xAI's native web_search, code
    # interpreter, etc.) are executed by the provider and reported as
    # discrete ``*_call`` output items.  xAI's /v1/responses surface
    # (e.g. grok-composer-2.5-fast on SuperGrok OAuth) routinely leaves
    # these items at ``status="in_progress"`` even when the overall
    # ``response.status == "completed"`` — the search ran to completion
    # server-side, the per-item status simply isn't reconciled.  These
    # are NOT a signal that the model's turn is unfinished, so they must
    # not flip ``has_incomplete_items``.  Only the response-level status
    # and genuine model output items (message/reasoning/function_call)
    # govern the incomplete verdict.  Without this guard, any turn where
    # grok-composer invokes server-side search is misclassified as
    # ``finish_reason="incomplete"`` and burns 3 fruitless continuation
    # retries before failing with "Codex response remained incomplete
    # after 3 continuation attempts".  client-side function/custom tool
    # calls keep their own in_progress handling below (they are skipped,
    # not awaited).
    _SERVER_SIDE_TOOL_CALL_TYPES = {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "computer_call",
        "local_shell_call",
        "mcp_call",
        *_HOSTED_MULTI_AGENT_ITEM_TYPES,
    }

    for item in output:
        item_type = str(_responses_field(item, "type", "") or "")
        item_status = str(_responses_field(item, "status", "") or "").strip().lower()

        if (
            item_status in {"queued", "in_progress", "incomplete"}
            and item_type not in _SERVER_SIDE_TOOL_CALL_TYPES
        ):
            has_incomplete_items = True
            saw_streaming_or_item_incomplete = True

        if item_type in _HOSTED_MULTI_AGENT_ITEM_TYPES:
            # OpenAI executes these delegation items server-side. They are
            # already captured in the full ordered Multi-agent replay list and
            # must never be dispatched as Hermes client tools.
            continue

        if item_type == "message":
            item_phase = _responses_field(item, "phase")
            item_agent_name = _responses_agent_name(item)
            is_root_message = item_agent_name in {None, "/root"}
            normalized_phase = None
            is_commentary_phase = False
            if isinstance(item_phase, str):
                normalized_phase = item_phase.strip().lower()
                if normalized_phase in {"commentary", "analysis"}:
                    is_commentary_phase = True
                    if not multi_agent_response or is_root_message:
                        saw_commentary_phase = True
                elif normalized_phase in {"final_answer", "final"}:
                    if not multi_agent_response or is_root_message:
                        saw_final_answer_phase = True
            message_text = _extract_responses_message_text(item)
            if message_text and (not multi_agent_response or is_root_message):
                # Multi-agent subagent messages are replay state, not user-facing
                # output. Root commentary remains reasoning/progress text; root
                # final_answer wins over phase-less root fallback messages.
                if is_commentary_phase:
                    reasoning_parts.append(message_text)
                elif multi_agent_response and normalized_phase in {
                    "final_answer",
                    "final",
                }:
                    root_final_parts.append(message_text)
                elif multi_agent_response:
                    root_fallback_parts.append(message_text)
                else:
                    content_parts.append(message_text)

            if message_text and not multi_agent_response:
                raw_message_item: Dict[str, Any] = {
                    "type": "message",
                    "role": "assistant",
                    "status": _normalize_responses_message_status(item_status),
                    "content": [{"type": "output_text", "text": message_text}],
                }
                item_id = _responses_field(item, "id")
                if isinstance(item_id, str) and item_id:
                    raw_message_item["id"] = item_id
                if normalized_phase:
                    raw_message_item["phase"] = normalized_phase
                message_items_raw.append(raw_message_item)
        elif item_type == "reasoning":
            saw_reasoning_item = True
            reasoning_text = _extract_responses_reasoning_text(item)
            item_agent_name = _responses_agent_name(item)
            if reasoning_text and (
                not multi_agent_response or item_agent_name in {None, "/root"}
            ):
                reasoning_parts.append(reasoning_text)
            # Capture the full reasoning item for multi-turn continuity.
            # encrypted_content is an opaque blob the API needs back on
            # subsequent turns to maintain coherent reasoning chains.
            encrypted = _responses_field(item, "encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                raw_item = {"type": "reasoning", "encrypted_content": encrypted}
                # Stamp the issuer so future turns can detect when a
                # model swap moved the conversation to an endpoint that
                # cannot decrypt this blob — see _chat_messages_to_responses_input
                # cross-issuer guard.
                if issuer_kind:
                    raw_item["_issuer_kind"] = issuer_kind
                item_id = _responses_field(item, "id")
                if isinstance(item_id, str) and item_id.startswith("rs_tmp_"):
                    logger.debug(
                        "Skipping transient Codex reasoning item during normalization: %s",
                        item_id,
                    )
                    continue
                if isinstance(item_id, str) and item_id:
                    raw_item["id"] = item_id
                # Capture summary — required by the API when replaying reasoning items
                summary = _responses_field(item, "summary")
                if isinstance(summary, list):
                    raw_summary = []
                    for part in summary:
                        text = _responses_field(part, "text")
                        if isinstance(text, str):
                            raw_summary.append({"type": "summary_text", "text": text})
                    raw_item["summary"] = raw_summary
                if not multi_agent_response:
                    reasoning_items_raw.append(raw_item)
        elif item_type == "function_call":
            if item_status in {"queued", "in_progress", "incomplete"}:
                continue
            fn_name = _responses_field(item, "name", "") or ""
            arguments = _responses_field(item, "arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            raw_call_id = _responses_field(item, "call_id")
            raw_item_id = _responses_field(item, "id")
            embedded_call_id, _ = _split_responses_tool_id(raw_item_id)
            call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = _deterministic_call_id(fn_name, arguments, len(tool_calls))
            call_id = call_id.strip()
            response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
            response_item_id = _derive_responses_function_call_id(call_id, response_item_id)
            tool_calls.append(SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=response_item_id,
                agent=_responses_value_to_plain_data(
                    _responses_field(item, "agent")
                ),
                type="function",
                function=SimpleNamespace(name=fn_name, arguments=arguments),
            ))
        elif item_type == "custom_tool_call":
            fn_name = _responses_field(item, "name", "") or ""
            arguments = _responses_field(item, "input", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            raw_call_id = _responses_field(item, "call_id")
            raw_item_id = _responses_field(item, "id")
            embedded_call_id, _ = _split_responses_tool_id(raw_item_id)
            call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = _deterministic_call_id(fn_name, arguments, len(tool_calls))
            call_id = call_id.strip()
            response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
            response_item_id = _derive_responses_function_call_id(call_id, response_item_id)
            tool_calls.append(SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=response_item_id,
                agent=_responses_value_to_plain_data(
                    _responses_field(item, "agent")
                ),
                type="function",
                function=SimpleNamespace(name=fn_name, arguments=arguments),
            ))

    if multi_agent_response:
        visible_parts = root_final_parts or root_fallback_parts
        final_text = "\n".join([part for part in visible_parts if part]).strip()
    else:
        final_text = "\n".join([part for part in content_parts if part]).strip()
    if (
        not final_text
        and not multi_agent_response
        and hasattr(response, "output_text")
        and not (saw_commentary_phase and not saw_final_answer_phase)
    ):
        out_text = getattr(response, "output_text", "")
        if isinstance(out_text, str):
            final_text = out_text.strip()

    # ── Tool-call leak recovery ──────────────────────────────────
    # gpt-5.x on the Codex Responses API sometimes degenerates and emits
    # what should be a structured `function_call` item as plain assistant
    # text using the Harmony/Codex serialization (``to=functions.foo
    # {json}`` or ``assistant to=functions.foo {json}``). The model
    # intended to call a tool, but the intent never made it into
    # ``response.output`` as a ``function_call`` item, so ``tool_calls``
    # is empty here. If we pass this through, the parent sees a
    # confident-looking summary with no audit trail (empty ``tool_trace``)
    # and no tools actually ran — the Taiwan-embassy-email incident.
    #
    # Detection: leaked tokens always contain ``to=functions.<name>`` and
    # the assistant message has no real tool calls. Treat it as incomplete
    # so the existing Codex-incomplete continuation path (3 retries,
    # handled in run_agent.py) gets a chance to re-elicit a proper
    # ``function_call`` item. The existing loop already handles message
    # append, dedup, and retry budget.
    leaked_tool_call_text = False
    if final_text and not tool_calls and _TOOL_CALL_LEAK_PATTERN.search(final_text):
        leaked_tool_call_text = True
        logger.warning(
            "Codex response contains leaked tool-call text in assistant content "
            "(no structured function_call items). Treating as incomplete so the "
            "continuation path can re-elicit a proper tool call. Leaked snippet: %r",
            final_text[:300],
        )
        # Clear the text so downstream code doesn't surface the garbage as
        # a summary. The encrypted reasoning items (if any) are preserved
        # so the model keeps its chain-of-thought on the retry.
        final_text = ""

    assistant_message = SimpleNamespace(
        content=final_text,
        tool_calls=tool_calls,
        reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
        reasoning_content=None,
        reasoning_details=None,
        codex_reasoning_items=reasoning_items_raw or None,
        codex_message_items=message_items_raw or None,
    )

    if tool_calls:
        finish_reason = "tool_calls"
    elif leaked_tool_call_text:
        finish_reason = "incomplete"
    elif saw_streaming_or_item_incomplete:
        finish_reason = "incomplete"
    elif (has_incomplete_items or saw_commentary_phase) and not saw_final_answer_phase:
        finish_reason = "incomplete"
    elif (reasoning_items_raw or reasoning_parts or saw_reasoning_item) and not final_text:
        # Response contains only reasoning (encrypted thinking state and/or
        # human-readable summary) with no visible content or tool calls. The
        # model is still thinking and needs another turn to produce the actual
        # answer. Marking this as "stop" would send it into the empty-content
        # retry loop which burns retries then fails — treat it as incomplete so
        # the Codex continuation path handles it correctly.
        finish_reason = "incomplete"
    else:
        finish_reason = "stop"
    return assistant_message, finish_reason
