"""Request-scoped, wire-only tool-name aliases for the chat-completions request builder.

Hermes keeps one canonical name per tool; a few routes cannot accept a canonical name on the
wire, so the outgoing copy is aliased and ``ChatCompletionsTransport._normalize_tool_call`` maps
replies back through ``transport._last_wire_aliases`` (reset per request, so only aliases THIS
request advertised are ever reversed and a real tool owning an alias keeps its identity).

- xAI reserves ``tool_search`` (#95003): the bridge declaration is renamed ``hermes_tool_search``.
- Ollama reattributes generated calls to the bridge tool ``tool_call``: with ``tool_call``
  advertised, a correctly generated ``web_search`` call came back named ``tool_call`` (0/9),
  and with the bridge advertised as ``call_deferred_tool`` the same generations came back as
  ``web_search`` (9/9) — Ollama 0.40.6, Qwen 3 and Hermes 3 templates whose call markup is the
  ``<tool_call>`` tag. The parser internals were not inspected. On a detected Ollama server the
  bridge is advertised as ``call_deferred_tool`` (``_2``/``_3`` if a real tool owns it),
  ``tool_call`` mentions in tool descriptions follow the wire name, and past assistant calls in
  the wire copy of history carry it too so the model sees one consistent name. Stored history
  is never rewritten. Nothing guesses that a call to the canonical name meant another tool.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.model_metadata import is_local_endpoint
from utils import base_url_host_matches

logger = logging.getLogger(__name__)

OLLAMA_TOOL_CALL_ALIAS = "call_deferred_tool"
_OLLAMA_COLLIDING_NAME = "tool_call"
# A standalone ``tool_call`` reference: not ``<tool_call>``/``</tool_call>`` markup, not
# ``tool_calls`` and not part of a longer identifier such as ``post_tool_call``.
_TOOL_CALL_REFERENCE = re.compile(r"(?<![A-Za-z0-9_<>/])tool_call(?![A-Za-z0-9_>])")


def _function_name(tool: Any) -> Any:
    return (tool.get("function") or {}).get("name") if isinstance(tool, dict) else None


def _alias_tool_search_bridge_for_xai(agent, transport, tools_for_api):
    """xAI chat-completions reserves ``tool_search`` and 400s when the bridge declares
    it (#95003): rename the wire declaration; ``normalize_response`` maps calls back
    via the transport's ``_last_wire_aliases`` (reset here so a stale map can't
    reverse-map a name this request never aliased). Deep-copy first (#27907)."""
    if transport is not None and hasattr(transport, "_last_wire_aliases"):
        transport._last_wire_aliases = {}
    is_xai_chat = agent.provider in {"xai", "xai-oauth"} or agent._base_url_hostname == "api.x.ai"
    if not (is_xai_chat and tools_for_api):
        return tools_for_api
    try:
        import copy as _copy_xai
        from agent.transports.chat_completions import _rename_tool_search_bridge_for_xai
        has_bridge = any(
            (t.get("function") or {}).get("name") == "tool_search" for t in tools_for_api if isinstance(t, dict)
        )
        if has_bridge:
            tools_for_api = _copy_xai.deepcopy(tools_for_api)
            tools_for_api, alias_map = _rename_tool_search_bridge_for_xai(tools_for_api)
            if transport is not None:
                transport._last_wire_aliases = alias_map
    except Exception as exc:
        logger.warning("%s⚠️ Failed to alias tool_search bridge for xAI: %s", getattr(agent, "log_prefix", ""), exc)
    return tools_for_api


def _detect_ollama_alias_route(agent) -> bool:
    """True for a chat-completions route whose server identifies itself as Ollama.

    Uses the existing endpoint probe (``detect_local_server_type``: ``/api/tags`` must return a
    ``models`` body; TTL- and disk-cached, the same verdict that drives ``num_ctx``), limited to
    endpoints ``is_local_endpoint`` accepts (loopback, LAN, Tailscale). A port number, hostname
    or a localhost address alone never qualifies. Ollama Cloud and public remote endpoints are
    excluded: the collision has not been observed there.
    """
    base_url = getattr(agent, "base_url", None) or ""
    if getattr(agent, "api_mode", None) != "chat_completions" or not base_url:
        return False
    if base_url_host_matches(base_url, "ollama.com") or not is_local_endpoint(base_url):
        return False
    from agent.model_metadata import detect_local_server_type

    api_key = agent.api_key if isinstance(getattr(agent, "api_key", None), str) else ""
    return detect_local_server_type(base_url, api_key=api_key) == "ollama"


def ollama_tool_call_alias_route(agent) -> bool:
    """Session-stable route verdict: computed once per (api_mode, base_url) so the advertised
    tool surface — and with it the cached prompt prefix — cannot flip mid-session."""
    key = (getattr(agent, "api_mode", None), getattr(agent, "base_url", None))
    cached = getattr(agent, "_ollama_tool_alias_route", None)
    if isinstance(cached, tuple) and cached[0] == key:
        return cached[1]
    try:
        verdict = _detect_ollama_alias_route(agent)
    except Exception as exc:
        logger.debug("Ollama tool alias route detection failed: %s", exc, exc_info=True)
        verdict = False
    agent._ollama_tool_alias_route = (key, verdict)
    return verdict


def _rewrite_descriptions(node: Any, alias: str) -> Any:
    """Copy of a schema node whose ``description`` strings name the bridge by its wire name."""
    if isinstance(node, dict):
        return {k: (_TOOL_CALL_REFERENCE.sub(alias, v) if k == "description" and isinstance(v, str)
                    else _rewrite_descriptions(v, alias)) for k, v in node.items()}
    if isinstance(node, list):
        return [_rewrite_descriptions(v, alias) for v in node]
    return node


def alias_tool_call_for_ollama(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Advertise ``tool_call`` as ``call_deferred_tool``; returns ``(tools, {alias: "tool_call"})``.

    The input is never mutated. Without a ``tool_call`` declaration nothing changes. A real tool
    already named ``call_deferred_tool`` keeps its name and the bridge takes ``_2``/``_3``.
    """
    names = {_function_name(t) for t in tools}
    if _OLLAMA_COLLIDING_NAME not in names:
        return tools, {}
    alias, suffix = OLLAMA_TOOL_CALL_ALIAS, 2
    while alias in names:
        alias, suffix = f"{OLLAMA_TOOL_CALL_ALIAS}_{suffix}", suffix + 1
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        wire = _rewrite_descriptions(tool, alias)
        if _function_name(tool) == _OLLAMA_COLLIDING_NAME:
            wire["function"]["name"] = alias
        out.append(wire)
    return out, {alias: _OLLAMA_COLLIDING_NAME}


def alias_history_tool_names(messages: list[dict[str, Any]], wire_names: dict[str, str]) -> list[dict[str, Any]]:
    """Wire copy of ``messages`` whose past tool calls use ``wire_names`` ({canonical: alias}).

    Only messages that change are copied; the stored history is never touched, and the result
    is deterministic for a given history, so the cached prefix stays byte-stable across turns.
    """
    out = None
    for index, msg in enumerate(messages):
        replacement = None
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            calls = msg["tool_calls"]
            new_calls = [
                {**tc, "function": {**tc["function"], "name": wire_names[tc["function"].get("name")]}}
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
                and tc["function"].get("name") in wire_names else tc
                for tc in calls
            ]
            if any(a is not b for a, b in zip(new_calls, calls)):
                replacement = {**msg, "tool_calls": new_calls}
        elif isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("name") in wire_names:
            replacement = {**msg, "name": wire_names[msg["name"]]}
        if replacement is not None:
            if out is None:
                out = list(messages)
            out[index] = replacement
    return messages if out is None else out


def apply_wire_tool_aliases(agent, transport, tools_for_api, api_messages):
    """Alias this request's wire tools/history for its route; returns ``(tools, messages)``.

    Resets the transport's reverse map (xAI step) and records ``agent._ollama_wire_tool_aliases``
    ({alias: canonical}) for stream progress display; both describe THIS request only.
    """
    tools_for_api = _alias_tool_search_bridge_for_xai(agent, transport, tools_for_api)
    agent._ollama_wire_tool_aliases = {}
    if not tools_for_api or not ollama_tool_call_alias_route(agent):
        return tools_for_api, api_messages
    try:
        wire_tools, alias_map = alias_tool_call_for_ollama(tools_for_api)
    except Exception as exc:
        logger.warning("%s⚠️ Failed to alias tool_call bridge for Ollama: %s", getattr(agent, "log_prefix", ""), exc, exc_info=True)
        return tools_for_api, api_messages
    if not alias_map:
        return tools_for_api, api_messages
    if transport is not None and hasattr(transport, "_last_wire_aliases"):
        transport._last_wire_aliases = {**(transport._last_wire_aliases or {}), **alias_map}
    agent._ollama_wire_tool_aliases = alias_map
    wire_names = {canonical: alias for alias, canonical in alias_map.items()}
    return wire_tools, alias_history_tool_names(api_messages, wire_names)


def canonical_stream_tool_name(agent, name):
    """Canonical name for a streamed tool-call name (progress display / partial-call stubs)."""
    if not name:
        return name
    return (getattr(agent, "_ollama_wire_tool_aliases", None) or {}).get(name, name)
