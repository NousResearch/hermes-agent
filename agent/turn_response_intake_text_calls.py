"""Tool calls a chat-completions server returned as answer text (#151373).

A server that binds tool calls only for names in the request's ``tools[]`` (correct per the OpenAI
contract) returns a ``<tool_call>`` block for any other name as plain content. With tool search on,
that is every call the model writes by a deferred tool's real name: the block comes back as text,
``strip_think_blocks`` later removes it, the reply reads as empty and the empty-response retries
replay a byte-identical, fully cached prompt that regenerates the same call.

``recover_text_tool_calls`` turns such blocks back into structured calls at intake, so the turn
continues on the normal path: ``validate_tool_calls`` gives them unique ids and routes a deferred
tool's real name through the ``tool_call`` bridge (``_route_deferred_call``), as it does for a call
the server parsed itself. The recovery is deliberately narrow; anything it does not accept stays
text exactly as before:

- chat-completions replies only, and only when the reply carries no structured tool calls;
- every block must be complete and parse (Qwen XML ``<function=NAME><parameter=...>`` or the JSON
  ``{"name", "arguments"}`` form); one bad block leaves the whole reply alone;
- every name must be one this session can call: a declared tool, or a deferred tool in scope
  behind the bridge. A hallucinated name is not turned into a call;
- a block inside a code fence or inline code is an example the model is showing, not a call;
- a block the model copied out of a recent tool result is data it is quoting, not a call.

Not the ACP bridge's ``extract_tool_calls_from_text``: that reads its own prompt contract's JSON
form and falls back to bare JSON anywhere in the text, which would run JSON a model merely quotes.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"^<function=([^>\s]+)>\s*(.*?)\s*(?:</function>)?\s*$", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_NUMBER = re.compile(r"-?\d+(\.\d+)?")
_RECENT_TOOL_RESULTS = 12


def _parameter_types(agent: Any, name: str) -> dict[str, Any]:
    """Parameter name -> JSON-schema type for ``name`` (declared schema first, then the registry)."""
    for tool in getattr(agent, "tools", None) or ():
        fn = (tool or {}).get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
            return {k: (v or {}).get("type") for k, v in props.items() if isinstance(v, dict)}
    try:
        from tools.tool_search_catalog import _registry_entry

        schema = getattr(_registry_entry(name), "schema", None) or {}
        props = (schema.get("parameters") or schema).get("properties") or {}
        return {k: (v or {}).get("type") for k, v in props.items() if isinstance(v, dict)}
    except Exception:  # an unreadable registry entry only costs the typing: values stay as written
        logger.debug("no parameter types for %s", name, exc_info=True)
        return {}


def _value(raw: str, declared: Any) -> Any:
    """A ``<parameter>`` body as the schema wants it: a string parameter stays text as written."""
    if declared == "string":
        return raw
    text = raw.strip()
    if text[:1] in "[{\"" or text in ("true", "false", "null") or _NUMBER.fullmatch(text):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return raw


def _in_code(content: str) -> bool:
    """Some ``<tool_call>`` sits inside a ``` fence or `inline code`: the model is showing the format."""
    for m in re.finditer(r"<tool_call>", content):
        before = content[:m.start()]
        if before.count("```") % 2 == 1 or before.split("\n")[-1].count("`") % 2 == 1:
            return True
    return False


def parse_text_tool_calls(agent: Any, content: str, *, allow_code: bool = False) -> Optional[list[tuple[str, dict]]]:
    """``[(name, arguments), ...]`` for the ``<tool_call>`` blocks in ``content``, or None when there are
    none, one is cut off or malformed, or (unless ``allow_code``) one is quoted inside code."""
    if content.count("<tool_call>") != content.count("</tool_call>"):
        return None
    if not allow_code and _in_code(content):
        return None
    calls: list[tuple[str, dict]] = []
    for body in _BLOCK.findall(content):
        m = _FUNCTION.match(body)
        if m:
            name = m.group(1)
            types = _parameter_types(agent, name)
            args: Any = {k: _value(v, types.get(k)) for k, v in _PARAMETER.findall(m.group(2))}
        else:
            try:
                obj = json.loads(body)
                name, args = obj.get("name"), obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args or "{}")
            except (ValueError, AttributeError):
                return None
        if not isinstance(name, str) or not _NAME.fullmatch(name.strip()) or not isinstance(args, dict):
            return None
        calls.append((name.strip(), args))
    return calls or None


def _strings(value: Any) -> list[str]:
    """Every string in a tool result; results are often stored as JSON, so their text arrives escaped."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return _strings(json.loads(text))
            except ValueError:
                pass
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _quotes_a_tool_result(agent: Any, calls: list[tuple[str, dict]], messages: Any) -> bool:
    """A recovered call equal to one written inside a recent tool result is data being quoted."""
    def key(call: tuple[str, dict]) -> tuple[str, str]:
        return call[0], json.dumps(call[1], sort_keys=True, default=str)

    wanted = {key(c) for c in calls}
    for msg in list(messages or [])[-_RECENT_TOOL_RESULTS:]:
        if not (isinstance(msg, dict) and msg.get("role") == "tool"):
            continue
        for text in _strings(msg.get("content") or ""):
            if "<tool_call>" in text and wanted & {key(c) for c in parse_text_tool_calls(agent, text, allow_code=True) or ()}:
                return True
    return False


def _callable_here(agent: Any, name: str) -> bool:
    """Declared in this request, or a deferred tool this session may call through the bridge."""
    valid = getattr(agent, "valid_tool_names", None) or ()
    if name in valid:
        return True
    from tools.tool_search_catalog import TOOL_CALL_NAME

    if TOOL_CALL_NAME not in valid:
        return False
    from agent.tool_executor import _canonical_tool_name, _tool_search_scoped_names

    return _canonical_tool_name(name) in _tool_search_scoped_names(agent)


def recover_text_tool_calls(agent: Any, message: Any, messages: Any = None) -> int:
    """Turn ``<tool_call>`` text in ``message`` into structured tool calls (see the module docstring).
    Returns how many were recovered; 0 leaves ``message`` untouched."""
    content = getattr(message, "content", None)
    if getattr(message, "tool_calls", None) or not isinstance(content, str) or "<tool_call>" not in content:
        return 0
    if (getattr(agent, "api_mode", None) or "chat_completions") != "chat_completions":
        return 0
    try:
        calls = parse_text_tool_calls(agent, content)
        if not calls or not all(_callable_here(agent, name) for name, _ in calls):
            return 0
        if _quotes_a_tool_result(agent, calls, messages):
            return 0
        from agent.transports.types import build_tool_call

        message.tool_calls = [build_tool_call(f"call_{uuid.uuid4().hex[:24]}", name, args) for name, args in calls]
        message.content = _BLOCK.sub("", content).strip()
        message.finish_reason = "tool_calls"
        logger.info("Recovered %d tool call(s) the server returned as text: %s",
                    len(calls), ", ".join(name for name, _ in calls))
        return len(calls)
    except Exception:
        logger.debug("text tool-call recovery skipped", exc_info=True)
        return 0
