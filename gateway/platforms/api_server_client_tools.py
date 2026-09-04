"""Client-tools bridge for the API server (Raycast AI Extensions & friends).

OpenAI-compatible clients may attach executable tool schemas to
``/v1/chat/completions`` (``tools`` / ``tool_choice``).  The CLIENT executes
these tools - e.g. Raycast runs ``@clipboard`` on the user's machine - so the
server must never run them.  It only needs to:

1. describe the schemas to the agent (system contract),
2. echo the agent's decision back as a protocol-correct ``tool_calls``
   response instead of prose, and
3. fold the client's follow-up ``role:"tool"`` result back into the
   conversation so the agent can continue to a final answer.

The decision channel reuses the proven ACP text contract from
``agent/acp_openai_bridge.py`` (``<tool_call>{...}</tool_call>``), so the
model-side behavior is identical to the IDE path and already covered by
``tests/agent/test_acp_openai_bridge.py``.

Kill-switch: ``gateway.platforms.api_server.client_tools: false`` in
config.yaml disables the bridge entirely (default: enabled).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Hard caps - a hostile or buggy client must not inflate the system prompt.
MAX_CLIENT_TOOLS = 32
MAX_SCHEMA_CHARS = 24_000
MAX_PARALLEL_CALLS = 6

# Conservative list of Hermes-native tool names.  A client tool whose name
# collides with one of these is renamed on the wire to ``client__<name>``
# and mapped back on emit, so the model can never confuse the two.
_RESERVED_HERMES_TOOL_NAMES = frozenset({
    "terminal", "read_file", "write_file", "patch", "search_files",
    "web_search", "web_extract", "memory", "fact_store", "session_search",
    "execute_code", "delegate_task", "cronjob", "todo", "image_generate",
    "text_to_speech", "browser_exec", "drive_preview", "desktop_preview",
    "vision_analyze", "skill_view", "skills_list", "skill_manage",
    "read_terminal", "focus_pane", "apply_layout", "tour", "tip",
})

_WIRE_PREFIX = "client__"

# Mirrors agent/acp_openai_bridge.py TOOL_CALL_BLOCK_RE.
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _function_schema(tool: Any) -> Optional[Dict[str, Any]]:
    """Return a normalised OpenAI function schema, or None if invalid."""
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if not isinstance(fn, dict):
        # Tolerate bare {name, description, parameters} shapes.
        fn = tool
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name.strip()):
        return None
    schema = {
        "name": name.strip(),
        "description": str(fn.get("description") or "")[:2000],
        "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}},
    }
    return schema


class ClientToolsBridge:
    """Represents the client-supplied tools for one request."""

    def __init__(self, tools: List[Any], tool_choice: Any = None):
        self._schemas: List[Dict[str, Any]] = []
        self._wire_to_original: Dict[str, str] = {}
        self._original_to_wire: Dict[str, str] = {}
        dropped = 0
        serialized = 0
        for tool in tools[: MAX_CLIENT_TOOLS * 4]:  # bound the scan itself
            if len(self._schemas) >= MAX_CLIENT_TOOLS:
                dropped += 1
                continue
            schema = _function_schema(tool)
            if schema is None:
                dropped += 1
                continue
            blob = json.dumps(schema)
            if serialized + len(blob) > MAX_SCHEMA_CHARS:
                dropped += 1
                continue
            serialized += len(blob)
            wire = self._wire_name(schema["name"])
            self._wire_to_original[wire] = schema["name"]
            self._original_to_wire[schema["name"]] = wire
            schema["wire_name"] = wire
            self._schemas.append(schema)
        if dropped:
            logger.warning(
                "client-tools bridge: dropped %d malformed/oversized tool schemas", dropped
            )
        self.dropped_count = dropped
        self.tool_choice = tool_choice

    # -- naming ------------------------------------------------------------

    @staticmethod
    def _wire_name(original: str) -> str:
        if original in _RESERVED_HERMES_TOOL_NAMES or original.startswith("_"):
            return _WIRE_PREFIX + original
        return original

    def wire_name(self, original: str) -> str:
        return self._original_to_wire.get(original, original)

    def original_name(self, wire: str) -> str:
        return self._wire_to_original.get(wire, wire)

    def is_wire_name(self, wire: str) -> bool:
        return wire in self._wire_to_original

    def recognizes(self, name: str) -> bool:
        """True for any client tool name in either form (original or wire)."""
        return name in self._original_to_wire or name in self._wire_to_original

    # -- state -------------------------------------------------------------

    @property
    def suppressed(self) -> bool:
        """True when no usable schema survived validation."""
        return not self._schemas

    @property
    def wire_names(self) -> List[str]:
        return [s["wire_name"] for s in self._schemas]

    def _forced_name(self) -> Optional[str]:
        """Function name when tool_choice names one, else None."""
        if isinstance(self.tool_choice, dict) and isinstance(self.tool_choice.get("function"), dict):
            name = self.tool_choice["function"].get("name")
            if isinstance(name, str) and name in self._original_to_wire:
                return self._original_to_wire[name]
        return None

    # -- system contract ----------------------------------------------------

    def system_contract(self) -> str:
        parts = [
            "## Client-side tools",
            "",
            "The CLIENT application can execute the following tools on the",
            "user's machine. You cannot run them yourself. When (and only when)",
            "one is needed to answer, emit instead of replying:",
            "",
            "<tool_call>{\"function\": {\"name\": \"<tool_name>\", \"arguments\": {...}}}</tool_call>",
            "",
            "One JSON object per block. Emit up to %d blocks for independent calls."
            % MAX_PARALLEL_CALLS,
            "Never invent or assume a tool result. Never wrap the block in prose",
            "or code fences. If no client tool is needed, simply answer normally.",
            "",
            "Available tools:",
        ]
        forced = self._forced_name()
        for s in self._schemas:
            parts.append(
                "- %s: %s | parameters: %s"
                % (s["wire_name"], s["description"] or "(no description)", json.dumps(s["parameters"]))
            )
        if forced:
            parts.append("")
            parts.append(
                "The client explicitly requested tool %r - emit its block immediately."
                % forced
            )
        return "\n".join(parts)

    # -- response extraction -------------------------------------------------

    def extract_calls(self, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """Pull ``<tool_call>`` decisions out of final text.

        Returns (openai_tool_calls, residual_text).  tool_calls entries use
        ORIGINAL client names (wire prefix stripped).  Empty list => the text
        is a plain answer.
        """
        if not text or "<tool_call>" not in text:
            return [], text
        try:
            from agent.acp_openai_bridge import extract_tool_calls_from_text
            calls, residual = extract_tool_calls_from_text(text)
        except Exception:
            logger.exception("client-tools bridge: block extraction failed")
            return [], text
        if not calls:
            return [], text
        out: List[Dict[str, Any]] = []
        for c in calls[:MAX_PARALLEL_CALLS]:
            fn = getattr(getattr(c, "function", None), "name", None) or ""
            args = getattr(getattr(c, "function", None), "arguments", None) or "{}"
            out.append({
                "id": getattr(c, "id", None) or f"call_{len(out)}",
                "type": "function",
                "function": {"name": self.original_name(fn), "arguments": args},
            })
        return out, residual.strip()


def fold_tool_result(messages: List[Any], bridge: "ClientToolsBridge") -> Tuple[List[Dict[str, Any]], bool]:
    """Fold a trailing assistant-tool_calls + role:tool pair into one user turn.

    Returns (new_messages, ok).  ok=False => nothing matched; caller proceeds
    with the original messages.
    """
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            continue
        if not all(
            isinstance(c, dict) and bridge.recognizes((c.get("function") or {}).get("name", ""))
            for c in calls
        ):
            continue
        j = i + 1
        results: List[Tuple[str, Any]] = []
        while j < len(messages) and isinstance(messages[j], dict) and messages[j].get("role") == "tool":
            results.append((messages[j].get("tool_call_id", ""), messages[j].get("content", "")))
            j += 1
        if not results:
            continue
        lines = ["[Client tool results]"]
        for call in calls:
            fn = (call.get("function") or {})
            tc_id = call.get("id", "")
            result_content = next((c for rid, c in results if rid == tc_id), "")
            lines.append(
                "You asked the client to run %s(%s).\nClient returned: %s"
                % (fn.get("name", "?"), fn.get("arguments", "{}"), result_content)
            )
        lines.append("")
        lines.append(
            "Continue. Use the results above to answer. If another listed client"
            " tool is genuinely required you may emit another <tool_call> block;"
            " otherwise reply normally and do not mention this exchange."
        )
        rendered = {"role": "user", "content": "\n".join(lines)}
        return messages[:i] + [rendered] + messages[j:], True
    return messages, False


def plan_stream_text(buffer: str, bridge: "ClientToolsBridge") -> Tuple[str, List[Dict[str, Any]]]:
    """Post-process the accumulated streamed text once the agent finishes.

    Returns (residual_text, tool_calls).  Writers emit residual as content
    when there are no calls; otherwise emit tool_calls chunks and finish with
    finish_reason="tool_calls".
    """
    return bridge.extract_calls(buffer)
