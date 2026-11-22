"""Tool history as assistant text, for requests that declare no tools.

Anthropic Messages ("Requests which include `tool_use` or `tool_result` blocks must define tools")
and Bedrock Converse ("The toolConfig field must be defined when using toolUse and toolResult
content blocks") reject tool blocks in a request without tool definitions, so a tools-free request
(the compaction summarizer's role-typed input, ``tool_choice="none"``) sends its tool history as
text: ``build_anthropic_kwargs`` and the aux-only ``call_converse`` apply this. Results go in the ASSISTANT role, never user: text a tool printed must not read
as a user turn (#146750).
"""

from __future__ import annotations

from typing import Any


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str)
            else (part.get("text") or "") if isinstance(part, dict) and part.get("type") in (None, "text")
            else "[non-text content]"
            for part in content
        )
    return "" if content is None else str(content)


def tool_history_for_request(messages: list[dict[str, Any]], tools: Any, tool_choice: Any = None) -> list[dict[str, Any]]:
    """``messages`` unchanged when the request declares ``tools`` (and ``tool_choice`` is not ``"none"``);
    otherwise a copy with ``tool_calls`` and ``role=tool`` rewritten as assistant text."""
    if tools and tool_choice != "none":
        return messages
    if not any(m.get("role") == "tool" or m.get("tool_calls") for m in messages):
        return messages
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            out.append({"role": "assistant", "content": f"[Tool result {msg.get('tool_call_id') or ''}]: {_text_of(msg.get('content'))}"})
        elif role == "assistant" and msg.get("tool_calls"):
            calls = "\n".join(
                f"  {(tc.get('function') or {}).get('name') or 'tool'}({(tc.get('function') or {}).get('arguments') or ''})"
                for tc in msg["tool_calls"] if isinstance(tc, dict)
            )
            text = _text_of(msg.get("content"))
            rewritten = {k: v for k, v in msg.items() if k != "tool_calls"}
            rewritten["content"] = (text + "\n" if text else "") + f"[Tool calls:\n{calls}\n]"
            out.append(rewritten)
        else:
            out.append(msg)
    return out
