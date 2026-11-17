"""Builders for the ACP session-update and content objects the adapter sends.

agent-client-protocol 1.0 dropped the top-level ``acp.text_block`` / ``acp.update_*`` helpers
that 0.x shipped; the schema models they wrapped are unchanged, so the adapter builds them here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    FileEditToolCallContent,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)


def text_block(text: str) -> TextContentBlock:
    return TextContentBlock(type="text", text=text)


def tool_content(block: Any) -> ContentToolCallContent:
    return ContentToolCallContent(type="content", content=block)


def tool_diff_content(path: str, new_text: str, old_text: str | None = None) -> FileEditToolCallContent:
    return FileEditToolCallContent(type="diff", path=path, new_text=new_text, old_text=old_text)


def update_user_message_text(text: str) -> UserMessageChunk:
    return UserMessageChunk(session_update="user_message_chunk", content=text_block(text))


def update_agent_message_text(text: str) -> AgentMessageChunk:
    return AgentMessageChunk(session_update="agent_message_chunk", content=text_block(text))


def update_agent_thought_text(text: str) -> AgentThoughtChunk:
    return AgentThoughtChunk(session_update="agent_thought_chunk", content=text_block(text))


def _tool_call_fields(content: Sequence[Any] | None, locations: Sequence[ToolCallLocation] | None) -> dict:
    return {"content": list(content) if content is not None else None,
            "locations": list(locations) if locations is not None else None}


def start_tool_call(tool_call_id: str, title: str, *, kind: Any = None, status: Any = None,
                    content: Sequence[Any] | None = None, locations: Sequence[ToolCallLocation] | None = None,
                    raw_input: Any = None, raw_output: Any = None) -> ToolCallStart:
    return ToolCallStart(session_update="tool_call", tool_call_id=tool_call_id, title=title, kind=kind,
                         status=status, raw_input=raw_input, raw_output=raw_output,
                         **_tool_call_fields(content, locations))


def update_tool_call(tool_call_id: str, *, title: str | None = None, kind: Any = None, status: Any = None,
                     content: Sequence[Any] | None = None, locations: Sequence[ToolCallLocation] | None = None,
                     raw_input: Any = None, raw_output: Any = None) -> ToolCallProgress:
    return ToolCallProgress(session_update="tool_call_update", tool_call_id=tool_call_id, title=title, kind=kind,
                            status=status, raw_input=raw_input, raw_output=raw_output,
                            **_tool_call_fields(content, locations))
