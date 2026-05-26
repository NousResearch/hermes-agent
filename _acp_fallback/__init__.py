"""Minimal in-repo fallback for the optional agent-client-protocol package.

The real ACP package is used when the optional ``hermes-agent[acp]`` extra is
installed.  Test and source imports still need to be import-safe in lean
environments, so this module implements only the small surface Hermes uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .exceptions import RequestError
from .schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    FileEditToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)

PROTOCOL_VERSION = 1


class Agent:
    pass


class Client:
    async def session_update(self, session_id: str, update: Any) -> None:
        pass

    async def request_permission(self, **kwargs: Any) -> Any:
        return None


def text_block(text: str) -> TextContentBlock:
    return TextContentBlock(type="text", text=text)


def tool_content(content: TextContentBlock) -> ContentToolCallContent:
    return ContentToolCallContent(type="content", content=content)


def tool_diff_content(
    *,
    path: str,
    new_text: str | None = None,
    old_text: str | None = None,
) -> FileEditToolCallContent:
    return FileEditToolCallContent(
        type="diff",
        path=path,
        old_text=old_text,
        new_text=new_text,
    )


def start_tool_call(
    tool_call_id: str,
    title: str,
    *,
    kind: str = "other",
    content: list[Any] | None = None,
    locations: list[Any] | None = None,
    raw_input: Any = None,
) -> ToolCallStart:
    return ToolCallStart(
        session_update="tool_call",
        tool_call_id=tool_call_id,
        title=title,
        kind=kind,
        content=content,
        locations=locations,
        raw_input=raw_input,
    )


def update_tool_call(
    tool_call_id: str,
    *,
    title: str | None = None,
    kind: str = "other",
    status: str = "completed",
    content: list[Any] | None = None,
    raw_output: Any = None,
    raw_input: Any = None,
) -> ToolCallProgress:
    return ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=tool_call_id,
        title=title,
        kind=kind,
        status=status,
        content=content,
        raw_input=raw_input,
        raw_output=raw_output,
    )


def update_agent_thought_text(text: str) -> AgentThoughtChunk:
    return AgentThoughtChunk(
        session_update="agent_thought_chunk",
        content=TextContentBlock(type="text", text=text),
    )


def update_agent_message_text(text: str) -> AgentMessageChunk:
    return AgentMessageChunk(
        session_update="agent_message_chunk",
        content=TextContentBlock(type="text", text=text),
    )


def update_user_message_text(text: str) -> UserMessageChunk:
    return UserMessageChunk(
        session_update="user_message_chunk",
        content=TextContentBlock(type="text", text=text),
    )


async def run_agent(
    agent: Agent,
    *,
    input_stream: Any | None = None,
    output_stream: Any | None = None,
    use_unstable_protocol: bool = False,
) -> None:
    """Tiny JSON-RPC loop used by tests when the external ACP package is absent."""
    if hasattr(agent, "on_connect"):
        agent.on_connect(Client())

    reader = output_stream
    writer = input_stream
    if reader is None or writer is None:
        return

    while True:
        line = await reader.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8"))
            method = request.get("method")
            req_id = request.get("id")
            if method not in {
                "initialize",
                "session/new",
                "session/prompt",
                "session/cancel",
                "authenticate",
            }:
                raise RequestError.method_not_found(str(method))
            response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        except RequestError as exc:
            logging.getLogger(__name__).exception("Background task failed")
            response = {
                "jsonrpc": "2.0",
                "id": locals().get("req_id", None),
                "error": {"code": exc.code, "message": exc.message, "data": exc.data},
            }
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(exc), "data": None},
            }
        writer.write((json.dumps(response) + "\n").encode("utf-8"))
        await writer.drain()
        await asyncio.sleep(0)
