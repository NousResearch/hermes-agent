from __future__ import annotations

import os
from typing import Any

from ..config import BrokerSettings
from ..core import ToolExecutionError, result_payload
from ..remote_mcp import RemoteMcpBridge


class GrainProvider:
    def __init__(self, settings: BrokerSettings) -> None:
        self.settings = settings
        self.bridge = RemoteMcpBridge(command="npx", args=["-y", "mcp-remote", settings.grain_mcp_url], env=os.environ.copy())

    async def meetings_search(self, *, query: str, limit: int = 10) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(
            os.getenv("GRAIN_TOOL_MEETINGS_SEARCH"),
            ["search_meetings_v2", "list_meetings"],
        )
        if remote == "search_meetings_v2":
            args = {
                "search_queries": [query],
                "limit": limit,
            }
        else:
            args = {
                "filters": {"title_search": query},
                "limit": limit,
            }
        return result_payload(data=await self.bridge.call_tool(remote, args))

    async def meeting_get(self, *, meeting_id: str) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(
            os.getenv("GRAIN_TOOL_MEETING_GET"),
            ["fetch_meeting"],
        )
        return result_payload(data=await self.bridge.call_tool(remote, {"meeting_id": meeting_id}))

    async def transcript_get(self, *, meeting_id: str) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(
            os.getenv("GRAIN_TOOL_TRANSCRIPT_GET"),
            ["fetch_meeting_transcript"],
        )
        return result_payload(data=await self.bridge.call_tool(remote, {"meeting_id": meeting_id}))

    async def highlights_list(self, *, meeting_id: str) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(
            os.getenv("GRAIN_TOOL_HIGHLIGHTS_LIST"),
            ["list_clips"],
        )
        data = await self.bridge.call_tool(remote, {"limit": 20})
        return result_payload(
            data=data,
            warnings=[
                "Grain's current MCP clip listing schema does not expose a direct meeting_id filter. This broker currently returns clip results without strict meeting scoping."
            ],
        )

    async def notes_get(self, *, meeting_id: str) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(
            os.getenv("GRAIN_TOOL_NOTES_GET"),
            ["fetch_meeting_notes"],
        )
        return result_payload(data=await self.bridge.call_tool(remote, {"meeting_id": meeting_id}))


class GranolaProvider:
    def __init__(self, settings: BrokerSettings) -> None:
        self.settings = settings
        self.bridge = RemoteMcpBridge(command="npx", args=["-y", "mcp-remote", settings.granola_mcp_url], env=os.environ.copy())

    async def meetings_search(self, *, query: str | None = None, folder_id: str | None = None, limit: int = 10) -> dict[str, Any]:
        if query:
            remote = await self.bridge.resolve_tool(os.getenv("GRANOLA_TOOL_MEETINGS_SEARCH"), ["get_meetings", "list_meetings"])
            args: dict[str, Any] = {"query": query, "limit": limit}
        else:
            remote = await self.bridge.resolve_tool(os.getenv("GRANOLA_TOOL_MEETINGS_LIST"), ["list_meetings"])
            args = {"limit": limit}
        if folder_id:
            args["folder_id"] = folder_id
        return result_payload(data=await self.bridge.call_tool(remote, args))

    async def meeting_get(self, *, meeting_id: str) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(os.getenv("GRANOLA_TOOL_MEETING_GET"), ["get_meetings"])
        return result_payload(data=await self.bridge.call_tool(remote, {"meeting_id": meeting_id}))

    async def transcript_get(self, *, meeting_id: str) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(os.getenv("GRANOLA_TOOL_TRANSCRIPT_GET"), ["get_meeting_transcript"])
        return result_payload(data=await self.bridge.call_tool(remote, {"meeting_id": meeting_id}))

    async def notes_get(self, *, meeting_id: str) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(os.getenv("GRANOLA_TOOL_NOTES_GET"), ["get_meetings"])
        return result_payload(data=await self.bridge.call_tool(remote, {"meeting_id": meeting_id}))

    async def folders_list(self) -> dict[str, Any]:
        remote = await self.bridge.resolve_tool(os.getenv("GRANOLA_TOOL_FOLDERS_LIST"), ["list_meeting_folders"])
        return result_payload(data=await self.bridge.call_tool(remote, {}))
