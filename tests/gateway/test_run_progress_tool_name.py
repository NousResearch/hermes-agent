"""Tool names in live gateway progress lines go through the adapter's ``format_tool_name``.

Discord renders progress as markdown, so a raw ``mcp_github_create_pull_request`` lost its
underscores to italic/underline parsing; other platforms keep the name unchanged.
"""

import pytest

from gateway.config import Platform
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _run_with_agent
from tests.waiting import wait_until

MCP_TOOL = "mcp_github_create_pull_request"


class _Capture(ProgressCaptureAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        _McpAgent.adapter = self


class _DiscordCapture(_Capture):
    """Capture sends while using Discord's real tool-name formatter."""

    supports_code_blocks = True

    def format_tool_name(self, name):
        from plugins.platforms.discord.adapter import DiscordAdapter

        return DiscordAdapter.format_tool_name(self, name)


class _McpAgent:
    """Emits one MCP tool call; ``preview`` is set per test."""

    adapter = None
    preview = None

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        args = {"title": self.preview} if self.preview else {}
        self.tool_progress_callback("tool.started", MCP_TOOL, self.preview, args)
        wait_until(lambda: _McpAgent.adapter.sent, what="the progress bubble was sent")
        return {"final_response": "done", "messages": [], "api_calls": 1}


async def _progress(monkeypatch, tmp_path, adapter_cls, platform, *, mode="all", preview=None) -> str:
    monkeypatch.setattr(_McpAgent, "preview", preview)
    adapter, _ = await _run_with_agent(
        monkeypatch, tmp_path, _McpAgent, session_id=f"sess-{adapter_cls.__name__}-{mode}",
        config_data={"display": {"tool_progress": mode}}, platform=platform, adapter_cls=adapter_cls,
    )
    return adapter.sent[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "preview", "expected"), [
    ("all", "Fix typo", f'`{MCP_TOOL}`: "Fix typo"'),
    ("all", None, f"`{MCP_TOOL}`..."),
    ("verbose", "Fix typo", f"`{MCP_TOOL}`(['title'])"),
])
async def test_discord_progress_renders_mcp_tool_names_as_inline_code(monkeypatch, tmp_path, mode, preview, expected):
    line = await _progress(monkeypatch, tmp_path, _DiscordCapture, Platform.DISCORD, mode=mode, preview=preview)

    assert expected in line, line


@pytest.mark.asyncio
async def test_other_platforms_keep_the_raw_tool_name(monkeypatch, tmp_path):
    line = await _progress(monkeypatch, tmp_path, _Capture, Platform.TELEGRAM, preview="Fix typo")

    assert line.endswith(f'{MCP_TOOL}: "Fix typo"') and "`" not in line, line
