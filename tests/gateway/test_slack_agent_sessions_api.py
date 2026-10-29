"""Slack adapter on slack-sdk 3.44+, which ships the Agent Sessions API (agents.sessions.*)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter, _slack_mod

_SESSION_STATUSES = {"active", "processing", "suspended", "closed"}


@pytest.fixture(autouse=True)
def _pin_legacy_assistant_threads_api():
    """Pin the SDK capability probe to the legacy assistant.threads API and
    restore it afterwards, so tests that flip the cached flag on don't leak."""
    prev = _slack_mod._AGENT_SESSIONS_SUPPORTED
    _slack_mod._AGENT_SESSIONS_SUPPORTED = False
    yield
    _slack_mod._AGENT_SESSIONS_SUPPORTED = prev


class _SlackResponse:
    """slack_sdk's SlackResponse shape: mapping-like (``.get``) but not a dict; the body is ``.data``."""

    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)


class _SlackApiError(Exception):
    """slack_sdk.SlackApiError's shape (``exc.response`` is a SlackResponse) without importing the
    SDK, which CI stubs as a bare module."""

    def __init__(self, body):
        super().__init__(f"The server responded with: {body}")
        self.response = _SlackResponse(body)


class _SlackSdk344Client:
    """slack-sdk 3.44 web client answering the two status methods the way Slack does.

    agents.sessions.setStatus takes only a lifecycle value and no text; while a thread's session is
    ``processing`` Slack shows its own "Working..." indicator in place of any custom text.
    assistant.threads.setStatus sets the free text and "" clears it. ``shown`` is what the user
    sees beneath the composer.
    """

    def __init__(self, legacy_error: str = ""):
        self.text: dict[tuple[str, str | None], str] = {}
        self.session: dict[tuple[str, str | None], str] = {}
        self.legacy_error = legacy_error

    def shown(self, channel_id: str, thread_ts: str) -> str | None:
        key = (channel_id, thread_ts)
        return "Working..." if self.session.get(key) == "processing" else self.text.get(key)

    async def agents_sessions_setStatus(self, *, channel_id, status, thread_ts=None, **_kwargs):
        if status not in _SESSION_STATUSES:
            raise _SlackApiError({
                "ok": False, "error": "invalid_arguments",
                "response_metadata": {"messages": ["[ERROR] must be a valid enum value [json-pointer:/status]"]}})
        self.session[(channel_id, thread_ts)] = status

    async def assistant_threads_setStatus(self, *, channel_id, thread_ts, status, **_kwargs):
        if self.legacy_error:
            raise _SlackApiError({"ok": False, "error": self.legacy_error})
        self.text[(channel_id, thread_ts)] = status


class TestAgentSessionsApiRouting:
    """The working-status line is free text, so it stays on assistant.threads.setStatus even when
    the SDK has agents.sessions.*; the thread title moves to agents.sessions.rename."""

    def _adapter(self, client=None):
        config = PlatformConfig(enabled=True, token="xoxb-fake-token")
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = client if client is not None else AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_status_line_shows_free_text_and_clears_on_agent_sessions_sdk(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True  # slack-sdk 3.44+ ships agents.sessions.*
        slack = _SlackSdk344Client()
        a = self._adapter(slack)
        meta = {"thread_id": "171234.0001"}

        await a.send_typing("C123", metadata=meta)
        assert slack.shown("C123", "171234.0001") == "is thinking..."

        a.set_status_text("C123", "is reading config.yaml…")
        await a.send_typing("C123", metadata=meta)
        assert slack.shown("C123", "171234.0001") == "is reading config.yaml…"

        await a.stop_typing("C123", metadata=meta)
        assert slack.shown("C123", "171234.0001") == ""

    @pytest.mark.asyncio
    async def test_rejected_status_write_warns_once_per_workspace_and_error(self, caplog):
        a = self._adapter(_SlackSdk344Client(legacy_error="missing_scope"))
        a._remember_channel_team("C123", "T1")
        meta = {"thread_id": "171234.0001"}
        with caplog.at_level(logging.WARNING):
            for _ in range(3):  # the typing refresh re-sends the status for the whole turn
                await a.send_typing("C123", metadata=meta)
            await a.stop_typing("C123", metadata=meta)
            await a.stop_typing("C123", metadata=meta)  # untracked now: a clear with no team id
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno >= logging.WARNING and "setStatus" in r.getMessage()]
        assert len(warnings) == 1
        assert "assistant.threads.setStatus" in warnings[0] and "missing_scope" in warnings[0]

    @pytest.mark.asyncio
    async def test_thread_title_uses_agents_sessions_rename(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        a._app.client.agents_sessions_rename = AsyncMock()
        a._app.client.assistant_threads_setTitle = AsyncMock()
        await a._set_assistant_thread_title("D123", "171234.0001", "Summarize the incident")
        a._app.client.agents_sessions_rename.assert_called_once_with(
            channel_id="D123",
            thread_ts="171234.0001",
            title="Summarize the incident",
        )
        a._app.client.assistant_threads_setTitle.assert_not_called()
