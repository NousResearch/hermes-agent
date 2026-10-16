"""Slack adapter routing onto the slack-sdk Agent Sessions API (agents.sessions.*).

``agents.sessions.setStatus`` accepts only the lifecycle enum
``active|processing|suspended|closed``. Free-text phrases belong on
``assistant.threads.setStatus``; a rejected Agent Sessions call falls back.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack import adapter_sessions as _sessions
from tests.gateway.test_slack import SlackAdapter, _slack_mod


class _SlackApiError(Exception):
    """slack_sdk.SlackApiError shape (``exc.response['error']``) without the SDK."""

    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"ok": False, "error": code}


@pytest.fixture(autouse=True)
def _pin_legacy_assistant_threads_api():
    """Pin the SDK capability probe to the legacy assistant.threads API and
    restore it afterwards, so tests that flip the cached flag on don't leak."""
    prev = _slack_mod._AGENT_SESSIONS_SUPPORTED
    _slack_mod._AGENT_SESSIONS_SUPPORTED = False
    _sessions.reset_status_runtime_state()
    yield
    _slack_mod._AGENT_SESSIONS_SUPPORTED = prev
    _sessions.reset_status_runtime_state()


class TestAgentSessionsApiRouting:
    """slack-sdk 3.44.0 Agent Sessions API (assistant_view deprecation Feb 2027).

    When the installed slack-sdk ships agents.sessions.* typed methods, status
    and title calls route through them; older SDKs keep using the legacy
    assistant.threads.* methods (compat bridge on Slack's side).
    """

    def _adapter(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake-token")
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_typing_uses_agent_sessions_when_supported(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="processing",
        )
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_typing_clears_via_agent_sessions(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.reset_mock()
        await a.stop_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="active",
        )
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_path_sends_the_human_phrase(self):
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="is thinking...",
        )
        a._app.client.agents_sessions_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejected_agent_sessions_falls_back_to_the_phrase(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=_SlackApiError("not_authorized")
        )
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="processing",
        )
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="is thinking...",
        )

        a._app.client.agents_sessions_setStatus.reset_mock()
        a._app.client.assistant_threads_setStatus.reset_mock()
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.assert_not_called()
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="is thinking...",
        )

    @pytest.mark.asyncio
    async def test_rejected_clear_falls_back_to_empty_phrase(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=_SlackApiError("not_authorized")
        )
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a.stop_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="active",
        )
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="",
        )

    @pytest.mark.asyncio
    async def test_agent_sessions_rejection_is_warned_once(self, caplog):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        a._app.client.assistant_threads_setStatus = AsyncMock()
        with caplog.at_level("WARNING", logger=_sessions.logger.name):
            await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
            await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        warnings = [
            rec for rec in caplog.records
            if rec.levelname == "WARNING" and "agents.sessions.setStatus" in rec.getMessage()
        ]
        assert len(warnings) == 1

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


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("is thinking...", "processing"),
        ("is reading docs/api.md…", "processing"),
        ("  still working… (2m03s)  ", "processing"),
        ("", "active"),
        ("   ", "active"),
    ],
)
def test_agent_session_status_enum_mapping(phrase, expected):
    assert _sessions.agent_session_status_enum(phrase) == expected
