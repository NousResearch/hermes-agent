"""Slack adapter routing onto the slack-sdk Agent Sessions API (agents.sessions.*)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter, _slack_mod


@pytest.fixture(autouse=True)
def _pin_legacy_assistant_threads_api():
    """Pin the SDK capability probe to the legacy assistant.threads API and
    restore it afterwards, so tests that flip the cached flag on don't leak."""
    prev = _slack_mod._AGENT_SESSIONS_SUPPORTED
    _slack_mod._AGENT_SESSIONS_SUPPORTED = False
    yield
    _slack_mod._AGENT_SESSIONS_SUPPORTED = prev


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
    async def test_agent_sessions_status_maps_free_text_to_processing(self):
        """The rendered status is free text; the sessions API takes only its lifecycle enum.

        Passing "is thinking..." straight through is rejected with `invalid_arguments`
        ("must be a valid enum value [status]"), so no loading indicator ever appeared.
        """
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a.set_status_text("C123", "is reading docs/api.md…")
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="processing",
        )

    @pytest.mark.asyncio
    async def test_agent_sessions_status_passes_lifecycle_values_through(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        await a._set_thread_status("C123", "T123", "parent_ts", "suspended", "test")
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status="suspended",
        )

    @pytest.mark.asyncio
    async def test_legacy_api_still_gets_the_rendered_status_text(self):
        """Without the sessions API there is no enum to satisfy — free text stays free text."""
        _slack_mod._AGENT_SESSIONS_SUPPORTED = False
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
