"""Slack semantic thread-title state transitions."""

from unittest.mock import AsyncMock

import pytest

import plugins.platforms.slack.adapter as slack_mod
from tests.gateway.test_slack import _pin_legacy_assistant_threads_api
from tests.gateway.test_slack_agent_sessions_api import (
    TestAgentSessionsApiRouting as _AgentSessionsHelpers,
)


class _SlackApiError(Exception):
    """``slack_sdk.SlackApiError`` shape: the adapter reads ``response["error"]``."""

    def __init__(self, error: str):
        super().__init__(error)
        self.response = {"error": error}


class _MappingResponse:
    """Mapping-like Slack body that is not a ``dict`` (``AsyncSlackResponse``)."""

    def __init__(self, **fields):
        self._fields = fields

    def get(self, key, default=None):
        return self._fields.get(key, default)


class _MappingSlackError(Exception):
    def __init__(self, error: str):
        super().__init__(error)
        self.response = _MappingResponse(error=error)


class TestSemanticThreadTitles:
    _adapter = _AgentSessionsHelpers._adapter

    @pytest.mark.asyncio
    async def test_semantic_title_replaces_raw_once_and_failed_call_retries(self):
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        rename = a._app.client.assistant_threads_setTitle = AsyncMock(
            side_effect=[None, RuntimeError("temporary"), None, None]
        )
        await a._set_assistant_thread_title("D123", "171234.0001", "Raw words", team_id="T1")
        assert a._titled_assistant_threads

        await a._set_assistant_thread_title("D123", "171234.0001", "Semantic title", team_id="T1", semantic=True)
        await a._set_assistant_thread_title("D123", "171234.0001", "Retry title", team_id="T1", semantic=True)
        assert rename.await_count == 3
        assert rename.await_args_list[1].kwargs["title"] == "Semantic title"
        assert rename.await_args_list[2].kwargs["title"] == "Retry title"
        assert len(a._semantically_titled_assistant_threads) == 1
        await a._set_assistant_thread_title("D123", "171234.0001", "Later duplicate", team_id="T1", semantic=True)
        assert rename.await_count == 3

    @pytest.mark.asyncio
    async def test_semantic_title_guard_is_opt_out_and_workspace_scoped(self):
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = False
        a._app.client.assistant_threads_setTitle = AsyncMock()
        await a._set_assistant_thread_title("D123", "171234.0001", "No title", team_id="T1")
        assert a._app.client.assistant_threads_setTitle.await_count == 0
        a.config.extra["assistant_thread_titles"] = True
        rename = a._app.client.assistant_threads_setTitle
        await a._set_assistant_thread_title("D123", "171234.0001", "T1 title", team_id="T1", semantic=True)
        await a._set_assistant_thread_title("D123", "171234.0001", "T2 title", team_id="T2", semantic=True)
        assert [call.kwargs["title"] for call in rename.await_args_list] == ["T1 title", "T2 title"]

    @pytest.mark.asyncio
    async def test_missing_agent_session_is_created_then_keeps_the_generated_title(self):
        slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        rename = a._app.client.agents_sessions_rename = AsyncMock(
            side_effect=[_SlackApiError("session_not_found"), None]
        )
        create = a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setTitle = AsyncMock()

        await a._set_assistant_thread_title(
            "D123", "171234.0001", "Generated title", team_id="T1", semantic=True)

        assert create.await_args.kwargs == {
            "channel_id": "D123",
            "thread_ts": "171234.0001",
            "status": "active",
            "title": "Generated title",
        }
        assert [call.kwargs["title"] for call in rename.await_args_list] == [
            "Generated title",
            "Generated title",
        ]
        await a._set_assistant_thread_title("D123", "171234.0001", "later raw message", team_id="T1")
        assert rename.await_count == 2
        a._app.client.assistant_threads_setTitle.assert_not_called()

    @pytest.mark.asyncio
    async def test_created_title_survives_a_failed_follow_up_rename(self):
        slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        rename = a._app.client.agents_sessions_rename = AsyncMock(
            side_effect=[_SlackApiError("session_not_found"), _SlackApiError("ratelimited")]
        )
        a._app.client.agents_sessions_setStatus = AsyncMock(
            return_value=_MappingResponse(ok=True, title="Generated title"))

        await a._set_assistant_thread_title(
            "D123", "171234.0001", "Generated title", team_id="T1", semantic=True)
        await a._set_assistant_thread_title("D123", "171234.0001", "later raw message", team_id="T1")

        assert rename.await_count == 2
        assert ("T1", "D123", "171234.0001") in a._semantically_titled_assistant_threads

    @pytest.mark.asyncio
    async def test_unconfirmed_create_does_not_block_a_later_title(self):
        slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        rename = a._app.client.agents_sessions_rename = AsyncMock(
            side_effect=[
                _SlackApiError("session_not_found"),
                _SlackApiError("ratelimited"),
                None,
            ])
        a._app.client.agents_sessions_setStatus = AsyncMock(
            return_value=_MappingResponse(ok=True, title="different"))

        await a._set_assistant_thread_title(
            "D123", "171234.0001", "Generated title", team_id="T1", semantic=True)
        await a._set_assistant_thread_title(
            "D123", "171234.0001", "Generated title", team_id="T1", semantic=True)

        assert rename.await_count == 3
        assert ("T1", "D123", "171234.0001") in a._semantically_titled_assistant_threads

    @pytest.mark.asyncio
    async def test_mapping_like_session_not_found_creates_the_session(self):
        slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        a._app.client.agents_sessions_rename = AsyncMock(
            side_effect=[_MappingSlackError("session_not_found"), None])
        create = a._app.client.agents_sessions_setStatus = AsyncMock()

        await a._set_assistant_thread_title(
            "D123", "171234.0001", "Generated title", team_id="T1", semantic=True)

        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_agent_session_typing_uses_the_status_enum(self):
        slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        status = a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()

        await a.send_typing("D123", metadata={"thread_id": "171234.0001"})
        await a.stop_typing("D123", metadata={"thread_id": "171234.0001"})

        assert [call.kwargs["status"] for call in status.await_args_list] == ["processing", "active"]
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_authorized_does_not_create_a_session(self):
        slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        a._app.client.agents_sessions_rename = AsyncMock(
            side_effect=_MappingSlackError("not_authorized"))
        create = a._app.client.agents_sessions_setStatus = AsyncMock()

        await a._set_assistant_thread_title(
            "D123", "171234.0001", "Generated title", team_id="T1", semantic=True)

        assert create.await_count == 0
        assert not a._semantically_titled_assistant_threads

    @pytest.mark.asyncio
    async def test_unrelated_title_error_is_warned_and_does_not_create(self, caplog):
        slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a.config.extra["assistant_thread_titles"] = True
        a._app.client.agents_sessions_rename = AsyncMock(side_effect=_SlackApiError("invalid_auth"))
        create = a._app.client.agents_sessions_setStatus = AsyncMock()

        with caplog.at_level("WARNING", logger="plugins.platforms.slack.adapter"):
            await a._set_assistant_thread_title(
                "D123", "171234.0001", "Nope", team_id="T1", semantic=True)

        assert create.await_count == 0
        assert not a._semantically_titled_assistant_threads
        assert any("invalid_auth" in record.message for record in caplog.records)
