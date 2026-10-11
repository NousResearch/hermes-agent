"""Slack semantic thread-title state transitions."""

from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_slack import (
    TestAgentSessionsApiRouting as _AgentSessionsHelpers,
    _pin_legacy_assistant_threads_api,
)


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
