"""Slack thread-status endpoint and payload selection.

The Agent Sessions API (``agents.sessions.setStatus``, slack-sdk >= 3.44.0) is
preferred with lifecycle values: a working status maps to ``processing`` and an
empty clear to ``active`` (it is the only API that works in Slack session
channels). When the sessions API is unavailable or rejects the call,
``_set_thread_status`` retries once on the legacy ``assistant.threads.setStatus``
with the ORIGINAL free text / empty-string clear, so older SDKs and session-less
channels keep working.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter, _slack_mod


@pytest.fixture(autouse=True)
def _pin_agent_sessions_supported():
    """Force the Agent Sessions path (the failing configuration) and restore
    the cached flag afterwards, so tests don't leak state into each other."""
    prev = _slack_mod._AGENT_SESSIONS_SUPPORTED
    _slack_mod._AGENT_SESSIONS_SUPPORTED = True
    yield
    _slack_mod._AGENT_SESSIONS_SUPPORTED = prev


def _adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    return a


FREE_TEXT = "still working (2m03s)"


class TestSetThreadStatusFallback:
    """_set_thread_status retries the legacy assistant.threads.setStatus when
    the Agent Sessions API rejects the call (e.g. invalid free-text status)."""

    @pytest.mark.asyncio
    async def test_new_api_rejects_free_text_falls_back_to_legacy(self):
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=Exception("invalid_arguments"))
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a._set_thread_status("C123", "", "171234.0001", FREE_TEXT, "failed")
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="171234.0001", status=FREE_TEXT)

    @pytest.mark.asyncio
    async def test_new_api_clear_maps_to_active(self):
        """Sessions-capable client: an empty clear maps to the 'active' lifecycle
        value, which clears the loading state on the sessions API."""
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a._set_thread_status("C123", "", "171234.0001", "", "failed")
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="171234.0001", status="active")
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_api_working_label_maps_to_processing(self):
        """Sessions-capable client: ANY non-empty working label — even one that
        happens to spell a lifecycle enum — is ongoing work from send_typing and
        maps to 'processing'."""
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a._set_thread_status("C123", "", "171234.0001", "active", "failed")
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="171234.0001", status="processing")
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_path_also_falls_back(self):
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=Exception("invalid_arguments"))
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a._set_thread_status("C123", "", "171234.0001", "", "clear failed")
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="171234.0001", status="")

    @pytest.mark.asyncio
    async def test_both_apis_failing_is_swallowed_and_debug_logged(self, caplog):
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=Exception("invalid_arguments"))
        a._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("missing_scope"))
        with caplog.at_level("DEBUG"):
            await a._set_thread_status("C123", "", "171234.0001", FREE_TEXT, "failed")
        a._app.client.assistant_threads_setStatus.assert_called_once()
        assert any("missing_scope" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_legacy_endpoint_failure_is_not_retried(self):
        """Legacy endpoint selected (older SDK): a failure is logged, not retried."""
        _slack_mod._AGENT_SESSIONS_SUPPORTED = False
        a = _adapter()
        a._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=Exception("missing_scope"))
        await a._set_thread_status("C123", "", "171234.0001", FREE_TEXT, "failed")
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="171234.0001", status=FREE_TEXT)

    @pytest.mark.asyncio
    async def test_send_typing_free_text_recovers_via_fallback(self):
        """End-to-end: the real send_typing free-text status succeeds on legacy."""
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=Exception("invalid_arguments"))
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        expected_status = a._default_status_text(None)
        # The sessions API saw the lifecycle enum for the free-text phrase...
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="parent_ts", status="processing")
        # ...and the legacy retry carried the original free text.
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123",
            thread_ts="parent_ts",
            status=expected_status,
        )
