"""Slack adapter routing onto the slack-sdk Agent Sessions API (agents.sessions.*)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base_status_anchor import STATUS_ANCHOR_MESSAGE_ID, STATUS_OWNER_KEY
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
    async def test_typing_without_thread_context_is_a_noop_by_default(self):
        """reply_in_thread defaults to true: a message ts alone is not a status target."""
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()
        await a.send_typing("C123", metadata={STATUS_ANCHOR_MESSAGE_ID: "171.1"})
        a._app.client.agents_sessions_setStatus.assert_not_called()
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


class TestFlatModeStatus:
    """reply_in_thread: false — the status rides on the inbound message's own ts through
    agents.sessions.setStatus (no thread is created). assistant.threads.setStatus on a top-level
    ts opens an assistant thread, so a flat target never reaches it."""

    MSG_TS = "1761300000.000100"

    def _adapter(self, **extra):
        config = PlatformConfig(enabled=True, token="xoxb-fake-token", extra={"reply_in_thread": False, **extra})
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = AsyncMock()
        a._app.client.agents_sessions_setStatus = AsyncMock()
        a._app.client.assistant_threads_setStatus = AsyncMock()
        return a

    def _flat_md(self, ts=None):
        return {"slack_team_id": "T1", STATUS_ANCHOR_MESSAGE_ID: ts or self.MSG_TS}

    @pytest.mark.asyncio
    async def test_flat_anchor_sets_processing_then_active_and_untracks(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        client = a._app.client
        await a.send_typing("C123", metadata=self._flat_md())
        client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts=self.MSG_TS, status="processing")
        assert a._active_status_threads[("T1", "C123", self.MSG_TS)]["flat"] is True
        client.agents_sessions_setStatus.reset_mock()
        await a.stop_typing("C123", metadata=self._flat_md())
        client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts=self.MSG_TS, status="active")
        assert a._active_status_threads == {}
        client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthetic_session_thread_is_the_flat_anchor(self):
        """DM sessions keyed per top-level message carry thread_id == message_id: still flat."""
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus.side_effect = Exception("invalid_arguments")
        await a.send_typing("D123", metadata={"thread_id": self.MSG_TS, "message_id": self.MSG_TS})
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="D123", thread_ts=self.MSG_TS, status="processing")
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_flat_without_sessions_api_never_uses_legacy(self, caplog):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = False
        a = self._adapter()
        with caplog.at_level("DEBUG"):
            await a.send_typing("C123", metadata=self._flat_md())
            await a.stop_typing("C123", metadata=self._flat_md())
        a._app.client.assistant_threads_setStatus.assert_not_called()
        assert a._active_status_threads == {}
        assert any("flat message" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_flat_sessions_failure_is_swallowed_without_legacy_retry(self, caplog):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus.side_effect = Exception("not_allowed")
        with caplog.at_level("DEBUG"):
            await a.send_typing("C123", metadata=self._flat_md())
            await a.stop_typing("C123", metadata=self._flat_md())
        assert a._app.client.agents_sessions_setStatus.call_count == 2
        a._app.client.assistant_threads_setStatus.assert_not_called()
        assert any("not_allowed" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_real_thread_keeps_thread_ts_and_legacy_fallback(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._app.client.agents_sessions_setStatus.side_effect = Exception("invalid_arguments")
        md = {"thread_id": "1761299999.000001", STATUS_ANCHOR_MESSAGE_ID: self.MSG_TS, "slack_team_id": "T1"}
        await a.send_typing("C123", metadata=md)
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="1761299999.000001", status="processing")
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="1761299999.000001", status=a._default_status_text(None))
        assert a._active_status_threads[("T1", "C123", "1761299999.000001")]["flat"] is False

    @pytest.mark.asyncio
    async def test_reply_in_thread_true_is_unchanged(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter(reply_in_thread=True)
        a._app.client.agents_sessions_setStatus.side_effect = Exception("invalid_arguments")
        await a.send_typing("C123", metadata={"thread_id": self.MSG_TS, STATUS_ANCHOR_MESSAGE_ID: self.MSG_TS})
        a._app.client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts=self.MSG_TS, status=a._default_status_text(None))

    @pytest.mark.asyncio
    async def test_teamless_clear_of_a_unique_flat_entry_stays_on_sessions(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        await a.send_typing("C123", metadata=self._flat_md())
        a._app.client.agents_sessions_setStatus.reset_mock()
        a._app.client.agents_sessions_setStatus.side_effect = Exception("invalid_arguments")
        await a.stop_typing("C123")
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts=self.MSG_TS, status="active")
        a._app.client.assistant_threads_setStatus.assert_not_called()
        assert a._active_status_threads == {}

    @pytest.mark.asyncio
    async def test_flat_entries_share_the_tracked_status_bounds(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._ACTIVE_STATUS_THREADS_MAX = 4
        anchors = [f"17613000{i:02d}.000100" for i in range(10)]
        for ts in anchors:
            await a.send_typing("C123", metadata=self._flat_md(ts))
        assert len(a._active_status_threads) <= 4
        assert ("T1", "C123", anchors[-1]) in a._active_status_threads
        for ts in anchors:
            await a.stop_typing("C123", metadata=self._flat_md(ts))
        assert a._active_status_threads == {}
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_owned_flat_entries_stay_bounded_and_clear_by_owner(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        a._ACTIVE_STATUS_THREADS_MAX = 4
        anchors = [f"17613000{i:02d}.000100" for i in range(10)]
        for ts in anchors:
            await a.send_typing("C123", metadata={**self._flat_md(ts), STATUS_OWNER_KEY: "session-a"})
        other = "1761300099.000100"
        await a.send_typing("C123", metadata={**self._flat_md(other), STATUS_OWNER_KEY: "session-b"})
        assert len(a._active_status_threads) <= 4
        await a.stop_typing("C123", metadata={**self._flat_md(anchors[0]), STATUS_OWNER_KEY: "session-a"})
        assert list(a._active_status_threads) == [("T1", "C123", other)]

    @pytest.mark.asyncio
    async def test_ownerless_flat_entries_never_clear_each_other(self):
        """An entry with no owner (pre-owner tracking, metadata without the key) clears only its own
        anchor: an empty owner is not a wildcard matching another ownerless entry."""
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = self._adapter()
        first, second = "1761300000.000100", "1761300050.000300"
        for ts in (first, second):
            await a.send_typing("C123", metadata=self._flat_md(ts))
        assert all(entry["owner"] == "" for entry in a._active_status_threads.values())
        await a.stop_typing("C123", metadata=self._flat_md(first))
        assert list(a._active_status_threads) == [("T1", "C123", second)]
        cleared = [c.kwargs["thread_ts"] for c in a._app.client.agents_sessions_setStatus.call_args_list
                   if c.kwargs["status"] == "active"]
        assert cleared == [first]


class _LegacyOnlyClient:
    """A real (non-mock) client WITHOUT ``agents_sessions_setStatus``: every attribute access
    yields a fresh bound method, as on slack-sdk's ``AsyncWebClient``."""

    def __init__(self):
        self.calls = []

    async def assistant_threads_setStatus(self, **kwargs):
        self.calls.append(kwargs)


class TestLegacyOnlyClient:
    """The SDK advertises the sessions API but this client lacks it: the legacy endpoint
    gets the original free text and the ``""`` clear, never lifecycle values."""

    @pytest.mark.asyncio
    async def test_thread_status_sends_free_text_then_empty_clear(self):
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake-token"))
        a._app = MagicMock()
        a._app.client = _LegacyOnlyClient()
        a.config.typing_status_text = "is digging in..."
        md = {"thread_id": "1761299999.000001", "slack_team_id": "T1"}
        await a.send_typing("C123", metadata=md)
        await a.stop_typing("C123", metadata=md)
        assert a._app.client.calls == [
            {"channel_id": "C123", "thread_ts": "1761299999.000001", "status": "is digging in..."},
            {"channel_id": "C123", "thread_ts": "1761299999.000001", "status": ""},
        ]

    @pytest.mark.asyncio
    async def test_flat_target_makes_no_call(self):
        """The legacy endpoint on a top-level ts would open an assistant thread."""
        _slack_mod._AGENT_SESSIONS_SUPPORTED = True
        a = SlackAdapter(PlatformConfig(
            enabled=True, token="xoxb-fake-token", extra={"reply_in_thread": False}))
        a._app = MagicMock()
        a._app.client = _LegacyOnlyClient()
        md = {"slack_team_id": "T1", STATUS_ANCHOR_MESSAGE_ID: "1761300000.000100"}
        await a.send_typing("C123", metadata=md)
        await a.stop_typing("C123", metadata=md)
        assert a._app.client.calls == []
