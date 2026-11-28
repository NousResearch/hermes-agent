"""Tests: chat.startStream recipient pair for Slack Connect authors (#159617).

Behaviour contract:
  * The recipient user must belong to the recipient team. Session scope is
    the installing workspace (#154560); a Connect author may belong to a
    partner org, so the pair is built from the author's HOME team (resolved
    via the cached users.info lookup), not from session scope.
  * Both stream lanes (native task card + draft stream) use the same pair.
  * Unknown home team: no doomed startStream call; a classified permanent
    refusal the gateway recognizes (quiet text fallback, no traceback spam).
  * Slack user_not_found on startStream: permanent refusal, not retryable.
  * Internal authors (home team == session team): pair unchanged.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.run_turn_runner import _CARD_DESTINATION_REFUSALS
from plugins.platforms.slack.adapter import SlackAdapter


class _SlackApiErrorShape(Exception):
    """slack_sdk.SlackApiError's shape (``exc.response["error"]``) without importing
    the SDK, which CI stubs as a bare module (same pattern as test_slack_native_streaming)."""

    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


def _make_adapter(home_team="T_B", users_info_error=None, start_error=None):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra={})
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    client.api_call = AsyncMock(
        return_value={"ts": "555.666"},
        side_effect=start_error,
    )
    client.chat_startStream = AsyncMock(return_value={"ok": True, "ts": "123.456"})
    if users_info_error is not None:
        client.users_info = AsyncMock(side_effect=users_info_error)
    else:
        client.users_info = AsyncMock(return_value={"user": {"team_id": home_team}})
    a._get_client = MagicMock(return_value=client)
    return a, client


CARD_META = {
    "team_id": "T_A",
    "recipient_team_id": "T_A",
    "recipient_user_id": "U_EXT",
    "thread_id": "111.000",
}
TASKS = [{"text": "do the thing", "status": "in_progress"}]


def _start_stream_payload(client):
    for call in client.api_call.call_args_list:
        if call.args and call.args[0] == "chat.startStream":
            return call.kwargs["json"]
    return None


class TestTaskCardRecipientPair:
    @pytest.mark.asyncio
    async def test_connect_author_gets_home_team_in_pair(self):
        a, client = _make_adapter(home_team="T_B")
        result = await a.send_native_task_card_progress(
            chat_id="C1", tasks=TASKS, reply_to="111.000", metadata=dict(CARD_META))
        assert result.success, result.error
        payload = _start_stream_payload(client)
        assert payload is not None, "chat.startStream was never called"
        assert payload["recipient_user_id"] == "U_EXT"
        assert payload["recipient_team_id"] == "T_B", (
            "recipient team must be the author's home team, not session scope")

    @pytest.mark.asyncio
    async def test_internal_author_pair_unchanged(self):
        a, client = _make_adapter(home_team="T_A")
        meta = dict(CARD_META, recipient_user_id="U_INTERNAL")
        result = await a.send_native_task_card_progress(
            chat_id="C1", tasks=TASKS, reply_to="111.000", metadata=meta)
        assert result.success, result.error
        payload = _start_stream_payload(client)
        assert payload is not None
        assert payload["recipient_team_id"] == "T_A"
        assert payload["recipient_user_id"] == "U_INTERNAL"

    @pytest.mark.asyncio
    async def test_unknown_home_team_fails_clean_without_api_call(self):
        a, client = _make_adapter(users_info_error=Exception("users.info network down"))
        result = await a.send_native_task_card_progress(
            chat_id="C1", tasks=TASKS, reply_to="111.000", metadata=dict(CARD_META))
        assert not result.success
        assert result.retryable is False, "unknown team must not be retried as transient"
        assert result.error in _CARD_DESTINATION_REFUSALS, (
            "gateway must recognize the refusal for a quiet fallback")
        assert _start_stream_payload(client) is None, (
            "no doomed chat.startStream call when the pair cannot be built")

    @pytest.mark.asyncio
    async def test_user_not_found_is_permanent_refusal(self):
        a, _client = _make_adapter(
            home_team="T_B",
            start_error=_SlackApiErrorShape(
                "The request to the Slack API failed.", {"error": "user_not_found"}))
        result = await a.send_native_task_card_progress(
            chat_id="C1", tasks=TASKS, reply_to="111.000", metadata=dict(CARD_META))
        assert not result.success
        assert result.retryable is False, "user_not_found is permanent, not transient"
        assert result.error in _CARD_DESTINATION_REFUSALS


class TestDraftStreamRecipientPair:
    @pytest.mark.asyncio
    async def test_connect_author_gets_home_team_in_pair(self):
        a, client = _make_adapter(home_team="T_B")
        result = await a._start_stream(
            client, ("T_A", "C1", "111.000"), 1, "hello", {"user_id": "U_EXT"})
        assert result.success, result.error
        kwargs = client.chat_startStream.call_args.kwargs
        assert kwargs["recipient_user_id"] == "U_EXT"
        assert kwargs["recipient_team_id"] == "T_B", (
            "draft lane must pair the author with their home team too")
