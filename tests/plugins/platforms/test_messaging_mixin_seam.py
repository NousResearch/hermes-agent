"""Seam tests for the R2-S1 extraction: messaging family -> SlackMessagingMixin.

Verifies the mixin seam (identity via MRO, isinstance probe, dispatch through
the adapter class) plus aggressive behavioral cases for send / edit / delete
flows that previously lived inline in SlackAdapter (adapter.py god-file slice
R2-S1, epic #78647 / target #78638).
"""

import asyncio
from unittest import mock

import pytest

from plugins.platforms.slack.messaging_mixin import SlackMessagingMixin
from plugins.platforms.slack.adapter import SlackAdapter

MESSAGING_NAMES = (
    "send",
    "send_private_notice",
    "send_or_update_status",
    "edit_message",
    "delete_message",
    "send_typing",
    "stop_typing",
)


class _FakeResponse:
    def __init__(self, ok=True, ts="1234.5678", **extra):
        self.ok = ok
        self.ts = ts
        self.extra = extra

    def get(self, key, default=None):
        if key == "ok":
            return self.ok
        if key == "ts":
            return self.ts
        return self.extra.get(key, default)

    def __getitem__(self, key):
        if key == "ok":
            return self.ok
        if key == "ts":
            return self.ts
        return self.extra[key]


def _make_adapter(**attrs):
    """Build a SlackAdapter without running __init__; callers stub helpers."""
    adapter = object.__new__(SlackAdapter)
    adapter._app = mock.Mock()
    adapter._client = mock.AsyncMock()
    adapter.config = mock.Mock(extra={})
    adapter.MAX_MESSAGE_LENGTH = 39000
    adapter._bot_message_ts = set()
    adapter._status_message_ids = {}
    adapter._active_status_threads = {}
    adapter._channel_team = {}
    adapter._status_text = {}
    adapter._ACTIVE_STATUS_THREADS_MAX = 1000
    adapter._STATUS_MESSAGE_IDS_MAX = 2000
    for k, v in attrs.items():
        setattr(adapter, k, v)
    return adapter


# --- seam identity -----------------------------------------------------------


@pytest.mark.parametrize("name", MESSAGING_NAMES)
def test_seam_identity_mixin_owns_all_seven(name):
    assert getattr(SlackAdapter, name) is getattr(SlackMessagingMixin, name)


def test_seam_isinstance_probe():
    adapter = _make_adapter()
    assert isinstance(adapter, SlackMessagingMixin)
    assert isinstance(adapter, SlackAdapter)


def test_seam_mro_order_mixin_before_base():
    # Mixin-first is mandatory: the 6 base-hook overrides must beat
    # BasePlatformAdapter's stubs in the MRO.
    from gateway.platforms.base import BasePlatformAdapter

    mro = SlackAdapter.__mro__
    assert mro.index(SlackMessagingMixin) < mro.index(BasePlatformAdapter)
    assert mro[1] is SlackMessagingMixin


# --- behavioral cases (aggressive; adapter helpers stubbed) ------------------


def _stub_send_helpers(adapter, client):
    adapter._is_ignored_channel = mock.Mock(return_value=False)
    adapter._ensure_dm_conversation = mock.AsyncMock(side_effect=lambda cid, team_id=None: cid)
    adapter._metadata_team_id = mock.Mock(return_value="T1")
    adapter._pop_slash_context = mock.Mock(return_value=None)
    adapter.format_message = mock.Mock(side_effect=lambda s: s)
    adapter.truncate_message = mock.Mock(side_effect=lambda s, lim: [s])
    adapter._resolve_thread_ts = mock.Mock(return_value=None)
    adapter._maybe_blocks = mock.Mock(return_value=None)
    adapter._get_client = mock.Mock(return_value=client)
    adapter._clear_thread_status_quietly = mock.AsyncMock()
    adapter._workspace_message_marker = mock.Mock(side_effect=lambda team, ts: f"{team}:{ts}")
    adapter._trim_bot_message_timestamps = mock.Mock()
    adapter._is_block_payload_rejection = mock.Mock(return_value=False)
    adapter._is_retryable_upload_error = mock.Mock(return_value=False)
    return adapter


def test_seam_send_flow_posts_via_client():
    client = _FakeClient()
    adapter = _stub_send_helpers(_make_adapter(), client)

    result = asyncio.run(adapter.send(chat_id="C123", content="hello from the mixin", metadata=None))

    assert result.success is True
    assert result.message_id == "1234.5678"
    client.chat_postMessage.assert_awaited_once()
    kwargs = client.chat_postMessage.await_args.kwargs
    assert kwargs["channel"] == "C123"
    assert kwargs["text"] == "hello from the mixin"
    assert kwargs["mrkdwn"] is True


def test_seam_send_ignored_channel_short_circuits():
    client = _FakeClient()
    adapter = _stub_send_helpers(_make_adapter(), client)
    adapter._is_ignored_channel = mock.Mock(return_value=True)

    result = asyncio.run(adapter.send(chat_id="C999", content="ignored", metadata=None))

    assert result.success is False
    assert result.error == "ignored_channel"
    client.chat_postMessage.assert_not_awaited()


def test_seam_send_not_connected_short_circuits():
    client = _FakeClient()
    adapter = _stub_send_helpers(_make_adapter(), client)
    adapter._app = None

    result = asyncio.run(adapter.send(chat_id="C123", content="nope", metadata=None))

    assert result.success is False
    assert result.error == "Not connected"
    client.chat_postMessage.assert_not_awaited()


def test_seam_edit_message_flow():
    client = _FakeClient()
    adapter = _make_adapter()
    adapter._is_ignored_channel = mock.Mock(return_value=False)
    adapter._get_client = mock.Mock(return_value=client)

    result = asyncio.run(
        adapter.edit_message(
            chat_id="C123", message_id="1111.2222", content="edited", metadata=None
        )
    )

    assert result.success is True
    client.chat_update.assert_awaited_once()
    kwargs = client.chat_update.await_args.kwargs
    assert kwargs["channel"] == "C123"
    assert kwargs["ts"] == "1111.2222"
    assert kwargs["text"] == "edited"


def test_seam_delete_message_flow():
    client = _FakeClient()
    adapter = _make_adapter()
    adapter._get_client = mock.Mock(return_value=client)

    ok = asyncio.run(adapter.delete_message(chat_id="C123", message_id="1111.2222"))

    assert ok is True
    client.chat_delete.assert_awaited_once()
    kwargs = client.chat_delete.await_args.kwargs
    assert kwargs["channel"] == "C123"
    assert kwargs["ts"] == "1111.2222"


def test_seam_send_typing_and_stop_typing_no_raise():
    client = _FakeClient()
    adapter = _make_adapter()
    adapter._is_ignored_channel = mock.Mock(return_value=False)
    adapter._get_client = mock.Mock(return_value=client)
    adapter._resolve_thread_ts = mock.Mock(return_value="1234.5678")
    adapter._metadata_team_id = mock.Mock(return_value="T1")
    adapter._channel_team = {}
    adapter._workspace_thread_key = mock.Mock(return_value=("T1", "C123", "1234.5678"))
    adapter._slack_timestamp_sort_key = mock.Mock(side_effect=lambda k: k)

    asyncio.run(adapter.send_typing(chat_id="C123", metadata={"message_id": "abc"}))
    asyncio.run(adapter.stop_typing(chat_id="C123", metadata={"message_id": "abc"}))

    # send_typing must have recorded the status entry; stop_typing must clear it
    assert adapter._active_status_threads == {}


class _FakeClient:
    """AsyncMock-based client with the four chat_* endpoints the mixin drives."""

    def __init__(self):
        self.chat_postMessage = mock.AsyncMock(return_value=_FakeResponse())
        self.chat_update = mock.AsyncMock(return_value=_FakeResponse())
        self.chat_delete = mock.AsyncMock(return_value=_FakeResponse(ok=True))
        self.chat_postEphemeral = mock.AsyncMock(return_value=_FakeResponse())
