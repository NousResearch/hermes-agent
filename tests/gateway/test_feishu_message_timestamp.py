import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.feishu.adapter import FeishuAdapter


def test_event_send_time_millisecond_int():
    msg = SimpleNamespace(create_time=1729332000000)
    dt = FeishuAdapter._event_send_time(msg)
    assert abs(dt.timestamp() - 1729332000.0) < 1.0


def test_event_send_time_millisecond_str():
    msg = SimpleNamespace(create_time="1729332000000")
    dt = FeishuAdapter._event_send_time(msg)
    assert abs(dt.timestamp() - 1729332000.0) < 1.0


def test_event_send_time_seconds_int():
    msg = SimpleNamespace(create_time=1729332000)
    dt = FeishuAdapter._event_send_time(msg)
    assert abs(dt.timestamp() - 1729332000.0) < 1.0


def test_event_send_time_missing_attribute_fallback_now():
    msg = SimpleNamespace()
    now = datetime.now()
    dt = FeishuAdapter._event_send_time(msg)
    assert abs((dt - now).total_seconds()) < 5.0


def test_event_send_time_missing_value_fallback_now():
    msg = SimpleNamespace(create_time=None)
    now = datetime.now()
    dt = FeishuAdapter._event_send_time(msg)
    assert abs((dt - now).total_seconds()) < 5.0


def test_event_send_time_invalid_fallback_now():
    msg = SimpleNamespace(create_time="not-a-timestamp")
    now = datetime.now()
    dt = FeishuAdapter._event_send_time(msg)
    assert abs((dt - now).total_seconds()) < 5.0


@pytest.mark.parametrize("raw", [0, "0", ""])
def test_event_send_time_empty_value_is_not_1970(raw):
    """A zero/empty create_time means "no value", not a 1970 epoch — it must fall back to now."""
    msg = SimpleNamespace(create_time=raw)
    now = datetime.now()
    dt = FeishuAdapter._event_send_time(msg)
    assert abs((dt - now).total_seconds()) < 5.0


def test_backfilled_message_keeps_its_platform_send_time(tmp_path, monkeypatch):
    """A message queued during an outage must not be stamped with the reconnect time.

    This is the regression the PR exists for: Feishu redelivers messages queued
    during a WebSocket outage right after reconnect. If the event timestamp comes
    from ``datetime.now()`` instead of ``create_time``, the backfilled message is
    indistinguishable from a fresh one, and ``_requeue_backoff_delay`` never sees
    the redelivery as the same event.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    adapter = FeishuAdapter(PlatformConfig())
    adapter._dispatch_inbound_event = AsyncMock()
    adapter._extract_message_content = AsyncMock(
        return_value=("sent during the outage", MessageType.TEXT, [], [], False, [])
    )
    adapter.get_chat_info = AsyncMock(
        return_value={"chat_id": "oc_chat", "name": "Feishu DM", "type": "dm"}
    )
    adapter._resolve_sender_profile = AsyncMock(
        return_value={"user_id": "ou_user", "user_name": "Zhang", "user_id_alt": None}
    )

    now = datetime.now()
    sent_ms = int((now - timedelta(hours=2)).timestamp() * 1000)
    message = SimpleNamespace(
        chat_id="oc_chat",
        thread_id=None,
        parent_id=None,
        upper_message_id=None,
        root_id=None,
        message_type="text",
        content='{"text":"sent during the outage"}',
        message_id="om_backfilled",
        create_time=str(sent_ms),
    )

    asyncio.run(
        adapter._process_inbound_message(
            data=SimpleNamespace(event=SimpleNamespace(message=message)),
            message=message,
            sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
            is_bot=False,
            chat_type="p2p",
            message_id="om_backfilled",
        )
    )

    event = adapter._dispatch_inbound_event.await_args.args[0]
    assert abs((now - event.timestamp).total_seconds() - 7200) < 5
