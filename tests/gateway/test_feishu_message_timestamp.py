from datetime import datetime
from types import SimpleNamespace

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
