"""Regression for #139017: rotated messaging routes have no Desktop controls."""

import asyncio
import threading
from types import SimpleNamespace


def test_messaging_control_read_returns_empty_snapshot_without_current_route(monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionSource
    import gateway.session_ancillary as ancillary

    session_id = 'wechat-session'
    live = SimpleNamespace(
        event_stream=SimpleNamespace(lock=threading.Lock()),
        route='weixin:fixture',
        source=SessionSource(Platform.WEIXIN, 'fixture-wechat'),
    )

    def metadata_read(_key):
        raise AssertionError('a messaging control read must not inspect persisted local controls')

    authority = SimpleNamespace(db=SimpleNamespace(get_meta=metadata_read), sessions={session_id: live})
    connection = SimpleNamespace(authority=authority)
    ref = SimpleNamespace(session_id=session_id)

    monkeypatch.setattr(ancillary, 'authorize', lambda *_args: None)

    def stale_route(*_args):
        raise AssertionError('a messaging control read must not inspect its reused route')

    monkeypatch.setattr(ancillary, '_require_current_route', stale_route)

    result = asyncio.run(ancillary.read(connection, ref, {'session_id': session_id}, kind='control'))

    assert result['control']['goal'] is None
    assert result['control']['loop'] is None
    assert result['control']['heartbeat'] is None
    assert isinstance(result['control']['revision'], str)
