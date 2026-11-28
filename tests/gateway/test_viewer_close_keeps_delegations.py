"""A TUI/desktop viewer that closes or reconnects does not kill its live session's background work.

The Ink TUI ends its view with ``session.detach {end_reason: tui_close | tui_shutdown}`` both when the
user means it (``/quit``, ``/new``) and when its transport fails and it tears itself down (a stalled
gateway loop, a dropped websocket, a reconnect). The owner cannot tell them apart. Interrupting the
session's async delegations at that instant therefore killed hours of subagent work whenever the
viewer merely went away and came back while the session itself carried on. The interrupt now waits
for a grace period and is cancelled by any viewer attaching to the session again; an exit nobody
takes back still ends the work. ``cli_close`` and ``new_session`` (the classic CLI's explicit
``/exit`` and ``/new``) keep ending it at once (tests/gateway/test_session_end_delegations.py).
"""
import pytest

from tests.gateway.test_session_end_delegations import _call, _no_delegations, _open, _owner, _running
from tools import async_delegation as delegation

TUI_REASONS = ['tui_close', 'tui_shutdown']


def _capture_timers(monkeypatch):
    """The grace timer's (delay, callback), fired by the test instead of after a real wait."""
    import gateway.session_viewer_close as viewer_close
    armed = []

    def call_later(loop, delay, callback):
        armed.append((delay, callback))
        return loop.call_later(3600, callback)
    monkeypatch.setattr(viewer_close, '_call_later', call_later, raising=False)
    return armed


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', TUI_REASONS)
async def test_viewer_that_drops_and_reconnects_keeps_the_sessions_subagents(tmp_path, monkeypatch, reason):
    """The 2026-11-26 sequence: the viewer's socket fails, its TUI sends the close, and the same session
    is attached again (it kept answering). Nothing the session started may have been interrupted."""
    db, authority = await _owner(tmp_path, monkeypatch)
    timers = _capture_timers(monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    stopped = []
    _running(stopped, 'deleg_parent', parent='chat')
    _running(stopped, 'deleg_route', route=authority.sessions['chat'].route)

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason=reason)
    assert stopped == [], 'the close must not interrupt delegations while a viewer may still come back'

    await _call(viewer, 'session.resume', session_id='chat')  # the reconnect
    for _delay, callback in timers:  # the grace elapses with a viewer attached
        callback()

    assert stopped == []


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', TUI_REASONS)
async def test_exit_nobody_takes_back_still_ends_the_sessions_subagents(tmp_path, monkeypatch, reason):
    """Quitting for good still ends the work, once the grace passes with no viewer; others' work stays."""
    db, authority = await _owner(tmp_path, monkeypatch)
    timers = _capture_timers(monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    await _open(db, authority, tmp_path, 'other')
    stopped = []
    _running(stopped, 'deleg_parent', parent='chat')
    _running(stopped, 'deleg_other', parent='other', route=authority.sessions['other'].route)

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason=reason)
    assert stopped == []
    assert [delay for delay, _ in timers] == [300.0]
    for _delay, callback in timers:
        callback()

    assert stopped == ['deleg_parent']


@pytest.mark.asyncio
async def test_new_work_is_not_refused_while_the_close_is_still_undecided(tmp_path, monkeypatch):
    """A turn still running when the viewer vanished may start another subagent during the grace."""
    db, authority = await _owner(tmp_path, monkeypatch)
    _capture_timers(monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    route = authority.sessions['chat'].route

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='tui_close')

    assert delegation._refused_by.get(('session_key', route)) is None
    assert delegation._refused_by.get(('parent_session_id', 'chat')) is None


@pytest.mark.asyncio
async def test_grace_zero_restores_the_immediate_end(tmp_path, monkeypatch):
    from gateway.session_persistence import authority_persistence
    db, authority = await _owner(tmp_path, monkeypatch)
    (authority_persistence(authority).scope.profile_dir / 'config.yaml').write_text('sessions:\n  viewer_close_delegation_grace_s: 0\n')
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    stopped = []
    _running(stopped, 'deleg_parent', parent='chat')

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='tui_close')

    assert stopped == ['deleg_parent']
