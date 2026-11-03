"""``/debug local`` on a local session never uploads (T7).

The messaging ``/debug`` handler ignored its argument and always uploaded the summary to a public
paste. A local session's ``/debug local`` asks for the report on this machine: the gateway returns
the redacted report for the client to show and makes no network call. From a messaging chat
``/debug local`` is refused and uploads nothing; bare ``/debug`` keeps the upload.
"""
import socket
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}

    async def _inline(fn, *args):
        return fn(*args)

    runner._run_in_executor_with_context = _inline
    return runner


def _event(platform, text):
    source = SessionSource(platform=platform, chat_id='chat-1', user_id='u1', chat_type='dm')
    return MessageEvent(text=text, source=source)


class _NoNetwork:
    """Any upload helper or socket connect fails the test."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError('network/upload call during /debug local')


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['/debug local', '/debug LOCAL', '/debug nous local'])
async def test_local_session_debug_local_returns_the_report_without_uploading(text):
    guard = _NoNetwork()
    bundle = {'report': 'REDACTED-REPORT-BODY', 'agent.log': 'FULL-LOG'}
    with patch('hermes_cli.debug.upload_to_pastebin', guard), \
            patch('hermes_cli.diagnostics_upload.share_to_nous', guard), \
            patch('hermes_cli.debug._schedule_auto_delete', guard), \
            patch('hermes_cli.debug.collect_share_bundle', return_value=bundle), \
            patch.object(socket.socket, 'connect', guard):
        out = await _runner()._handle_debug_command(_event(Platform.LOCAL, text))
    assert guard.calls == []
    assert 'REDACTED-REPORT-BODY' in out
    assert 'http' not in out.lower()


@pytest.mark.asyncio
async def test_messaging_debug_still_uploads_the_summary():
    uploaded = []
    with patch('hermes_cli.debug.upload_to_pastebin', lambda report: uploaded.append(report) or 'https://paste.rs/x'), \
            patch('hermes_cli.debug._schedule_auto_delete', lambda urls: None), \
            patch('hermes_cli.debug._best_effort_sweep_expired_pastes', lambda: None), \
            patch('hermes_cli.debug._capture_dump', lambda: 'dump'), \
            patch('hermes_cli.debug.collect_debug_report', lambda **kw: 'SUMMARY'):
        out = await _runner()._handle_debug_command(_event(Platform.TELEGRAM, '/debug'))
    assert uploaded == ['SUMMARY']
    assert 'https://paste.rs/x' in out


@pytest.mark.asyncio
async def test_messaging_debug_local_is_refused_without_uploading():
    """``/debug local`` from a chat is a named refusal pointing at bare ``/debug``; nothing is uploaded."""
    guard = _NoNetwork()
    with patch('hermes_cli.debug.upload_to_pastebin', guard), \
            patch('hermes_cli.diagnostics_upload.share_to_nous', guard), \
            patch('hermes_cli.debug._schedule_auto_delete', guard), \
            patch('hermes_cli.debug._best_effort_sweep_expired_pastes', guard), \
            patch('hermes_cli.debug.collect_debug_report', guard), \
            patch.object(socket.socket, 'connect', guard):
        out = await _runner()._handle_debug_command(_event(Platform.TELEGRAM, '/debug local'))
    assert guard.calls == []
    assert '/debug' in out and 'http' not in out.lower()


@pytest.mark.asyncio
async def test_local_debug_local_real_collector_opens_no_socket(tmp_path, monkeypatch):
    """The real collector, unpatched: no socket connect and no upload helper is reached."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    guard = _NoNetwork()
    with patch('hermes_cli.debug.upload_to_pastebin', guard), \
            patch('hermes_cli.diagnostics_upload.share_to_nous', guard), \
            patch.object(socket.socket, 'connect', guard), \
            patch.object(socket.socket, 'connect_ex', guard), \
            patch('socket.create_connection', guard):
        out = await _runner()._handle_debug_command(_event(Platform.LOCAL, '/debug local'))
    assert guard.calls == []
    assert 'hermes dump' in out
