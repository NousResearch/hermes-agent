"""A cold resume of a messaging chat reports the effort its next turn resolves (#142336).

Desktop marks a resumed slice's reasoning effort pending and expects the level back — from the
resume reply for a LOCAL session (``snapshot_info``), or from a later ``session.info`` event the
policy refresh publishes on attach. A messaging chat has no frozen local policy, so neither
happened: the composer's reasoning-effort pill spun forever until the viewer picked a level once
(the #141152 half). A re-open after a desktop restart spun again, because the pick path is not
re-run on resume.

The fix reports what the chat's next turn resolves (``messaging_route_info``) in the resume reply's
``info``, the same fields ``session.info`` already answers for the same chat. Real ``GatewayRunner``,
``SessionAuthority`` and SQLite store.
"""
import pytest

from gateway.run import GatewayRunner


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))


async def _telegram_chat():
    """A registered Telegram DM with history and a viewer that can control it, as Desktop attaches."""
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    runner = GatewayRunner(GatewayConfig())
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='4242', user_id='u-4242', chat_type='dm')
    root = authority.register(source).session_id
    authority.db.append_message(root, 'user', 'hello from telegram')
    viewer = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'viewer', 'profile_id': 'default', 'instance_id': 'owner',
        'capabilities': ['session:read', 'session:control']})
    return runner, source, viewer, root


@pytest.mark.asyncio
async def test_resuming_a_messaging_chat_reports_its_next_turn_reasoning_level():
    runner, source, viewer, root = await _telegram_chat()
    try:
        runner._apply_reasoning_selection(runner.session_store._generate_session_key(source), 'telegram', 'high')

        resumed = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': root}})

        info = resumed['result']['info']
        assert info.get('reasoning_effort') == 'high', info
        assert info.get('reasoning_effort_wire') == 'high', info
        assert 'model' in info and info.get('desktop_protocol'), info
        for field in ('service_tier', 'fast'):
            assert field in info, (field, info)
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_the_resume_reply_and_session_info_agree_for_a_messaging_chat():
    """One source of truth: the reply must answer what ``session.info`` answers for the same chat."""
    runner, source, viewer, root = await _telegram_chat()
    try:
        runner._apply_reasoning_selection(runner.session_store._generate_session_key(source), 'telegram', 'low')

        resumed = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': root}})
        polled = await viewer.dispatch({'id': 2, 'method': 'session.info', 'params': {'session_id': root}})

        for field in ('model', 'reasoning_effort', 'reasoning_effort_wire', 'service_tier', 'fast'):
            assert resumed['result']['info'].get(field) == polled['result'].get(field), (field, resumed, polled)
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
