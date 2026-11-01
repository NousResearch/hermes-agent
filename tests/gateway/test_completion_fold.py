"""Process completions that pile up behind a busy turn run as ONE queued turn, not one turn each."""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import list_session_admissions


async def _discord_session(tmp_path, monkeypatch):
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_state import SessionDB
    from plugins.platforms.discord.adapter import DiscordAdapter
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # nothing claims: all stay queued
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fold', db=db)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')

    async def human(text):
        event = MessageEvent(text=text, source=source, message_id=text)
        with native_callback(runner, event, tmp_path):
            await authority.admit_native(event)
    await human('running turn')
    route = runner.session_store._generate_session_key(source)
    sid = runner.session_store.peek_session_id(route)
    owner = authority.logical_owner(sid)

    async def complete(name):
        evt = {'type': 'completion', 'session_id': name, 'started_at': 1.0, 'session_key': route, 'parent_session_id': sid,
               'platform': 'discord', 'chat_type': 'dm', 'chat_id': '42', 'user_id': '42',
               'exit_code': 0, 'output': f'{name} output'}
        assert await runner._inject_watch_notification(f'[IMPORTANT: {name} finished]', evt) is True

    async def tick(name):
        evt = {'type': 'heartbeat', 'session_id': name, 'event_id': f'{name}-tick', 'started_at': 1.0,
               'session_key': route, 'parent_session_id': sid, 'platform': 'discord', 'chat_type': 'dm',
               'chat_id': '42', 'user_id': '42', 'output': f'{name} still running', 'omitted': 0}
        assert await runner._inject_watch_notification(f'[IMPORTANT: {name} still running]', evt) is True
    complete.tick = tick

    def queued():
        return [row for row in list_session_admissions(authority.db, session_id=owner) if row['status'] == 'queued']
    def history():
        return list_session_admissions(authority.db, session_id=owner, pending_only=False)
    complete.history = history
    return authority, human, complete, queued


@pytest.mark.asyncio
async def test_completions_queued_behind_a_turn_fold_into_one_turn(tmp_path, monkeypatch):
    authority, _human, complete, queued = await _discord_session(tmp_path, monkeypatch)
    try:
        await complete('proc_a')
        await complete.tick('proc_b')  # the finishing process's own tick never blocks the fold
        await complete('proc_b')
        await complete('proc_c')
        rows = queued()
        assert [row['principal_id'].split(':')[0] for row in rows] == ['messaging', 'automation']
        text = rows[-1]['payload']['text']
        assert text.startswith('[IMPORTANT: 3 background process results arrived')
        assert text.count('background process results arrived') == 1  # re-folding replaces the header
        assert all(f'{name} finished' in text for name in ('proc_a', 'proc_b', 'proc_c'))
        assert text.count('INTERNAL NOTIFICATION') == 1
        assert len(rows[-1]['payload']['native_text_v1']['automation']['identities']) == 3
        # Each replaced row keeps only its own identity (the batch carries text and identities):
        # storage grows with N, not N^2.
        folded = [row for row in complete.history() if row['outcome'] == 'cancelled'
                  and row['request_id'].startswith('["completion"')]
        assert [row['payload']['text'] for row in folded] == ['[folded into a later completion batch]'] * 2
        assert not [row for row in folded if 'identities' in row['payload']['native_text_v1']['automation']]
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('stale_read', [False, True])
async def test_a_prompt_queued_between_completions_keeps_them_apart(tmp_path, monkeypatch, stale_read):
    """``stale_read``: the prompt commits after the tail was read but before the fold is written."""
    from gateway import session_automation
    authority, human, complete, queued = await _discord_session(tmp_path, monkeypatch)
    try:
        await complete('proc_a')
        if stale_read:
            tail = queued()[-1]
            monkeypatch.setattr(session_automation, '_waiting_completion', lambda *args: {**tail, '_skip': []})
        await human('queued follow-up')
        await complete('proc_b')
        rows = queued()
        assert [row['payload']['text'].split('\n')[0] for row in rows] == [
            'running turn', '[IMPORTANT: proc_a finished]', 'queued follow-up', '[IMPORTANT: proc_b finished]']
    finally:
        authority.db.close()
