"""Desktop and CLI ``/handoff`` on the shared owner (#143608): the owner serves ``handoff.request`` /
``handoff.state`` / ``handoff.fail`` and writes the pending row in its own store, where the gateway's
handoff watcher claims it. They were refused by name ("not available on the shared gateway yet")."""
import asyncio

import pytest

from tests.gateway.test_owner_loop_writes_class import _local, _owner


def _telegram_home(tmp_path, monkeypatch, home_channel=True):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path)  # the fixture authority's symbolic profile id is a relative home
    for key in ('TELEGRAM_BOT_TOKEN', 'TELEGRAM_HOME_CHANNEL', 'TELEGRAM_HOME_CHANNEL_NAME'):
        monkeypatch.delenv(key, raising=False)
    home = "    home_channel:\n      platform: telegram\n      chat_id: '4242'\n      name: Phone\n" if home_channel else ''
    (tmp_path / 'config.yaml').write_text(  # single-profile owner: dispatch reads the ambient home
        "platforms:\n  telegram:\n    enabled: true\n    token: '123456789:AAFakeTokenForTestsOnly_abcdefghij'\n" + home,
        encoding='utf-8')


async def _call(owner, method, **params):
    return await owner.dispatch({'id': 1, 'method': method, 'params': params})


@pytest.mark.asyncio
async def test_handoff_to_telegram_queues_the_row_the_watcher_claims(tmp_path, monkeypatch):
    _telegram_home(tmp_path, monkeypatch)
    _store, authority, owner = _owner(tmp_path, monkeypatch, handle=None)
    ref = _local(authority, owner, tmp_path)
    reply = await _call(owner, 'handoff.request', session_id=ref.session_id, platform='Telegram')
    assert reply.get('result') == {'queued': True, 'session_key': ref.session_id, 'platform': 'telegram',
                                   'home_name': 'Phone'}, reply
    # The watcher's own read of this store sees it (gateway/run_adapters.py::_handoff_watcher).
    assert [row['id'] for row in authority.db.list_pending_handoffs()] == [ref.session_id]
    state = await _call(owner, 'handoff.state', session_id=ref.session_id)
    assert state['result'] == {'state': 'pending', 'platform': 'telegram', 'error': ''}
    again = await _call(owner, 'handoff.request', session_id=ref.session_id, platform='telegram')
    assert again['error']['message'] == 'session_busy', again
    # A claimed row is the watcher's: the client's timeout cannot fail it.
    assert authority.db.claim_handoff(ref.session_id)
    late = await _call(owner, 'handoff.fail', session_id=ref.session_id, error='client poll timeout')
    assert late['result'] == {'failed': False, 'state': 'running'}
    authority.db.close()


@pytest.mark.asyncio
async def test_handoff_refuses_before_queueing_what_the_watcher_could_never_deliver(tmp_path, monkeypatch):
    _telegram_home(tmp_path, monkeypatch, home_channel=False)
    _store, authority, owner = _owner(tmp_path, monkeypatch, handle=None)
    ref = _local(authority, owner, tmp_path)
    for platform, field in (('telegram', 'platform'), ('nowhere', 'platform')):
        reply = await _call(owner, 'handoff.request', session_id=ref.session_id, platform=platform)
        assert reply['error']['message'] == 'invalid_params' and reply['error']['data']['field'] == field, reply
    unknown = await _call(owner, 'handoff.request', session_id='no-such-session', platform='telegram')
    assert unknown['error']['message'] == 'not_found', unknown
    assert authority.db.list_pending_handoffs() == []
    # An unclaimed row times out on the client: failed, so /handoff can be retried.
    _telegram_home(tmp_path, monkeypatch)
    assert (await _call(owner, 'handoff.request', session_id=ref.session_id, platform='telegram'))['result']['queued']
    timed_out = await _call(owner, 'handoff.fail', session_id=ref.session_id, error='client poll timeout')
    assert timed_out['result'] == {'failed': True, 'state': 'failed'}
    authority.db.close()


@pytest.mark.asyncio
async def test_handoff_is_refused_to_a_viewer_without_control(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    _telegram_home(tmp_path, monkeypatch)
    _store, authority, owner = _owner(tmp_path, monkeypatch, handle=None)
    ref = _local(authority, owner, tmp_path)
    reader = AuthorityConnection(authority, object(), {'user_id': 'human', 'capabilities': ['session:read']})
    reply = await _call(reader, 'handoff.request', session_id=ref.session_id, platform='telegram')
    assert reply['error']['message'] == 'permission_denied', reply
    assert authority.db.list_pending_handoffs() == []
    await asyncio.sleep(0)
    authority.db.close()
