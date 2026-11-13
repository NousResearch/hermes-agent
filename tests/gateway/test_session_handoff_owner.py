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
    # The watcher's own read of this store sees it (hermes_agent/gateway/run_adapters.py::_handoff_watcher).
    assert [row['id'] for row in authority.db.list_pending_handoffs()] == [ref.session_id]
    state = await _call(owner, 'handoff.state', session_id=ref.session_id)
    assert state['result'] == {'state': 'pending', 'platform': 'telegram', 'error': ''}
    again = await _call(owner, 'handoff.request', session_id=ref.session_id, platform='telegram')
    assert again['error']['data']['reason'] == 'session_busy', again
    assert again['error']['message'] == 'a handoff of this session is already in flight; wait for it to settle'
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
        assert reply['error']['data']['reason'] == 'invalid_params' and reply['error']['data']['field'] == field, reply
    # Desktop shows ``error.message`` as is: it says what to fix, never the bare code.
    assert reply['error']['message'] == "unknown platform 'nowhere'", reply
    no_home = await _call(owner, 'handoff.request', session_id=ref.session_id, platform='telegram')
    assert no_home['error']['message'].startswith('no home channel configured for telegram'), no_home
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
    from hermes_agent.gateway.session_controls import AuthorityConnection
    _telegram_home(tmp_path, monkeypatch)
    _store, authority, owner = _owner(tmp_path, monkeypatch, handle=None)
    ref = _local(authority, owner, tmp_path)
    reader = AuthorityConnection(authority, object(), {'user_id': 'human', 'capabilities': ['session:read']})
    reply = await _call(reader, 'handoff.request', session_id=ref.session_id, platform='telegram')
    assert reply['error']['message'] == 'permission_denied', reply
    assert authority.db.list_pending_handoffs() == []
    await asyncio.sleep(0)
    authority.db.close()


@pytest.mark.asyncio
async def test_handoff_to_a_relay_fronted_platform_is_queued(tmp_path, monkeypatch):
    """A relay-fronted gateway has only a RELAY block; the watcher still delivers Discord through it
    (``resolve_delivery_transport``), so the owner queues it as the classic CLI always did."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GATEWAY_RELAY_PLATFORMS', 'discord')
    monkeypatch.delenv('DISCORD_HOME_CHANNEL', raising=False)
    (tmp_path / 'config.yaml').write_text(
        "platforms:\n  relay:\n    enabled: true\n  discord:\n    enabled: false\n"
        "    home_channel:\n      platform: discord\n      chat_id: '777'\n      name: Lab\n", encoding='utf-8')
    _store, authority, owner = _owner(tmp_path, monkeypatch, handle=None)
    ref = _local(authority, owner, tmp_path)
    reply = await _call(owner, 'handoff.request', session_id=ref.session_id, platform='discord')
    assert reply.get('result', {}).get('home_name') == 'Lab', reply
    monkeypatch.setenv('GATEWAY_RELAY_PLATFORMS', 'slack')  # a platform the relay does not front
    authority.db.fail_handoff(ref.session_id, 'reset')
    refused = await _call(owner, 'handoff.request', session_id=ref.session_id, platform='discord')
    assert refused['error']['data']['reason'] == 'invalid_params', refused
    authority.db.close()


@pytest.mark.asyncio
async def test_handoff_waits_for_durable_work_the_owner_has_no_task_for(tmp_path, monkeypatch):
    """A turn recovered ``unknown`` after an owner restart has no resident drain task, yet still owns
    the conversation: the request is refused, and so is the watcher's repoint of a row queued before
    the turn was admitted, so the route never moves while its outcome is unresolved."""
    import hermes_agent.hermes_state_runtime as rt
    from tests.gateway.test_authority_cold_route_recovery import _source
    from tests.gateway.test_handoff_local_receipt import _handoff
    _telegram_home(tmp_path, monkeypatch)
    _store, authority, owner = _owner(tmp_path, monkeypatch, handle=None)
    ref = _local(authority, owner, tmp_path)
    assert (await _call(owner, 'handoff.request', session_id=ref.session_id, platform='telegram'))['result']['queued']
    rt.admit_session_input(authority.db, epoch=authority.epoch, principal_id=owner.actor.subject,
                           session_id=ref.session_id, request_id='lost', payload={'text': 'pending work'})
    rt.claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
    authority.epoch = rt.begin_runtime_epoch(authority.db, instance_id='restarted')
    rt.recover_session_inputs(authority.db, epoch=authority.epoch)
    with pytest.raises(RuntimeError, match='/discard'):
        await _handoff(authority.runner, authority, ref.session_id, _source())
    assert authority.db.get_session(ref.session_id)['source'] != 'telegram'
    authority.db.fail_handoff(ref.session_id, 'refused')
    refused = await _call(owner, 'handoff.request', session_id=ref.session_id, platform='telegram')
    assert refused['error']['data']['reason'] == 'unknown_execution', refused
    assert authority.db.get_handoff_state(ref.session_id)['state'] == 'failed'
    authority.db.close()
