"""Ordinary intake ownership: real HTTP save / runner / base, fake I/O only."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from tests.hermes_cli.test_buzz_config_policy import client, URL
from tests.gateway.test_live_buzz_authorization import registered_buzz

A, B, SELF = 'a' * 64, 'b' * 64, '1' * 64


def instrument(adapter, runner):
    adapter._self_pubkey = SELF
    adapter._display_name = 'TestAgent'
    adapter._channel_meta['group'] = {'name': 'Group', 'description': 'shared'}
    adapter._channel_names['group'] = 'Group'
    adapter._resolve_user_name = AsyncMock(return_value='Synthetic')
    adapter._cache_inbound_attachments = AsyncMock(return_value=[])
    adapter.send_reaction = AsyncMock()
    adapter.send_typing = AsyncMock()
    forwarded, model_bound = [], []
    async def model_boundary(event):
        verdict = runner._is_user_authorized(event.source)
        forwarded.append((event.source.user_id, verdict))
        if verdict is True:
            model_bound.append(event.source.user_id)
    adapter.set_message_handler(model_boundary)
    state = adapter._new_channel_state('group')
    sequence = 0
    async def emit(user):
        nonlocal sequence
        sequence += 1
        event = dict(id=f'ownership-{sequence}', pubkey=user, kind=9, created_at=sequence,
                     content='@TestAgent inspect', tags=[['h', 'group'], ['p', SELF],
                     ['imeta', 'url https://test.relay/media/file.bin',
                      'm application/octet-stream', 'x ' + 'a' * 64, 'size 1', 'filename file.bin']])
        await adapter._handle_event('group', state, event)
        tasks = tuple(adapter._session_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
    def io_counts():
        return (adapter._resolve_user_name.await_count,
                adapter._cache_inbound_attachments.await_count, adapter.send_reaction.await_count)
    return emit, forwarded, model_bound, io_counts


def save(c, home, **policy):
    import yaml
    response = c.put(URL, json={'policy': policy})
    assert response.status_code == 200, response.text
    actual = yaml.safe_load((home / 'config.yaml').read_text())['gateway']['platforms']['buzz']['extra']
    assert all(actual[k] == v for k, v in policy.items())


@pytest.mark.asyncio
async def test_live_new_sender_grant_revoke_pairing_forwarding(registered_buzz):
    home, runner, adapter = registered_buzz
    identities = id(runner), id(adapter)
    emit, forwarded, model_bound, io = instrument(adapter, runner)
    c = client()
    assert adapter._allowed_pubkeys == {A}
    await emit(A)
    assert model_bound == [A] and io() == (1, 1, 1)
    before = io()
    await emit(B)
    assert forwarded[-1] == (B, False) and model_bound == [A] and io() == before
    save(c, home, allowed_users=[B], allow_all_users=False)
    await emit(A)
    assert forwarded[-1] == (A, False) and io() == before
    await emit(B)
    assert forwarded[-1] == (B, True) and model_bound == [A, B] and io() == (2, 2, 2)
    save(c, home, allowed_users=[], allow_all_users=False)
    before = io()
    await emit(B)
    assert forwarded[-1] == (B, False) and model_bound == [A, B] and io() == before
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_: True)
    await emit(B)
    assert forwarded[-1] == (B, True) and model_bound == [A, B, B] and io() == (3, 3, 3)
    assert adapter._allowed_pubkeys == {A} and (id(runner), id(adapter)) == identities


@pytest.mark.asyncio
@pytest.mark.parametrize('authority', ['missing', 'throws', 'string', 'integer', 'false', 'zero'])
async def test_nonliteral_authority_outside_startup_list_forwards_without_io(registered_buzz, authority):
    _, runner, adapter = registered_buzz
    emit, forwarded, model_bound, io = instrument(adapter, runner)
    def throws(*_):
        raise RuntimeError('synthetic authority failure')
    callbacks = {'missing': None, 'throws': throws, 'string': lambda *_: 'yes',
                 'integer': lambda *_: 1, 'false': lambda *_: False, 'zero': lambda *_: 0}
    adapter.set_authorization_check(callbacks[authority])
    await emit(B)
    assert forwarded == [(B, False)]
    assert model_bound == [] and io() == (0, 0, 0)
