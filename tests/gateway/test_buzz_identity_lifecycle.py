"""Independent synthetic identity-lock regressions. No live state/transport."""
import asyncio
import json
from unittest.mock import AsyncMock
import pytest
from gateway.config import PlatformConfig
from gateway import status
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

buzz = load_plugin_adapter('buzz')
KEY = '00' * 31 + '03'
PUB = 'f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9'
CHANNEL = '11111111-1111-1111-1111-111111111111'

@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(buzz, '_resolve_private_key', lambda extra: KEY)
    monkeypatch.setattr(buzz, '_resolve_auth_tag', lambda extra: '')
    adapter = buzz.BuzzAdapter(PlatformConfig(enabled=True, extra={'relay_url':'https://test.invalid','transport':'poll'}))
    adapter.cli_path = '/synthetic/buzz'
    released = []
    monkeypatch.setattr(status, 'release_scoped_lock', lambda scope,key,metadata=None: released.append((scope,key)))
    adapter._load_cursors = lambda: None
    adapter._save_cursors = lambda: None
    adapter._discover_dms = AsyncMock()
    async def seed(channel_id, chat_type):
        adapter._channel_state[channel_id] = adapter._new_channel_state(chat_type)
    adapter._seed_channel = seed
    return adapter,released

@pytest.mark.asyncio
async def test_real_contract_false_tuple_prevents_post_lock_io(setup, monkeypatch):
    adapter,released=setup
    # Current status API returns tuple[bool, Optional[dict]], not bool.
    monkeypatch.setattr(status, 'acquire_scoped_lock', lambda scope,key,metadata=None: (False, {'pid':123,'profile':'other'}))
    adapter._run_cli = AsyncMock(side_effect=[(0,json.dumps([{'pubkey':PUB}]),''),(0,json.dumps([{'channel_id':CHANNEL}]),'')])
    try:
        result = await adapter.connect()
        print(json.dumps({'probe':'conflict','result':result,'post_lock_cli':adapter._run_cli.await_count-1,'lock_key_retained':bool(adapter._platform_lock_identity)}))
        assert result is False, 'tuple(False, holder) must not be treated as a grant'
        assert adapter._run_cli.await_count == 1
        assert adapter._platform_lock_identity is None
        assert adapter._poll_task is None
        adapter._discover_dms.assert_not_awaited()
        assert released == []
    finally:
        await adapter.disconnect()
    assert released == []

@pytest.mark.asyncio
async def test_positive_acquisition_connects_and_explicit_disconnect_releases(setup, monkeypatch):
    adapter,released = setup
    monkeypatch.setattr(status, 'acquire_scoped_lock', lambda scope,key,metadata=None: (True,None))
    adapter._run_cli = AsyncMock(side_effect=[(0,json.dumps([{'pubkey':PUB}]),''),(0,json.dumps([{'channel_id':CHANNEL}]),'')])
    try:
        assert await adapter.connect() is True
        assert released == []
    finally:
        await adapter.disconnect()
    assert released == [('buzz',f'https://test.invalid:{PUB}')]
    assert adapter._platform_lock_identity is None
    await adapter.disconnect()
    assert released == [('buzz',f'https://test.invalid:{PUB}')]

@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['return_false','exception','cancellation'])
async def test_post_acquire_failure_releases_owned_lock(setup,monkeypatch,failure):
    adapter,released=setup
    monkeypatch.setattr(status,'acquire_scoped_lock',lambda scope,key,metadata=None:(True,None))
    async def cli(args, **kwargs):
        if args == ['users','get']:
            return 0,json.dumps([{'pubkey':PUB}]),''
        if failure == 'return_false':
            return 0,'{}',''
        if failure == 'exception':
            raise RuntimeError('synthetic roster failure')
        raise asyncio.CancelledError()
    adapter._run_cli=AsyncMock(side_effect=cli)
    try:
        if failure=='return_false':
            assert await adapter.connect() is False
        else:
            with pytest.raises(RuntimeError if failure=='exception' else asyncio.CancelledError):
                await adapter.connect()
        print(json.dumps({'probe':failure,'release_count':len(released),'lock_key_retained':bool(adapter._platform_lock_identity)}))
        assert released == [('buzz', f'https://test.invalid:{PUB}')], 'failed connect must release its acquired lock'
        assert adapter._platform_lock_identity is None
    finally:
        await adapter.disconnect()
    assert released == [('buzz', f'https://test.invalid:{PUB}')]
    assert adapter._poll_task is None
