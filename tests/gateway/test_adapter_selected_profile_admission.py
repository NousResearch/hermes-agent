"""A device-selected profile must pass native admission, not only ingress routing.

Hermes Gadget's opt-in selector stamps source.profile on a primary adapter's
event. Previously _binding recomputed the primary profile without consulting
the adapter, rejecting an otherwise valid secondary turn with profile_mismatch.
Use real authorities and disposable ledgers; only the device policy is a fake.
"""
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource
from gateway.session_authorities import active_authority
from gateway.session_ingress_context import (
    capture_provenance, native_callback, register_transport_home, restore_provenance)
from hermes_state_runtime import RuntimeStoreError


class _DeviceAdapter:
    trusted = True
    allowed = frozenset({'default', 'work'})

    def validate_native_profile_route(self, source):
        return (self.trusted and source.user_id == 'paired-device'
                and source.profile in self.allowed)


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    work = root / 'profiles' / 'work'
    work.mkdir(parents=True)
    (work / 'config.yaml').write_text('{}')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = (('default', root), ('work', work))
    config = GatewayConfig(multiplex_profiles=True)
    config._runtime_profile_homes = homes
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    try:
        await initialize_gateway_runtime(runner)
        register_transport_home(runner, None, root)
        adapter = _DeviceAdapter()
        runner.adapters = {Platform.BLUEBUBBLES: adapter}
        runner._is_user_authorized_for_source = lambda *args, **kwargs: True
        source = SessionSource(platform=Platform.BLUEBUBBLES, chat_id='paired-device',
                               chat_type='dm', user_id='paired-device', profile='work')
        source._transport_adapter_ref = weakref.ref(adapter)
        event = MessageEvent(text='hello', message_type=MessageType.TEXT,
                             source=source, message_id='device-message')
        yield SimpleNamespace(root=root, work=work, runner=runner, adapter=adapter,
                              source=source, event=event)
    finally:
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_selected_profile_owns_its_durable_input_and_revalidates_replay(rig, monkeypatch):
    r = rig
    provenances = {}
    for name, home in (('default', r.root), ('work', r.work)):
        r.source.profile = name
        with _profile_runtime_scope(home, hydrate_secrets=False):
            authority = active_authority(r.runner)
            # Exercise durable admission without starting a model/agent turn.
            monkeypatch.setattr(authority, '_schedule_admitted', lambda *args: None)
            with native_callback(r.runner, r.event, r.root):
                receipt = await authority.admit_native(r.event)
                provenances[name] = capture_provenance(r.runner, r.event)
            assert receipt.status == 'queued'
            rows = authority.db._read_all('SELECT principal_id FROM session_admissions')
            assert len(rows) == 1 and name in rows[0]['principal_id']
            assert Path(authority.db.db_path).parent == home
            assert provenances[name]['transport_home'] == str(r.root)
            assert restore_provenance(r.runner, r.source, provenances[name]) == home
    # A captured row is not a permanent grant. Current policy still decides on replay.
    r.adapter.trusted = False
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        restore_provenance(r.runner, r.source, provenances['work'])
    r.adapter.trusted = True
    r.adapter.allowed = {'default'}
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        restore_provenance(r.runner, r.source, provenances['work'])


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['no_hook', 'truthy', 'untrusted', 'unserved', 'no_authority',
                                 'route_conflict', 'swapped_adapter'])
async def test_selection_never_replaces_transport_policy_or_authority(rig, monkeypatch, case):
    r = rig
    mutations = {
        'no_hook': lambda: monkeypatch.delattr(_DeviceAdapter, 'validate_native_profile_route'),
        'truthy': lambda: monkeypatch.setattr(
            _DeviceAdapter, 'validate_native_profile_route', lambda *args: 'yes'),
        'untrusted': lambda: monkeypatch.setattr(r.adapter, 'trusted', False),
        'unserved': lambda: monkeypatch.setattr(
            r.runner.config, '_runtime_profile_homes', (('default', r.root),)),
        'no_authority': lambda: monkeypatch.setattr(
            r.runner.session_authorities, 'for_home', lambda home: None),
        'route_conflict': lambda: monkeypatch.setattr(r.runner.config, 'profile_routes',
            parse_profile_routes([{'platform': 'bluebubbles', 'profile': 'default'}])),
        'swapped_adapter': lambda: monkeypatch.setattr(
            r.runner, 'adapters', {Platform.BLUEBUBBLES: _DeviceAdapter()}),
    }
    mutations[case]()
    with native_callback(r.runner, r.event, r.root), pytest.raises(RuntimeStoreError):
        capture_provenance(r.runner, r.event)


@pytest.mark.asyncio
@pytest.mark.parametrize('transport', ['secondary', 'relay'])
async def test_selection_is_only_for_the_primary_direct_transport(rig, transport):
    r = rig
    profile, home = None, r.root
    if transport == 'secondary':
        profile, home = 'work', r.work
        r.source.profile = 'default'
        r.runner.adapters = {}
        r.runner._profile_adapters = {'work': {Platform.BLUEBUBBLES: r.adapter}}
        register_transport_home(r.runner, profile, home)
    else:
        r.source.delivered_via_upstream_relay = True
        r.runner.adapters = {Platform.RELAY: r.adapter}
    with native_callback(r.runner, r.event, home, profile), pytest.raises(RuntimeStoreError):
        capture_provenance(r.runner, r.event)


@pytest.mark.asyncio
async def test_profile_selection_cannot_mint_or_edit_callback_provenance(rig):
    r = rig
    assert capture_provenance(r.runner, r.event) is None
    with native_callback(r.runner, r.event, r.root):
        provenance = capture_provenance(r.runner, r.event)
    provenance['runtime_home'] = str(r.root)
    with pytest.raises(RuntimeStoreError):
        restore_provenance(r.runner, r.source, provenance)
