"""A producer wake enters its route as the committed human turn, whatever the route's stored origin
says about labels and speakers, from a subagent's process, and over the relay."""
import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_identity import replace_source
from hermes_state_runtime import RuntimeStoreError, get_session_admission, list_session_admissions
from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner
from tests.gateway.test_automation_notification_category import _SOURCE, _authority, _native_owner


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', [{'chat_name': 'renamed thread'}, {'user_name': None},
                                   {'user_id': '7'}, {'profile': 'work'}])
async def test_wake_admits_as_the_committed_turn_when_the_origin_drifted(tmp_path, monkeypatch, drift):
    runner, authority, _adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, 'human-1')
    try:
        route, sid = metadata['gateway_session_key'], metadata['gateway_session_id']
        runner.session_store._entries[route].origin = replace_source(source, **drift)
        evt = {'type': 'completion', 'session_id': 'proc_x', 'session_key': route, 'platform': 'discord',
               'chat_id': '42', 'chat_type': 'dm', 'exit_code': 0, 'output': 'done', 'started_at': 1.0,
               'parent_session_id': sid}
        assert await runner._inject_watch_notification('[done]', evt) is True
        rows = list_session_admissions(authority.db, session_id=authority.logical_owner(sid), pending_only=False)
        human, wake = rows[0]['payload']['native_text_v1'], rows[-1]['payload']['native_text_v1']
        assert wake['source'] == {k: v for k, v in human['source'].items() if k != 'message_id'}
        assert wake['provenance'] == human['provenance']
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('thread_id', [None, '9'])
async def test_wake_refuses_another_user_on_a_per_user_route(tmp_path, monkeypatch, thread_id):
    fields = {**_SOURCE, 'chat_type': 'group', 'thread_id': thread_id}
    _runner, authority, adapter, _source, metadata = await _native_owner(
        tmp_path, monkeypatch, 'human-1', fields, group_sessions_per_user=True, thread_sessions_per_user=True)
    try:
        other = adapter.build_source(**{**fields, 'user_id': '7'})
        event = MessageEvent(text='notice', source=other, internal=True, metadata=dict(metadata))
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await authority.admit_automation(adapter, event, 'notice')
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('compressed', [False, True])
async def test_subagent_process_notice_enters_its_parent_route_but_not_after_reset(tmp_path, monkeypatch, compressed):
    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    root = (await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))).ref.session_id
    key = runner.session_store._generate_session_key(source)
    worker, nested = root + '-worker', root + '-nested'
    # Shaped like reserve_child_execution's rows (gateway/session_worker_child.py).
    authority.db.create_session(worker, source='subagent', parent_session_id=root, model_config={'_delegate_from': root})
    authority.db.create_session(nested, source='subagent', parent_session_id=worker,
                                model_config={'_delegate_from': worker})
    if compressed:
        authority.db.append_message(root, 'user', 'human')
        authority.db.publish_compression_child(parent_session_id=root, child_session_id=root + '-tip', source='discord',
                                               messages=[{'role': 'user', 'content': 'summary'}],
                                               require_compression_lease=False)
        assert runner.session_store.advance_compression_session(key, root, root + '-tip') is not None
    event = {'type': 'completion', 'session_id': 'proc-worker', 'started_at': 1,
             'session_key': key, 'parent_session_id': nested}
    assert await runner._inject_watch_notification('worker-finished', event) is True
    rows = list_session_admissions(authority.db, session_id=root, pending_only=False)
    assert len(rows) == 2 and 'worker-finished' in rows[-1]['payload']['text']
    runner.session_store.reset_session(key)
    # A notice pinned to a session /new closed is never admitted to the fresh one: dropped, or requeued
    # while the completion pre-flight reads a reset compression tip as uncertain.
    stale = dict(event, session_id='proc-old-worker')
    assert await runner._inject_watch_notification('old-worker', stale) is (False if compressed else None)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_subagent_row_without_delegate_provenance_owns_no_route(tmp_path, monkeypatch):
    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    root = (await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))).ref.session_id
    authority.db.create_session(root + '-orphan', source='subagent', parent_session_id=root)
    event = {'type': 'completion', 'session_id': 'proc-orphan', 'started_at': 1,
             'session_key': runner.session_store._generate_session_key(source), 'parent_session_id': root + '-orphan'}
    # The completion pre-flight calls this pin gone, so admission must not take it either.
    assert await runner._inject_watch_notification('orphan', event) is None
    assert len(list_session_admissions(authority.db, session_id=root, pending_only=False)) == 1


@pytest.mark.asyncio
async def test_subagent_notice_is_dropped_after_its_resumed_child_was_switched_away(tmp_path, monkeypatch):
    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    root = (await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))).ref.session_id
    key = runner.session_store._generate_session_key(source)
    worker = root + '-worker'
    authority.db.create_session(worker, source='subagent', parent_session_id=root, model_config={'_delegate_from': root})
    # /resume the child, then /resume its parent: the child ends at a user boundary (session_switch).
    runner.session_store.switch_session(key, worker)
    runner.session_store.switch_session(key, root)
    event = {'type': 'completion', 'session_id': 'proc-worker', 'started_at': 1,
             'session_key': key, 'parent_session_id': worker}
    assert await runner._inject_watch_notification('worker-finished', event) is None
    assert len(list_session_admissions(authority.db, session_id=root, pending_only=False)) == 1


async def _resumed_delegate_route(tmp_path, monkeypatch):
    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    root = (await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='h1'))).ref.session_id
    key = runner.session_store._generate_session_key(source)
    child = root + '-worker'
    # A delegate run (shaped like reserve_child_execution's row) resumed on the chat with /resume <id> --all.
    authority.db.create_session(child, source='subagent', parent_session_id=root, model_config={'_delegate_from': root})
    assert runner.session_store.switch_session(key, child) is not None
    await _admit_human(runner, authority, MessageEvent(text='on child', source=source, message_id='h2'))
    return runner, authority, source, key, child


def _compress(runner, authority, key, child):
    tip = child + '-tip'
    authority.db.append_message(child, 'user', 'on child')
    authority.db.publish_compression_child(parent_session_id=child, child_session_id=tip, source='discord',
                                           messages=[{'role': 'user', 'content': 'summary'}],
                                           require_compression_lease=False)
    assert runner.session_store.advance_compression_session(key, child, tip) is not None
    return tip


@pytest.mark.asyncio
async def test_wake_on_a_resumed_delegate_passes_preflight_after_compression(tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_envelope import check_native_route
    runner, authority, source, key, child = await _resumed_delegate_route(tmp_path, monkeypatch)
    tip = _compress(runner, authority, key, child)
    event = {'type': 'completion', 'session_id': 'proc', 'started_at': 1, 'session_key': key, 'parent_session_id': tip}
    assert await runner._inject_watch_notification('done', event) is True
    wake = list_session_admissions(authority.db, session_id=child)[-1]
    # The drain's pre-claim check (and startup recovery's): a refusal pauses the chat behind the wake.
    await check_native_route(runner, wake['payload'], authority.physical_target(SessionRef(authority.profile_id, child)),
                             source, runner._adapter_for_source(source))
    authority.db.close()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_notice_pinned_before_compression_reaches_the_resumed_delegate_chat(tmp_path, monkeypatch):
    runner, authority, _, key, child = await _resumed_delegate_route(tmp_path, monkeypatch)
    tip = _compress(runner, authority, key, child)
    # A process the resumed run started before compressing: the pre-flight and admission agree on the tip.
    assert (await runner._classify_completion_owner(child, key))[::2] == ('deliver', tip)
    event = {'type': 'completion', 'session_id': 'proc-before', 'started_at': 1, 'session_key': key,
             'parent_session_id': child}
    assert await runner._inject_watch_notification('before', event) is True
    authority.db.close()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_process_notice_from_a_subagent_of_a_resumed_delegate_reaches_the_chat(tmp_path, monkeypatch):
    runner, authority, _, key, child = await _resumed_delegate_route(tmp_path, monkeypatch)
    nested = child + '-nested'  # delegate_task inside the resumed conversation
    authority.db.create_session(nested, source='subagent', parent_session_id=child, model_config={'_delegate_from': child})
    event = {'type': 'completion', 'session_id': 'proc-nested', 'started_at': 1, 'session_key': key,
             'parent_session_id': nested}
    assert await runner._inject_watch_notification('nested done', event) is True
    assert 'nested done' in list_session_admissions(authority.db, session_id=child)[-1]['payload']['text']
    authority.db.close()
    await asyncio.sleep(0)


def _relay_adapter():
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    descriptor = CapabilityDescriptor(1, 'relay', 'Fixture', 4096, False, True, False, 'plain', 'chars')
    transport = WebSocketRelayTransport('http://127.0.0.1:1/relay', 'relay', '',
                                        gateway_id='fixture-gateway', upgrade_secret='fixture-secret')
    transport._ws = object()  # the connector proof requires a live socket
    transport._descriptors_by_platform['relay'] = descriptor
    return RelayAdapter(PlatformConfig(), descriptor, transport=transport), transport


def _relay_source(message_id=None):
    return SessionSource(platform=Platform.RELAY, chat_id='hub', chat_type='group', user_id='alex',
                         message_id=message_id, delivered_via_upstream_relay=True)


@pytest.mark.asyncio
async def test_relay_wake_row_reverifies_the_source_it_stores(tmp_path, monkeypatch):
    from gateway.session_envelope import restore_native
    from gateway.session_ingress_context import native_callback, register_transport_home, relay_callback
    runner, authority = await _authority(tmp_path, monkeypatch)
    adapter, transport = _relay_adapter()
    runner.adapters = {Platform.RELAY: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    try:
        human = MessageEvent(text='human', source=_relay_source('human-1'), message_id='human-1')
        with native_callback(runner, human, tmp_path), relay_callback(transport, human):
            await authority.admit_native(human)
        route = runner.session_store._generate_session_key(human.source)
        event = MessageEvent(text='notice', source=_relay_source(), internal=True, metadata={
            'gateway_session_key': route, 'gateway_session_id': runner.session_store.peek_session_id(route)})
        row = get_session_admission(authority.db, admission_id=(
            await authority.admit_automation(adapter, event, 'notice')).admission_id)
        assert row is not None
        assert restore_native(row['payload'], runner).source.message_id is None
    finally:
        authority.db.close()


# ── A Discord-thread wake rebuilt from the session-key fallback (#144006) ──────────────────
# Every committed human envelope in a Discord thread carries ``parent_chat_id`` (the parent
# channel) and ``scope_id`` (the guild). When the route has no stored origin to restore and no
# cached source, the wake is rebuilt from the session key and the completion event alone, so the
# rebuild only keeps the anchors the event carries: a producer that omits them rebuilds a
# degraded copy of the route the wake is admitted on.

_THREAD = dict(chat_id='900', chat_type='thread', thread_id='900', user_id='42',
               user_name='human', parent_chat_id='800', guild_id='700')
# Display labels a platform may change at any time (#143282); everything else identifies a route.
_ROUTE_LABELS = frozenset({'message_id', 'user_name', 'chat_name', 'chat_topic',
                           'auto_thread_initial_name', 'auto_thread_created'})


def _route_identity(source):
    """The fields that identify a route: what the wake rebuild must reproduce, never invent."""
    return {key: value for key, value in source.to_dict().items() if key not in _ROUTE_LABELS}


async def _thread_owner(tmp_path, monkeypatch):
    from gateway.session_ingress_context import native_callback, register_transport_home
    from plugins.platforms.discord.adapter import DiscordAdapter
    runner, authority = await _authority(tmp_path, monkeypatch)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='***', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    await _admit_human(runner, authority, MessageEvent(
        text='human', source=adapter.build_source(**_THREAD, message_id='human-1'),
        message_id='human-1'))
    return runner, authority, adapter


def _thread_completion_event(route, session_id, **anchors):
    event = {'type': 'completion', 'session_id': 'proc_x', 'session_key': route, 'platform': 'discord',
             'chat_type': 'thread', 'chat_id': '900', 'thread_id': '900', 'user_id': '42',
             'user_name': 'human', 'parent_chat_id': '800', 'scope_id': '700', 'exit_code': 0,
             'output': 'done', 'started_at': 1.0, 'parent_session_id': session_id}
    event.update(anchors)
    return event


def _drop_the_stored_origin(runner, route):
    """Leave the wake only the session-key fallback: no persisted origin to restore (the routing
    entry lost it) and no cached live source, so it is rebuilt from the key and the event alone."""
    entry = runner.session_store._entries[route]
    entry.origin = None
    entry.transport_profile = None  # ``_route_owner`` reads the origin only for a mounted LOCAL route
    runner._session_sources = {}


@pytest.mark.asyncio
async def test_thread_wake_rebuilt_from_the_session_key_is_the_committed_route(tmp_path, monkeypatch):
    """No stored origin and no cached source: the wake is rebuilt from the session key and the
    completion event. That rebuild must carry the thread's parent channel and guild scope, so it
    names the same route the session's last committed human turn does instead of a degraded copy."""
    from gateway.session_automation import committed_route_envelope
    runner, authority, adapter = await _thread_owner(tmp_path, monkeypatch)
    try:
        route = runner.session_store._generate_session_key(adapter.build_source(**_THREAD))
        sid = runner.session_store.peek_session_id(route)
        committed = SessionSource.from_dict(committed_route_envelope(
            authority, runner.session_store.lookup_by_session_key(route))['source'])
        _drop_the_stored_origin(runner, route)
        rebuilt = runner._build_process_event_source(_thread_completion_event(route, sid))
        assert rebuilt is not None
        assert _route_identity(rebuilt) == _route_identity(committed)
        assert runner.session_store._generate_session_key(rebuilt) == route
        assert await runner._inject_watch_notification('[done]', _thread_completion_event(route, sid)) is True
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_thread_wake_rebuild_keeps_its_parent_channel_profile_route(tmp_path, monkeypatch):
    """``match_profile_route`` matches a ``chat_id`` route against a thread's parent channel, so a
    rebuilt source that dropped ``parent_chat_id`` matches no profile route at all and the wake
    would run under the wrong profile. The anchor has to survive the rebuild for that match to hold."""
    from gateway.profile_routing import ProfileRoute, match_profile_route
    runner, authority, adapter = await _thread_owner(tmp_path, monkeypatch)
    try:
        route = runner.session_store._generate_session_key(adapter.build_source(**_THREAD))
        routes = [ProfileRoute(name='thread-parent', platform='discord', profile='code', chat_id='800')]
        entry = runner.session_store.lookup_by_session_key(route)
        assert entry is not None
        committed = entry.origin
        assert committed is not None
        assert match_profile_route(routes, platform='discord', guild_id=committed.guild_id,
                                   chat_id=committed.chat_id, thread_id=committed.thread_id,
                                   parent_chat_id=committed.parent_chat_id,
                                   user_id=committed.user_id) is not None
        _drop_the_stored_origin(runner, route)
        rebuilt = runner._build_process_event_source(_thread_completion_event(
            route, runner.session_store.peek_session_id(route)))
        assert rebuilt is not None
        assert match_profile_route(routes, platform='discord', guild_id=rebuilt.guild_id,
                                   chat_id=rebuilt.chat_id, thread_id=rebuilt.thread_id,
                                   parent_chat_id=rebuilt.parent_chat_id,
                                   user_id=rebuilt.user_id) is not None
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_thread_wake_rebuild_reports_the_parent_the_event_carries(tmp_path, monkeypatch):
    """The rebuild copies the anchors, it never derives them: a parent the event names is reported
    verbatim, so a route whose parent really drifted stays visibly drifted rather than silently
    corrected into agreement with the committed envelope."""
    runner, authority, adapter = await _thread_owner(tmp_path, monkeypatch)
    try:
        route = runner.session_store._generate_session_key(adapter.build_source(**_THREAD))
        _drop_the_stored_origin(runner, route)
        rebuilt = runner._build_process_event_source(_thread_completion_event(
            route, runner.session_store.peek_session_id(route), parent_chat_id='other-parent'))
        assert rebuilt is not None
        assert rebuilt.parent_chat_id == 'other-parent'
        assert rebuilt.scope_id == '700'
    finally:
        authority.db.close()


def test_completion_watcher_carries_the_thread_route_anchors(tmp_path, monkeypatch):
    """The producer chain: a Discord-thread turn's session context is stamped onto the process
    session, armed as a watcher, and the completion event that watcher produces still rebuilds the
    thread's parent channel and guild scope."""
    from types import SimpleNamespace
    import gateway.run as gateway_run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_notifications import GatewayNotificationsMixin
    from gateway.session_authority import SessionAuthority
    from gateway.session_context import get_session_env
    from hermes_state import SessionDB
    from tools.process_registry import ProcessRegistry
    from tools.terminal_tool_background import _register_completion_watcher, _stamp_gateway_routing

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    for name, value in (('HERMES_SESSION_PLATFORM', 'discord'), ('HERMES_SESSION_CHAT_ID', '900'),
                        ('HERMES_SESSION_THREAD_ID', '900'), ('HERMES_SESSION_USER_ID', '42'),
                        ('HERMES_SESSION_USER_NAME', 'human'),
                        ('HERMES_SESSION_PARENT_CHAT_ID', '800'), ('HERMES_SESSION_SCOPE_ID', '700'),
                        ('HERMES_SESSION_ID', 'session-1')):
        monkeypatch.setenv(name, value)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = db
    proc_session = SimpleNamespace(id='proc_thread', parent_session_id='')
    _stamp_gateway_routing(proc_session, get_session_env)
    armed = []
    monkeypatch.setattr(gateway_run, '_gateway_runner_ref',
                        lambda: SimpleNamespace(arm_process_watcher=lambda w: armed.append(w) or True))
    _register_completion_watcher(ProcessRegistry(), proc_session, 'agent:main:discord:thread:900:900')
    event = GatewayNotificationsMixin._build_process_completion_event(
        armed[0], SimpleNamespace(command='sleep', output_buffer='done', exit_code=0,
                                  completion_reason='exited', termination_source='', started_at=1.0,
                                  owner_task_id='', task_id='', parent_session_id='session-1'),
        'proc_thread')
    rebuilt = runner._build_process_event_source(event)
    assert rebuilt is not None
    assert (rebuilt.parent_chat_id, rebuilt.scope_id) == ('800', '700')
    assert runner.session_store._generate_session_key(rebuilt) == 'agent:main:discord:thread:900:900'


def test_async_delegation_dispatch_snapshots_the_thread_parent(monkeypatch):
    """The durable delegation ledger persists the parent channel of the thread that dispatched the
    child, so a restart-recovered completion event still rebuilds the route it belongs to."""
    from tools import async_delegation as delegation
    for name, value in (('HERMES_SESSION_SCOPE_ID', '700'), ('HERMES_SESSION_PARENT_CHAT_ID', '800'),
                        ('HERMES_SESSION_USER_ID', '42'), ('HERMES_SESSION_USER_NAME', 'human')):
        monkeypatch.setenv(name, value)
    assert delegation._capture_routing_origin() == {
        'scope_id': '700', 'parent_chat_id': '800', 'user_id': '42', 'user_name': 'human'}
