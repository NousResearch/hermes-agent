"""Background producers reach the chat that started them, through the real authorities, adapters,
turn path and SQLite. Only the model call is stubbed; it records the frozen session policy each turn
resolved. Regression coverage for #139377 and #139699.

- A ``terminal(background=true, heartbeat=N)`` liveness tick is admitted into the local or messaging
  chat that started the process. A tick that finds an earlier tick of its process still queued
  replaces it, carrying both outputs, instead of stacking another full turn, and no tick runs after
  its process's completion.
- A served (non-launch) profile's Bot Chat takes every relayed DM with no client attached, minted by
  the relay door itself, also after a restart over rows that store the launch profile's transport.
- A served profile's Desktop session admits its own process's watch match and completion.
"""
import asyncio
from contextlib import closing
import json
import queue
import sqlite3
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until

AUTHOR = {'id': 'bot:alpha', 'name': 'alpha', 'is_bot': True}
LEGACY_DESKTOP_ID = '20261015_112020_548787'


class _Gateway:
    """Two profiles on one multiplexed gateway, each with its own model; restartable over its homes."""

    def __init__(self, tmp_path, monkeypatch):
        from pathlib import Path
        self.tmp_path, root = tmp_path, tmp_path / '.hermes'
        self.homes = (('default', root), ('beta', root / 'profiles' / 'beta'))
        for name, home in self.homes:
            home.mkdir(parents=True, exist_ok=True, mode=0o700)
            (home / 'config.yaml').write_text(json.dumps({'model': {
                'default': f'{name}-model', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1',
                'api_key': 'loopback-only'}}))
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        monkeypatch.setenv('HERMES_HOME', str(root))
        import hermes_state
        monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
        self.turns, self.gate = [], None
        self.runner, self.connections = None, []

        async def model(runner, message, context_prompt, history, source, session_id, **_kwargs):
            from gateway.session_policy import policy_for_source
            policy = policy_for_source(runner, source)
            self.turns.append(SimpleNamespace(profile=source.profile, chat=source.chat_id, text=message,
                                              policy=policy and (policy.model, policy.cwd)))
            if self.gate is not None:
                await self.gate.wait()
            return {'final_response': 'ack', 'messages': [], 'api_calls': 1, 'tools': [],
                    'history_offset': len(history), 'session_id': session_id}

        monkeypatch.setattr(GatewayRunner, '_run_agent', model)

    async def start(self):
        from gateway.run_runtime import initialize_gateway_runtime
        from gateway.runtime_ownership import process_ownership
        config = GatewayConfig()
        config.multiplex_profiles = True
        config._runtime_profile_homes = self.homes
        self.runner = GatewayRunner(config)
        process_ownership.reserve([home for _, home in self.homes])
        await initialize_gateway_runtime(self.runner)
        self.primary, self.secondary = (self.runner.session_authorities.for_home(home) for _, home in self.homes)
        # The launch profile has a local session of its own, as on any install that used it, so the
        # launch profile's local adapter exists next to the served profile's.
        self.launch = self.create(self.primary, self.connect(self.primary), 'launch', 'cli')

    async def stop(self):
        from gateway.runtime_ownership import process_ownership
        for connection in self.connections:
            await connection.close()
        self.connections = []
        for authority in getattr(self.runner, 'session_authorities', None) or []:
            authority.db.close()
        for _, home in self.homes:
            process_ownership.release(home)

    def connect(self, authority, **identity):
        from gateway.session_controls import AuthorityConnection
        connection = AuthorityConnection(authority, object(), {'user_id': 'uid:1000', **identity})
        self.connections.append(connection)
        return connection

    def create(self, authority, connection, request_id, source, **params):
        from gateway.session_authorities import owner_scope
        from gateway.session_local import create_local_session
        with owner_scope(authority):
            return create_local_session(authority, connection.actor, dict(
                request_id=request_id, source=source, cwd=str(self.tmp_path), toolsets=[], **params))

    async def watch(self, authority, session_id):
        """Attach a viewer to the session, as an open CLI or Desktop pane is."""
        viewer = self.connect(authority)
        reply = await viewer.dispatch({'id': 'watch', 'method': 'session.resume', 'params': {'session_id': session_id}})
        assert 'result' in reply, reply
        return viewer

    def rows(self, authority, session_id):
        return list_session_admissions(authority.db, session_id=session_id, pending_only=False)

    async def settled(self, authority, session_id, count):
        def done():
            rows = self.rows(authority, session_id)
            return len(rows) >= count and all(row['status'] == 'terminal' for row in rows)
        await async_wait_until(done, timeout=30, what=f'{count} settled admissions')

    def process_event(self, authority, session_id, kind, process='proc_job', **fields):
        live = authority.sessions[session_id]
        target = self.runner.session_store.lookup_by_session_key(live.route).session_id
        return {'type': kind, 'session_id': process, 'started_at': 1.0, 'session_key': live.route,
                'parent_session_id': target, 'command': 'make test', 'output': 'tick', **fields}

    def tick(self, authority, session_id, seq, process='proc_job', output=None):
        return self.process_event(authority, session_id, 'heartbeat', process=process,
                                  **_tick_fields(process, seq, output))

    async def drain(self, *events):
        waiting = queue.Queue()
        for event in events:
            waiting.put(event)
        await self.runner._drain_watch_notifications(waiting)
        return [waiting.get_nowait() for _ in range(waiting.qsize())]


@pytest_asyncio.fixture
async def gateway(tmp_path, monkeypatch):
    gateway = _Gateway(tmp_path, monkeypatch)
    await gateway.start()
    try:
        yield gateway
    finally:
        if gateway.gate is not None:
            gateway.gate.set()
        await gateway.stop()


def _tick_fields(process, seq, output=None):
    return {'event_id': f'{process}-tick-{seq}', 'seq': seq, 'interval': 60, 'elapsed': 60 * seq,
            'output': f'{process}/{seq}\n' if output is None else output}


def _tick_output(text):
    return text.split('Output since last heartbeat:\n', 1)[1].split(']', 1)[0]


@pytest.mark.asyncio
async def test_a_liveness_tick_is_admitted_into_the_session_that_started_the_process(gateway):
    from gateway.session_automation import admission_display
    from hermes_cli.gateway_tools_notify import _card
    sid = gateway.launch.session_id

    assert await gateway.drain(gateway.tick(gateway.primary, sid, 1)) == [], 'the tick was refused and requeued'
    await gateway.settled(gateway.primary, sid, 1)
    [turn] = gateway.turns
    assert (turn.chat, turn.text.startswith('[Background process proc_job heartbeat #1 ')) == (sid, True)
    # The claim's message.start card: the wake stays machinery, and an idle CLI opens the turn with a
    # one-line receipt.
    [row] = gateway.rows(gateway.primary, sid)
    card = _card({'payload': admission_display(row['payload'])})
    assert card is not None and (card['display_kind'], card['display_text']) == (
        'internal_notification', 'Background Process Output after 1m: make test')


@pytest.mark.asyncio
async def test_waiting_ticks_fold_into_one_turn_and_none_outlives_its_process(gateway):
    primary, sid = gateway.primary, gateway.launch.session_id
    viewer = await gateway.watch(primary, sid)
    gateway.gate = asyncio.Event()
    await gateway.drain(gateway.tick(primary, sid, 1, process='proc_1'))
    await async_wait_until(lambda: gateway.turns, timeout=30, what='the first tick running')

    # While proc_1's first tick runs: proc_12 queues two ticks (they fold), the viewer queues inputs
    # whose ids look like ticks, then proc_1 queues two ticks of its own (proc_1 is a prefix of
    # proc_12), the second carrying both outputs past the 2,000-character tail.
    for seq in (1, 2):
        assert await gateway.drain(gateway.tick(primary, sid, seq, process='proc_12')) == []
    for count, lookalike in enumerate(('["heartbeat","proc_12","typed"]', '["heartbeat" typed'), start=1):
        submitted = await viewer.dispatch({'id': count, 'method': 'prompt.submit', 'params': {
            'session_id': sid, 'input_id': lookalike, 'text': 'typed'}})
        assert 'result' in submitted, submitted
    for seq, output, omitted in ((2, 'a' * 2000, 300), (3, 'b' * 1000, 0)):
        tick = dict(gateway.tick(primary, sid, seq, process='proc_1', output=output), omitted=omitted)
        assert await gateway.drain(tick) == []
    # proc_12 exits: its waiting tick is withdrawn, and a late tick is dropped.
    exited = gateway.process_event(primary, sid, 'completion', process='proc_12', exit_code=0)
    assert await gateway.runner._deliver_completion_notification('[proc_12 finished]', exited) is True
    assert await gateway.drain(gateway.tick(primary, sid, 3, process='proc_12')) == []
    gateway.gate.set()
    await gateway.settled(primary, sid, 8)

    assert {row['request_id']: row['outcome'] for row in gateway.rows(primary, sid)} == {
        '["heartbeat","proc_1","proc_1-tick-1"]': 'completed',
        '["heartbeat","proc_12","proc_12-tick-1"]': 'cancelled',
        '["heartbeat","proc_12","proc_12-tick-2"]': 'cancelled',
        '["heartbeat","proc_12","typed"]': 'completed', '["heartbeat" typed': 'completed',
        '["heartbeat","proc_1","proc_1-tick-2"]': 'cancelled',
        '["heartbeat","proc_1","proc_1-tick-3"]': 'completed',
        '["completion","proc_12",1.0]': 'completed'}
    assert [_tick_output(turn.text) for turn in gateway.turns if 'heartbeat #' in turn.text] == [
        'proc_1/1\n', '...(1300 earlier characters omitted)\n' + 'a' * 1000 + 'b' * 1000]


@pytest_asyncio.fixture
async def messaging(tmp_path, monkeypatch):
    """A Discord DM under the session authority with one committed human message; inputs stay queued."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_state import SessionDB
    from plugins.platforms.discord.adapter import DiscordAdapter
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='ticks', db=db)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='***', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    chat = dict(chat_id='42', chat_type='dm', user_id='42', user_name='human')
    human = MessageEvent(text='human', source=adapter.build_source(**chat, message_id='m1'), message_id='m1')
    with native_callback(runner, human, tmp_path):
        await authority.admit_native(human)
    route = runner.session_store._generate_session_key(adapter.build_source(**chat))
    owner = authority.logical_owner(runner.session_store.peek_session_id(route))
    try:
        yield SimpleNamespace(runner=runner, authority=authority, route=route, owner=owner, event=lambda kind, **fields: {
            'type': kind, 'session_id': 'proc_job', 'started_at': 1.0, 'session_key': route, 'parent_session_id': owner,
            'platform': 'discord', 'command': 'make test', **chat, **fields})
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_messaging_chat_gets_its_ticks_folded_and_none_after_the_completion(messaging):
    async def drain(event):
        waiting = queue.Queue()
        waiting.put(event)
        await messaging.runner._drain_watch_notifications(waiting)
        return waiting.qsize()

    for seq in (1, 2):
        assert await drain(messaging.event('heartbeat', **_tick_fields('proc_job', seq))) == 0, 'refused and requeued'
    exited = messaging.event('completion', exit_code=0, output='done')
    assert await messaging.runner._inject_watch_notification('[proc_job finished]', exited) is True
    assert await drain(messaging.event('heartbeat', **_tick_fields('proc_job', 3))) == 0

    rows = list_session_admissions(messaging.authority.db, session_id=messaging.owner, pending_only=False)
    automation = {row['request_id']: row for row in rows if row['principal_id'] == 'automation:' + messaging.route}
    assert {key: row['outcome'] for key, row in automation.items()} == {
        '["heartbeat","proc_job","proc_job-tick-1"]': 'cancelled',
        '["heartbeat","proc_job","proc_job-tick-2"]': 'cancelled', '["completion","proc_job",1.0]': None}
    folded = automation['["heartbeat","proc_job","proc_job-tick-2"]']['payload']['text']
    assert _tick_output(folded) == 'proc_job/1\nproc_job/2\n'


async def _drain_one(runner, event):
    waiting = queue.Queue()
    waiting.put(event)
    await runner._drain_watch_notifications(waiting)
    return waiting.qsize()


def _automation_rows(messaging):
    rows = list_session_admissions(messaging.authority.db, session_id=messaging.owner, pending_only=False)
    return {row['request_id']: row for row in rows if row['principal_id'] == 'automation:' + messaging.route}


@pytest.mark.asyncio
async def test_a_refused_replacement_leaves_the_earlier_tick_queued_and_the_fold_survives_a_restart(messaging):
    """Cancelling the waiting tick and admitting its replacement are one ledger write, and what the
    replacement folds in is read from the stored tick, not from the runner's memory."""
    tick = lambda seq: messaging.event('heartbeat', **_tick_fields('proc_job', seq))
    assert await _drain_one(messaging.runner, tick(1)) == 0
    messaging.authority.retiring = True  # the replacement is refused
    assert await _drain_one(messaging.runner, tick(2)) == 1, 'a refused tick is requeued'
    assert {key: row['status'] for key, row in _automation_rows(messaging).items()} == {
        '["heartbeat","proc_job","proc_job-tick-1"]': 'queued'}, 'the accepted tick was cancelled'

    messaging.authority.retiring = False
    messaging.runner.__dict__.pop('_admitted_heartbeat_ticks', None)  # what a restart forgets
    assert await _drain_one(messaging.runner, tick(2)) == 0
    rows = _automation_rows(messaging)
    assert {key: row['outcome'] for key, row in rows.items()} == {
        '["heartbeat","proc_job","proc_job-tick-1"]': 'cancelled', '["heartbeat","proc_job","proc_job-tick-2"]': None}
    assert _tick_output(rows['["heartbeat","proc_job","proc_job-tick-2"]']['payload']['text']) == 'proc_job/1\nproc_job/2\n'
    # A retry of the admitted replacement (its acknowledgement lost) is dropped, never folded again.
    assert await _drain_one(messaging.runner, tick(2)) == 0
    assert len(_automation_rows(messaging)) == 2


@pytest.mark.asyncio
async def test_a_tick_drained_during_a_restart_drain_leaves_the_committed_tick_queued(messaging):
    """A restart's drain refuses the replacement (``runtime_draining``): the committed tick the
    next owner runs stays queued, never cancelled for a replacement held only in memory."""
    tick = lambda seq: messaging.event('heartbeat', **_tick_fields('proc_job', seq))
    assert await _drain_one(messaging.runner, tick(1)) == 0
    messaging.runner._draining = True
    try:
        assert await _drain_one(messaging.runner, tick(2)) == 1, 'a refused tick is requeued'
    finally:
        messaging.runner._draining = False
    assert {key: (row['status'], row['outcome']) for key, row in _automation_rows(messaging).items()} == {
        '["heartbeat","proc_job","proc_job-tick-1"]': ('queued', None)}


@pytest.mark.asyncio
async def test_a_tick_whose_replaced_tick_starts_first_is_admitted_with_only_its_own_output(messaging, monkeypatch):
    authority = messaging.authority
    tick = lambda seq: messaging.event('heartbeat', **_tick_fields('proc_job', seq))
    assert await _drain_one(messaging.runner, tick(1)) == 0
    admit = authority.admit_automation

    async def claimed_first(adapter, event, identity, replaces=None):
        if replaces is not None:  # the drain claims the waiting tick between the check and the write
            # (The fixture's human message still heads the FIFO, so the claim is stamped directly.)
            with authority.db._lock:
                authority.db._conn.execute("UPDATE session_admissions SET status='started',owner_epoch=?,generation=1 "
                                           "WHERE admission_id=?", (authority.epoch, replaces))
                authority.db._conn.commit()
        return await admit(adapter, event, identity, replaces)
    monkeypatch.setattr(authority, 'admit_automation', claimed_first)

    assert await _drain_one(messaging.runner, tick(2)) == 0
    rows = _automation_rows(messaging)
    assert {key: row['status'] for key, row in rows.items()} == {
        '["heartbeat","proc_job","proc_job-tick-1"]': 'started', '["heartbeat","proc_job","proc_job-tick-2"]': 'queued'}
    assert _tick_output(rows['["heartbeat","proc_job","proc_job-tick-2"]']['payload']['text']) == 'proc_job/2\n'


@pytest.mark.asyncio
async def test_a_tick_admitted_while_its_completion_lands_never_queues_behind_it(messaging, monkeypatch):
    """The tick's check-and-admit and the completion's admit-and-withdraw are one critical section: a
    completion that lands while a checked tick is still on its way in withdraws it once it is in."""
    authority, gate, entered = messaging.authority, asyncio.Event(), asyncio.Event()
    admit = authority.admit_automation

    async def held(adapter, event, identity, **kwargs):
        if identity.startswith('["heartbeat"'):
            entered.set()
            await gate.wait()
        return await admit(adapter, event, identity, **kwargs)
    monkeypatch.setattr(authority, 'admit_automation', held)

    tick = asyncio.ensure_future(_drain_one(messaging.runner, messaging.event('heartbeat', **_tick_fields('proc_job', 1))))
    await asyncio.wait_for(entered.wait(), 10)
    exited = messaging.event('completion', exit_code=0, output='done')
    completion = asyncio.ensure_future(messaging.runner._inject_watch_notification('[proc_job finished]', exited))
    # Unserialized, the completion admits and withdraws (nothing yet) while the tick is held.
    await asyncio.wait({completion}, timeout=1)
    gate.set()
    assert (await tick, await completion) == (0, True)
    assert {key: row['outcome'] for key, row in _automation_rows(messaging).items()} == {
        '["heartbeat","proc_job","proc_job-tick-1"]': 'cancelled', '["completion","proc_job",1.0]': None}


@pytest.mark.asyncio
async def test_job_output_that_looks_like_a_cut_marker_keeps_the_earlier_output(messaging):
    """The omitted count rides the tick as its own field; text the job printed is never read as one."""
    literal = '...(10 earlier characters omitted)\nprinted by the job\n'
    for seq, output in ((1, 'proc_job/1\n'), (2, literal)):
        assert await _drain_one(messaging.runner, messaging.event(
            'heartbeat', **_tick_fields('proc_job', seq, output))) == 0
    folded = _automation_rows(messaging)['["heartbeat","proc_job","proc_job-tick-2"]']['payload']['text']
    assert _tick_output(folded) == 'proc_job/1\n' + literal


@pytest.mark.asyncio
async def test_a_late_tick_of_a_settled_batch_member_is_dropped(messaging):
    """A completion batch names only its primary in ``request_id``; the siblings live in the payload.
    Once the batch turn has finished, a late tick of a sibling process is still stale."""
    from gateway.session_automation import producer_identity
    batch = [messaging.event('completion', session_id=process, exit_code=0, output='done')
             for process in ('proc_a', 'proc_b')]
    primary = dict(batch[0], _automation_identities=[producer_identity(messaging.runner, e) for e in batch])
    assert await messaging.runner._inject_watch_notification('[2 finished]', primary) is True
    with messaging.authority.db._lock:
        messaging.authority.db._conn.execute(
            "UPDATE session_admissions SET status='terminal',outcome='completed' WHERE request_id=?",
            ('["completion","proc_a",1.0]',))
        messaging.authority.db._conn.commit()

    assert await _drain_one(messaging.runner, messaging.event(
        'heartbeat', session_id='proc_b', **_tick_fields('proc_b', 1))) == 0
    assert list(_automation_rows(messaging)) == ['["completion","proc_a",1.0]'], 'the late tick was admitted'


@pytest.mark.asyncio
async def test_a_served_profiles_bot_chat_takes_every_dm_with_no_client_attached(gateway):
    sender = gateway.connect(gateway.secondary)

    async def deliver(count):
        reply = await sender.dispatch({'id': count, 'method': 'bot_relay.deliver', 'params': {
            'id': f'{count:032x}', 'profile': 'beta', 'message': f'DM {count}', 'author': AUTHOR}})
        assert 'result' in reply, reply
        await gateway.settled(gateway.secondary, reply['result']['session_id'], count)
        return reply['result']['session_id']

    # The first DM mints the Bot Chat through the door; the second follows a turn that ran in it.
    chat = await deliver(1)
    assert await deliver(2) == chat

    # A turn in a served profile's local chat records the launch profile as its transport; a restart
    # reads the chat back over that row.
    db = gateway.secondary.db.db_path
    await gateway.stop()
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute('SELECT transport_profile FROM sessions WHERE id=?', (chat,)).fetchone() == ('default',)
    await gateway.start()
    sender = gateway.connect(gateway.secondary)
    assert [await deliver(count) for count in (3, 4)] == [chat, chat]

    frozen = ('beta-model', str(gateway.tmp_path))
    assert [(turn.profile, turn.chat, turn.text, turn.policy) for turn in gateway.turns] == [
        ('beta', chat, f'DM {count}', frozen) for count in (1, 2, 3, 4)]


@pytest.mark.asyncio
@pytest.mark.parametrize('origin', ['created', 'legacy'])
async def test_a_served_profiles_desktop_session_admits_its_own_process_notices(gateway, origin):
    secondary = gateway.secondary
    if origin == 'created':
        desktop = gateway.connect(secondary)
        sid = gateway.create(secondary, desktop, 'desktop', 'gui', model='desk-model').session_id
    else:
        # A row written by a Desktop before the session authority existed; resuming it adopts it.
        desktop = gateway.connect(secondary, provider='local', profile_id=secondary.profile_id,
                                  instance_id=secondary.instance_id)
        sid = secondary.db.create_session(LEGACY_DESKTOP_ID, source='desktop', model='desk-model',
                                          cwd=str(gateway.tmp_path))
    resumed = await desktop.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
    assert 'result' in resumed, resumed
    submitted = await desktop.dispatch({'id': 2, 'method': 'prompt.submit', 'params': {
        'session_id': sid, 'input_id': 'start', 'text': 'run the job in the background'}})
    assert 'result' in submitted, submitted
    await gateway.settled(secondary, sid, 1)

    match = gateway.process_event(secondary, sid, 'watch_match', event_id='match-1', pattern='READY')
    assert await gateway.drain(match) == [], 'the watch match was refused and requeued'
    completion = gateway.process_event(secondary, sid, 'completion', exit_code=0)
    assert await gateway.runner._deliver_completion_notification('[job finished]', completion) is True
    await gateway.settled(secondary, sid, 3)
    assert [(turn.profile, turn.chat, turn.policy) for turn in gateway.turns] == [
        ('beta', sid, ('desk-model', str(gateway.tmp_path)))] * 3
