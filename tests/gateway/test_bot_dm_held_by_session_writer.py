"""A Bot Chat DM the owner refuses before any admission, because another session writer holds the
chat (a pre-#106742 Desktop backend's lease on a legacy Bot Chat), is a definite, named no-send:
the tool says the message was NOT sent, and the locally pinned intent is settled so nothing replays
it. A refusal after the owner may have admitted stays ambiguous (#134889 MSG-19)."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
import pytest_asyncio

from hermes_state_runtime import RuntimeStoreError, list_session_admissions

CHAT = 'old-bot-chat'


@pytest_asyncio.fixture
async def held(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import gateway_runtime
    from hermes_cli.active_sessions import release_active_session, try_acquire_active_session
    from hermes_cli.gateway_client import rpc_error
    from hermes_state import SessionDB
    from tools import bot_live_delivery, bot_mode_dm

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    db = store._db
    # A Desktop Bot Chat from before the one-gateway runtime: adopted on its first delivery.
    db.create_session(CHAT, source='desktop', model='old', cwd=str(tmp_path))
    db.set_session_title(CHAT, 'Bot Chat')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'uid:1000', 'provider': 'local', 'profile_id': 'default', 'instance_id': 'fixture',
        'capabilities': ['session:read', 'session:submit', 'session:create']})
    monkeypatch.setattr(connection, '_sibling_for', lambda profile: None)
    authority._native_legacy_transports = {connection.actor.transport_id: connection.actor}
    executed = []

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    loop = asyncio.get_running_loop()
    # A test may hold the next refusal on its way back to the tool (``gate``), after the owner
    # answered and released its serialization: the window a concurrent same-id attempt races.
    gate = SimpleNamespace(event=None, reached=threading.Event())

    def rpc(method):
        def call(home, params):
            # What the tool's authority client speaks: the contract-checked dispatch, then the
            # client's own refusal mapping.
            frame = asyncio.run_coroutine_threadsafe(connection.dispatch(
                {'id': 1, 'method': method, 'params': params}), loop).result()
            if 'error' in frame:
                event, gate.event = gate.event, None
                if event is not None:
                    gate.reached.set()
                    event.wait(30)
                raise rpc_error(frame['error'])
            return frame['result']
        return call

    monkeypatch.setattr(bot_live_delivery, 'authority_delivery', rpc('bot_relay.deliver'))
    monkeypatch.setattr(bot_live_delivery, 'authority_settle', rpc('bot_relay.settle'))
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint', lambda home, timeout: SimpleNamespace(
        state='ready', endpoint=SimpleNamespace(instance_id='fixture')))
    dm_dir = tmp_path / 'dm'
    dm_dir.mkdir(mode=0o700)
    monkeypatch.setattr(bot_mode_dm, '_dm_dir', lambda: dm_dir)
    monkeypatch.setattr(bot_mode_dm, '_spawn_delivery', lambda command, label, **kwargs: '{}')
    # tui_gateway/session_lifecycle._claim_active_session_slot: the only 'desktop' lease writer.
    lease, refusal = try_acquire_active_session(session_id=CHAT, surface='desktop', config={},
        registry_home=tmp_path, metadata={'live_session_id': 'desk'}, track_liveness=True)
    assert lease is not None, refusal
    leases = [lease]

    def release():
        while leases:
            release_active_session(leases.pop())

    try:
        yield SimpleNamespace(authority=authority, db=db, home=tmp_path, dm_dir=dm_dir,
                              executed=executed, release=release, gate=gate)
    finally:
        release()
        db.close()


async def _send(held, text):
    """``message_agent``'s local send: pin the intent, then call the owner's door."""
    from tools import bot_mode_dm
    raw = await asyncio.to_thread(bot_mode_dm._start_delivery, ['hermes', '-p', 'default'], text, '@default',
                                  stdin_file=False, task_id=None, agent=None, profile_home=held.home)
    return json.loads(raw)


def _admissions(held):
    return [row['request_id'] for row in list_session_admissions(held.db, session_id=CHAT, pending_only=False)]


@pytest.mark.asyncio
async def test_a_dm_refused_while_another_writer_holds_the_chat_is_a_definite_no_send(held):
    from gateway.session_bot import recover_bot_deliveries
    from tools.bot_live_delivery import read_delivery_result

    refused = await _send(held, 'ping')
    assert refused.get('reason') == 'runtime_coordination_required', refused
    assert 'status' not in refused and 'NOT sent' in refused['error'], refused
    assert 'Do not resend' not in refused['error'] and 'session' in refused['error'], refused
    # The pinned intent is settled with its payload, and the owner holds the id's terminal
    # not-sent receipt: no runner, retry or sweep can replay it.
    assert sorted(path.name for path in held.dm_dir.iterdir()) == []
    receipt = read_delivery_result(held.home, refused['delivery_id'])
    assert receipt['refused'] is True and receipt['status'] == 'failed' and not receipt.get('admission_id'), receipt
    held.release()
    await recover_bot_deliveries(held.authority)
    assert _admissions(held) == [] and held.executed == []
    # Once the other writer closes, a new send is adopted and admitted exactly once.
    sent = await _send(held, 'ping again')
    assert sent.get('status') in {'queued', 'claimed', 'settled'}, sent
    assert sent['delivery_id'] != refused['delivery_id']
    assert _admissions(held) == ['bot:' + sent['delivery_id']]


@pytest.mark.asyncio
async def test_a_refusal_after_the_owner_may_have_admitted_stays_ambiguous(held, monkeypatch):
    held.release()
    real = type(held.authority).admit_automation

    async def committed_then_refused(self, *args):
        await real(self, *args)
        raise RuntimeStoreError('runtime_coordination_required')

    monkeypatch.setattr(type(held.authority), 'admit_automation', committed_then_refused)
    held.authority._schedule = lambda ref: None
    result = await _send(held, 'ping')
    assert result.get('status') == 'ambiguous' and 'Do not resend' in result['error'], result
    assert [path.name for path in held.dm_dir.glob('*.live.json')], 'the unknown admission keeps its intent'
    assert _admissions(held) == ['bot:' + result['delivery_id']]


def _resubmit(held, delivery_id, text):
    """A second attempt under the SAME delivery id and payload (the runner retrying the pinned
    intent), straight at the owner's door."""
    from tools.bot_live_delivery import deliver_to_live_owner, find_canonical_live_owner
    return deliver_to_live_owner(held.home, find_canonical_live_owner(held.home), text, delivery_id=delivery_id)


def _runner(held, dm_file):
    from tools import bot_mode_dm
    return bot_mode_dm._run_delivery(['hermes', '-p', 'default'], dm_file, stdin_file=False, profile_home=held.home)


@pytest.mark.asyncio
async def test_a_same_id_attempt_racing_a_delayed_refusal_never_coexists_with_not_sent(held):
    """The refusal leaves the owner's serialization, the lease is released and the pinned intent
    is resubmitted under the same id before the tool sees the refusal: the tool must not then say
    'NOT sent' (and drop the intent) for a delivery the owner admitted."""
    hold = held.gate.event = threading.Event()
    first = asyncio.create_task(_send(held, 'ping'))
    assert await asyncio.to_thread(held.gate.reached.wait, 30)
    [intent] = held.dm_dir.glob('*.live.json')
    delivery_id = json.loads(intent.read_text())['delivery_id']
    held.release()
    second = await asyncio.to_thread(_resubmit, held, delivery_id, 'ping')
    assert second.get('admission_id') and _admissions(held) == ['bot:' + delivery_id], second
    hold.set()  # the first refusal reaches the tool now
    answer = await first
    assert 'NOT sent' not in answer.get('error', ''), answer
    assert answer.get('status') in {'queued', 'claimed', 'settled', 'ambiguous'}, answer
    assert _admissions(held) == ['bot:' + delivery_id]


@pytest.mark.asyncio
async def test_a_not_sent_answer_is_the_delivery_ids_durable_disposition(held):
    """Once the tool said 'NOT sent', a later attempt under that id (an already-running runner
    retry) is answered with that disposition, never admitted."""
    refused = await _send(held, 'ping')
    assert 'NOT sent' in refused['error'], refused
    held.release()
    again = await asyncio.to_thread(_resubmit, held, refused['delivery_id'], 'ping')
    assert again.get('refused') is True and again['status'] == 'failed', again
    assert _admissions(held) == [] and held.executed == []


@pytest.mark.asyncio
async def test_the_runner_settles_a_held_refusal_with_its_payload(held, capsys):
    """The background runner's definite no-send removes the payload too: rerunning the same
    runner command after the writer closes can neither rebuild the intent nor admit the id."""
    from tools import bot_mode_dm
    dm_file = bot_mode_dm._write_dm_file('ping')
    assert await asyncio.to_thread(_runner, held, dm_file) == 1
    refused = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert 'NOT sent' in refused['error'], refused
    assert sorted(path.name for path in held.dm_dir.iterdir()) == []
    held.release()
    await asyncio.to_thread(_runner, held, dm_file)
    capsys.readouterr()
    assert _admissions(held) == [] and held.executed == []


@pytest.mark.asyncio
async def test_an_older_owner_without_the_marker_stays_ambiguous(held, monkeypatch):
    from gateway import session_bot

    async def old_owner(home, key, exc):
        return None

    monkeypatch.setattr(session_bot, '_held_by_another_writer', old_owner)
    result = await _send(held, 'ping')
    assert result.get('status') == 'ambiguous' and 'Do not resend' in result['error'], result
    assert [path.name for path in held.dm_dir.glob('*.live.json')], 'an unproven refusal keeps its intent'


@pytest.mark.asyncio
async def test_a_settled_receipt_states_only_what_settle_serialized(held):
    """``settle`` doesn't re-check the writer (it may release between the refusal and the settle),
    so the stored receipt says the id was settled as not sent, never that a writer held the chat."""
    from tools.bot_live_delivery import find_canonical_live_owner, read_delivery_result, settle_held_delivery
    held.release()
    delivery_id = 'd' * 64
    answer = await asyncio.to_thread(settle_held_delivery, held.home, find_canonical_live_owner(held.home),
                                     'ping', delivery_id=delivery_id)
    assert answer['refused'] is True and answer['reason'] == 'runtime_coordination_required', answer
    receipt = read_delivery_result(held.home, delivery_id)
    assert 'held' not in receipt['error'] and 'not sent' in receipt['error'], receipt
    assert _admissions(held) == [] and held.executed == []


@pytest.mark.asyncio
async def test_a_refusal_alone_leaves_the_id_open_for_a_same_id_retry(held):
    """Only the Bot DM tool's settle records a not-sent disposition. A producer that retries its
    queued record under the same id (the cron drain) after a held refusal still admits it once."""
    from hermes_cli.gateway_client import GatewayRPCError
    delivery_id = 'c' * 64
    with pytest.raises(GatewayRPCError) as refused:
        await asyncio.to_thread(_resubmit, held, delivery_id, 'cron report')
    assert str(refused.value) == 'runtime_coordination_required' and refused.value.data.get('admitted') is False
    held.release()
    admitted = await asyncio.to_thread(_resubmit, held, delivery_id, 'cron report')
    assert admitted.get('admission_id') and _admissions(held) == ['bot:' + delivery_id], admitted
