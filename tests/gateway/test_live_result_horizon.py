"""A live session's settled results hold their transcript copies for the exact-retry horizon, not forever (LC-9).

Each turn runs through API admission, the owner's drain and settlement; the model is stubbed at
the runner's message handler. The wall clock moves forward between turns, as days would pass.
"""
import asyncio
import json
import time

import pytest

from gateway.session_api_turn import admit_api_turn, observe_api_turn
from hermes_state_mutation_retirement import RETRY_HORIZON_S
from hermes_state_runtime import RuntimeStoreError

PAD = 'x' * 4000
HOUR = 3600
_PREFIX = 'gateway.admission.result.v1.'


@pytest.fixture
def clock(monkeypatch):
    offset = [0.0]
    real = time.time
    monkeypatch.setattr(time, 'time', lambda: real() + offset[0])
    return offset


def _admit(api, session_id, request_id):
    return admit_api_turn(api, session_id=session_id, request_id=request_id,
                          user_message=request_id, conversation_history=[])


async def _turn(api, owner, session_id, request_id, result):
    admitted = _admit(api, session_id, request_id)

    async def handle(event):
        from gateway.session_results import execution_result
        execution_result.get()['result'] = json.loads(json.dumps(result))
        return result.get('final_response', '')
    owner.runner._handle_message = handle
    await asyncio.wait_for(owner._drain(admitted[1]), 10)
    await _maintenance(owner)
    return admitted


async def _maintenance(owner):
    """The background work a settlement spawned (the drain never awaits it)."""
    from gateway.session_runtime_workers import mutation_tasks
    await asyncio.wait_for(asyncio.gather(*mutation_tasks(owner)), 10)


async def _expire(api, owner, clock):
    """A settlement in another session one retry horizon later: its post-settlement pass runs."""
    handler = owner.runner._handle_message
    clock[0] += RETRY_HORIZON_S + HOUR
    try:
        await _turn(api, owner, 'gc-tick', str(clock[0]), {'final_response': 'tick', 'completed': True})
    finally:
        owner.runner._handle_message = handler


def _stored(owner):
    with owner.db._read_ctx() as conn:
        return {key[len(_PREFIX):]: raw for key, raw in conn.execute(
            'SELECT key, value FROM state_meta WHERE key GLOB ?', (_PREFIX + '*',))}


@pytest.mark.asyncio
async def test_live_session_results_shed_their_transcript_copies_past_the_retry_horizon(api, owner, clock):
    from gateway.platforms.api_server_authority_runs import run_projection
    ids, first = [], []
    for index in range(5):
        clock[0] = index * 2 * HOUR  # turns two hours apart: the post-settlement pass runs at most hourly
        reply = f'reply-{index}-' + PAD + ' café 中文 "quoted" \\'
        admitted = await _turn(api, owner, 'live', f'turn-{index}', {
            'final_response': reply, 'completed': True, 'estimated_cost_usd': 1e-05,
            'messages': [{'role': 'assistant', 'content': reply}]})
        ids.append(admitted[2]['admission_id'])
        first.append(await observe_api_turn(admitted))
    fresh = _stored(owner)
    # The reply and its transcript copy: what made a live session's ledger grow per turn.
    assert all(fresh[aid].count(PAD) == 2 for aid in ids)

    # Seven days and five hours after the first turn: turns 0-2 are past the horizon, 3-4 inside it.
    clock[0] = RETRY_HORIZON_S + 5 * HOUR
    await _turn(api, owner, 'live', 'turn-5', {'final_response': 'reply-5', 'completed': True})
    stored = _stored(owner)
    for index, aid in enumerate(ids[:3]):
        saved, before = json.loads(stored[aid]), json.loads(fresh[aid])
        # The one reply stays byte for byte, with its flags and accounting; the transcript copy goes.
        assert saved['result']['final_response'] == before['result']['final_response'] == first[index][0]['final_response']
        assert stored[aid].count(PAD) == 1 and 'messages' not in saved['result']
        assert {k: saved['result'][k] for k in ('completed', 'estimated_cost_usd')} == {'completed': True, 'estimated_cost_usd': 1e-05}
        assert len(stored[aid]) - len(json.dumps(saved['result']['final_response'], ensure_ascii=False)) < 512
    assert [stored[aid] for aid in ids[3:]] == [fresh[aid] for aid in ids[3:]]

    # Inside the horizon an exact retry replays the settled reply byte for byte.
    again = await observe_api_turn(_admit(api, 'live', 'turn-3'))
    assert json.dumps(again, sort_keys=True) == json.dumps(first[3], sort_keys=True)
    # Past it the retry neither re-runs nor replays an empty reply as the answer: the key is spent,
    # refused at admission, before any door prepares a response.
    with pytest.raises(RuntimeStoreError) as refused:
        _admit(api, 'live', 'turn-0')
    assert refused.value.reason == 'admission_conflict'
    projection = run_projection(api, 'turn-0')
    assert (projection['status'], projection['output'], projection['result_reclaimed']) == (
        'completed', first[0][0]['final_response'], True)
    assert 'result_reclaimed' not in run_projection(api, 'turn-3')


@pytest.mark.asyncio
async def test_bot_receipts_keep_their_own_answer_past_the_horizon(api, owner, clock, tmp_path):
    from gateway.session_bot import _result, _retry_identity
    settled, failed, retried = 'a' * 32, 'b' * 32, 'c' * 32
    transient = {'final_response': 'The model failed.', 'failed': True,
                 'error': 'Error code: 503 - upstream unavailable', 'failure_reason': None}
    records = {}
    turns = ((settled, {'final_response': 'the reply ' + PAD, 'completed': True}), (failed, transient),
             (retried, transient))
    for index, (key, result) in enumerate(turns):
        clock[0] = index * 2 * HOUR
        _, _, row = await _turn(api, owner, 'bot', 'bot:' + key, result)
        records[key] = dict(status='canonical', admission_id=row['admission_id'], delivery_id=key,
                            profile_home=str(tmp_path), session_id='bot', message=key, principal_id='api')
    clock[0] = 6 * HOUR
    await _turn(api, owner, 'bot', _retry_identity(retried), {'final_response': 'retried ' + PAD, 'completed': True})
    fresh = {key: _result(owner, record) for key, record in records.items()}
    assert [fresh[key]['status'] for key in (settled, failed, retried)] == ['settled', 'failed', 'settled']
    assert fresh[retried]['reply'] == 'retried ' + PAD and fresh[failed]['error'] == transient['error']
    # What each receipt file holds once its delivery settled.
    held = {key: {**records[key], **fresh[key]} for key in records}

    clock[0] = RETRY_HORIZON_S + 8 * HOUR
    await _turn(api, owner, 'bot', 'bot:later', {'final_response': 'later', 'completed': True})
    stored = _stored(owner)
    assert 'reclaimed_at' in stored[records[settled]['admission_id']]
    assert sum('reclaimed_at' in raw for raw in stored.values()) == 2  # the settled turn and the retry
    # Owner start re-reads every receipt: each keeps its own answer.
    assert {key: _result(owner, held[key]) for key in held} == fresh


@pytest.mark.asyncio
async def test_pending_bot_receipt_of_a_reduced_turn_still_gets_its_reply(api, owner, clock, tmp_path):
    from gateway.session_bot import _result
    key = 'e' * 32
    _, _, row = await _turn(api, owner, 'bot', 'bot:' + key, {'final_response': 'the reply ' + PAD, 'completed': True})
    # The receipt was never refreshed after the turn settled (its owner was down for the week).
    pending = dict(status='canonical', admission_id=row['admission_id'], delivery_id=key,
                   profile_home=str(tmp_path), session_id='bot', message=key, principal_id='api')
    await _expire(api, owner, clock)
    assert 'reclaimed_at' in _stored(owner)[row['admission_id']]
    assert (_result(owner, pending)['status'], _result(owner, pending)['reply']) == ('settled', 'the reply ' + PAD)


@pytest.mark.asyncio
async def test_interrupted_and_failed_results_keep_their_diagnosis(api, owner, clock, tmp_path):
    """An interrupted turn explains itself in ``final_response`` with no ``error``; Bot receipts
    derive their error from it, so such results are never reduced."""
    from gateway.session_bot import _result
    records = {}
    for key, result in (('d' * 32, {'interrupted': True, 'completed': False,
                                    'final_response': 'Stopped before this turn started. ' + PAD}),
                        ('f' * 32, {'failed': True, 'final_response': 'The model failed. ' + PAD})):
        _, _, row = await _turn(api, owner, 'bot', 'bot:' + key, result)
        records[key] = dict(status='canonical', admission_id=row['admission_id'], delivery_id=key,
                            profile_home=str(tmp_path), session_id='bot', message=key, principal_id='api')
    original = {key: _result(owner, record) for key, record in records.items()}
    assert original['d' * 32]['error'].startswith('Stopped before this turn started.')
    held = {key: {**records[key], **original[key]} for key in records}
    # Unfinished turns that still settle ``completed`` (partial, iteration budget) keep theirs too.
    unfinished = [(await _turn(api, owner, 'live', name, {'final_response': name + PAD, **flags}))[2]['admission_id']
                  for name, flags in (('partial', {'partial': True}), ('budget', {'completed': False}))]
    await _expire(api, owner, clock)
    assert {key: _result(owner, held[key]) for key in held} == original
    stored = _stored(owner)
    assert all(PAD in stored[aid] for aid in unfinished)


@pytest.mark.asyncio
async def test_expired_streaming_responses_retry_is_refused_before_the_stream(api, owner, clock):
    """An Idempotency-Key retry whose replay body was evicted and whose result was reduced: refused
    with 409 before SSE starts, so no failed snapshot replaces the original response."""
    from aiohttp.test_utils import TestClient, TestServer
    from tests.gateway.test_responses_identity_retention import _app
    calls = []
    store = api._current_response_store()
    headers = {'Idempotency-Key': 'original'}
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        first = await client.post('/v1/responses', json={'input': 'work'}, headers=headers)
        original = await first.json()
        await _maintenance(owner)
        # The replay body is evicted; the identity and the chain snapshot survive.
        store._conn.execute("DELETE FROM responses WHERE response_id LIKE 'idem:%'")
        store._conn.commit()
        await _expire(api, owner, clock)
        for stream in (True, False):
            retry = await client.post('/v1/responses', json={'input': 'work', 'stream': stream}, headers=headers)
            body = await retry.text()
            assert retry.status == 409 and 'admission_conflict' in body, (stream, retry.status, body)
        saved = store.get(original['id'])
    assert calls == ['work']
    assert saved['response'] == original


def _evict_replay_bodies(store):
    store._conn.execute("DELETE FROM responses WHERE response_id LIKE 'idem:%'")
    store._conn.commit()


@pytest.mark.asyncio
async def test_responses_replay_streams_the_snapshot_read_at_admission(api, owner, clock, monkeypatch):
    """A reclaim pass between the admission check and the observation cannot turn the replay into
    a failure over the original response: the replay streams what admission read."""
    import hermes_state_terminal as terminal
    from aiohttp.test_utils import TestClient, TestServer
    from tests.gateway.test_responses_identity_retention import _app
    calls = []
    store = api._current_response_store()
    headers = {'Idempotency-Key': 'original'}
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        original = await (await client.post('/v1/responses', json={'input': 'work'}, headers=headers)).json()
        await _maintenance(owner)
        snapshot = store.get(original['id'])
        _evict_replay_bodies(store)
        clock[0] += RETRY_HORIZON_S + HOUR
        spawn = api._spawn_stream_agent

        def reclaim_then_spawn(*args, **kwargs):
            owner.db._execute_write(lambda conn: terminal.reclaim_settled_results(conn, now=time.time(), limit=200))
            return spawn(*args, **kwargs)
        monkeypatch.setattr(api, '_spawn_stream_agent', reclaim_then_spawn)
        retry = await client.post('/v1/responses', json={'input': 'work', 'stream': True}, headers=headers)
        wire = await retry.text()
        assert retry.status == 200 and 'response.completed' in wire and 'answer work' in wire, wire
        assert store.get(original['id']) == snapshot
    assert calls == ['work']


@pytest.mark.asyncio
async def test_a_failed_responses_replay_never_replaces_the_original_snapshot(api, owner, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer
    from gateway import session_api_turn
    from tests.gateway.test_responses_identity_retention import _app
    calls = []
    store = api._current_response_store()
    headers = {'Idempotency-Key': 'original'}
    async with TestClient(TestServer(_app(api, owner, calls))) as client:
        original = await (await client.post('/v1/responses', json={'input': 'work'}, headers=headers)).json()
        snapshot = store.get(original['id'])
        _evict_replay_bodies(store)

        async def unavailable(*args, **kwargs):
            raise RuntimeStoreError('storage_unavailable')
        monkeypatch.setattr(session_api_turn, 'observe_api_turn', unavailable)
        retry = await client.post('/v1/responses', json={'input': 'work', 'stream': True}, headers=headers)
        assert 'response.failed' in await retry.text()
        assert store.get(original['id']) == snapshot
    assert calls == ['work']


@pytest.mark.asyncio
async def test_a_retry_repairs_the_snapshot_an_original_observer_failure_wrote(api, owner, monkeypatch):
    """The turn completed but the original stream's result read failed (``failed`` snapshot). A
    retry that replays the committed result repairs it, and ``previous_response_id`` chains on it."""
    import sqlite3
    from aiohttp.test_utils import TestClient, TestServer
    from gateway import session_api_turn as turns
    from tests.gateway.test_responses_identity_retention import _app
    calls, histories = [], []
    app = _app(api, owner, calls)
    handle = owner.runner._handle_message

    async def record(event):
        histories.append(list(turns.api_execution.get()['history']))
        return await handle(event)
    owner.runner._handle_message = record
    read_result = turns._settled_api_result

    def unavailable(*args):
        result, _ = read_result(*args)
        assert result['final_response'] == 'answer work'
        raise sqlite3.OperationalError('disk I/O error')
    headers = {'Idempotency-Key': 'work'}
    body = {'input': 'work', 'stream': True}
    store = api._current_response_store()
    async with TestClient(TestServer(app)) as client:
        with monkeypatch.context() as patch:
            patch.setattr(turns, '_settled_api_result', unavailable)
            wire = await (await client.post('/v1/responses', json=body, headers=headers)).text()
        assert 'response.failed' in wire
        rid = next(json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: '))['response']['id']
        assert store.get(rid)['response']['status'] == 'failed'
        assert 'response.completed' in await (await client.post('/v1/responses', json=body, headers=headers)).text()
        chained = await client.post('/v1/responses', json={'input': 'next', 'previous_response_id': rid})
        assert chained.status == 200, await chained.text()
        await chained.text()
        await _maintenance(owner)
    assert calls == ['work', 'next']
    assert (store.get(rid)['response']['status'], histories[-1][-1]) == (
        'completed', {'role': 'assistant', 'content': 'answer work'})
