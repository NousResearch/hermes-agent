"""Real Output consumer on the lower Route/API NEW admission seam; no live execution."""
import asyncio
import json
import sqlite3
from dataclasses import replace
from types import MethodType

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.test_canonical_peer_text_admission import dispatch, run_request
from tests.gateway.test_peer_output_product import register_routes


@pytest.mark.asyncio
async def test_real_output_consumer_authorizes_new_on_exact_fenced_owner(files_target, monkeypatch):
    from gateway import session_peer_output
    from gateway.platforms import api_server_runs
    t = files_target
    checked, launches = [], []
    actual = session_peer_output.authorize_output_consent

    def observed(adapter, authority, shared, conn, token, value, policy, evidence):
        assert adapter is t.adapter and authority is t.authority
        assert conn is t.db._conn and conn.in_transaction and shared.in_transaction
        assert evidence.adapter is adapter and evidence.owner[:3] == (authority, t.db, authority.epoch)
        assert json.loads(evidence.record_json)['dispatch'] == value.as_mapping()
        for path in (t.home / 'shared-state.db', t.home / 'state.db'):
            with sqlite3.connect(path, timeout=0) as contender:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    contender.execute('BEGIN IMMEDIATE')
        result = actual(adapter, authority, shared, conn, token, value, policy, evidence)
        checked.append(json.loads(evidence.record_json))
        return result

    monkeypatch.setattr(session_peer_output, 'authorize_output_consent', observed)
    register_routes(t)
    session_peer_output.initialize_peer_output(t.adapter)
    issued = await invite(t)

    async def inert(adapter, launch, **kwargs):
        launches.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    value = dispatch(issued)
    response = await t.adapter._handle_runs(run_request(issued['grant'], value))
    assert response.status == 202, response.text
    await asyncio.sleep(0)
    assert len(checked) == len(launches) == 1
    row = launches[0][2]
    assert row['payload']['api_turn_v1']['output_consent'] == checked[0]
    replay = await t.adapter._handle_runs(run_request(issued['grant'], value))
    assert replay.status == 202 and json.loads(replay.text)['replayed'] is True
    assert json.loads(replay.text)['run_id'] == json.loads(response.text)['run_id']
    assert len(checked) == len(launches) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['absent', 'foreign', 'replacement', 'owner_db', 'registry',
                                   'evidence_owner', 'evidence_adapter', 'evidence_record', 'evidence_missing',
                                   'replacement_before_api', 'replacement_during_callback'])
async def test_output_consumer_drift_at_new_write_rolls_back_admission(files_target, monkeypatch, change):
    from gateway import session_api_turn, session_peer_output
    from gateway.platforms import api_server_runs
    t = files_target
    register_routes(t)
    session_peer_output.initialize_peer_output(t.adapter)
    issued = await invite(t)
    original = session_api_turn.admit_session_input
    entered = []
    if change == 'replacement_before_api':
        api_admit = session_api_turn.admit_api_turn

        def replace_before_api(*args, **kwargs):
            t.adapter._room_output_admission = MethodType(lambda *args: True, t.adapter)
            return api_admit(*args, **kwargs)
        monkeypatch.setattr(session_api_turn, 'admit_api_turn', replace_before_api)
    if change == 'evidence_missing':
        capture = session_peer_output.capture_output_consent

        def missing_at_probe(*args, **kwargs):
            return capture(*args, **kwargs) if kwargs.get('connection') is not None else None
        monkeypatch.setattr(session_peer_output, 'capture_output_consent', missing_at_probe)

    def interleave(*args, **kwargs):
        entered.append(True)
        assert bool(kwargs['payload']['api_turn_v1'].get('output_consent')) == (change != 'evidence_missing')
        if change == 'absent':
            t.adapter._room_output_admission = None
        elif change == 'foreign':
            t.adapter._room_output_admission = MethodType(session_peer_output.authorize_output_consent, object())
        elif change == 'replacement':
            t.adapter._room_output_admission = MethodType(lambda *args: True, t.adapter)
        elif change == 'owner_db':
            t.adapter._peer_output_owner = (t.authority, object(), *t.adapter._peer_output_owner[2:])
        elif change == 'registry':
            t.runner.session_authorities._by_key.clear()
        elif change not in {'evidence_missing', 'replacement_before_api'}:
            capture = session_peer_output.capture_output_consent

            def changed(*args, **kwargs):
                current = capture(*args, **kwargs)
                assert current is not None
                if change == 'replacement_during_callback':
                    t.adapter._room_output_admission = MethodType(lambda *args: True, t.adapter)
                    return current
                if change == 'evidence_owner':
                    return replace(current, owner=(object(), *current.owner[1:]))
                if change == 'evidence_adapter':
                    return replace(current, adapter=object())
                return replace(current, record_json=current.record_json + ' ')
            monkeypatch.setattr(session_peer_output, 'capture_output_consent', changed)
        return original(*args, **kwargs)

    monkeypatch.setattr(session_api_turn, 'admit_session_input', interleave)

    async def forbidden(*args, **kwargs):
        pytest.fail('refused output admission reached execution')
    monkeypatch.setattr(api_server_runs, '_execute_run', forbidden)
    response = await t.adapter._handle_runs(run_request(issued['grant'], dispatch(issued)))
    assert response.status in (409, 503), response.text
    assert entered == [True]
    assert t.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert t.db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
    assert t.adapter._run_idempotency_store._conn.execute('SELECT count(*) FROM run_idempotency').fetchone()[0] == 0
    assert not t.adapter._active_run_tasks
