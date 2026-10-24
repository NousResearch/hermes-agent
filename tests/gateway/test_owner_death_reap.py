"""Owner-death orphan reap: a client transport's close stops the admissions it can no longer stop.

The canonical Stop only exists if some client sends prompt.cancel / session.interrupt. A
client that submits a turn and then dies before its cancel lands leaves that turn running
with nobody left to stop it, renewing its lease. On transport close the gateway owns
stopping them: a grace window (a reload reconnects and attach() cancels the reap), then the
canonical Stop.
"""
import asyncio

import pytest

# Pre-import at collection time, before the home-I/O guard fixture installs (see
# test_prompt_cancel_started.py: the drain's lazy import of gateway.session_managed_worker
# pulls hermes_bootstrap, whose module-level probe kills a worktree drain mid-run).
import gateway.session_managed_worker

from gateway import session_controls, session_finite
from gateway.session_contract import Principal, SessionRef
from gateway.session_controls import AuthorityConnection
from tests.gateway.test_canonical_stop_hard_cancel import Agent
from tests.gateway.test_session_authority_cancel_settlement import (
    ACTOR, REF, _authority, _submit, _wire_turn_agent)

# The submit principal's subject must equal the connection's subject for the reap's ledger
# query to find its admissions: this identity shape makes authenticated_subject return 'human'.
HUMAN_IDENTITY = {'user_id': 'human', 'provider': 'local', 'profile_id': 'owned',
                  'instance_id': 'current', 'capabilities': ['session:read', 'session:submit',
                                                              'session:control']}


def _reap_task(authority):
    task = authority._pending_orphan_reaps[(ACTOR.subject, REF.session_id)]
    return asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_owner_death_reap_stops_the_started_turn(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    agent = Agent()
    authority.runner._cached_agent_for = {}.get
    claimed, stopped = asyncio.Event(), asyncio.Event()

    async def execute(authority, ref, row):
        _wire_turn_agent(authority, row['generation'], agent)
        claimed.set()
        await asyncio.wait_for(stopped.wait(), 5)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(session_controls, 'ORPHAN_REAP_GRACE_SECONDS', 0.05)

    with db:
        receipt = await _submit(authority, 'orphan-started')
        await asyncio.wait_for(claimed.wait(), 5)
        conn = AuthorityConnection(authority, object(), HUMAN_IDENTITY)
        # The submitting transport dies with the turn still running and no cancel ever sent.
        await conn.close()
        await _reap_task(authority)
        assert agent.stops == ['hard'], f'the orphaned turn kept running: {agent.stops}'
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        with db._read_ctx() as conn2:
            row = conn2.execute('SELECT status, outcome FROM session_admissions WHERE admission_id=?',
                                (receipt.admission_id,)).fetchone()
    # The load-bearing claim is the Stop delivery (agent.stops above); the ledger outcome depends
    # on how the turn reacts to it, which this fake execute does not model (the real worker's
    # The load-bearing claim is the Stop delivery (agent.stops above). The ledger outcome
    # depends on how the turn reacts to it, which this fake execute does not model.
    assert row['status'] == 'terminal', f'the orphaned admission never settled: {tuple(row)}'

@pytest.mark.asyncio
async def test_owner_death_reap_honours_a_reconnect(tmp_path, monkeypatch):
    """The desktop dashboard reloads, not dies: re-attach inside the grace keeps the turn."""
    db, authority = _authority(tmp_path, monkeypatch)
    agent = Agent()
    authority.runner._cached_agent_for = {}.get
    claimed = asyncio.Event()

    async def execute(authority, ref, row):
        _wire_turn_agent(authority, row['generation'], agent)
        claimed.set()
        await asyncio.sleep(0.05)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(session_controls, 'ORPHAN_REAP_GRACE_SECONDS', 0.05)

    with db:
        receipt = await _submit(authority, 'reload')
        await asyncio.wait_for(claimed.wait(), 5)
        conn = AuthorityConnection(authority, object(), HUMAN_IDENTITY)
        await conn.close()
        # Same principal, back before the grace expires: the reload re-attaches its session.
        reconnected = Principal(ACTOR.subject, 'owned',
                                frozenset({'session:read', 'session:control'}), 'reconnect')
        await authority.attach(reconnected, REF)
        assert (ACTOR.subject, REF.session_id) not in authority._pending_orphan_reaps
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        with db._read_ctx() as conn2:
            row = conn2.execute('SELECT status, outcome FROM session_admissions WHERE admission_id=?',
                                (receipt.admission_id,)).fetchone()
    assert agent.stops == [], f'the reload killed its own turn: {agent.stops}'
    assert (row['status'], row['outcome']) == ('terminal', 'completed')


@pytest.mark.asyncio
async def test_owner_death_reap_cancels_queued_admissions(tmp_path, monkeypatch):
    """A queued prompt of the dead submitter must not be claimed into a new ownerless turn."""
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    monkeypatch.setattr(session_controls, 'ORPHAN_REAP_GRACE_SECONDS', 0.05)

    with db:
        receipt = await _submit(authority, 'orphan-queued')
        conn = AuthorityConnection(authority, object(), HUMAN_IDENTITY)
        await conn.close()
        await _reap_task(authority)
        with db._read_ctx() as conn2:
            row = conn2.execute('SELECT status, outcome FROM session_admissions WHERE admission_id=?',
                                (receipt.admission_id,)).fetchone()
    assert (row['status'], row['outcome']) == ('terminal', 'cancelled'), \
        f'the dead submitter stayed queued for the drain to claim: {tuple(row)}'
