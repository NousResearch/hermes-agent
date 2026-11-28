"""A spared admission is resumed, not sent through new-turn routing."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run_runtime, session_detached_workers as detached, session_route_recovery
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import LiveSession, SessionAuthority
from persistence.core.errors import AuthorityRefusal


@pytest.fixture
def reserved(tmp_path, monkeypatch):
    registry = SessionAuthorities(tmp_path, multiplexed=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='1')
    runner = SimpleNamespace(
        session_authorities=registry, _running=True, _draining=False,
        session_runtime_descriptor={'state': 'ready'},
        session_api=SimpleNamespace(task=SimpleNamespace(done=lambda: False)),
        session_ticket_store=SimpleNamespace(profile_ids=frozenset({str(tmp_path)})),
        _restored_source=lambda entry: source,
    )
    owner = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='successor', epoch=2, persistence=None)
    registry.add(tmp_path, owner, name='default')
    row = dict(admission_id='a', generation=3, principal_id='owner', payload={'text': 'already started'})
    record = dict(pid=42, birth=1, session_id='s', execution_id='admission-worker:a', generation=3, socket='unused')
    owner.detached_resume_slots['s'] = (row, record)
    monkeypatch.setattr(owner, '_publish_pending', lambda *_: None)
    monkeypatch.setattr('gateway.session_open.settled', AsyncMock())
    return owner, source, row, record


@pytest.mark.asyncio
@pytest.mark.parametrize('automation', [False, True], ids=['interactive', 'local-automation-no-viewer'])
async def test_resume_bypasses_inprocess_and_note_routes(reserved, automation, monkeypatch):
    owner, source, row, record = reserved
    if automation:
        row['payload']['local_automation_v1'] = {'identity': '["cron", "fixture"]'}
    owner.sessions['s'] = live = LiveSession(source, 'route')
    policy = SimpleNamespace(source='telegram', config=lambda: {'gateway': {'managed_workers': False}})
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda *_: policy)
    events = []

    async def note(_owner, _ref, _row):
        if automation:
            events.append('completion note')
            return 'note'
        return None

    async def inprocess(*args):
        events.append('in-process turn')
        return 'wrong producer'

    async def refuse_attach(authority, ref, claimed, candidate, worker_class):
        assert authority is owner and ref.session_id == 's'
        assert claimed is row and candidate is record
        events.append('attach attempt')
        raise AuthorityRefusal('managed_worker_lost')

    async def lose(authority, claimed, scope, recording):
        assert authority is owner and claimed is row and scope['execution_id'] == record['execution_id']
        events.append('native unknown')

    monkeypatch.setattr('gateway.session_completion_notes.note_if_unviewed', note)
    monkeypatch.setattr(owner.runner, '_handle_message', inprocess, raising=False)
    monkeypatch.setattr('gateway.session_ingress._admission_scope', lambda *a: _admission_scope())
    monkeypatch.setattr('gateway.session_a2a.forward_author', lambda *_: None)
    monkeypatch.setattr(detached, 'reattach', refuse_attach)
    monkeypatch.setattr(detached, 'kill_recorded', lambda candidate: events.append(('kill', candidate)))
    monkeypatch.setattr('gateway.session_managed_worker._settle_lost', lose)
    monkeypatch.setattr('gateway.session_unattended_unknown.settle_unattended_unknown', AsyncMock(return_value=False))
    monkeypatch.setattr(owner, '_pause', lambda *args: None)
    monkeypatch.setattr('gateway.session_settlement_recovery.settle_and_deliver', AsyncMock(return_value=False))
    detached.start_reserved_resume_slots(owner)
    await asyncio.wait_for(live.task, 5)
    assert events == ['attach attempt', ('kill', record), 'native unknown']
    assert not owner.executing


@asynccontextmanager
async def _admission_scope():
    yield


@pytest.mark.asyncio
async def test_reserved_fifo_starts_when_cold_route_restores(reserved, monkeypatch):
    owner, _, row, _ = reserved
    order = []

    async def execute_claim(_ref, live, claimed, **_kwargs):
        assert owner.sessions['s'] is live and claimed is row
        order.append('resume')
        return True

    async def drain(_ref):
        order.append('follower')

    monkeypatch.setattr(owner, '_run_claimed', execute_claim)
    monkeypatch.setattr(owner, '_drain', drain)
    monkeypatch.setattr(session_route_recovery, '_mount_plan', lambda *_: ('route', object(), object()))
    run_runtime._start_detached_resumes(owner.runner, owner)
    assert not owner.sessions and 's' in owner.detached_resume_slots
    ref = session_route_recovery.restore_routed_session(owner, 's')
    owner._schedule(ref)
    task = owner.sessions['s'].task
    assert task is not None, 'reservation still blocks the FIFO after the route was restored'
    session_route_recovery.restore_routed_session(owner, 's')
    assert owner.sessions['s'].task is task
    await asyncio.wait_for(task, 5)
    assert order == ['resume', 'follower']
    assert not owner.detached_resume_slots
