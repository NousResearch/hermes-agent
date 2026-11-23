"""Detached startup must not break the normal hot-serve lifecycle."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run_runtime, session_detached_workers
from gateway.session_authorities import SessionAuthorities


@pytest.fixture
def hot_serve(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    home.mkdir()
    registry = SessionAuthorities(tmp_path, multiplexed=True)
    runner = SimpleNamespace(
        session_authorities=registry, session_control_server=None,
        session_runtime_descriptor={'served_profiles': []},
        session_ticket_store=SimpleNamespace(profile_ids=frozenset()),
    )
    authority = SimpleNamespace(
        runner=runner, profile_id=str(home), retiring=False, sessions={},
        waiters={}, native_waiters=set(), hosted_room_service=None,
    )

    async def build(_runner, name, profile_home, *, register):
        registry.add(profile_home, authority, name=name)
        return authority

    monkeypatch.setattr(run_runtime, '_build_profile_authority', build)
    monkeypatch.setattr(run_runtime, 'recover_bot_mailbox', AsyncMock())
    monkeypatch.setattr(run_runtime, '_drop_retired_local_adapter', lambda _: None)
    monkeypatch.setattr('gateway.session_local_startup.recover_local_sessions_async', AsyncMock())
    monkeypatch.setattr('gateway.platforms.webhook_ingress.recover_webhook_finalizations', AsyncMock())
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda _: None)
    monkeypatch.setattr('gateway.session_hosted_service.start_ready_hosted_services', lambda _: None)
    return runner, authority, home


@pytest.mark.asyncio
async def test_default_hot_serve_publishes_without_reading_detach_dependencies(hot_serve):
    runner, authority, home = hot_serve
    assert await run_runtime.serve_profile_runtime(runner, 'secondary', home) is authority
    assert registry_contains(runner, authority, home)
    assert authority.profile_id in runner.session_ticket_store.profile_ids


def registry_contains(runner, authority, home):
    return runner.session_authorities.for_home(home) is authority


@pytest.mark.asyncio
async def test_failed_detached_start_withdraws_hot_served_profile(hot_serve, monkeypatch):
    runner, authority, home = hot_serve
    runner.session_runtime_descriptor['state'] = 'ready'
    runner._running, runner._draining = True, False
    runner.session_api = SimpleNamespace(task=SimpleNamespace(done=lambda: False))
    authority.detached_resume_slots = {'s': ({}, {})}

    def fail(owner):
        assert owner is authority
        assert authority.profile_id in runner.session_ticket_store.profile_ids
        raise RuntimeError('resume scheduling failed')

    monkeypatch.setattr(session_detached_workers, 'start_reserved_resume_slots', fail)
    with pytest.raises(RuntimeError, match='resume scheduling failed'):
        await run_runtime.serve_profile_runtime(runner, 'secondary', home)
    assert authority.retiring
    assert runner.session_authorities.for_home(home) is None
    assert runner.session_runtime_descriptor['served_profiles'] == []
    assert not runner.session_ticket_store.profile_ids
