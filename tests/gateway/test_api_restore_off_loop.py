"""The API admission's route restore must not wait on the owner loop."""
import asyncio

import pytest

from gateway.session_api_turn import admit_api_turn_async


@pytest.mark.asyncio
async def test_api_admission_keeps_owner_loop_live_while_restoring_route(api, owner, monkeypatch):
    loop = asyncio.get_running_loop()
    store = owner.runner.session_store
    route_lock = store._lock
    waiting = asyncio.Event()

    class ObservedLock:
        def __enter__(self):
            loop.call_soon_threadsafe(waiting.set)
            # A regression must fail rather than hang the test runner.
            assert route_lock.acquire(timeout=3), 'API route lock blocked the owner loop'

        def __exit__(self, *_):
            route_lock.release()

    monkeypatch.setattr(store, '_lock', ObservedLock())
    route_lock.acquire()
    admission = asyncio.create_task(admit_api_turn_async(
        api, session_id='restore', user_message='hello', conversation_history=[]))
    try:
        await asyncio.wait_for(waiting.wait(), 10)
        assert not admission.done(), 'the owner loop could not run while restore waited on the route lock'
        assert 'restore' not in owner.sessions
    finally:
        route_lock.release()
        # Drain the task before the fixture closes its store, including on regression.
        await asyncio.wait_for(admission, 10)
    _, ref, row = admission.result()
    assert ref.session_id == row['target_session_id'] == 'restore'
    assert store._entries[owner.sessions['restore'].route].session_id == 'restore'


@pytest.mark.asyncio
async def test_delete_during_api_restore_does_not_remount_session(api, owner, monkeypatch):
    import threading

    from gateway import session_api
    from gateway.session_ingress_media import native_inputs_root
    from gateway.session_mutations import retire_live_sessions
    from persistence.core.errors import AuthorityRefusal

    session_id = 'deleted-during-restore'
    session_api.bind_api_session(owner, session_id)
    store = owner.runner.session_store
    route = owner.sessions[session_id].route
    loop = asyncio.get_running_loop()
    route_installed = asyncio.Event()
    publish = threading.Event()
    restore_route = session_api._restore_api_route

    def pause_after_install(*args):
        entry = restore_route(*args)
        loop.call_soon_threadsafe(route_installed.set)
        assert publish.wait(timeout=10), 'test never released the restore worker'
        return entry

    monkeypatch.setattr(session_api, '_restore_api_route', pause_after_install)
    admission = asyncio.create_task(admit_api_turn_async(
        api, session_id=session_id, user_message='hello', conversation_history=[]))
    try:
        await asyncio.wait_for(route_installed.wait(), 10)
        removed = owner.persistence.sessions.delete(session_id, media_root=str(native_inputs_root()))
        assert session_id in removed
        retire_live_sessions(owner, removed)
    finally:
        publish.set()
        with pytest.raises(AuthorityRefusal, match='not_found'):
            await asyncio.wait_for(admission, 10)
    assert session_id not in owner.sessions
    assert route not in store._entries
