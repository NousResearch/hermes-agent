"""Post-delivery callbacks of turns the session authority runs are released after the reply.

A local (Desktop/TUI/CLI) or receipt-only turn is executed by the authority's drain, not by an
adapter's message task, so the adapter-side release in ``_process_message_background`` never ran:
each such session kept its last turn's callback (the background-review release, closing over the
whole turn context) for the life of the gateway (~250 KB per session on a live gateway), and
notices held for that release were never sent.
"""
import pytest

from tests.gateway.test_session_end_delegations import _deliver, _local, _PROCESS


def _turn_registering_a_callback(runner, authority, ref, fired):
    """A turn that, like TurnRunner, registers a release on its delivery adapter mid-turn."""
    live = authority.sessions[ref.session_id]
    adapter = runner._delivery_adapter_for(live.source)

    async def handle(event):
        adapter.register_post_delivery_callback(live.route, lambda: fired.append(event.text), generation=7)
        return 'ok'
    runner._handle_message = handle
    return adapter


@pytest.mark.asyncio
async def test_an_authority_run_turn_fires_and_clears_its_post_delivery_callback(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        authority.sessions[ref.session_id].subscribers['watching'] = actor
        fired = []
        adapter = _turn_registering_a_callback(runner, authority, ref, fired)
        await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert fired == [f'[finished] RESULT_{_PROCESS[1]}'], 'the release runs once the turn settled'
        assert adapter._post_delivery_callbacks == {}, 'nothing of the turn outlives it'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_turn_that_never_settles_drops_its_callback_unfired(tmp_path, monkeypatch):
    from gateway import session_authority
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        authority.sessions[ref.session_id].subscribers['watching'] = actor
        fired = []
        adapter = _turn_registering_a_callback(runner, authority, ref, fired)

        async def unknown(self, *args, **kwargs):
            raise RuntimeError('settlement store unavailable')
        monkeypatch.setattr(session_authority, 'settle_with_retry', unknown, raising=False)
        monkeypatch.setattr('gateway.session_settlement_recovery.settle_with_retry', unknown)
        monkeypatch.setattr('gateway.session_settlement_recovery.recover_failed_settlement',
                            lambda *a, **k: (_ for _ in ()).throw(session_authority.RuntimeStoreError('moved')))
        await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert fired == [], 'no reply followed, so nothing announces after it'
        assert adapter._post_delivery_callbacks == {}
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_turn_that_raises_drops_its_callback_unfired(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        authority.sessions[ref.session_id].subscribers['watching'] = actor
        fired = []
        live = authority.sessions[ref.session_id]
        adapter = runner._delivery_adapter_for(live.source)

        async def handle(event):
            adapter.register_post_delivery_callback(live.route, lambda: fired.append(event.text), generation=7)
            raise RuntimeError('agent crashed mid-turn')
        runner._handle_message = handle
        await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert fired == [] and adapter._post_delivery_callbacks == {}
        # A registration arriving after the turn ended (an abandoned executor thread) is refused.
        from gateway.platforms.base_post_delivery import close_owed_slots
        from gateway.session_results import execution_result
        token = execution_result.set({})
        try:
            close_owed_slots(execution_result.get())
            adapter.register_post_delivery_callback(live.route, lambda: None, generation=8)
        finally:
            execution_result.reset(token)
        assert adapter._post_delivery_callbacks == {}
    finally:
        authority.db.close()


def test_a_registration_racing_the_turn_end_is_either_owed_or_refused():
    """An executor thread mid-registration when the drain closes the turn: its slot must land in
    the closing snapshot (so it is released) or be refused, never be added after the snapshot."""
    import threading
    from gateway.platforms.base_post_delivery import PostDeliveryMixin, close_owed_slots
    from gateway.session_results import execution_result

    entered, proceed = threading.Event(), threading.Event()

    class PausingSlots(dict):
        def get(self, key, default=None):
            entered.set()
            proceed.wait(5)
            return super().get(key, default)

    adapter = object.__new__(PostDeliveryMixin)
    adapter._post_delivery_callbacks = PausingSlots()
    captured = {}

    def register():
        token = execution_result.set(captured)
        try:
            adapter.register_post_delivery_callback('route', lambda: None, generation=1)
        finally:
            execution_result.reset(token)

    worker = threading.Thread(target=register)
    worker.start()
    assert entered.wait(5)
    closing = {}
    closer = threading.Thread(target=lambda: closing.setdefault('slots', close_owed_slots(captured)))
    closer.start()
    closer.join(0.3)
    proceed.set()
    worker.join(5)
    closer.join(5)
    in_registry = 'route' in adapter._post_delivery_callbacks
    in_snapshot = any(key == 'route' for _adapter, key, _gen in closing['slots'])
    assert in_registry == in_snapshot, 'a slot in the registry must be one the drain will release'
