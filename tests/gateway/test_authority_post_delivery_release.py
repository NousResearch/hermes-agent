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
