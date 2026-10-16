"""Local divergence: continuing a historical messaging row adopts it as a local session (TUI-8, #134889).

Opt-in through ``gateway.session.viewer_replies_stay_local`` (set by ``build_lane``). A viewer's submission to a Telegram row its lane's route has moved past binds the row as the viewer's own
local session (``gateway/session_viewer_adoption.py``): the turn lands in that row, answers in the panel
and leaves the lane alone. The original identity is kept for ``scripts/local_adopt_rollback.py``.
"""
import importlib.util
import json
import logging
from pathlib import Path
import uuid

import pytest
import pytest_asyncio

from gateway.config import Platform
from tests.gateway.test_authority_viewer_turn import _telegram, build_lane

ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture
async def lane(tmp_path, monkeypatch):
    return await build_lane(tmp_path, monkeypatch)


@pytest_asyncio.fixture
async def moved(lane, monkeypatch):
    """The lane after ``/new``: ``lane.ref`` is a historical row, ``current`` the lane's session."""
    from gateway import run
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'frozen')
    lane.current = lane.runner.session_store.get_or_create_session(_telegram(), force_new=True)
    lane.db.end_session(lane.ref.session_id, 'session_reset')  # historical rows are ended
    lane.original = lane.db.get_session(lane.ref.session_id)
    return lane


def _meta(db, key):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    return None if row is None else json.loads(row[0])


@pytest.mark.asyncio
async def test_continuing_a_historical_row_adopts_it_and_stays_in_the_panel(moved, caplog):
    sid = moved.ref.session_id
    with caplog.at_level(logging.WARNING, logger='gateway.session_viewer_adoption'):
        await moved.submit(moved.viewer, moved.ref, 'pick this up again')
        await moved.settle(moved.ref)

    live = moved.authority.sessions[sid]
    assert live.source.platform == Platform.LOCAL
    assert [turn.session_id for turn in moved.runner.turns] == [sid]
    assert moved.db.message_count(sid) == 1
    assert moved.db.message_count(moved.current.session_id) == 0
    assert moved.runner.bot.sent == []
    row = moved.db.get_session(sid)
    assert (row['source'], row['session_key'], row['chat_id']) == ('telegram', live.route, sid)
    assert live.route != moved.original['session_key']
    preimage = _meta(moved.db, 'local_adopt_preimage:' + sid)
    assert preimage['columns']['session_key'] == moved.original['session_key']
    assert preimage['actor'] == moved.viewer.subject
    assert any(sid in record.getMessage() and moved.original['session_key'] in record.getMessage()
               for record in caplog.records)


@pytest.mark.asyncio
async def test_adoption_happens_once(moved):
    sid = moved.ref.session_id
    await moved.submit(moved.viewer, moved.ref, 'one')
    await moved.settle(moved.ref)
    receipt = _meta(moved.db, 'gateway.local_policy.v1:' + sid)
    await moved.submit(moved.viewer, moved.ref, 'two')
    await moved.settle(moved.ref)

    assert [turn.session_id for turn in moved.runner.turns] == [sid, sid]
    assert _meta(moved.db, 'gateway.local_policy.v1:' + sid) == receipt
    assert _meta(moved.db, 'local_adopt_preimage:' + sid)['columns']['session_key'] == moved.original['session_key']
    assert moved.db.message_count(moved.current.session_id) == 0


@pytest.mark.asyncio
async def test_two_concurrent_submissions_adopt_once(moved):
    """Both wait on the mount's mutation lock (held by e.g. a rewind); the second finds it adopted."""
    import asyncio
    sid = moved.ref.session_id
    mount = moved.authority.sessions[sid]
    await mount.mutation_lock.acquire()
    submits = [asyncio.create_task(moved.submit(moved.viewer, moved.ref, text)) for text in ('a', 'b')]
    while len(mount.mutation_lock._waiters or ()) < 2:
        await asyncio.sleep(0)
    mount.mutation_lock.release()
    await asyncio.wait_for(asyncio.gather(*submits), 10)
    await moved.settle(moved.ref)
    assert [turn.session_id for turn in moved.runner.turns] == [sid, sid]
    assert moved.db.message_count(moved.current.session_id) == 0


@pytest.mark.asyncio
async def test_without_read_the_turn_is_refused_visibly_before_any_adoption(moved):
    """Adoption hands the actor the whole transcript: a submit-only principal never enters it; the
    submission gate refuses its turn as for any row the lane moved past."""
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    writer = Principal('viewer', 'fixture', frozenset({'session:submit'}), 'socket')
    with pytest.raises(RuntimeStoreError) as refused:
        await moved.submit(writer, moved.ref, 'blind write')
    assert refused.value.reason == 'session_not_current'
    assert moved.authority.sessions[moved.ref.session_id].source.platform == Platform.TELEGRAM
    assert _meta(moved.db, 'local_adopt_preimage:' + moved.ref.session_id) is None
    assert moved.runner.turns == []


@pytest.mark.asyncio
async def test_input_without_the_operator_envelope_never_adopts(moved):
    """The lane's own sender (platform input, relay, automation acting as the chat) carries no
    ``local_operator_v1``: its submission to the historical row runs as the lane does, the row keeps
    its identity and no preimage is written."""
    sid = moved.ref.session_id
    await moved.submit(moved.owner, moved.ref, 'from the chat itself')
    await moved.settle(moved.ref)

    assert moved.authority.sessions[sid].source.platform == Platform.TELEGRAM
    assert _meta(moved.db, 'local_adopt_preimage:' + sid) is None
    row = moved.db.get_session(sid)
    for column in ('source', 'session_key', 'chat_id', 'user_id'):
        assert row[column] == moved.original[column], column
    assert [turn.session_id for turn in moved.runner.turns] == [moved.current.session_id]


@pytest.mark.asyncio
async def test_a_row_its_route_still_names_is_never_adopted(lane):
    sid = lane.ref.session_id
    await lane.submit(lane.viewer, lane.ref, 'current lane')
    await lane.settle(lane.ref)
    assert lane.authority.sessions[sid].source.platform == Platform.TELEGRAM
    assert _meta(lane.db, 'local_adopt_preimage:' + sid) is None


@pytest.mark.asyncio
async def test_adoption_itself_refuses_a_row_its_route_still_names(lane, monkeypatch):
    """The adoption entry point's own fence, independent of the submit-time trigger."""
    from gateway import run
    from gateway.session_viewer_adoption import adopt_viewer_row
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'frozen')
    with pytest.raises(RuntimeStoreError, match='not_found'):
        adopt_viewer_row(lane.authority, lane.ref.session_id, lane.viewer)
    assert lane.db.get_session(lane.ref.session_id)['source'] == 'telegram'
    assert _meta(lane.db, 'local_adopt_preimage:' + lane.ref.session_id) is None


@pytest.mark.asyncio
async def test_adoption_keeps_the_mount_its_locks_and_pending_prompts(moved):
    """The messaging mount becomes the local one in place: subscribers, event stream, locks and an
    already registered prompt all stay valid for whoever holds them."""
    sid = moved.ref.session_id
    snapshot = await moved.authority.attach(moved.viewer, moved.ref)
    before = moved.authority.sessions[sid]
    locks, stream, controls = (before.mutation_lock, before.claim_gate), before.event_stream, before.controls
    controls.register(sid, before.route, 7, {'request_id': 'held-prompt', 'command': 'ls', 'description': 'list'})
    import asyncio
    moved.runner.hold = asyncio.Event()
    await moved.submit(moved.viewer, moved.ref, 'still watching')
    await asyncio.wait_for(moved.runner.started.wait(), 10)

    after = moved.authority.sessions[sid]
    assert after is before and after.source.platform == Platform.LOCAL
    assert (after.mutation_lock, after.claim_gate) == locks and after.controls is controls
    assert 'held-prompt' in after.controls.pending  # read while the adopted turn runs
    assert after.subscribers == {snapshot.subscription_id: moved.viewer}
    assert after.event_stream is stream
    moved.runner.hold.set()
    await moved.settle(moved.ref)
    epoch, sequence = after.event_stream.watermark()
    # The adopted turn's events were published to the stream the viewer is attached to.
    assert (epoch, sequence > snapshot.last_sequence) == (snapshot.replay_epoch, True)


def _rollback_module():
    spec = importlib.util.spec_from_file_location('local_adopt_rollback', ROOT / 'scripts' / 'local_adopt_rollback.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_rollback_restores_the_original_identity(moved):
    sid = moved.ref.session_id
    await moved.submit(moved.viewer, moved.ref, uuid.uuid4().hex)
    await moved.settle(moved.ref)
    adopted = moved.db.get_session(sid)
    local_route = moved.authority.sessions[sid].route
    rollback = _rollback_module().rollback

    rollback(moved.db.db_path, sid, dry_run=True)
    assert moved.db.get_session(sid)['session_key'] == adopted['session_key']
    assert _meta(moved.db, 'gateway.local_policy.v1:' + sid) is not None

    rollback(moved.db.db_path, sid)
    row = moved.db.get_session(sid)
    assert moved.original['end_reason'] is not None and adopted['end_reason'] is None  # reopened by its turn
    for column in ('session_key', 'chat_id', 'user_id', 'chat_type', 'origin_json', 'ended_at', 'end_reason'):
        assert row[column] == moved.original[column], column
    for key in ('gateway.local_policy.v1:', 'gateway.local_legacy.v1:', 'local_adopt_preimage:'):
        assert _meta(moved.db, key + sid) is None, key
    with moved.db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM gateway_routing WHERE session_key=?', (local_route,)).fetchone() is None


@pytest.mark.asyncio
async def test_a_pruned_route_adopts_the_row_instead_of_writing_into_the_successor(moved):
    """Review probe P10 under the divergence: the row is adopted and the turn lands in it."""
    store = moved.runner.session_store
    with store._lock:
        store._entries.pop(moved.current.session_key)
    await moved.submit(moved.viewer, moved.ref, 'continue the old chat')
    await moved.settle(moved.ref)

    assert [turn.session_id for turn in moved.runner.turns] == [moved.ref.session_id]
    assert moved.authority.sessions[moved.ref.session_id].source.platform == Platform.LOCAL
    assert moved.db.message_count(moved.current.session_id) == 0 and moved.runner.bot.sent == []


@pytest.mark.asyncio
async def test_no_lane_can_be_repointed_onto_an_adopted_row(moved):
    store = moved.runner.session_store
    await moved.submit(moved.viewer, moved.ref, 'mine now')
    await moved.settle(moved.ref)

    assert store.switch_session(moved.current.session_key, moved.ref.session_id) is None
    with store._lock:
        assert store._entry_locked(moved.current.session_key).session_id == moved.current.session_id
