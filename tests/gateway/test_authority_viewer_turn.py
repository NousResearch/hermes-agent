"""A local viewer's turn on a messaging session stays in the panel (TUI-8, NousResearch/hermes-agent#134889).

Before the canonical authority (#106742) a viewer continued a Telegram row in its own TUI session
(``platform="tui"``): nothing reached the chat. Through the authority, every egress of a viewer's turn
resolves ``_delivery_adapter_for`` to the profile's ``LocalSessionAdapter`` instead of the bot, while
the platform's own input still answers on the platform. And since a messaging turn runs on whichever
session the lane's route names, a viewer may only submit to that session: a historical row is
refused, never silently continued in the lane's current session.
"""
import asyncio
import dataclasses
from types import SimpleNamespace
import uuid

import pytest
import pytest_asyncio

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _Bot(BasePlatformAdapter):
    """A Telegram transport that records what reached the chat."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=uuid.uuid4().hex)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        self.sent.append((chat_id, '<typing>'))

    async def get_chat_info(self, chat_id):
        return {'id': chat_id, 'type': 'dm'}


class _Runner(GatewayAuthorizationMixin):
    """The turn resolves its session from the source and its transport from the delivery seam, as
    ``run_turn_prepare`` and ``run_turn`` do."""

    def __init__(self, tmp_path):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        self.config = GatewayConfig()
        self.session_store = SessionStore(tmp_path / 'sessions', self.config)
        self._session_db = self.session_store._db
        self.bot = _Bot()
        self.adapters = {Platform.TELEGRAM: self.bot}
        self._draining = False
        self.turns = []
        self.hold = None
        self.started = asyncio.Event()

    def _adapter_for_source(self, source):
        return self._delivery_adapter_for(source)

    async def _handle_message(self, event):
        entry = self.session_store.get_or_create_session(event.source)
        self._session_db.append_message(entry.session_id, 'user', event.text)
        adapter = self._delivery_adapter_for(event.source)
        prompts = None
        if adapter is not None:
            approval = await adapter.send_exec_approval(chat_id=event.source.chat_id, command='ls',
                                                        session_key=entry.session_key, description='list')
            clarify = await adapter.send_clarify(chat_id=event.source.chat_id, question='which?', choices=None,
                                                 clarify_id=uuid.uuid4().hex[:10], session_key=entry.session_key)
            await adapter.send_typing(event.source.chat_id)
            prompts = (approval.success, clarify.success)
        # A send_message-style egress the turn addresses to another chat of the same platform.
        elsewhere = dataclasses.replace(event.source, chat_id='9999')
        self.turns.append(SimpleNamespace(session_id=entry.session_id, adapter=adapter, prompts=prompts,
                                          elsewhere=self._delivery_adapter_for(elsewhere)))
        self.started.set()
        if self.hold is not None:
            await self.hold.wait()
        return 'reply'


def _telegram():
    from gateway.session import SessionSource
    return SessionSource(platform=Platform.TELEGRAM, chat_id='4242', user_id='u-4242', chat_type='dm')


@pytest_asyncio.fixture
async def lane(tmp_path, monkeypatch):
    return await build_lane(tmp_path, monkeypatch)


@pytest_asyncio.fixture
async def upstream(tmp_path, monkeypatch):
    """The lane with ``gateway.session.viewer_replies_stay_local`` left at its default (false)."""
    return await build_lane(tmp_path, monkeypatch, stay_local=False)


async def build_lane(tmp_path, monkeypatch, *, stay_local=True):
    """A registered Telegram lane on a live authority, a viewer and the lane's own sender. The
    profile's config.yaml opts into the local divergence unless *stay_local* is false."""
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    (tmp_path / 'config.yaml').write_text(
        'platform_toolsets:\n  tui: []\n'
        + ('gateway:\n  session:\n    viewer_replies_stay_local: true\n' if stay_local else ''))
    runner = _Runner(tmp_path)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    ref = authority.register(_telegram())
    caps = frozenset({'session:read', 'session:submit'})
    viewer = Principal('viewer', 'fixture', caps, 'socket')
    owner = Principal('u-4242', 'fixture', caps, 'bot')  # the lane's own sender: no operator envelope

    async def submit(actor, target, text):
        from gateway.session_contract import Submission
        await authority.submit(actor, Submission(uuid.uuid4().hex, target, {'text': text}, 'queue'))

    async def settle(target):
        task = authority.sessions[target.session_id].task
        if task is not None:
            await asyncio.wait_for(task, 10)

    return SimpleNamespace(runner=runner, authority=authority, ref=ref, viewer=viewer, owner=owner,
                           submit=submit, settle=settle, db=authority.db)


@pytest.mark.asyncio
async def test_a_viewer_turn_on_the_current_lane_answers_only_in_the_panel(lane):
    from gateway.session_local import LocalSessionAdapter
    await lane.submit(lane.viewer, lane.ref, 'from the panel')
    await lane.settle(lane.ref)

    [turn] = lane.runner.turns
    assert turn.session_id == lane.ref.session_id
    assert lane.db.message_count(lane.ref.session_id) == 1
    assert isinstance(turn.adapter, LocalSessionAdapter)
    assert lane.runner.bot.sent == []


@pytest.mark.asyncio
async def test_approval_and_clarify_prompts_are_acknowledged_for_the_panel(lane):
    """The turn runner registers a prompt in PendingControls only once its card was sent
    (``run_turn_runner._send_shared_clarify`` / ``_approval_notify_sync``): both must succeed."""
    await lane.submit(lane.viewer, lane.ref, 'needs approval')
    await lane.settle(lane.ref)
    assert lane.runner.turns[0].prompts == (True, True)
    assert lane.runner.bot.sent == []


@pytest.mark.asyncio
async def test_the_lanes_own_input_still_answers_on_the_platform(lane):
    await lane.submit(lane.owner, lane.ref, 'from telegram')
    await lane.settle(lane.ref)

    [turn] = lane.runner.turns
    assert turn.adapter is lane.runner.bot
    assert ('4242', 'reply') in lane.runner.bot.sent


@pytest.mark.asyncio
async def test_egress_the_viewer_turn_addresses_to_another_chat_is_not_swallowed(lane):
    await lane.submit(lane.viewer, lane.ref, 'tell the other chat')
    await lane.settle(lane.ref)
    assert lane.runner.turns[0].elsewhere is lane.runner.bot


@pytest.mark.asyncio
async def test_a_turn_without_a_local_channel_tells_its_viewers_instead_of_the_chat(lane):
    lane.runner.adapters[Platform.LOCAL] = object()  # the LOCAL slot is held by something else
    frames = []
    lane.authority.sessions[lane.ref.session_id].event_stream.observers.add(frames.append)
    await lane.submit(lane.viewer, lane.ref, 'from the panel')
    await lane.settle(lane.ref)

    assert lane.runner.turns[0].adapter is None and lane.runner.bot.sent == []
    notices = [frame['params']['payload'] for frame in frames if frame['params']['type'] == 'notification.show']
    assert len(notices) == 1 and notices[0]['level'] == 'warning'


def test_lane_state_never_raises_and_fails_closed():
    """No session store (a bare runner) or a store that cannot be read is ``'unknown'``, not an error;
    a source without a messaging platform is not judged at all."""
    from gateway.session_contract import SessionRef
    from gateway.session_viewer_turn import lane_state
    ref = SessionRef('p', 's')

    def authority(runner, source):
        return SimpleNamespace(runner=runner, sessions={'s': SimpleNamespace(source=source, route='r')})
    broken = SimpleNamespace(session_store=SimpleNamespace(_lock=None))
    assert lane_state(authority(SimpleNamespace(), _telegram()), ref) == 'unknown'
    assert lane_state(authority(broken, _telegram()), ref) == 'unknown'
    assert lane_state(authority(SimpleNamespace(), SimpleNamespace(platform=None, user_id='u')), ref) is None


def test_peeking_at_recovery_never_spends_the_legacy_slack_claim(tmp_path):
    """The legacy (pre-workspace) Slack key is claimed once per process by whichever transition
    reaches it first; a dry run that consumed it would rob the lane's real next message of it."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionSource, SessionStore
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    source = SessionSource(platform=Platform.SLACK, chat_id='D1', user_id='U1', chat_type='dm', scope_id='T1')
    legacy = store._legacy_slack_session_key(source)

    assert store.peek_recovery(store._generate_session_key(source), source) == (False, None)
    assert store._claim_legacy_slack_key(legacy)


def _prune_route(lane):
    """The lane after ``/new`` with its routing entry gone (pruned, or never restored)."""
    store = lane.runner.session_store
    current = store.get_or_create_session(_telegram(), force_new=True)
    with store._lock:
        store._entries.pop(current.session_key)
    return current


def _no_adoption(monkeypatch):
    """Keep the replies-stay-local gate but not its adoption, to see what the gate alone decides."""
    from gateway import session_viewer_adoption
    monkeypatch.setattr(session_viewer_adoption, 'adopt_moved_row', lambda *args: None)


async def _refused(lane, actor=None):
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    with pytest.raises(RuntimeStoreError, match='session_not_current'):
        await lane.submit(actor or lane.viewer, lane.ref, 'continue the old chat')
    assert list_session_admissions(lane.db, session_id=lane.ref.session_id) == []
    assert (lane.runner.turns, lane.runner.bot.sent) == ([], [])


@pytest.mark.asyncio
async def test_a_pruned_route_never_lets_a_viewer_turn_into_the_successor(lane, monkeypatch):
    """Review probe P10: no routing entry, a ``/new`` successor in the store. The turn would resolve
    through recovery into the successor, so the row is refused rather than continued there."""
    from gateway.session_viewer_turn import lane_state
    _no_adoption(monkeypatch)
    current = _prune_route(lane)
    assert lane_state(lane.authority, lane.ref) == 'moved'

    await _refused(lane)
    assert (lane.db.message_count(lane.ref.session_id), lane.db.message_count(current.session_id)) == (0, 0)


@pytest.mark.asyncio
async def test_a_pruned_route_that_recovers_the_row_itself_admits_the_viewer_turn(lane, monkeypatch):
    """No entry, but recovery reopens this very row (no successor): the turn runs in it."""
    from gateway.session_viewer_turn import lane_state
    _no_adoption(monkeypatch)
    store = lane.runner.session_store
    with store._lock:
        store._entries.pop(lane.authority.sessions[lane.ref.session_id].route)
    assert lane_state(lane.authority, lane.ref) == 'current'

    await lane.submit(lane.viewer, lane.ref, 'continue')
    await lane.settle(lane.ref)
    assert [turn.session_id for turn in lane.runner.turns] == [lane.ref.session_id]


@pytest.mark.asyncio
async def test_an_unresolvable_lane_store_fails_closed(lane, monkeypatch):
    _prune_route(lane)
    store = lane.runner.session_store
    monkeypatch.setattr(store, '_db_for_key', lambda key: None)  # e.g. a profile home that is gone

    await _refused(lane)
    assert lane.authority.sessions[lane.ref.session_id].source.platform == Platform.TELEGRAM


@pytest.mark.asyncio
async def test_a_viewer_cannot_continue_a_row_the_route_moved_past(upstream):
    """Upstream behaviour: the lane's route names the ``/new`` successor."""
    from gateway.session_viewer_turn import lane_state
    current = upstream.runner.session_store.get_or_create_session(_telegram(), force_new=True)
    assert current.session_id != upstream.ref.session_id
    assert lane_state(upstream.authority, upstream.ref) == 'moved'

    await _refused(upstream)
    assert upstream.db.message_count(current.session_id) == 0


@pytest.mark.parametrize('stay_local', [False, True])
@pytest.mark.asyncio
async def test_a_new_between_admission_and_claim_refuses_the_viewer_turn_visibly(tmp_path, monkeypatch, stay_local):
    """Refused before the claim, the row ends terminal with its reason: never left queued behind a
    paused FIFO that ``/reset`` (which acts on the lane's current session) could not clear."""
    from hermes_state_runtime import list_session_admissions
    lane = await build_lane(tmp_path, monkeypatch, stay_local=stay_local)
    frames = []
    lane.authority.sessions[lane.ref.session_id].event_stream.observers.add(frames.append)
    lane.runner.hold = asyncio.Event()
    await lane.submit(lane.viewer, lane.ref, 'first')
    await asyncio.wait_for(lane.runner.started.wait(), 10)
    await lane.submit(lane.viewer, lane.ref, 'second')
    current = lane.runner.session_store.get_or_create_session(_telegram(), force_new=True)
    lane.runner.hold.set()
    await lane.settle(lane.ref)

    assert len(lane.runner.turns) == 1
    rows = list_session_admissions(lane.db, session_id=lane.ref.session_id, pending_only=False)
    assert [(row['status'], row['outcome']) for row in rows] == [('terminal', 'completed'), ('terminal', 'refused')]
    refused = [frame['params']['payload'] for frame in frames
               if frame['params']['payload'].get('admission_id') == rows[1]['admission_id']]
    assert refused and refused[-1]['outcome'] == 'refused' and 'nothing is left queued' in refused[-1]['error']
    assert lane.db.message_count(current.session_id) == 0


@pytest.mark.asyncio
async def test_a_suspended_route_refuses_the_viewer_turn(upstream):
    """``/stop`` suspends the route: the lane's next turn resets into a fresh session."""
    from gateway.session_viewer_turn import lane_state
    store = upstream.runner.session_store
    assert store.suspend_session(upstream.authority.sessions[upstream.ref.session_id].route)
    assert lane_state(upstream.authority, upstream.ref) == 'moved'
    await _refused(upstream)


@pytest.mark.asyncio
async def test_a_routed_row_ended_by_a_boundary_refuses_the_viewer_turn(upstream):
    """Still routed, but ended past recovery: the stale entry is dropped and a fresh session minted."""
    from gateway.session_viewer_turn import lane_state
    upstream.db.end_session(upstream.ref.session_id, 'session_reset')
    assert lane_state(upstream.authority, upstream.ref) == 'moved'
    await _refused(upstream)


@pytest.mark.asyncio
async def test_a_routed_row_ended_by_an_accident_is_still_its_lanes_current_session(upstream):
    """``agent_close`` is recoverable: the lane reopens this very row, so the viewer turn runs in it."""
    from gateway.session_viewer_turn import lane_state
    upstream.db.end_session(upstream.ref.session_id, 'agent_close')
    assert lane_state(upstream.authority, upstream.ref) == 'current'
    await upstream.submit(upstream.viewer, upstream.ref, 'still here')
    await upstream.settle(upstream.ref)
    assert [turn.session_id for turn in upstream.runner.turns] == [upstream.ref.session_id]


@pytest.mark.asyncio
async def test_a_route_on_the_rows_compression_continuation_is_current(upstream):
    """Compression copies ``session_key`` onto the continuation: one logical owner, not a move."""
    from gateway.session_viewer_turn import lane_state
    sid = upstream.ref.session_id
    row = upstream.db.get_session(sid)
    tip = sid + '-tip'
    upstream.db.create_session(tip, 'telegram', parent_session_id=sid, session_key=row['session_key'])
    upstream.db.end_session(sid, 'compression')
    store = upstream.runner.session_store
    with store._lock:
        store._entry_locked(row['session_key']).session_id = tip
    assert upstream.authority.logical_owner(tip) == sid
    assert lane_state(upstream.authority, upstream.ref) == 'current'


@pytest.mark.asyncio
async def test_a_suspended_route_left_on_a_compressed_parent_runs_the_turn_in_the_tip(upstream):
    """``_apply_route_checks`` heals before it reads the suspension: the entry is rewritten to the
    compression tip, the snapshot taken on the parent no longer applies, and the turn runs in the tip."""
    from gateway.session_viewer_turn import lane_state
    sid = upstream.ref.session_id
    row = upstream.db.get_session(sid)
    tip = sid + '-tip'
    upstream.db.create_session(tip, 'telegram', parent_session_id=sid, session_key=row['session_key'])
    upstream.db.end_session(sid, 'compression')
    assert upstream.runner.session_store.suspend_session(row['session_key'])
    assert lane_state(upstream.authority, upstream.ref) == 'current'

    await upstream.submit(upstream.viewer, upstream.ref, 'after /stop and a restart')
    await upstream.settle(upstream.ref)
    assert [turn.session_id for turn in upstream.runner.turns] == [tip]


def _failing_recovery_lookup(lane, monkeypatch):
    """A routed row ended by an accident (recovery decides) whose recovery lookup fails: the real
    transition swallows the error (``_peer_row``), the dry run raises it."""
    from hermes_state_errors import StateDbReplacedError
    lane.db.end_session(lane.ref.session_id, 'agent_close')

    def replaced(**_):
        raise StateDbReplacedError('state.db was replaced')
    monkeypatch.setattr(lane.runner.session_store._db, 'find_latest_gateway_session_for_peer', replaced)


@pytest.mark.asyncio
async def test_a_failing_recovery_lookup_is_unknown_not_an_error(upstream, monkeypatch):
    from gateway.session_viewer_turn import lane_state
    _failing_recovery_lookup(upstream, monkeypatch)
    assert lane_state(upstream.authority, upstream.ref) == 'unknown'
    await upstream.submit(upstream.viewer, upstream.ref, 'continue')  # admitted as upstream admits it
    await upstream.settle(upstream.ref)
    assert len(upstream.runner.turns) == 1


@pytest.mark.asyncio
async def test_without_a_routing_entry_the_viewer_turn_is_admitted_as_upstream(upstream):
    """No entry: recovery decides, as for the platform's own next message (#134889); here it lands
    in the ``/new`` successor, visibly, exactly as upstream does."""
    from gateway.session_viewer_turn import lane_state
    current = _prune_route(upstream)
    assert lane_state(upstream.authority, upstream.ref) == 'current'
    await upstream.submit(upstream.viewer, upstream.ref, 'continue')
    await upstream.settle(upstream.ref)
    assert [turn.session_id for turn in upstream.runner.turns] == [current.session_id]


@pytest.mark.asyncio
async def test_by_default_a_viewer_turn_answers_on_the_platform(upstream):
    """Upstream delivery: without the opt-in nothing reroutes the viewer turn's egress."""
    await upstream.submit(upstream.viewer, upstream.ref, 'from the panel')
    await upstream.settle(upstream.ref)
    [turn] = upstream.runner.turns
    assert turn.adapter is upstream.runner.bot
    assert ('4242', 'reply') in upstream.runner.bot.sent


@pytest.mark.asyncio
async def test_by_default_a_moved_row_is_refused_never_adopted(upstream):
    from tests.gateway.test_authority_viewer_adoption import _meta
    upstream.runner.session_store.get_or_create_session(_telegram(), force_new=True)
    await _refused(upstream)
    assert upstream.authority.sessions[upstream.ref.session_id].source.platform == Platform.TELEGRAM
    assert _meta(upstream.db, 'local_adopt_preimage:' + upstream.ref.session_id) is None


@pytest.mark.asyncio
async def test_with_replies_kept_local_a_failing_recovery_lookup_is_refused_visibly(lane, monkeypatch):
    """Fail closed, never adopted: the operator sees ``session_not_current``, not the lookup error."""
    _failing_recovery_lookup(lane, monkeypatch)
    await _refused(lane)
    assert lane.authority.sessions[lane.ref.session_id].source.platform == Platform.TELEGRAM


@pytest.mark.parametrize('value, expected', [
    ('true', True), (None, False), ('false', False), ("'true'", False), ('1', False)])
def test_the_switch_is_read_from_the_owning_profiles_config(tmp_path, monkeypatch, value, expected):
    """Through the gateway's own loader, from the profile home the authority names (multiplex), not the
    active one; anything but a YAML ``true`` is upstream."""
    from gateway.session_viewer_turn import viewer_replies_stay_local
    active, profile = tmp_path / 'active', tmp_path / 'profile'
    active.mkdir(), profile.mkdir()
    (active / 'config.yaml').write_text('gateway:\n  session:\n    viewer_replies_stay_local: true\n')
    monkeypatch.setenv('HERMES_HOME', str(active))
    (profile / 'config.yaml').write_text(
        f'gateway:\n  session:\n    viewer_replies_stay_local: {value}\n' if value else 'gateway: {}\n')
    assert viewer_replies_stay_local(SimpleNamespace(profile_id=str(profile))) is expected


def _replaced_state_db_under_lane_state(monkeypatch):
    """``lane_state``'s lineage read raises a RuntimeError subclass rather than a sqlite3/OS error —
    what a state.db read can do when the file is replaced under the process. Injected synthetically:
    the guard under test is lane_state's outer ``except Exception``, and the frame filter keeps every
    other ``logical_owner`` caller (the admission's FIFO identity) reading normally."""
    import sys
    from gateway.session_authority import SessionAuthority
    from hermes_state_errors import StateDbReplacedError
    real = SessionAuthority.logical_owner

    def logical_owner(self, session_id):
        if sys._getframe(1).f_globals.get('__name__') == 'gateway.session_viewer_turn':
            raise StateDbReplacedError('state.db was replaced')
        return real(self, session_id)
    monkeypatch.setattr(SessionAuthority, 'logical_owner', logical_owner)


@pytest.mark.asyncio
async def test_a_replaced_state_db_under_lane_state_is_unknown_and_refused_visibly(lane, monkeypatch):
    """The outer guard: a read failure while resolving the lane answers ``'unknown'`` (never raises),
    and with the switch on that refuses the viewer's turn visibly."""
    from gateway.session_viewer_turn import lane_state
    _replaced_state_db_under_lane_state(monkeypatch)
    assert lane_state(lane.authority, lane.ref) == 'unknown'
    await _refused(lane)


def _route_on_a_deleted_row(lane):
    """The lane's entry names a row hard-deleted from state.db."""
    store = lane.runner.session_store
    with store._lock:
        store._entry_locked(lane.authority.sessions[lane.ref.session_id].route).session_id = 'deleted-row'
    assert lane.db.get_session('deleted-row') is None


@pytest.mark.asyncio
async def test_a_route_on_a_deleted_row_recovers_the_live_row_itself(upstream):
    """Upstream: the entry is a stale hit (row gone), popped, and recovery reopens *ref*'s own live
    row, so the viewer turn runs in it rather than being refused."""
    from gateway.session_viewer_turn import lane_state
    _route_on_a_deleted_row(upstream)
    assert lane_state(upstream.authority, upstream.ref) == 'current'
    await upstream.submit(upstream.viewer, upstream.ref, 'continue')
    await upstream.settle(upstream.ref)
    assert [turn.session_id for turn in upstream.runner.turns] == [upstream.ref.session_id]


@pytest.mark.asyncio
async def test_a_route_on_a_deleted_row_whose_recovery_picks_another_row_is_moved(upstream):
    """Pins the outcome (a deleted row whose recovery picks another row is refused), not the new
    mechanism: the owner compare reached the same verdict before ``_row_exists`` existed."""
    from gateway.session_viewer_turn import lane_state
    successor = upstream.runner.session_store.get_or_create_session(_telegram(), force_new=True).session_id
    _route_on_a_deleted_row(upstream)
    route = upstream.authority.sessions[upstream.ref.session_id].route
    assert upstream.runner.session_store.peek_recovery(route, _telegram()) == (True, successor)
    assert lane_state(upstream.authority, upstream.ref) == 'moved'
    await _refused(upstream)
