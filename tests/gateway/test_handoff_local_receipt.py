"""A local chat handed off to a messaging lane keeps opening there (#143286).

``/handoff`` repoints the platform's route at the local session (``SessionStore.switch_session``),
which rewrites the row's routing identity. Its local receipt must go with that write, or the row
matches neither lane: every open, pin or rename refused ``storage_unavailable`` (``stored identity
mismatch``) and its title (a Bot Chat's registry lookup) answered ``not_found``. A store an older
build left that way heals when the gateway starts.
"""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_authority_cold_route_recovery import _Runner, _source


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    from gateway import run
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *_path: {'platform_toolsets': {'cli': []}, 'model': {'default': 'fixture'}})


def _compress(authority, sid):
    """Two real compression splits of the local chat, as its turns publish them."""
    db, live = authority.db, authority.sessions[sid]
    store = authority.runner.session_store
    for parent, child in ((sid, sid + '-c1'), (sid + '-c1', sid + '-c2')):
        store._record_gateway_session_peer(parent, live.route, live.source)
        assert db.try_acquire_compression_lock(parent, 'fixture')
        db.publish_compression_child(parent_session_id=parent, child_session_id=child, source='local',
                                     messages=[{'role': 'user', 'content': 'kept'}], compression_lock_holder='fixture')
        db.release_compression_lock(parent, 'fixture')
        store._record_gateway_session_peer(child, live.route, live.source)


async def _handoff(runner, authority, sid, source):
    """The handoff watcher's repoint (``GatewayStartupMixin._handoff_repoint``) on the real store."""
    from gateway.run_startup import GatewayStartupMixin
    from gateway.session import AsyncSessionStore
    from hermes_state import AsyncSessionDB
    route = runner.session_store._generate_session_key(source)
    gateway = SimpleNamespace(_session_db=AsyncSessionDB(authority.db), session_authority=authority,
                              async_session_store=AsyncSessionStore(runner.session_store))
    await GatewayStartupMixin._handoff_repoint(gateway, SimpleNamespace(source=source), route, sid)
    return route


def _as_an_older_build(monkeypatch):
    """An older build's repoint: the row's routing identity moves, its local receipt stays."""
    import hermes_state_local
    for name in ('hand_off_conversations', 'release_receipts'):
        monkeypatch.setattr(hermes_state_local, name, lambda *_args: None, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('history', ['plain', 'compressed', 'reset'])
@pytest.mark.parametrize('handed_off_by', ['this build', 'an older build'])
async def test_a_handed_off_bot_chat_opens_on_its_lane_after_a_restart(tmp_path, monkeypatch, handed_off_by, history):
    """``reset``: a local ``/new`` moved the chat to a new segment and its receipt names that one. A
    handoff carries the conversation the user is looking at: the lane continues the post-``/new``
    segment (the title follows it), and the earlier segment is that lane's read-only history, never a
    local row no receipt restores."""
    from gateway.config import Platform
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import transcript_target
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_local import POLICY_PREFIX, local_receipt
    from hermes_state_local_migration import LEGACY_PREFIX
    from hermes_state_runtime import RuntimeStoreError

    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    warm.db.set_session_title(ref.session_id, 'Bot Chat')
    if history == 'reset':
        runner.session_store.reset_session(warm.sessions[ref.session_id].route)
        # Its first turn there gets the segment a generated title of its own.
        warm.db.set_auto_title(local_receipt(warm.db, ref.session_id)['entry']['session_id'], 'Generated', source='llm')
    if history == 'compressed':
        _compress(warm, ref.session_id)
        runner = _Runner(tmp_path)  # a restart records the steps on the receipt (#140060)
        warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='adopted')
        warm.authorize(owner, ref, 'session:read')  # the receipt now names the tip
    keys = (POLICY_PREFIX + ref.session_id, LEGACY_PREFIX + ref.session_id)
    with warm.db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)', keys).fetchone()
    # The lane's conversation: the reset segment the receipt now names, else the chat itself.
    head = SessionRef('fixture', local_receipt(warm.db, ref.session_id)['entry']['session_id']
                      if history == 'reset' else ref.session_id)

    telegram = _source()
    with monkeypatch.context() as patch:
        if handed_off_by == 'an older build':
            _as_an_older_build(patch)
        route = await _handoff(runner, warm, ref.session_id, telegram)
    assert warm.logical_owner(runner.session_store.lookup_by_session_key(route).session_id) == head.session_id
    if handed_off_by == 'this build':
        # The Desktop that ran /handoff still holds the chat open.
        warm.authorize(owner, ref, 'session:read')
        if history == 'reset':  # its pre-/new segment is the lane's history now
            with pytest.raises(RuntimeStoreError, match='session_superseded'):
                warm.authorize(owner, ref, 'session:control')
        else:
            warm.authorize(owner, ref, 'session:control')
            assert transcript_target(warm, ref) == warm.physical_target(ref)

    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    # A browser Webapp viewer: never a native legacy transport, so legacy adoption is not its way in.
    webapp = Principal('webapp', 'fixture', frozenset({'session:read', 'session:control'}), 'browser')
    assert resolve_titled_session(cold, webapp, 'Bot Chat') == head
    cold.authorize(webapp, head, 'session:control')  # pin, rename, archive

    live = cold.sessions[head.session_id]
    assert (live.source.platform, live.route) == (Platform.TELEGRAM, route)
    with cold.db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)', keys).fetchone() is None
    tip = cold.db.get_session(cold.physical_target(head))
    assert json.loads(tip['origin_json'])['platform'] == 'telegram'
    if history == 'reset':  # the earlier segment opens read-only on that lane
        cold.authorize(webapp, ref, 'session:read')
        assert cold.sessions[ref.session_id].route == route
        with pytest.raises(RuntimeStoreError, match='session_superseded'):
            cold.authorize(webapp, ref, 'session:control')


@pytest.mark.asyncio
async def test_a_receipt_whose_current_segment_is_still_local_is_never_released(tmp_path, monkeypatch):
    """An older build handed off a chat's FIRST segment after a local ``/new``: that row moved, but
    the receipt names the post-``/new`` segment, still local. Releasing it would leave that
    transcript reachable from nowhere, so the boot heal keeps it and the conversation still opens."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    runner.session_store.reset_session(warm.sessions[ref.session_id].route)
    head = local_receipt(warm.db, ref.session_id)['entry']['session_id']
    store, telegram = runner.session_store, _source()
    with monkeypatch.context() as patch:  # the older build's watcher: switch_session(route, <the id>)
        _as_an_older_build(patch)
        store.get_or_create_session(telegram)
        store.switch_session(store._generate_session_key(telegram), ref.session_id, preserve_prompt_pin=False)

    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    assert local_receipt(cold.db, ref.session_id)['entry']['session_id'] == head
    assert cold.db.get_session(head)['source'] == 'gui'


async def _handed_off_bot_chat(tmp_path):
    """A Bot Chat the Desktop minted and a ``/handoff`` moved to Telegram, still open in that Desktop."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control',
                                                     'session:submit'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    warm.db.set_session_title(ref.session_id, 'Bot Chat')
    telegram = _source()
    route = await _handoff(runner, warm, ref.session_id, telegram)
    warm.authorize(owner, ref, 'session:control')
    return runner, warm, owner, ref, telegram, route


@pytest.mark.asyncio
async def test_a_handed_off_bot_chat_stays_the_bots_chat_while_its_lane_holds_it(tmp_path):
    """Bot Mode's one canonical chat per profile: a Bot or cron delivery after the restart goes to
    the handed-off Bot Chat on its lane; it is never renamed aside for a freshly minted one."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_bot import _target, displaceable_bot_chat
    from gateway.session_local_title import resolve_titled_session
    _runner, _warm, owner, ref, _telegram, route = await _handed_off_bot_chat(tmp_path)
    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    assert displaceable_bot_chat(cold, cold.db.get_session_by_title('Bot Chat')) is None
    target, live, entry = await _target(cold, owner)
    assert target == ref and live.route == route == entry.session_key
    assert resolve_titled_session(cold, owner, 'Bot Chat') == ref
    assert cold.db.get_session(ref.session_id)['title'] == 'Bot Chat'


async def _bot_row(authority, actor):
    """What the Desktop Bots row reads for the profile: the roster's ``canonical_session`` (the
    ``profiles.list`` reader of that profile's store) and the registry lookup ``session.list
    title="Bot Chat"`` its click runs. ``canonical-chat.ts`` refuses to mint when the lookup is
    empty but the roster names a chat, so the two must agree."""
    from pathlib import Path
    from gateway.session_controls import AuthorityConnection
    from tui_gateway.server import _profile_session_fields
    from tui_gateway.profile_roster_cache import invalidate
    invalidate()
    roster = {}
    _profile_session_fields(roster, Path(authority.db.db_path).parent)
    canonical = roster['canonical_session']
    listed = await AuthorityConnection(authority, object(), {'user_id': actor.subject}).list_sessions(
        None, {'title': 'Bot Chat', 'include_hidden': True, 'limit': 50})
    found = [row['id'] for row in listed['sessions']]
    assert (canonical and canonical['id']) == (found[0] if found else None), (canonical, found)
    return found[0] if found else None


@pytest.mark.asyncio
@pytest.mark.parametrize('door', ['Desktop create', 'Bot delivery'])
async def test_a_handed_off_bot_chat_whose_lane_moved_on_is_moved_aside(tmp_path, door):
    """After the platform's own ``/new`` the handed-off chat is that lane's history: the canonical
    title moves aside at once, so the Bots row's roster and its registry lookup agree at every step
    (warm, cold) and its click mints a fresh Bot Chat; the Desktop still viewing the old one is
    read-only (its route names the new conversation)."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_bot import _target
    from gateway.session_controls import AuthorityConnection
    from hermes_state_runtime import RuntimeStoreError
    runner, warm, owner, ref, telegram, route = await _handed_off_bot_chat(tmp_path)
    assert await _bot_row(warm, owner) == ref.session_id
    runner.session_store.reset_session(route)  # Telegram's /new
    assert await _bot_row(warm, owner) is None
    newer = warm.register(telegram)
    assert newer != ref and warm.sessions[newer.session_id].route == route
    assert warm.sessions[ref.session_id].superseded and warm.agent(ref) is None
    with pytest.raises(RuntimeStoreError, match='session_superseded'):
        warm.authorize(owner, ref, 'session:control')

    cold_runner = _Runner(tmp_path)
    cold = await initialize_session_authority(cold_runner, profile_id='fixture', instance_id='cold')
    cold_runner._adapter_for_source = lambda source: cold_runner.adapters.get(source.platform)
    cold_runner._cached_agent_for = lambda route: None
    assert await _bot_row(cold, owner) is None
    if door == 'Desktop create':  # the click: nothing found, so it mints
        opened = await AuthorityConnection(cold, object(), {'user_id': owner.subject}).create(
            None, {'title': 'Bot Chat', 'request_id': 'bot-chat-2', 'source': 'gui', 'cwd': str(tmp_path),
                   'model': 'frozen', 'toolsets': [], 'hidden': True})
        fresh = opened['session_id']
    else:  # the bot's own delivery, as the Desktop's principal (whose roster reads it)
        from gateway.session_contract import Principal
        desktop = AuthorityConnection(cold, object(), {'user_id': owner.subject}).actor
        fresh = (await _target(cold, Principal(desktop.subject, 'fixture', owner.capabilities, 'socket')))[0].session_id
    assert fresh not in (ref.session_id, newer.session_id)
    assert await _bot_row(cold, owner) == fresh
    moved = cold.db.get_session(ref.session_id)
    assert moved['title'] == f'Bot Chat (telegram {ref.session_id})' and not moved['hidden']


@pytest.mark.asyncio
async def test_a_second_chat_handed_off_to_the_same_home_chat_supersedes_the_first(tmp_path):
    """Desktop A handed off its Bot Chat H to the Telegram home chat; Desktop B then hands off its
    own open chat Y there. The lane names Y now: H is history, read-only for the Desktop still
    viewing it (a Stop or turn through it would act on Y), and its canonical title moves aside."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from hermes_state_runtime import RuntimeStoreError
    runner, warm, owner, ref, telegram, route = await _handed_off_bot_chat(tmp_path)
    other = create_local_session(warm, owner, {'request_id': 'other-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                               'model': 'frozen', 'toolsets': []})
    warm.authorize(owner, other, 'session:control')
    assert await _handoff(runner, warm, other.session_id, telegram) == route
    assert warm.logical_owner(runner.session_store.lookup_by_session_key(route).session_id) == other.session_id
    warm.authorize(owner, other, 'session:submit')
    with pytest.raises(RuntimeStoreError, match='session_superseded'):
        warm.authorize(owner, ref, 'session:submit')
    assert warm.agent(ref) is None
    assert await _bot_row(warm, owner) is None

    # H handed off there again: the lane names H once more, and Y is the one that is history now.
    await _handoff(runner, warm, ref.session_id, telegram)
    warm.authorize(owner, ref, 'session:submit')
    with pytest.raises(RuntimeStoreError, match='session_superseded'):
        warm.authorize(owner, other, 'session:submit')

    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    cold.authorize(owner, ref, 'session:control')
    cold.authorize(owner, other, 'session:read')
    with pytest.raises(RuntimeStoreError, match='session_superseded'):
        cold.authorize(owner, other, 'session:control')


@pytest.mark.asyncio
async def test_the_bots_own_bot_chat_after_a_reset_handoff_and_the_lanes_new_is_replaced(tmp_path):
    """The Bot Chat a delivery minted (its automatic creation id), reset locally, handed off, then the
    platform's ``/new``: the next delivery mints the next generation. The handed-off creation row
    (its name went to the head) is spent, never reused as an unfinished creation."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_bot import _target
    from gateway.session_contract import Principal
    runner = _Runner(tmp_path)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._cached_agent_for = lambda route: None
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control',
                                                     'session:submit'}), 'socket')
    first = (await _target(warm, owner))[0]
    runner.session_store.reset_session(warm.sessions[first.session_id].route)
    telegram = _source()
    route = await _handoff(runner, warm, first.session_id, telegram)
    head = runner.session_store.lookup_by_session_key(route).session_id
    assert warm.db.get_session_by_title('Bot Chat')['id'] == head
    runner.session_store.reset_session(route)  # Telegram's /new
    fresh = (await _target(warm, owner))[0].session_id
    assert fresh not in (first.session_id, head)
    assert warm.db.get_session_by_title('Bot Chat')['id'] == fresh


@pytest.mark.asyncio
@pytest.mark.parametrize('lands', ['after the title read', 'before the displace write'])
async def test_a_delivery_racing_the_lanes_new_mints_the_fresh_bot_chat(tmp_path, monkeypatch, lands):
    """The lane's ``/new`` frees the handed-off Bot Chat's title from the store's worker thread,
    outside the titled-create lock. A delivery it lands in the middle of still lands in exactly one
    canonical Bot Chat, a fresh one, instead of failing on the title that vanished under it."""
    from gateway import session_bot
    runner, warm, owner, ref, _telegram, route = await _handed_off_bot_chat(tmp_path)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._cached_agent_for = lambda route: None
    release, displaceable = session_bot.release_superseded_bot_chat, session_bot.displaceable_bot_chat
    if lands == 'before the displace write':  # the /new switched the lane; its rename is still pending
        monkeypatch.setattr(session_bot, 'release_superseded_bot_chat', lambda *_args: None)
        runner.session_store.reset_session(route)
    seen = []

    def racing(authority, row):
        found = displaceable(authority, row)
        seen.append(found)
        if lands == 'after the title read' and len(seen) == 1:
            runner.session_store.reset_session(route)  # the lane held it when the delivery read it
        elif lands == 'before the displace write' and len(seen) == 2:  # under the lock
            release(warm.db, runner.session_store, route, ref.session_id)
        return found
    monkeypatch.setattr(session_bot, 'displaceable_bot_chat', racing)
    fresh = (await session_bot._target(warm, owner))[0].session_id
    assert fresh != ref.session_id
    assert warm.db.get_session_by_title('Bot Chat')['id'] == fresh
    assert warm.db.get_session(ref.session_id)['title'] == f'Bot Chat (telegram {ref.session_id})'


def _resume_gateway(runner, authority):
    """The platform's ``/resume`` handler (``GatewayRunner._handle_resume_command``) on the real store,
    run by an admin (``--all`` widens it to every titled conversation)."""
    from gateway.run import GatewayRunner
    from hermes_state import AsyncSessionDB
    gateway = object.__new__(GatewayRunner)
    gateway.adapters, gateway.config, gateway._voice_mode, gateway._running_agents = {}, runner.config, {}, {}
    gateway._session_db, gateway.session_store = AsyncSessionDB(authority.db), runner.session_store
    gateway.session_authority = authority
    gateway._is_user_authorized = gateway._resume_caller_is_admin = lambda _source: True
    return gateway


@pytest.mark.asyncio
async def test_a_platform_resume_of_a_local_conversation_continues_its_current_head(tmp_path):
    """Bot Chat X, then a local ``/new`` (the receipt names the new segment X2), then an admin's
    ``/resume --all Bot Chat`` on Telegram: the title still names X, the first segment. The lane
    continues the conversation's head, as ``/handoff`` carries it, so X2 never stays a local row
    whose receipt was released (refused ``storage_unavailable`` after a restart)."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_local import create_local_session
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_local import local_receipt
    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    warm.db.set_session_title(ref.session_id, 'Bot Chat')
    runner.session_store.reset_session(warm.sessions[ref.session_id].route)
    head = SessionRef('fixture', local_receipt(warm.db, ref.session_id)['entry']['session_id'])
    telegram = _source()
    reply = await _resume_gateway(runner, warm)._handle_resume_command(
        MessageEvent(text='/resume --all Bot Chat', source=telegram))
    route = runner.session_store._generate_session_key(telegram)
    assert runner.session_store.lookup_by_session_key(route).session_id == head.session_id, reply

    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    cold.authorize(owner, head, 'session:control')
    assert resolve_titled_session(cold, owner, 'Bot Chat') == head
    assert cold.sessions[head.session_id].route == route


@pytest.mark.asyncio
@pytest.mark.parametrize('opened', ['followed by the Desktop', 'a viewer after a restart', 'the lane after a restart'])
@pytest.mark.parametrize('takeover', ['another chat handed off', "the platform's /new"])
@pytest.mark.parametrize('door', ['/handoff', 'platform /resume'])
@pytest.mark.parametrize('history', ['plain', 'reset'])
async def test_a_moved_conversation_is_read_only_once_its_lane_is_taken_over(tmp_path, history, door, takeover,
                                                                             opened):
    """Bot Chat X (``reset``: a local ``/new`` moved it to X2) moves onto a Telegram lane, which
    continues its head, writable there. Once that lane is taken over (another chat handed off to
    it, or the platform's ``/new`` and its next inbound) the head is the lane's history, however it
    was mounted: the Desktop that followed it, a viewer or the lane's inbound after a restart. Its
    only route is the lane's, so a turn or a Stop through it would act on the new conversation:
    read stays open, submit and control are refused, warm and after a restart."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError
    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control',
                                                     'session:submit'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    warm.db.set_session_title(ref.session_id, 'Bot Chat')
    if history == 'reset':
        runner.session_store.reset_session(warm.sessions[ref.session_id].route)
    head = SessionRef('fixture', local_receipt(warm.db, ref.session_id)['entry']['session_id'])
    telegram = _source()
    if door == '/handoff':
        route = await _handoff(runner, warm, ref.session_id, telegram)
    else:
        await _resume_gateway(runner, warm)._handle_resume_command(
            MessageEvent(text='/resume --all Bot Chat', source=telegram))
        route = runner.session_store._generate_session_key(telegram)
    assert runner.session_store.lookup_by_session_key(route).session_id == head.session_id
    if opened != 'followed by the Desktop':
        runner = _Runner(tmp_path)
        warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='restarted')
    if opened == 'a viewer after a restart':
        warm.authorize(owner, head, 'session:read')
    warm.register(telegram)  # the lane's next inbound
    warm.authorize(owner, head, 'session:submit')  # the lane's conversation: writable there

    if takeover == 'another chat handed off':
        other = create_local_session(warm, owner, {'request_id': 'other-chat', 'source': 'gui',
                                                   'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        await _handoff(runner, warm, other.session_id, telegram)
    else:
        runner.session_store.reset_session(route)
        warm.register(telegram)
    assert warm.logical_owner(runner.session_store.lookup_by_session_key(route).session_id) != head.session_id

    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    for authority in (warm, cold):
        authority.authorize(owner, head, 'session:read')
        for capability in ('session:submit', 'session:control'):
            with pytest.raises(RuntimeStoreError, match='session_superseded'):
                authority.authorize(owner, head, capability)
        assert authority.agent(head) is None  # never the lane's running agent


@pytest.mark.asyncio
async def test_a_moved_receipt_with_a_malformed_lineage_never_stops_the_boot(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state_local import POLICY_PREFIX
    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    with monkeypatch.context() as patch:
        _as_an_older_build(patch)
        await _handoff(runner, warm, ref.session_id, _source())
    key = POLICY_PREFIX + ref.session_id
    warm.db._execute_write(lambda conn: conn.execute(
        "UPDATE state_meta SET value=json_set(value, '$.lineage', json(?)) WHERE key=?",
        (json.dumps([ref.session_id, {}]), key)))
    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    with cold.db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone()


def _serve(tmp_path, monkeypatch):
    """A multiplexed gateway's homes: the default profile and a named ``bot`` profile, each with its
    own ``state.db``; ``bot`` delivers to a Telegram home channel."""
    from pathlib import Path
    import hermes_state
    root = tmp_path / '.hermes'
    bot = root / 'profiles' / 'bot'
    bot.mkdir(parents=True, mode=0o700)
    (bot / 'config.yaml').write_text(
        'platforms:\n  telegram:\n    enabled: true\n    token: fixture\n'
        '    home_channel: {platform: telegram, chat_id: "-1001", name: Bot Home}\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)  # the profiles root is HOME-anchored
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    return root, bot


async def _boot(root, bot):
    """One gateway process: a real runner whose runtime builds one authority per served home."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = (('default', root), ('bot', bot))
    runner = GatewayRunner(config)
    await initialize_gateway_runtime(runner)
    return runner


def _shutdown(runner):
    for authority in list(runner.session_authorities):
        authority.db.close()
    runner.session_store.close_all_db_handles()
    runner.close_all_session_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('handed_off_by', ['this build', 'an older build'])
async def test_a_named_profiles_handed_off_bot_chat_opens_on_its_lane_after_a_restart(
        tmp_path, monkeypatch, handed_off_by):
    """The reporter's topology: a named bot profile under a multiplexed gateway. The chat lives in
    that profile's own store, the handoff watcher's ``_process_handoff`` runs in its scope and keys
    the lane ``agent:bot:...``; after a restart the Webapp still finds and opens it there."""
    import sqlite3
    from unittest.mock import AsyncMock
    from gateway.config import Platform
    from gateway.run import _async_profile_runtime_scope
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import active_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import transcript_target
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX

    root, bot = _serve(tmp_path, monkeypatch)
    process_ownership.reserve([root, bot])
    try:
        runner = await _boot(root, bot)
        runner._profile_adapters = {'bot': {Platform.TELEGRAM: SimpleNamespace(
            create_handoff_thread=AsyncMock(return_value=None))}}
        runner._handle_message = AsyncMock(return_value='')  # the synthetic handoff turn itself
        async with _async_profile_runtime_scope(bot):
            warm = active_authority(runner)
            caps = frozenset({'session:create', 'session:read', 'session:control'})
            owner = Principal('owner', warm.profile_id, caps, 'socket')
            ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                                     'model': 'frozen', 'toolsets': []})
            warm.db.set_session_title(ref.session_id, 'Bot Chat')
            keys = (POLICY_PREFIX + ref.session_id, LEGACY_PREFIX + ref.session_id)
            with warm.db._read_ctx() as conn:
                receipt = conn.execute('SELECT key, value FROM state_meta WHERE key IN (?,?)', keys).fetchall()
            assert receipt
            await runner._process_handoff({'id': ref.session_id, 'title': 'Bot Chat', 'handoff_platform': 'telegram'},
                                          'bot')
            if handed_off_by == 'an older build':
                warm.db._execute_write(lambda conn: conn.executemany(
                    'INSERT OR REPLACE INTO state_meta(key, value) VALUES(?,?)', [tuple(row) for row in receipt]))
            else:
                warm.authorize(owner, ref, 'session:control')  # the Desktop that ran /handoff
                assert transcript_target(warm, ref) == warm.physical_target(ref)
        _shutdown(runner)

        runner = await _boot(root, bot)
        async with _async_profile_runtime_scope(bot):
            cold = active_authority(runner)
            webapp = Principal('webapp', cold.profile_id, frozenset({'session:read', 'session:control'}), 'browser')
            assert resolve_titled_session(cold, webapp, 'Bot Chat') == ref
            cold.authorize(webapp, ref, 'session:control')  # pin, rename, archive
            live = cold.sessions[ref.session_id]
            assert live.source.platform == Platform.TELEGRAM and live.route.startswith('agent:bot:telegram:')
            with cold.db._read_ctx() as conn:
                assert conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)', keys).fetchone() is None
        _shutdown(runner)
        with sqlite3.connect(root / 'state.db') as conn:  # never the default profile's store
            assert conn.execute('SELECT 1 FROM sessions WHERE id=?', (ref.session_id,)).fetchone() is None
    finally:
        for home in (root, bot):
            process_ownership.release(home)
