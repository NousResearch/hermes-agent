"""A Bot Chat follows its profile's current route, whoever creates it.

Desktop's create asks for ``follow_profile_config``. The Bot/cron delivery door, a peer DM
create, a CLI ``chat -c "Bot Chat" --create-if-missing`` and the adoption of a pre-upgrade
Bot Chat used to freeze the route of that moment. Desktop later reuses the chat by title, so
a profile change never reached it, and retiring the old key broke it. A session that is not
a Bot Chat keeps the route it was created or saved with. A safe-mode Bot Chat stays frozen:
that launch never reads the profile. A Bot Chat an earlier build already froze follows from
the first time a client opens it; its creation request is not rewritten.

Real: SessionStore/SessionAuthority/AuthorityConnection, ``session_bot._target`` (the delivery
door), ``resolve_local_target`` -> ``adopt_legacy_session``, ``session.resume`` (the policy
refresh on open), a real config.yaml and ``GatewayTurnPrepareMixin._resolve_session_agent_runtime``.
Seam: a SimpleNamespace runner. No endpoint is contacted.
"""
import json
from pathlib import Path

import pytest

from tests.gateway.test_follow_profile_followups import _create_bot_chat, _profile
from tests.gateway.test_session_follow_profile_config import _owner, _route


@pytest.fixture
def home(monkeypatch):
    """The profile home the follow-profile tests write ``config.yaml`` into."""
    from gateway import run
    from hermes_constants import get_hermes_home
    profile = get_hermes_home()
    profile.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', profile)
    return profile


def _follows(runner, live):
    from gateway.session_policy import policy_for_source
    return policy_for_source(runner, live.source).follows_profile


def _key(runner, authority, live):
    """The api_key the session's next turn sends, from its policy's one effective read.

    ``_route`` reports ``no-key-required`` for an adopted session, so a key check goes
    through ``effective``."""
    from gateway.session_policy import policy_for_source
    model = policy_for_source(runner, live.source).effective(authority)[0].get('model') or {}
    return model.get('api_key')


@pytest.mark.asyncio
@pytest.mark.parametrize('first', ['delivery', 'desktop', 'cli'])
async def test_bot_chat_follows_profile_whoever_creates_it(home, first):
    from gateway.session_bot import _target
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        if first == 'delivery':
            _ref, live, _entry = await _target(authority, owner.actor)
        elif first == 'desktop':
            live = await _create_bot_chat(owner, authority, home, 'gui-first', title='Bot Chat', hidden=True)
        else:
            response = await owner.dispatch({'id': 'cli', 'method': 'session.create', 'params': dict(
                request_id='cli-first', source='cli', cwd=str(home), title='Bot Chat')})
            assert 'result' in response, response
            live = authority.sessions[response['result']['session_id']]
        # Desktop opening Bot Chat later is a titled create: it lands on the same session.
        reopened = await _create_bot_chat(owner, authority, home, 'gui-open', title='Bot Chat', hidden=True)
        assert reopened is live
        assert _route(runner, live) == ('old-model', 'http://127.0.0.1:1/v1', 'old-key')
        _profile(home, 'new-model', 2, 'new-key')
        assert _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
        assert _key(runner, authority, live) == 'new-key'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [
    {'title': 'Notes'},
    {'title': 'Bot Chat', 'safe_mode': True, 'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1'},
    {'title': 'Bot Chat', 'ignore_user_config': True, 'provider': 'custom',
     'base_url': 'http://127.0.0.1:1/v1'},
])
async def test_other_titles_and_bypass_launches_stay_frozen(home, extra):
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        response = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': dict(
            request_id='create', source='cli', cwd=str(home), model='old-model', **extra)})
        assert 'result' in response, response
        live = authority.sessions[response['result']['session_id']]
        assert not _follows(runner, live)
        _profile(home, 'new-model', 2, 'new-key')
        assert not _follows(runner, live)
        assert _route(runner, live)[0] == 'old-model'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_delivery_resumes_a_creation_staged_without_the_follow_flag(home):
    """An older build staged the delivery's Bot Chat and stopped before its title. The next
    delivery reuses that creation identity and must not be refused for a new flag."""
    from gateway.session_bot import _bot_chat_creation_id, _target
    from gateway.session_local import stage_local_session
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import commit_local_session
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        request_id = _bot_chat_creation_id(authority, owner.actor)
        receipt = stage_local_session(authority, owner.actor, {
            'request_id': request_id, 'source': 'cli', 'cwd': str(Path.home())})
        commit_local_session(authority.db, epoch=authority.epoch, receipt=receipt)
        staged = restore_local_session(authority, receipt['session_id'])
        ref, live, _entry = await _target(authority, owner.actor)
        assert ref.session_id == staged.session_id
        assert authority.db.get_session_by_title('Bot Chat')['id'] == staged.session_id
        assert not _follows(runner, live)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_cli_retries_a_titled_create_staged_without_the_follow_flag(home):
    """Same trap on ``session.create``: an exact retry of a request staged before the flag
    resumes that request instead of being refused. The CLI then opens its titled Bot Chat, which
    follows the profile; the request it was created with stays as staged."""
    from gateway.session_local import stage_local_session
    from hermes_state_local import commit_local_session, local_receipt
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        receipt = stage_local_session(authority, owner.actor, {
            'request_id': 'cli-retry', 'source': 'cli', 'cwd': str(home)})
        commit_local_session(authority.db, epoch=authority.epoch, receipt=receipt)
        response = await owner.dispatch({'id': 'retry', 'method': 'session.create', 'params': dict(
            request_id='cli-retry', source='cli', cwd=str(home), title='Bot Chat')})
        assert 'result' in response, response
        assert response['result']['session_id'] == receipt['session_id']
        live = authority.sessions[receipt['session_id']]
        assert authority.db.get_session_by_title('Bot Chat')['id'] == receipt['session_id']
        stored = local_receipt(authority.db, receipt['session_id'])['policy']
        assert stored['request_json'] == receipt['policy']['request_json']
        assert _follows(runner, live)
    finally:
        await owner.close()
        store.close_all_db_handles()


def _legacy(db, profile, sid, **row):
    title = row.pop('title', None)
    db.create_session(sid, cwd=str(profile), **row)
    if title:
        db.set_session_title(sid, title)
    db.end_session(sid, 'tui_close')


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', ['title', 'follow-marker'])
async def test_adopted_legacy_bot_chat_keeps_following(home, marker):
    from gateway.session_local_migration import resolve_local_target
    from gateway.session_policy import policy_for_source
    _profile(home, 'profile-model', 2, 'k2', agent={'reasoning_effort': 'low'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    db = authority.db
    _legacy(db, home, 'old-tui', source='tui', model='saved-model', model_config={
        'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'api_mode': 'chat_completions',
        'reasoning_config': {'enabled': True, 'effort': 'high'}})
    bot_config = {'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'api_mode': 'codex_responses'}
    if marker == 'follow-marker':
        bot_config['follow_profile_config'] = True
    _legacy(db, home, 'old-bot', source='gui', model='old-bot-model', model_config=bot_config,
            title='Bot Chat' if marker == 'title' else 'Renamed later')
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        tui = authority.sessions[resolve_local_target(authority, owner.actor, 'old-tui').session_id]
        bot = authority.sessions[resolve_local_target(authority, owner.actor, 'old-bot').session_id]
        route = _route(runner, tui)
        reasoning = policy_for_source(runner, tui.source).current_reasoning_config(runner)
        assert (route[:2], reasoning.get('effort')) == (('saved-model', 'http://127.0.0.1:1/v1'), 'high')
        assert _key(runner, authority, tui) not in ('k2', 'k3')
        assert _follows(runner, bot)
        assert _route(runner, bot)[:2] == ('profile-model', 'http://127.0.0.1:2/v1')
        assert _key(runner, authority, bot) == 'k2'
        frozen = json.loads(policy_for_source(runner, bot.source).config_json).get('model') or {}
        assert frozen.get('api_mode') != 'codex_responses'
        _profile(home, 'moved-model', 3, 'k3', agent={'reasoning_effort': 'low'})
        assert _route(runner, bot)[:2] == ('moved-model', 'http://127.0.0.1:3/v1')
        assert _key(runner, authority, bot) == 'k3'
        assert _key(runner, authority, tui) not in ('k2', 'k3')
        assert _route(runner, tui)[:2] == ('saved-model', 'http://127.0.0.1:1/v1')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_adopted_desktop_chat_keeps_its_saved_route(home):
    from gateway.session_local_migration import resolve_local_target
    from gateway.session_policy import policy_for_source
    _profile(home, 'profile-model', 2, 'k2', agent={'reasoning_effort': 'low'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    db = authority.db
    _legacy(db, home, 'old-desktop', source='desktop', model='saved-model', model_config={
        'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'api_mode': 'chat_completions',
        'reasoning_config': {'enabled': True, 'effort': 'high'}})
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        live = authority.sessions[resolve_local_target(authority, owner.actor, 'old-desktop').session_id]
        reasoning = policy_for_source(runner, live.source).current_reasoning_config(runner)
        route = _route(runner, live)
        assert not _follows(runner, live)
        assert (route[:2], reasoning.get('effort')) == (('saved-model', 'http://127.0.0.1:1/v1'), 'high')
        assert _key(runner, authority, live) not in ('k2', 'k3')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['empty', 'provider-only'])
async def test_adopting_a_bot_chat_with_no_profile_model_stays_openable(home, shape):
    """A profile that names no model must not leave a Bot Chat receipt that restore rejects.

    ``{}``, and a provider plus key with no ``model.default``, both resolve to a blank model.
    Committing that blank name makes the first open ``storage_unavailable``, and every later
    open stays there after a model is added. The saved row's model is enough to open, and the
    chat follows once the profile has a default.
    """
    from gateway.session_local_migration import resolve_local_target
    from hermes_state_local import local_receipt
    if shape == 'empty':
        (home / 'config.yaml').write_text('{}\n')
    else:
        (home / 'config.yaml').write_text(json.dumps({'model': {
            'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'sk-provider-only'}}))
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    _legacy(authority.db, home, 'old-bot', source='gui', model='saved-model', title='Bot Chat',
            model_config={'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'})
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        opened = resolve_local_target(authority, owner.actor, 'old-bot')
        stored = local_receipt(authority.db, opened.session_id)['policy']['model']
        assert isinstance(stored, str) and stored.strip()
        _profile(home, 'added-model', 2, 'sk-added')
        authority.sessions.pop(opened.session_id, None)
        again = resolve_local_target(authority, owner.actor, 'old-bot')
        live = authority.sessions[again.session_id]
        assert _follows(runner, live)
        assert _route(runner, live)[:2] == ('added-model', 'http://127.0.0.1:2/v1')
        assert _key(runner, authority, live) == 'sk-added'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_adopted_bot_chat_keeps_a_composer_pick_until_the_profile_moves(home):
    from gateway.session_local_migration import resolve_local_target
    _profile(home, 'profile-model', 2, 'k2')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    _legacy(authority.db, home, 'old-bot', source='gui', model='picked-model', title='Bot Chat', model_config={
        'composer_override_profile': {'model': 'profile-model', 'provider': 'custom'}})
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        bot = authority.sessions[resolve_local_target(authority, owner.actor, 'old-bot').session_id]
        assert _follows(runner, bot)
        assert _route(runner, bot)[:2] == ('picked-model', 'http://127.0.0.1:2/v1')
        assert _key(runner, authority, bot) == 'k2'
        _profile(home, 'moved-model', 3, 'k3')
        assert _route(runner, bot)[:2] == ('moved-model', 'http://127.0.0.1:3/v1')
        assert _key(runner, authority, bot) == 'k3'
    finally:
        await owner.close()
        store.close_all_db_handles()


def _frozen_bot_chat(authority, actor, home, title='Bot Chat', **extra):
    """A chat an earlier build committed without the flag (its delivery door, a CLI titled create)."""
    from gateway.session_local import stage_local_session
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_local_title import title_new_session
    from hermes_state_local import commit_local_session
    params = {'request_id': 'earlier-build', 'source': 'cli', 'cwd': str(home), **extra}
    receipt = stage_local_session(authority, actor, dict(params))
    commit_local_session(authority.db, epoch=authority.epoch, receipt=receipt)
    ref = restore_local_session(authority, receipt['session_id'])
    title_new_session(authority, ref, title)
    return ref, params


async def _open(authority, ref):
    """Desktop opening a stored Bot Chat: ``session.resume`` by id on its own connection."""
    from gateway.session_controls import AuthorityConnection
    desktop = AuthorityConnection(authority, object(), {'user_id': 'human'})
    try:
        response = await desktop.dispatch({'id': 'open', 'method': 'session.resume',
                                           'params': {'session_id': ref.session_id}})
        assert 'result' in response, response
    finally:
        await desktop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('moved', ['before-open', 'after-open'])
async def test_opening_a_frozen_bot_chat_makes_it_follow(home, moved):
    """A Bot Chat an earlier build froze kept its old route after any open. The open now adopts
    it once, durably, and the creation request stays the exact-retry identity."""
    from gateway.session_local import stage_local_session
    from hermes_state_local import local_receipt
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        ref, params = _frozen_bot_chat(authority, owner.actor, home)
        live = authority.sessions[ref.session_id]
        request = local_receipt(authority.db, ref.session_id)['policy']['request_json']
        if moved == 'before-open':
            _profile(home, 'new-model', 2, 'new-key')
        await _open(authority, ref)
        if moved == 'after-open':
            _profile(home, 'new-model', 2, 'new-key')
        assert _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
        stored = local_receipt(authority.db, ref.session_id)['policy']
        assert stored['request_json'] == request and 'follow_adopted' not in stored
        assert json.loads(stored['config_json'])['follow_adopted'] is True
        generation = authority.db.get_session(ref.session_id)['runtime_generation']
        await _open(authority, ref)
        assert authority.db.get_session(ref.session_id)['runtime_generation'] == generation
        assert stage_local_session(authority, owner.actor, params).session_id == ref.session_id
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [{}, {'api_key': 'sk-launch-for-one'}], ids=['profile-key', 'launch-key'])
async def test_a_flipped_bot_chat_never_sends_its_frozen_key_to_the_profiles_new_endpoint(home, extra):
    """A frozen key (the profile's, or a CLI ``--api-key``) belongs to the old endpoint. After the
    open, the profile's new endpoint gets no key until the profile names one, and the private-value
    reference is re-derived: a fresh owner recovers the chat's other secrets, no frozen model key."""
    from gateway.session_policy import policy_for_source
    from hermes_state_local import local_receipt
    aux = {'auxiliary': {'vision': {'provider': 'custom', 'api_key': 'aux-key'}}}
    _profile(home, 'old-model', 1, 'old-key', **aux)
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        ref, _params = _frozen_bot_chat(authority, owner.actor, home, **extra)
        live = authority.sessions[ref.session_id]
        (home / 'config.yaml').write_text(json.dumps({'model': {
            'provider': 'custom', 'base_url': 'http://127.0.0.1:2/v1', 'default': 'new-model'}, **aux}))
        await _open(authority, ref)
        authority._local_config_secrets = {}  # a fresh owner reads the reference against config.yaml
        model, url, key = _route(runner, live)
        assert (model, url) == ('new-model', 'http://127.0.0.1:2/v1') and key not in ('old-key', 'sk-launch-for-one')
        assert _key(runner, authority, live) is None
        ref_json = local_receipt(authority.db, ref.session_id)['policy']['config_secret_ref']
        entries = [path for path, _ in json.loads(ref_json.split(':', 1)[1])['entries']]
        assert entries == [['auxiliary', 'vision', 'api_key']]
        config = policy_for_source(runner, live.source).config(authority)
        assert config['auxiliary']['vision']['api_key'] == 'aux-key'
        _profile(home, 'new-model', 2, 'new-key', **aux)
        assert _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['other-title', 'chose-frozen', 'safe-mode'])
async def test_opening_leaves_other_chats_and_deliberate_freezes_frozen(home, kind):
    from hermes_state_local import local_receipt
    extra = {'other-title': {'title': 'Notes'}, 'chose-frozen': {'follow_profile_config': False},
             'safe-mode': {'safe_mode': True, 'model': 'old-model', 'provider': 'custom',
                           'base_url': 'http://127.0.0.1:1/v1'}}[kind]
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        ref, _params = _frozen_bot_chat(authority, owner.actor, home, **extra)
        live = authority.sessions[ref.session_id]
        _profile(home, 'new-model', 2, 'new-key')
        await _open(authority, ref)
        assert not _follows(runner, live)
        assert _route(runner, live)[:2] == ('old-model', 'http://127.0.0.1:1/v1')
        # An older owner restores this receipt: it gains no field.
        assert 'follow_adopted' not in local_receipt(authority.db, ref.session_id)['policy']
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_opening_a_frozen_adopted_bot_chat_holds_its_composer_pick_until_the_profile_moves(home, monkeypatch):
    """An earlier build adopted this pre-upgrade Bot Chat frozen. The open applies adoption's rule."""
    from gateway import session_local_migration
    from gateway.session_local_migration import resolve_local_target
    _profile(home, 'profile-model', 2, 'k2')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    _legacy(authority.db, home, 'old-bot', source='gui', model='picked-model', title='Bot Chat', model_config={
        'composer_override_profile': {'model': 'profile-model', 'provider': 'custom'}})
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        with monkeypatch.context() as earlier:
            earlier.setattr(session_local_migration, 'legacy_bot_chat_launch',
                            lambda row, params, mode, config: (params, mode))
            ref = resolve_local_target(authority, owner.actor, 'old-bot')
        bot = authority.sessions[ref.session_id]
        assert not _follows(runner, bot)
        await _open(authority, ref)
        assert _follows(runner, bot)
        assert _route(runner, bot)[:2] == ('picked-model', 'http://127.0.0.1:2/v1')
        assert _key(runner, authority, bot) == 'k2'
        _profile(home, 'moved-model', 3, 'k3')
        assert _route(runner, bot)[:2] == ('moved-model', 'http://127.0.0.1:3/v1')
        assert _key(runner, authority, bot) == 'k3'
    finally:
        await owner.close()
        store.close_all_db_handles()


def _named_profile(home, monkeypatch, section, credential, url, secret):
    """The profile runs ``custom:primary``; a second named entry ``picked`` serves *url* with *secret*
    held inline, through ``key_env`` or through ``key_cmd``."""
    entry = {'base_url': url}
    if credential == 'inline':
        entry['api_key'] = secret
    elif credential == 'env':
        monkeypatch.setenv('PICKED_ENTRY_KEY', secret)
        entry['key_env'] = 'PICKED_ENTRY_KEY'
    else:
        entry['key_cmd'] = f'echo {secret}'
    primary = {'base_url': 'http://127.0.0.1:2/v1', 'api_key': 'primary-key-long'}
    config = {'model': {'provider': 'custom:primary', 'default': 'profile-model'}}
    if section == 'providers':
        config['providers'] = {'primary': primary, 'picked': entry}
    else:
        config['custom_providers'] = [{'name': 'primary', **primary}, {'name': 'picked', **entry}]
    (home / 'config.yaml').write_text(json.dumps(config))


@pytest.mark.asyncio
@pytest.mark.parametrize('section, credential, path', [
    ('providers', 'inline', 'adoption'), ('providers', 'env', 'open'),
    ('custom_providers', 'inline', 'open'), ('custom_providers', 'command', 'adoption')])
async def test_a_held_pick_never_sends_its_named_entrys_new_key_to_the_old_endpoint(
        home, monkeypatch, section, credential, path):
    """A held composer pick on the named entry ``picked`` keeps the endpoint it was made on. Once that
    entry moves to another endpoint and key, its credential belongs there: the hold ends."""
    from gateway import session_local_migration
    from gateway.session_local_migration import resolve_local_target
    _named_profile(home, monkeypatch, section, credential, 'http://127.0.0.1:5/v1', 'picked-key-one')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    _legacy(authority.db, home, 'old-bot', source='gui', model='picked-model', title='Bot Chat', model_config={
        'provider': 'custom:picked', 'base_url': 'http://127.0.0.1:5/v1',
        'composer_override_profile': {'model': 'profile-model', 'provider': 'custom:primary'}})
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        with monkeypatch.context() as earlier:
            if path == 'open':
                earlier.setattr(session_local_migration, 'legacy_bot_chat_launch',
                                lambda row, params, mode, config: (params, mode))
            ref = resolve_local_target(authority, owner.actor, 'old-bot')
        if path == 'open':
            await _open(authority, ref)
        bot = authority.sessions[ref.session_id]
        assert _route(runner, bot)[:2] == ('picked-model', 'http://127.0.0.1:5/v1')
        _named_profile(home, monkeypatch, section, credential, 'http://127.0.0.1:6/v1', 'picked-key-two')
        assert _route(runner, bot) == ('profile-model', 'http://127.0.0.1:2/v1', 'primary-key-long')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('pick', [{'provider': 'custom:other'}, {'base_url': 'http://127.0.0.1:7/v1'}],
                         ids=['provider-only', 'endpoint-only'])
async def test_a_route_pick_without_a_model_holds_on_a_following_create(home, pick):
    """``chat -c "Bot Chat" --create-if-missing --provider X`` follows the profile, and its pick holds
    like a ``-m`` pick until the profile's model or provider changes."""
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'default': 'shared-model',
                  'api_key': 'profile-key-one'},
        'providers': {'other': {'base_url': 'http://127.0.0.1:7/v1', 'api_key': 'other-key-long'}}}))
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        response = await owner.dispatch({'id': 'cli', 'method': 'session.create', 'params': dict(
            request_id='cli-pick', source='cli', cwd=str(home), title='Bot Chat', **pick)})
        assert 'result' in response, response
        live = authority.sessions[response['result']['session_id']]
        assert _follows(runner, live)
        model, url, key = _route(runner, live)
        assert (model, url) == ('shared-model', 'http://127.0.0.1:7/v1') and key != 'profile-key-one'
        _profile(home, 'moved-model', 3, 'moved-key-long')
        assert _route(runner, live) == ('moved-model', 'http://127.0.0.1:3/v1', 'moved-key-long')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('by', ['session_id', 'title'])
async def test_a_busy_first_open_leaves_adoption_to_the_next_open_on_that_connection(home, monkeypatch, by):
    """Desktop keeps its connection: a warm reopen (``session.activate`` is ``session.resume``) must still
    adopt a chat whose first open found a turn running."""
    import hermes_state_mutation_guards as guards
    from gateway.session_controls import AuthorityConnection
    from hermes_state_runtime import RuntimeStoreError
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    real, refused = guards.require_idle, []

    def busy_once(db, conn, ids):
        if not refused:
            refused.append(ids)
            raise RuntimeStoreError('session_busy')
        return real(db, conn, ids)
    monkeypatch.setattr(guards, 'require_idle', busy_once)
    desktop = AuthorityConnection(authority, object(), {'user_id': 'human'})
    try:
        ref, _params = _frozen_bot_chat(authority, owner.actor, home)
        live = authority.sessions[ref.session_id]
        _profile(home, 'new-model', 2, 'new-key')
        params = {'session_id': ref.session_id} if by == 'session_id' else {'title': 'Bot Chat'}
        for attempt in ('busy', 'idle'):
            response = await desktop.dispatch({'id': attempt, 'method': 'session.resume', 'params': dict(params)})
            assert 'result' in response, response
            assert _follows(runner, live) is (attempt == 'idle')
        assert refused and _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
    finally:
        await desktop.close()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_opening_a_frozen_bot_chat_keeps_its_committed_reasoning_choice(home):
    """A ``/reasoning high`` committed on the frozen chat is the session's choice: the open adopts the
    profile's route, not the profile's ``low``."""
    from gateway.session_policy import policy_for_source
    _profile(home, 'old-model', 1, 'old-key', agent={'reasoning_effort': 'low'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._set_session_reasoning_override = lambda route, value: None
    try:
        ref, _params = _frozen_bot_chat(authority, owner.actor, home)
        live = authority.sessions[ref.session_id]
        snap = authority.db.get_session(ref.session_id)
        response = await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': dict(
            session_id=ref.session_id, request_id='high', operation='reasoning', payload={'effort': 'high'},
            expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])})
        assert 'result' in response, response
        _profile(home, 'new-model', 2, 'new-key', agent={'reasoning_effort': 'low'})
        await _open(authority, ref)
        assert _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
        assert policy_for_source(runner, live.source).current_reasoning_config(runner).get('effort') == 'high'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('first', ['delivery', 'opened'])
async def test_a_reasoning_choice_after_the_follow_beats_the_profile_level(home, first):
    """``/reasoning high`` on a Bot Chat that already follows its profile is the session's choice:
    the next turn runs ``high``, not the profile's ``low`` (a delivery-minted chat, and a frozen
    chat that an open adopted)."""
    from gateway.session_bot import _target
    from gateway.session_policy import policy_for_source
    _profile(home, 'old-model', 1, 'old-key', agent={'reasoning_effort': 'low'})
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._set_session_reasoning_override = lambda route, value: None
    try:
        if first == 'delivery':
            ref, live, _entry = await _target(authority, owner.actor)
        else:
            ref, _params = _frozen_bot_chat(authority, owner.actor, home)
            live = authority.sessions[ref.session_id]
            await _open(authority, ref)
        assert _follows(runner, live)
        snap = authority.db.get_session(ref.session_id)
        response = await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': dict(
            session_id=ref.session_id, request_id='high', operation='reasoning', payload={'effort': 'high'},
            expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])})
        assert 'result' in response, response
        policy = policy_for_source(runner, live.source)
        assert policy.follows_profile
        assert policy.current_reasoning_config(runner).get('effort') == 'high'
        _profile(home, 'new-model', 2, 'new-key', agent={'reasoning_effort': 'minimal'})
        policy = policy_for_source(runner, live.source)
        assert _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
        assert policy.current_reasoning_config(runner).get('effort') == 'high'
    finally:
        await owner.close()
        store.close_all_db_handles()


def test_follows_profile_parses_each_frozen_blob_once_per_policy(monkeypatch):
    """``follows_profile`` is read ~8-10 times a turn on the event loop. A frozen session's policy holds
    a whole profile config (tens of KB), so each read must not re-parse it."""
    import json as real_json
    from dataclasses import replace
    from types import SimpleNamespace
    from gateway import session_policy
    from gateway.session_policy import FOLLOW_ADOPTED, LocalSessionPolicy
    config = real_json.dumps({'model': {'default': 'm'}, 'pad': 'x' * 20000})
    parsed = []

    def loads(text, *a, **kw):
        parsed.append(text)
        return real_json.loads(text, *a, **kw)
    monkeypatch.setattr(session_policy, 'json', SimpleNamespace(loads=loads, dumps=real_json.dumps))
    policy = LocalSessionPolicy(source='cli', platform='cli', cwd='/', model='m', toolsets=(),
                                config_json=config, request_json='{}', terminal_json='{}')
    assert [policy.follows_profile for _ in range(10)] == [False] * 10
    assert parsed.count(config) <= 1 and parsed.count('{}') <= 1
    adopted = replace(policy, config_json=real_json.dumps({FOLLOW_ADOPTED: True}))
    assert adopted.follows_profile is True and policy.follows_profile is False


@pytest.mark.asyncio
async def test_an_mcp_reload_keeps_a_busy_first_opens_adoption_pending(home, monkeypatch):
    """``reload.mcp`` refreshes only the MCP section (it never runs ``follow_bot_chat``): it must not
    forget an adoption a busy first open left pending, or the warm reopen returns early on the old route."""
    import hermes_state_mutation_guards as guards
    from gateway.session_controls import AuthorityConnection
    from gateway.session_policy_refresh import commit_refresh
    from hermes_state_runtime import RuntimeStoreError
    _profile(home, 'old-model', 1, 'old-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    real, refused = guards.require_idle, []

    def busy_once(db, conn, ids):
        if not refused:
            refused.append(ids)
            raise RuntimeStoreError('session_busy')
        return real(db, conn, ids)
    monkeypatch.setattr(guards, 'require_idle', busy_once)
    desktop = AuthorityConnection(authority, object(), {'user_id': 'human'})
    try:
        ref, _params = _frozen_bot_chat(authority, owner.actor, home)
        live = authority.sessions[ref.session_id]
        _profile(home, 'new-model', 2, 'new-key')
        params = {'session_id': ref.session_id}
        response = await desktop.dispatch({'id': 'busy', 'method': 'session.resume', 'params': dict(params)})
        assert 'result' in response and refused and not _follows(runner, live), response
        await commit_refresh(authority, ref, mcp_only=True)  # reload.mcp on that idle chat
        assert not _follows(runner, live)
        response = await desktop.dispatch({'id': 'warm', 'method': 'session.resume', 'params': dict(params)})
        assert 'result' in response, response
        assert _follows(runner, live) and _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
    finally:
        await desktop.close()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_deferred_model_pick_that_was_refused_is_no_pick_when_an_open_adopts_the_chat(home, monkeypatch):
    """A ``/model`` sent while a turn ran is deferred to the turn boundary, where it no longer resolved and
    was dropped. Its receipt stays (exact retries), but no selection was committed: the open follows the
    profile's current model."""
    import asyncio
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from gateway.session_contract import Submission
    from hermes_cli import model_switch
    from tests.waiting import async_wait_until
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=False, new_model='', target_provider=''))
    _profile(home, 'old-model', 1, 'profile-key')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    runner._set_session_reasoning_override = lambda route, value: None
    runner._resolve_session_agent_runtime = (
        lambda **k: GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, **k))
    entered, release, published = asyncio.Event(), asyncio.Event(), []

    async def turn(event):
        entered.set()
        await release.wait()
        return 'ok'
    runner._handle_message = turn
    try:
        ref, _params = _frozen_bot_chat(authority, owner.actor, home)
        live = authority.sessions[ref.session_id]
        publish = live.event_stream.publish
        monkeypatch.setattr(live.event_stream, 'publish', lambda sid, payload, **kw: (
            published.append(payload), publish(sid, payload, **kw))[1])
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await asyncio.wait_for(entered.wait(), 10)
        snap = authority.db.get_session(ref.session_id)
        response = await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': dict(
            session_id=ref.session_id, request_id='pick', operation='model', payload={'model': 'gone-model'},
            expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])})
        assert response['result']['status'] == 'deferred', response
        release.set()
        await async_wait_until(lambda: any(p.get('status') == 'refused' for p in published), timeout=10,
                               what='the deferred pick to be refused at the turn boundary')
        _profile(home, 'new-model', 2, 'new-key')
        await _open(authority, ref)
        assert _follows(runner, live) and _route(runner, live) == ('new-model', 'http://127.0.0.1:2/v1', 'new-key')
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()
