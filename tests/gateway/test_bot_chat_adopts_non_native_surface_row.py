"""The canonical ``Bot Chat`` is adopted whoever minted it, not only by the local creation lanes.

A Bot Chat written by a surface with no local creation policy of its own — the dashboard's
basic-auth login (``source='mobile'``, ``user_id='basic:<login>'``), a mobile client — is stored
with no route columns. ``validate_legacy_row`` then refused it for every caller on two independent
counts: its ``source`` is not one of ``LEGACY_LOCAL_SOURCES``, and its ``user_id`` stamp is a
foreign principal. ``adopt_legacy_session`` could never run, the restore chain has nothing to
rebuild from, and every read answered ``not_found``.

That is fatal for the Bot Chat specifically: Bot Mode resolves the forever-chat by exact title and
Desktop fails closed on a lookup error rather than minting a replacement, so one row written by
another device made the bot unopenable on every device, permanently — the same bot's chats ended up
segregated by creation surface, with one of them a black hole.

The chat belongs to the profile (that IS the identity contract), so the profile's operator may adopt
it from any surface. Ownership for everyone else is unchanged: a login-stamped chat stays that
login's, so a non-operator caller still meets a foreign owner, and no exemption applies to a row that
carries a route or belongs to another profile.
"""
import pytest

from tests.gateway.test_follow_profile_followups import _profile
from tests.gateway.test_session_follow_profile_config import _owner

BOT_CHAT = 'Bot Chat'


@pytest.fixture
def home(monkeypatch):
    """The profile home the adoption tests write ``config.yaml`` into."""
    from gateway import run
    from hermes_constants import get_hermes_home
    profile = get_hermes_home()
    profile.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', profile)
    return profile


def _legacy(db, profile, sid, **row):
    title = row.pop('title', None)
    db.create_session(sid, cwd=str(profile), **row)
    if title:
        db.set_session_title(sid, title)
    db.end_session(sid, 'tui_close')


def _stamp(db, sid, **columns):
    db._execute_write(lambda conn: conn.execute(
        'UPDATE sessions SET ' + ','.join(f'{name}=?' for name in columns) + ' WHERE id=?',
        (*columns.values(), sid)))


def _bot_row(db, home, sid, **overrides):
    """An unrouted Bot Chat minted by the dashboard/mobile surface: no route, foreign stamp."""
    row = {'source': 'mobile', 'user_id': 'basic:herald', 'model': 'saved-model', 'title': BOT_CHAT,
           'model_config': {'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1',
                            'api_mode': 'chat_completions'}}
    row.update(overrides)
    _legacy(db, home, sid, **row)
    return sid


def _operator(authority):
    """A verified transport gate of this profile — the capability the desktop's own connection holds.

    ``AuthorityConnection`` strips ``session:operator`` from identity data and re-adds it only for
    ``operator=True``, so this is the real gate, not a forged capability set.
    """
    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'}, operator=True)
    authority._native_legacy_transports = {connection.actor.transport_id: connection.actor}
    return connection


@pytest.mark.asyncio
async def test_unrouted_bot_chat_minted_by_a_non_native_surface_is_adopted(home):
    from gateway.session_local_migration import resolve_local_target
    from gateway.session_policy import policy_for_source
    from hermes_state_local import local_receipt

    _profile(home, 'profile-model', 2, 'k2')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    db = authority.db
    sid = _bot_row(db, home, '20261027_112018_c9b976')
    operator = _operator(authority)
    try:
        ref = resolve_local_target(authority, operator.actor, sid)
        # Adopted in place: the transcript keeps its id, so no history is copied or lost.
        assert ref.session_id == sid
        row = db.get_session(sid)
        assert str(row['chat_id']) == sid
        assert str(row['session_key']).startswith('agent:')
        stored = local_receipt(db, sid)
        assert stored['session_id'] == sid and stored['policy']['model']
        # The chat follows its profile, exactly as one minted through the delivery door does.
        assert policy_for_source(runner, authority.sessions[sid].source).follows_profile
        # A second open resolves the adopted live session: no second adoption, no second chat.
        assert resolve_local_target(authority, operator.actor, sid).session_id == sid
        assert db.get_session_by_title(BOT_CHAT)['id'] == sid
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_login_stamped_bot_chat_stays_that_logins_for_a_non_operator(home):
    """#140183 F1: the exemption is the operator's, not a licence for any caller to take a chat a
    login stamped as its own."""
    from gateway.session_local_migration import resolve_local_target
    from hermes_state_runtime import RuntimeStoreError

    _profile(home, 'profile-model', 2, 'k2')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    db = authority.db
    sid = _bot_row(db, home, 'stamped-bot-chat', user_id='github:alice')
    authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
    try:
        with pytest.raises(RuntimeStoreError) as refused:
            resolve_local_target(authority, owner.actor, sid)
        assert refused.value.reason == 'not_found'
        assert db.get_session(sid)['title'] == BOT_CHAT
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_non_bot_chat_row_from_the_same_surface_still_refuses_adoption(home):
    """The relaxation is the canonical title's, not the surface's: an ordinary mobile chat on a
    foreign login stays unadoptable."""
    from gateway.session_local_migration import resolve_local_target
    from hermes_state_runtime import RuntimeStoreError

    _profile(home, 'profile-model', 2, 'k2')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    db = authority.db
    _legacy(db, home, 'mobile-notes', source='mobile', user_id='basic:herald', title='Notes')
    operator = _operator(authority)
    try:
        with pytest.raises(RuntimeStoreError) as refused:
            resolve_local_target(authority, operator.actor, 'mobile-notes')
        assert refused.value.reason == 'not_found'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_the_bot_chat_still_has_to_be_unrouted_and_on_this_profile(home):
    """The Bot Chat exemption lifts the source/principal tests only."""
    from gateway.session_local_migration import resolve_local_target
    from hermes_state_runtime import RuntimeStoreError

    _profile(home, 'profile-model', 2, 'k2')
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    db = authority.db
    _bot_row(db, home, 'held-route')
    _stamp(db, 'held-route', session_key='agent:main:tui:dm:stale')
    _bot_row(db, home, 'other-profile')
    _stamp(db, 'other-profile', profile_name='another-profile')
    operator = _operator(authority)
    try:
        for sid in ('held-route', 'other-profile'):
            with pytest.raises(RuntimeStoreError) as refused:
                resolve_local_target(authority, operator.actor, sid)
            assert refused.value.reason == 'not_found', sid
    finally:
        await owner.close()
        store.close_all_db_handles()
