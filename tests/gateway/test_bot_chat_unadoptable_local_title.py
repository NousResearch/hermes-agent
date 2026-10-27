"""A local row that holds the canonical "Bot Chat" title but no native open can ever adopt does not
refuse every Bot Chat delivery for good (#141243, #143715).

A crash or reconnect can leave a CLI/TUI/Desktop row titled "Bot Chat" with a route column but no
local binding; another profile or (outside Desktop) another login may have stamped it.
``adopt_legacy_session`` refuses such a row for every caller, so a cron ``deliver: bot-chat``
failed ``not_found`` on every beat. The door now moves it aside, as it
does an API or messaging row, and mints the profile's local Bot Chat; adoptable history stays put.
"""
from pathlib import Path

import pytest

from hermes_state_runtime import RuntimeStoreError

from tests.gateway import test_peer_dm_bot_chat_headless as _headless_gateway

# The headless gateway fixture: one real authority over a SessionDB in a temp HERMES_HOME.
headless = _headless_gateway.headless
_local_owner = _headless_gateway._local_owner


def _squatter(db, sid, *, column, value, source='tui'):
    db.create_session(sid, source)
    db.set_session_title(sid, 'Bot Chat')
    if column:
        db._execute_write(lambda conn: conn.execute(f'UPDATE sessions SET {column}=? WHERE id=?', (value, sid)))
    return sid


async def _deliver(authority, key):
    from gateway.session_authorities import served_profile_name
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import _bot_chat_tip
    home = Path(authority.db.db_path).parent.resolve()
    # The cron producer's door: ``bot_relay.deliver`` on a local interactive connection, no author,
    # pinned to the Bot Chat it discovered first (``find_canonical_live_owner``).
    pinned = _bot_chat_tip(home)
    return await deliver(_local_owner(authority), dict(id=key * 32, profile=served_profile_name(home),
                                                       message='cron beat', **({'session_id': pinned} if pinned else {})))


@pytest.mark.parametrize('column, value', [
    ('session_key', 'agent:main:tui:dm:stale'),   # route left by a crash, never bound
    ('profile_name', 'another-profile'),          # stamped by another profile
])
@pytest.mark.asyncio
async def test_an_unadoptable_local_bot_chat_is_moved_aside_and_the_delivery_lands(headless, column, value):
    from tests.waiting import async_wait_until
    db = headless.authority.db
    old = _squatter(db, '20261018_071332_c8e3e4', column=column, value=value)
    first = await _deliver(headless.authority, 'a')  # pinned to the squatter: the beat still lands
    new = first['session_id']
    assert new != old and str(db.get_session(new)['chat_id']).startswith('local-'), first
    assert db.get_session_by_title('Bot Chat')['id'] == new
    assert db.get_session(old)['title'] == f'Bot Chat (tui {old})', 'the old row keeps its id under a visible name'
    # Every later beat finds the minted chat by title: no further move, no second Bot Chat.
    second = await _deliver(headless.authority, 'b')
    assert second['session_id'] == new
    await async_wait_until(lambda: len(headless.executed) == 2, timeout=10, interval=.05)
    assert [sid for sid, _, _ in headless.executed] == [new, new]
    assert len(db._read_all('SELECT id FROM sessions')) == 2


@pytest.mark.parametrize('source, stamp, route', [
    ('tui', 'github:alice', None),
    ('tui', 'nous:alice@example.com', None),
    ('cli', 'basic:alice', None),
    ('tui', 'github:alice', 'agent:main:tui:dm:stale'),  # a crash's route column, never bound
    ('tui', 'uid:1001', None),                           # another OS user, not this host's identity
    ('desktop', 'auth:v1:["basic","","alice"]', 'agent:main:tui:dm:stale'),  # a subject, but routed
])
@pytest.mark.asyncio
async def test_a_bot_chat_another_login_stamped_is_moved_aside_and_the_delivery_lands(
        headless, monkeypatch, source, stamp, route):
    # Outside Desktop, a stamp that is neither this host's identity nor a login subject names no
    # subject a native ticket carries; a crash's route column makes a subject-stamped row (Desktop
    # too) unadoptable. No caller can adopt such a row, so it held the title and refused every
    # delivery for good (#143715). Moving it keeps its id, stamp and messages.
    from tests.waiting import async_wait_until
    import gateway.session_worker_reservation as reservation
    monkeypatch.setattr(reservation, 'owner_worker_subject', lambda: 'uid:1000')  # the fixture's native subject
    db = headless.authority.db
    old = _squatter(db, '20260924_173840_909979', column='user_id', value=stamp, source=source)
    if route:
        db._execute_write(lambda conn: conn.execute('UPDATE sessions SET session_key=? WHERE id=?', (route, old)))
    db.append_message(old, 'user', 'earlier')
    first = await _deliver(headless.authority, 'd')
    new = first['session_id']
    assert new != old and str(db.get_session(new)['chat_id']).startswith('local-'), first
    assert db.get_session_by_title('Bot Chat')['id'] == new
    moved = db.get_session(old)
    assert (moved['title'], moved['user_id'], moved['message_count']) == (f'Bot Chat ({source} {old})', stamp, 1)
    second = await _deliver(headless.authority, 'e')
    assert second['session_id'] == new
    await async_wait_until(lambda: len(headless.executed) == 2, timeout=10, interval=.05)
    assert [sid for sid, _, _ in headless.executed] == [new, new]
    assert len(db._read_all('SELECT id FROM sessions')) == 2


@pytest.mark.asyncio
async def test_a_desktop_bot_chat_a_login_stamped_is_never_moved_aside(headless):
    # A served login's own Desktop row stays that login's (#140183): refused, never renamed.
    db = headless.authority.db
    old = _squatter(db, 'desktop_bot_chat', column='user_id', value='github:alice', source='desktop')
    db.append_message(old, 'user', 'earlier')
    with pytest.raises(RuntimeStoreError) as refused:
        await _deliver(headless.authority, 'f')
    assert refused.value.reason == 'not_found'
    assert db.get_session_by_title('Bot Chat')['id'] == old and len(db._read_all('SELECT id FROM sessions')) == 1


@pytest.mark.parametrize('stamp', [None, 'uid:1000'])
@pytest.mark.asyncio
async def test_kept_local_bot_chat_history_is_adopted_in_place(headless, monkeypatch, stamp):
    # A plain pre-upgrade TUI Bot Chat, or one stamped with the host's own OS identity (the subject
    # a same-user native ticket carries, ``owner_worker_subject``), is adopted in place with its
    # history: the bot's forever-chat keeps its context instead of restarting empty.
    import gateway.session_worker_reservation as reservation
    monkeypatch.setattr(reservation, 'owner_worker_subject', lambda: 'uid:1000')  # the fixture's native subject
    db = headless.authority.db
    old = _squatter(db, 'tui_bot_chat', column='user_id' if stamp else None, value=stamp)
    db.append_message(old, 'user', 'earlier')
    record = await _deliver(headless.authority, 'c')
    assert record['session_id'] == old and db.get_session(old)['chat_id'] == old
    assert db.get_session_by_title('Bot Chat')['id'] == old and len(db._read_all('SELECT id FROM sessions')) == 1


@pytest.mark.asyncio
async def test_a_bot_chat_stamped_with_a_forwarded_login_subject_is_never_moved_aside(headless):
    # A same-user peer may mint a native ticket for a structured login subject (#138915), and that
    # ticket adopts its own row, so the row stays where it is, as on main.
    db = headless.authority.db
    old = _squatter(db, 'tui_bot_chat', column='user_id', value='auth:v1:["basic","","alice"]')
    with pytest.raises(RuntimeStoreError) as refused:
        await _deliver(headless.authority, 'f')
    assert refused.value.reason == 'not_found'
    assert db.get_session_by_title('Bot Chat')['id'] == old and len(db._read_all('SELECT id FROM sessions')) == 1
