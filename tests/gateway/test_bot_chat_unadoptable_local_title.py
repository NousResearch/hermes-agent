"""A local row that holds the canonical "Bot Chat" title but no native open can ever adopt does not
refuse every Bot Chat delivery for good (#141243).

A crash or reconnect can leave a CLI/TUI/Desktop row titled "Bot Chat" with a route column but no
local binding (or another profile's stamp). ``adopt_legacy_session`` refuses it for every caller, so
a cron ``deliver: bot-chat`` failed ``not_found`` on every beat. The door now moves it aside, as it
does an API or messaging row, and mints the profile's local Bot Chat; adoptable history stays put.
"""
from pathlib import Path

import pytest

from hermes_state_runtime import RuntimeStoreError

from tests.gateway import test_peer_dm_bot_chat_headless as _headless_gateway

# The headless gateway fixture: one real authority over a SessionDB in a temp HERMES_HOME.
headless = _headless_gateway.headless
_local_owner = _headless_gateway._local_owner


def _squatter(db, sid, *, column, value):
    db.create_session(sid, 'tui')
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


@pytest.mark.parametrize('column, value', [(None, None), ('user_id', 'github:alice')])
@pytest.mark.asyncio
async def test_kept_local_bot_chat_history_is_never_moved_aside(headless, column, value):
    # A plain pre-upgrade TUI Bot Chat is adopted in place; a row a login stamped as its own stays
    # that login's (refused, never renamed). Neither loses its title.
    db = headless.authority.db
    old = _squatter(db, 'tui_bot_chat', column=column, value=value)
    db.append_message(old, 'user', 'earlier')
    try:
        record = await _deliver(headless.authority, 'c')
    except RuntimeStoreError as exc:
        assert column == 'user_id' and exc.reason == 'not_found'
    else:
        assert column is None and record['session_id'] == old
    assert db.get_session_by_title('Bot Chat')['id'] == old and len(db._read_all('SELECT id FROM sessions')) == 1
