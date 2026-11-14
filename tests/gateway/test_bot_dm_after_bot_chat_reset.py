"""A Bot Chat ``/new`` rotates the chat's session id under a stable title.

The producer pins the id it read before it asks (``_bot_chat_tip``: the titled row's *compression*
tip). A reset fork is deliberately not a compression continuation (``_is_compression_child_row``),
so after a ``/new`` that pin is the reset root while the chat's live session is the fork. The door
answers a pin that names another session ``admission_conflict``; for the same chat that refused
every bot DM for good (silently, as ``ambiguous``), so the door now admits a pin that is the same
chat's older session into the live one.
"""
import json

import pytest

from hermes_state_runtime import list_session_admissions

pytest_plugins = ['tests.gateway.test_session_bot_retry']

KEY = 'd' * 32


def _session_key(bot):
    return next(iter(bot.authority.runner.session_store._entries))


@pytest.mark.asyncio
async def test_a_pin_on_the_reset_bot_chat_admits_into_the_live_session(bot):
    from gateway.session_bot import deliver

    store = bot.authority.runner.session_store
    pinned = bot.chat
    new_entry = store.reset_session(_session_key(bot))
    assert new_entry is not None and new_entry.session_id != pinned
    # The reset left the title on the root (both rows are the same chat): the producer still pins it.
    assert bot.authority.db.get_session_by_title('Bot Chat')['id'] == pinned

    receipt = await deliver(bot.connection, dict(id=KEY, profile='default', message='ping', session_id=pinned))

    assert receipt['admission_id'] and not receipt.get('refused'), receipt
    # The turn landed in the chat's live session, not the reset root the producer pinned.
    rows = list_session_admissions(bot.authority.db, session_id=pinned, pending_only=False)
    assert [r['request_id'] for r in rows] == ['bot:' + KEY], rows
    with bot.authority.db._read_ctx() as conn:
        payload = conn.execute('SELECT payload_json FROM session_admissions WHERE request_id=?',
                               ('bot:' + KEY,)).fetchone()[0]
    assert json.loads(payload)['local_automation_v1']['target'] == new_entry.session_id, payload


@pytest.mark.asyncio
async def test_a_pin_on_another_chat_is_still_refused(bot):
    """Same-chat tolerance only: a pin naming a different chat stays an ``admission_conflict``."""
    from gateway.session_bot import deliver
    from hermes_state_runtime import RuntimeStoreError

    db = bot.authority.db
    other = db.create_session('local-otherchat', 'local', session_key='agent:default:local:dm:local-otherchat',
                              chat_id='local-otherchat', chat_type='dm',
                              user_id=bot.connection.actor.subject)
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await deliver(bot.connection, dict(id=KEY, profile='default', message='ping',
                                           session_id=other))
