"""Cron thread-seed must key EXACTLY like the reply that will continue it.

Live incident (Alice, 2026-08-20 01:08, job 8e21a957b77b): the continuable
cron thread seed created its session row with chat_type="thread", but a
Slack DM thread reply arrives with chat_type="dm" — build_session_key puts
them in different rows (agent:main:slack:thread:D...:<ts> vs
agent:main:slack:dm:D...:<ts>), so the user's reply hit a session that had
never seen the brief. Channels are unaffected (channel thread replies carry
chat_type="thread"); the DM lane is the unswept sibling of the flat-seed
is_dm fix (dcca9d8cfe).

Contract under test: the KEY of the seeded session equals the KEY the
user's in-thread reply will build. Asserting on build_session_key output —
not on SessionSource field shapes — pins the end-to-end contract.
"""

from unittest.mock import MagicMock, patch

from cron.scheduler import _seed_cron_thread_session
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _seeded_source(store):
    store.get_or_create_session.assert_called_once()
    return store.get_or_create_session.call_args[0][0]


def test_dm_thread_seed_key_matches_dm_reply_key():
    """A brief threaded under a Slack DM must seed the same session row a
    DM in-thread reply resolves to (chat_type='dm', not 'thread')."""
    store = MagicMock()
    adapter = MagicMock()
    adapter._session_store = store

    with patch("gateway.mirror.mirror_to_session", return_value=True):
        _seed_cron_thread_session(
            {"id": "j1", "name": "digest"}, adapter, "slack",
            "D0BJTDCSR7C", "1787188136.448949", "Three bullets",
            chat_name=None, is_dm=True,
        )

    reply_source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D0BJTDCSR7C",
        chat_type="dm",
        user_id="U0B5F8EEYAD",
        thread_id="1787188136.448949",
    )
    assert build_session_key(_seeded_source(store)) == build_session_key(
        reply_source
    ), (
        "seeded key diverges from the DM reply's key — the brief lands in a "
        "row no reply ever resolves to (continuation amnesia)"
    )


def test_channel_thread_seed_key_matches_thread_reply_key():
    """Channel behavior must NOT regress: a channel thread reply keys as
    chat_type='thread' (participant-shared), and the seed must keep matching
    it."""
    store = MagicMock()
    adapter = MagicMock()
    adapter._session_store = store

    with patch("gateway.mirror.mirror_to_session", return_value=True):
        _seed_cron_thread_session(
            {"id": "j2", "name": "digest"}, adapter, "slack",
            "C0AAAAAAAA", "1787188000.000100", "Three bullets",
            chat_name="ops", is_dm=False,
        )

    reply_source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0AAAAAAAA",
        chat_type="thread",
        user_id="U0B5F8EEYAD",
        thread_id="1787188000.000100",
    )
    assert build_session_key(_seeded_source(store)) == build_session_key(
        reply_source
    )


def test_dm_seed_default_is_backward_compatible():
    """Callers that don't pass is_dm keep today's thread-keyed behavior —
    the new parameter must not silently rekey non-DM call sites."""
    store = MagicMock()
    adapter = MagicMock()
    adapter._session_store = store

    with patch("gateway.mirror.mirror_to_session", return_value=True):
        _seed_cron_thread_session(
            {"id": "j3"}, adapter, "telegram", "123", "9001", "brief",
        )

    assert _seeded_source(store).chat_type == "thread"
