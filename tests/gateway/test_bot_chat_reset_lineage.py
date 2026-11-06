"""The API door resolves a Bot Chat's title row to its local lineage owner first.

A local reset (``/new``, compression exhaustion) forks a physical segment: the receipt
keeps the creation id as owner and advances ``lineage`` to the new head, while the title
stays on the creation id (or on a superseded segment after a re-title). ``get_compression_tip``
deliberately does not follow reset forks, so keying either door off the title row id alone
missed the post-reset head: a peer turn addressed at the head segment was not admitted to
the FIFO and ran as an API turn, and the peer-dm create-if-missing recovery returned ``[]``.

Same bug class as #145371 (``tools/bot_live_delivery.py`` discovery), different lane.
"""
from contextlib import closing
from dataclasses import asdict
from types import SimpleNamespace

from hermes_state import SessionDB
import hermes_state_runtime as rt
from gateway.platforms import api_server_bot_chat as bot_chat


def _local_session(db, epoch, cwd):
    from hermes_state_local import commit_local_session
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity
    from gateway.session_policy import build_policy
    sid = local_identity('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(cwd), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    return sid


def _mutate(db, epoch, sid, operation):
    session = db.get_session(sid)
    return rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=sid,
        request_id=operation + '-' + str(session['runtime_revision']), operation=operation,
        expected_revision=session['runtime_revision'], expected_generation=session['runtime_generation'],
        payload={})


def test_a_head_addressed_peer_turn_resolves_after_a_reset(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        owner = _local_session(db, epoch, tmp_path)
        db.set_session_title(owner, 'Bot Chat')
        head = _mutate(db, epoch, owner, 'reset')['target_session_id']
        authority = SimpleNamespace(db=db)

        assert db.get_session_by_title('Bot Chat')['id'] == owner
        # A client that lists the reset lineage as its latest segment addresses the head.
        assert bot_chat.bot_chat_tip(authority, head)['id'] == head
        # The owner id is what `hermes peer dm` resolves by title; it admits to the same FIFO.
        assert bot_chat.bot_chat_tip(authority, owner)['id'] == head
        assert bot_chat.bot_chat_tip(authority, 'not-the-bot-chat') is None


def test_a_title_on_a_superseded_segment_still_finds_the_reset_lineage(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        owner = _local_session(db, epoch, tmp_path)
        db.set_session_title(owner, 'Bot Chat')
        # Both resets run from the logical owner: each forks a new physical segment and
        # ends the previous one, so `mid` is itself a reset segment after the second.
        mid = _mutate(db, epoch, owner, 'reset')['target_session_id']
        head = _mutate(db, epoch, owner, 'reset')['target_session_id']
        # A re-title leaves the name on the superseded segment, whose own key has no receipt.
        db.set_session_title(owner, '')
        db.set_session_title(mid, 'Bot Chat')

        assert db.get_session_by_title('Bot Chat')['id'] == mid
        assert [row['id'] for row in bot_chat._reset_local_title_owner(db, 'Bot Chat', True)] == [mid]

        authority = SimpleNamespace(db=db)
        assert bot_chat.bot_chat_tip(authority, head)['id'] == head
        assert bot_chat.bot_chat_tip(authority, owner)['id'] == head
        assert bot_chat.bot_chat_tip(authority, mid)['id'] == head
