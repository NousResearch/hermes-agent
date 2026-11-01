"""FIX-2087: the bot-chat preflight consistency guard must refuse only on a stable source-identity
change, never on a per-message field.

A restored live source keeps the ``message_id`` of the turn that created it, while every delivery
rebuilds the canonical ``local_source`` without one. Comparing the full wire dict (``to_dict()``)
used to false-diverge on that message_id and kill the profile's bot-chat delivery for the whole
process lifetime — 118 consecutive ``storage_unavailable`` refusals, 10-25 -> 10-30, one per hourly
cron tick per gateway generation, healed only by a restart. The guard now compares the stable source
identity (the same tuple ``LocalSessionAdapter.authorize_source`` gates on).
"""
from types import SimpleNamespace

import pytest


def _runner(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    return SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)


async def _registered_local_session(tmp_path):
    """A local session whose restore has taken the register branch: canonical ``local_source``
    (``message_id=None``) live and registered with the LOCAL adapter, ready to be re-restored."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    authority = await initialize_session_authority(_runner(tmp_path), profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(authority, actor, {'request_id': 'r1', 'source': 'gui',
                                                  'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
    sid = ref.session_id
    # Force the register branch (the launch authority may already hold it live after create): the
    # first restore then registers the canonical local_source with the LOCAL adapter.
    authority.sessions.pop(sid, None)
    restore_local_session(authority, sid)
    return authority, sid


@pytest.mark.asyncio
async def test_message_id_only_source_difference_does_not_refuse(tmp_path):
    """The live source carrying a turn's message_id is the SAME session (stable identity intact);
    the preflight must admit the next delivery, not refuse it. Red on the base to_dict() comparison,
    green with the _identity comparison."""
    from gateway.session_local_recovery import restore_local_session
    authority, sid = await _registered_local_session(tmp_path)
    live = authority.sessions[sid]
    assert live.source.message_id is None

    # A turn processing a message stamps the live source's message_id (the real Bot Chat origin
    # carries one); every later delivery rebuilds the local_source without it.
    live.source.message_id = 'bot:cafef00d'

    # Second restore: a message_id-only difference is not an identity change -> must not refuse.
    restore_local_session(authority, sid)
    assert sid in authority.sessions


@pytest.mark.asyncio
async def test_identity_change_still_refuses(tmp_path):
    """Sanity: a genuine session change (different chat) must STILL refuse, so narrowing the guard
    to the stable identity did not over-broaden it. This refusal comes from ``authorize_source``
    (the object/identity check), independent of the comparison under test."""
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_runtime import RuntimeStoreError
    authority, sid = await _registered_local_session(tmp_path)

    authority.sessions[sid].source.chat_id = 'some-other-chat'
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        restore_local_session(authority, sid)
