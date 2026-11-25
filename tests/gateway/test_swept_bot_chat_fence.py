"""A Bot Chat retired by a housekeeping sweep keeps its deletion fence past the retry horizon.

Sweep markers age out at the horizon, but a Bot Chat's must not: its mailbox receipts outlive the
horizon, and the generation walk would otherwise reissue the swept id to a new Bot Chat.
"""
from tests.fixtures.storage_holder import reclaim_tombstones, retired_session_key


def test_a_swept_bot_chat_generation_is_never_reissued(owner, tmp_path):
    from gateway.session_bot import _bot_chat_creation_id
    from gateway.session_controls import AuthorityConnection
    from gateway.session_ingress_media import native_inputs_root
    from gateway.session_local import create_local_session
    from session_retention import after_sessions_removed
    owner.runner.adapters = {}
    actor = AuthorityConnection(owner, object(), {'user_id': 'human'}).actor
    request_id = _bot_chat_creation_id(owner, actor)
    ref = create_local_session(owner, actor, dict(source='cli', cwd=str(tmp_path), model='m', toolsets=[],
                                                  request_id=request_id))
    store = owner.persistence
    store.sessions.end(ref.session_id, 'test')
    outcome = store.maintenance.prune_sessions(older_than_days=None, media_root=str(native_inputs_root()))
    after_sessions_removed(store, store.scope.profile_dir, outcome.deleted_ids, reason='prune')
    assert outcome.count == 1
    reclaim_tombstones(store)
    assert store.state_meta.get(retired_session_key(ref.session_id)) is not None
    assert _bot_chat_creation_id(owner, actor) != request_id
