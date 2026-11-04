"""A Bot Chat retired by a housekeeping sweep keeps its deletion fence past the retry horizon.

Sweep markers age out at the horizon, but a Bot Chat's must not: its mailbox receipts outlive the
horizon, and the generation walk would otherwise reissue the swept id to a new Bot Chat.
"""


def test_a_swept_bot_chat_generation_is_never_reissued(owner, tmp_path):
    from gateway.session_bot import _bot_chat_creation_id
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_state_mutation_retirement import RETIRED_PREFIX, prune_retired_tombstones
    owner.runner.adapters = {}
    actor = AuthorityConnection(owner, object(), {'user_id': 'human'}).actor
    request_id = _bot_chat_creation_id(owner, actor)
    ref = create_local_session(owner, actor, dict(source='cli', cwd=str(tmp_path), model='m', toolsets=[],
                                                  request_id=request_id))
    owner.db.end_session(ref.session_id, 'test')
    assert owner.db.prune_sessions(older_than_days=None) == 1
    owner.db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
    assert owner.db.get_meta(RETIRED_PREFIX + ref.session_id) is not None
    assert _bot_chat_creation_id(owner, actor) != request_id
