"""A staged local creation's prefill snapshot is held before its creating write commits, so a reader of the
new session never loads the file around it, and an attempt that does not create the session keeps none."""
import json

import pytest
from persistence_assembly import close_stores_under

CONFIG = ('platform_toolsets: {cli: []}\n'
          'model: {provider: custom, default: m, base_url: "http://127.0.0.1:9/v1"}\n'
          'prefill_messages_file: prefill.json\n')


def _prefill(home, mark):
    (home / 'prefill.json').write_text(json.dumps([{'role': 'user', 'content': mark}]))


async def _owner(tmp_path, monkeypatch):
    from tests.gateway.test_local_create_env_refs import _owner
    monkeypatch.delenv('HERMES_PREFILL_MESSAGES_FILE', raising=False)
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG)
    _prefill(home, 'FIRST')
    owner, connection = await _owner(home, monkeypatch)
    return home, owner, connection


def _stage(owner, connection, home, request_id):
    from gateway.session_local import stage_local_session
    return stage_local_session(owner, connection.actor, {'request_id': request_id, 'source': 'cli',
                                                         'cwd': str(home), 'toolsets': []})


@pytest.mark.asyncio
async def test_a_reader_at_the_creating_commit_gets_the_attempts_snapshot(tmp_path, monkeypatch):
    from gateway.session_local import commit_local_session
    from gateway.session_persistence import authority_persistence
    from gateway.session_policy import restore_policy
    from gateway.session_worker_construct import session_prefill
    home, owner, connection = await _owner(tmp_path, monkeypatch)
    try:
        staged = _stage(owner, connection, home, 'reader')
        sid, policy = staged['session_id'], restore_policy(staged['policy'])
        store = authority_persistence(owner)
        repo = type(store.bindings)
        real, seen = repo.reserve, {}

        def reserve(self, **kwargs):
            reserved = real(self, **kwargs)  # committed: the row is visible to every reader from here
            _prefill(home, 'EDITED')
            seen['reader'] = session_prefill(owner, policy, sid)[0]['content']
            return reserved
        monkeypatch.setattr(repo, 'reserve', reserve)
        commit_local_session(store, epoch=owner.epoch, receipt=staged)
        assert seen == {'reader': 'FIRST'}
        assert session_prefill(owner, policy, sid)[0]['content'] == 'FIRST'
    finally:
        await connection.close()
        close_stores_under(owner.persistence.scope.profile_dir)


@pytest.mark.asyncio
async def test_an_attempt_that_does_not_create_the_session_keeps_no_snapshot(tmp_path, monkeypatch):
    """An overlapping exact retry that commits after the session's snapshot is gone (released) and a refused
    attempt each withdraw the snapshot they held during their write."""
    from gateway.session_local import commit_local_session
    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import AuthorityRefusal
    home, owner, connection = await _owner(tmp_path, monkeypatch)
    store = authority_persistence(owner)
    try:
        # All three staged before any commits: same identity; the last with a different request.
        staged, retry, loser = (_stage(owner, connection, home, 'retry'), _stage(owner, connection, home, 'retry'),
                                _stage(owner, connection, tmp_path, 'retry'))
        commit_local_session(store, epoch=owner.epoch, receipt=staged)
        assert staged['session_id'] in owner._local_prefill
        owner._local_prefill.clear()
        commit_local_session(store, epoch=owner.epoch, receipt=retry)
        assert owner._local_prefill == {}
        with pytest.raises(AuthorityRefusal) as refused:
            commit_local_session(store, epoch=owner.epoch, receipt=loser)
        assert refused.value.reason == 'invalid_params'
        assert owner._local_prefill == {}
    finally:
        await connection.close()
        close_stores_under(owner.persistence.scope.profile_dir)
