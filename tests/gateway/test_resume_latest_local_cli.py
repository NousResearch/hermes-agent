"""``session.resume latest='cli'`` finds the CLI's own gateway sessions.

The owner stores a CLI-created session as ``source='local'`` with ``created_source='cli'``.
``resolve_latest_session`` searched ``source IN ('cli', 'oneshot')`` only, so ``-z --resume
latest``, ``chat -c --output-format stream-json`` and ``chat -c --safe-mode`` (all routed
through ``gateway_chat._resume_latest``) reported "No previous CLI session" for every session
the gateway had created.
"""
import json
from types import SimpleNamespace

import pytest

from gateway.session_policy import SURFACES


def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps({'model': {'default': 'm'}}))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(home / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _handle_message=None, _peek_session_state=lambda key: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    runner.session_authority = authority
    return store, authority, AuthorityConnection(authority, object(), {'user_id': 'human'})


async def _call(owner, method, **params):
    response = await owner.dispatch({'id': params.get('request_id', method), 'method': method, 'params': params})
    if 'error' in response:
        return response['error']['data']['reason']
    return response['result']


@pytest.mark.asyncio
async def test_resume_latest_matches_the_owners_local_rows_by_creating_surface(tmp_path, monkeypatch):
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        created = {}
        # Desktop last, so it is the newest row: it must never be the CLI's or TUI's "latest".
        for surface in ('cli', 'tui', 'gui'):
            result = await _call(owner, 'session.create', request_id=surface, source=surface,
                                 cwd=str(tmp_path), toolsets=[], model='m')
            created[surface] = sid = result['session_id']
            # The first turn's two writers: the agent's lazy create stamps ``created_source`` with
            # the agent surface (run_agent; gui runs as ``desktop``), then the peer refresh relabels
            # the row ``local`` (session_a2a.storage_source).
            authority.db.create_session(sid, source=SURFACES[surface])
            live = authority.sessions[sid]
            store._record_gateway_session_peer(sid, live.route, live.source)
            authority.db.append_message(sid, 'user', 'hi')
        rows = {sid: authority.db.get_session(sid) for sid in created.values()}
        assert {(row['source'], row['created_source']) for row in rows.values()} == {
            ('local', 'cli'), ('local', 'tui'), ('local', 'desktop')}
        authority.sessions.clear()

        for family in ('cli', 'tui'):
            result = await _call(owner, 'session.resume', latest=family)
            assert isinstance(result, dict) and result['session_id'] == created[family], (family, result)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_auxiliary_usage_cannot_replace_the_insert_stamp(tmp_path, monkeypatch):
    """Auxiliary usage creates a missing row as created_source='unknown'.

    The creator upsert repairs 'unknown', but session.create's DO NOTHING insert does
    not, so it stamps the surface itself; a later record_auxiliary_usage must leave it,
    and session.resume latest='cli' still finds the row.
    """
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        created = await _call(owner, 'session.create', request_id='chat', source='cli',
                              cwd=str(tmp_path), toolsets=[], model='m')
        sid = created['session_id']
        authority.db.record_auxiliary_usage(sid, 'compression', model='m', input_tokens=4, output_tokens=1)
        live = authority.sessions[sid]
        store._record_gateway_session_peer(sid, live.route, live.source)
        authority.db.append_message(sid, 'user', 'hi')
        authority.sessions.clear()
        resumed = await _call(owner, 'session.resume', latest='cli')
        assert isinstance(resumed, dict) and resumed['session_id'] == sid, resumed
        row = authority.db.get_session(sid)
        assert (row['source'], row['created_source']) == ('local', 'cli')
    finally:
        await owner.close()
        store.close_all_db_handles()


def _set_created(db, sid, value):
    with db._lock:
        db._conn.execute("UPDATE sessions SET created_source=? WHERE id=?", (value, sid))
        db._conn.commit()


def _reset(authority, store, sid, child):
    from gateway.session import SessionEntry
    from hermes_state_local import local_receipt
    from hermes_state_local_lineage import reset_local_target

    live = authority.sessions[sid]
    store._record_gateway_session_peer(sid, live.route, live.source)
    old = SessionEntry.from_dict(local_receipt(authority.db, sid)['entry'])
    entry = SessionEntry(old.session_key, child, old.created_at, old.updated_at, origin=old.origin,
                         platform=old.platform, chat_type=old.chat_type, is_fresh_reset=True)
    reset_local_target(authority.db, epoch=authority.epoch, parent_session_id=sid, entry=entry.to_dict())
    return child


@pytest.mark.asyncio
async def test_reset_copies_the_parent_surface_and_falls_back_to_policy(tmp_path, monkeypatch):
    """A real parent created_source is copied. NULL, local and unknown are not.

    /new publishes with the policy label (cli), not the live local source. Copying
    tui is not the policy fallback; a parent with no surface becomes cli.
    """
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        copied = await _call(owner, 'session.create', request_id='copy', source='cli',
                             cwd=str(tmp_path), toolsets=[], model='m')
        sid = copied['session_id']
        _set_created(authority.db, sid, 'tui')
        child = _reset(authority, store, sid, 'reset-copied')
        assert authority.db.get_session(child)['created_source'] == 'tui'

        for parent_created, name in ((None, 'reset-null'), ('unknown', 'reset-unknown'), ('local', 'reset-local')):
            created = await _call(owner, 'session.create', request_id=name, source='cli',
                                  cwd=str(tmp_path), toolsets=[], model='m')
            sid = created['session_id']
            _set_created(authority.db, sid, parent_created)
            child = _reset(authority, store, sid, name + '-child')
            row = authority.db.get_session(child)
            assert row['created_source'] == 'cli', (parent_created, row['source'], row['created_source'])
            assert row['source'] == 'cli'
    finally:
        await owner.close()
        store.close_all_db_handles()


def _listed(db, source, capsys):
    from argparse import Namespace
    from hermes_cli.sessions_cmd import _cmd_list
    _cmd_list(db, Namespace(limit=20, source=source, workspace=None))
    return capsys.readouterr().out


@pytest.mark.asyncio
async def test_oneshot_keeps_its_own_created_source_and_stays_out_of_cli(tmp_path, monkeypatch, capsys):
    """created_source is provenance: a oneshot run is stamped oneshot, not the cli it runs as."""
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        created = await _call(owner, 'session.create', request_id='z', source='oneshot',
                              cwd=str(tmp_path), toolsets=[], model='m')
        sid = created['session_id']
        assert authority.db.get_session(sid)['created_source'] == 'oneshot'
        assert sid not in _listed(authority.db, 'cli', capsys)
        assert sid in _listed(authority.db, 'oneshot', capsys)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('admission', [None, 'queued', 'unknown'])
async def test_latest_skips_a_newer_empty_launch_stub(tmp_path, monkeypatch, admission):
    """CLI-31: the insert stamps the surface, so a 0-message launch stub left by a crash or a
    failed resume is the newest CLI row. Neither ``session.resume latest`` nor ``-c`` picks it.
    A newer session whose first prompt is admitted (queued, or ``unknown`` after a restart) but
    not yet written is a conversation, not a stub: both still pick it."""
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        ids = []
        for name in ('real', 'stub'):
            created = await _call(owner, 'session.create', request_id=name, source='cli',
                                  cwd=str(tmp_path), toolsets=[], model='m')
            ids.append(sid := created['session_id'])
            authority.db.create_session(sid, source='cli')
            live = authority.sessions[sid]
            store._record_gateway_session_peer(sid, live.route, live.source)
        real, stub = ids
        authority.db.append_message(real, 'user', 'hello')
        with authority.db._lock:
            authority.db._conn.execute("UPDATE sessions SET started_at = started_at + 60 WHERE id = ?", (stub,))
            if admission:
                authority.db._conn.execute(
                    "INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,"
                    "lineage_json,payload_json,payload_digest,intent,status) "
                    "VALUES('adm','req','human',?,'[]','{}','d','queue',?)", (stub, admission))
            authority.db._conn.commit()
        authority.sessions.clear()
        expected = stub if admission else real

        from hermes_cli.main import _resolve_last_session
        monkeypatch.setattr('hermes_cli.main._resolve_workspace_key', lambda: None)
        monkeypatch.setattr('hermes_cli.main.get_hermes_home', lambda: authority.db.db_path.parent)
        assert _resolve_last_session('cli') == expected

        resumed = await _call(owner, 'session.resume', latest='cli')
        assert isinstance(resumed, dict) and resumed['session_id'] == expected, resumed
    finally:
        await owner.close()
        store.close_all_db_handles()
