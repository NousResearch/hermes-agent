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
from persistence.authority.runtime import Compress, PreparedMutation
from persistence.conversation.usage import TokenCounts
from persistence_assembly import store_location
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from pathlib import Path


def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps({'model': {'default': 'm'}}))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _handle_message=None, _peek_session_state=lambda key: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    persistence = storage_handle(home)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', persistence=persistence,
                                 epoch=persistence.runtime.begin_epoch(instance_id='owner'))
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
            seed_session(authority.persistence, sid, SURFACES[surface])
            live = authority.sessions[sid]
            store._record_gateway_session_peer(sid, live.route, live.source)
            seed_messages(authority.persistence, sid, {'role': 'user', 'content': 'hi'})
        rows = {sid: session_row(authority.persistence, sid) for sid in created.values()}
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
        authority.persistence.usage.record_auxiliary_usage(sid, 'compression', model='m',
                                                           tokens=TokenCounts(input_tokens=4, output_tokens=1))
        live = authority.sessions[sid]
        store._record_gateway_session_peer(sid, live.route, live.source)
        seed_messages(authority.persistence, sid, {'role': 'user', 'content': 'hi'})
        authority.sessions.clear()
        resumed = await _call(owner, 'session.resume', latest='cli')
        assert isinstance(resumed, dict) and resumed['session_id'] == sid, resumed
        row = session_row(authority.persistence, sid)
        assert (row['source'], row['created_source']) == ('local', 'cli')
    finally:
        await owner.close()
        store.close_all_db_handles()


def _set_created(db, sid, value):
    raw_holder(db)._execute_write(lambda conn: conn.execute("UPDATE sessions SET created_source=? WHERE id=?", (value, sid)))


def _reset(authority, store, sid, child):
    from gateway.session import SessionEntry

    live = authority.sessions[sid]
    store._record_gateway_session_peer(sid, live.route, live.source)
    old = SessionEntry.from_dict(authority.persistence.bindings.receipt(sid).entry)
    entry = SessionEntry(old.session_key, child, old.created_at, old.updated_at, origin=old.origin,
                         platform=old.platform, chat_type=old.chat_type, is_fresh_reset=True)
    authority.persistence.bindings.reset_target(epoch=authority.epoch, parent_session_id=sid, entry=entry.to_dict())
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
        _set_created(authority.persistence, sid, 'tui')
        child = _reset(authority, store, sid, 'reset-copied')
        assert session_row(authority.persistence, child)['created_source'] == 'tui'

        for parent_created, name in ((None, 'reset-null'), ('unknown', 'reset-unknown'), ('local', 'reset-local')):
            created = await _call(owner, 'session.create', request_id=name, source='cli',
                                  cwd=str(tmp_path), toolsets=[], model='m')
            sid = created['session_id']
            _set_created(authority.persistence, sid, parent_created)
            child = _reset(authority, store, sid, name + '-child')
            row = session_row(authority.persistence, child)
            assert row['created_source'] == 'cli', (parent_created, row['source'], row['created_source'])
            assert row['source'] == 'cli'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_compress_mutation_copies_parent_or_uses_the_policy_surface(tmp_path, monkeypatch):
    """The compress mutation copies the parent's surface. A NULL or ``local`` parent is not a
    surface, so the child falls back to the policy label."""
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        def compress(sid):
            row = session_row(authority.persistence, sid)
            ask = dict(principal_id=authority.persistence.bindings.receipt(sid).principal_id, session_id=sid,
                       request_id=f'compress-{sid}', expected_revision=row['runtime_revision'], command=Compress(),
                       expected_generation=row['runtime_generation'])
            snapshot = authority.persistence.runtime.prepare_mutation(authority.epoch, **ask)
            prepared = PreparedMutation(snapshot, {'in_place': False, 'messages': [{'role': 'user', 'content': 'hi'}]})
            return authority.persistence.runtime.mutate(authority.epoch, **ask, prepared=prepared, owner=None).result['target_session_id']

        for parent_created, expected in (('desktop', 'desktop'), (None, 'cli'), ('local', 'cli')):
            created = await _call(owner, 'session.create', request_id=f'compress-{parent_created}',
                                  source='cli', cwd=str(tmp_path), toolsets=[], model='m')
            sid = created['session_id']
            _set_created(authority.persistence, sid, parent_created)
            row = session_row(authority.persistence, compress(sid))
            assert row['created_source'] == expected, (parent_created, row['created_source'])
    finally:
        await owner.close()
        store.close_all_db_handles()


def _listed(db, source, capsys):
    from argparse import Namespace
    from hermes_cli.sessions_cmd import _cmd_list
    from persistence_assembly import open_persistence_reader
    with open_persistence_reader(store_location(db.scope)) as store:
        _cmd_list(Namespace(limit=20, source=source, workspace=None), persistence=store)
    return capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize(('surface', 'labels'), [('gui', ('gui', 'desktop')), ('acp', ('acp',))])
async def test_gui_and_acp_list_under_the_policy_label_and_the_agent_surface(
        tmp_path, monkeypatch, capsys, surface, labels):
    """session.create stamps the agent surface (gui is desktop). --source accepts
    either name, before the peer refresh and after the row is stored as local."""
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        created = await _call(owner, 'session.create', request_id=surface, source=surface,
                              cwd=str(tmp_path), toolsets=[], model='m')
        sid = created['session_id']
        for label in labels:
            assert sid in _listed(authority.persistence, label, capsys), (surface, label, 'before peer')
        live = authority.sessions[sid]
        store._record_gateway_session_peer(sid, live.route, live.source)
        for label in labels:
            assert sid in _listed(authority.persistence, label, capsys), (surface, label, 'after peer')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_oneshot_keeps_its_own_created_source_and_stays_out_of_cli(tmp_path, monkeypatch, capsys):
    """created_source is provenance: a oneshot run is stamped oneshot, not the cli it runs as."""
    store, authority, owner = _owner(tmp_path / 'home', monkeypatch)
    try:
        created = await _call(owner, 'session.create', request_id='z', source='oneshot',
                              cwd=str(tmp_path), toolsets=[], model='m')
        sid = created['session_id']
        assert session_row(authority.persistence, sid)['created_source'] == 'oneshot'
        assert sid not in _listed(authority.persistence, 'cli', capsys)
        assert sid in _listed(authority.persistence, 'oneshot', capsys)
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
            seed_session(authority.persistence, sid, 'cli')
            live = authority.sessions[sid]
            store._record_gateway_session_peer(sid, live.route, live.source)
        real, stub = ids
        seed_messages(authority.persistence, real, {'role': 'user', 'content': 'hello'})

        def stub_rows(conn):
            conn.execute("UPDATE sessions SET started_at = started_at + 60 WHERE id = ?", (stub,))
            if admission:
                conn.execute(
                    "INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,"
                    "lineage_json,payload_json,payload_digest,intent,status) "
                    "VALUES('adm','req','human',?,'[]','{}','d','queue',?)", (stub, admission))
        raw_holder(authority.persistence)._execute_write(stub_rows)
        authority.sessions.clear()
        expected = stub if admission else real

        from hermes_cli.main import _resolve_last_session
        monkeypatch.setattr('hermes_cli.main._resolve_workspace_key', lambda: None)
        monkeypatch.setattr('hermes_cli.main.get_hermes_home', lambda: Path(store_location(authority.persistence.scope)).parent)
        assert _resolve_last_session('cli') == expected

        resumed = await _call(owner, 'session.resume', latest='cli')
        assert isinstance(resumed, dict) and resumed['session_id'] == expected, resumed
    finally:
        await owner.close()
        store.close_all_db_handles()
