"""`hermes sessions prune` while the always-on gateway owns state.db (live smoke LS-F7).

The gateway is always up, so a CLI prune that refuses whenever a process holds the store (and
deletes under it with ``--force``) is unusable or a second writer. With a ready owner the CLI only
reads its preview; the owner deletes through ``session.prune`` with the housekeeping retirement
path, which keeps a retained reset child's logical owner. A REAL foreign process holds the store,
as the gateway does; the owner end is the production ``AuthorityConnection`` dispatch.
"""
import subprocess
import sys
import time
from argparse import Namespace
from contextlib import asynccontextmanager, closing
from types import SimpleNamespace

import pytest

import hermes_state_runtime as rt
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_cli import sessions_cmd
from hermes_state import SessionDB
from tests.hermes_state.test_target_advance_fence import _local_session, _mutate

pytestmark = pytest.mark.platforms("posix")  # the holder scan the old path refused on is POSIX-only

_HOLDER = ("import sqlite3, sys\nconn = sqlite3.connect(sys.argv[1])\n"
           "conn.execute('SELECT count(*) FROM sqlite_master')\nprint('ready', flush=True)\nsys.stdin.readline()\n")


def _aged(db, sid, days, *, ended=True):
    at = time.time() - days * 86400

    def write(conn):  # last activity = freshest of row activity and the latest message
        conn.execute('UPDATE sessions SET started_at=?, last_activity_at=?, ended_at=? WHERE id=?',
                     (at, at, at if ended else None, sid))
        conn.execute('UPDATE messages SET timestamp=? WHERE session_id=?', (at, sid))
    db._execute_write(write)


def _connection(authority, *, operator, native=True):
    identity = {'user_id': 'local-operator', 'provider': 'local', 'profile_id': authority.profile_id,
                'instance_id': authority.instance_id, 'native_bootstrap': native,
                'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']}
    return AuthorityConnection(authority, object(), identity, operator=operator)


@pytest.fixture
def owned_store(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    retired = []
    runner = SimpleNamespace(_draining=False, session_store=SimpleNamespace(retire_runtime_sessions=retired.extend))
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owner', db=db, epoch=epoch)
    owner = _local_session(db, epoch, tmp_path)
    child = _mutate(db, epoch, owner, 'reset')['target_session_id']  # ends `owner`, keeps it as the FIFO owner
    db.append_message(child, 'user', 'kept turn')
    _aged(db, owner, 200)
    for sid, source, days in (('old-cli', 'cli', 200), ('old-cron', 'cron', 200), ('fresh-cli', 'cli', 1)):
        db.create_session(sid, source)
        db.append_message(sid, 'user', sid)
        _aged(db, sid, days)
    authority.sessions['old-cli'] = LiveSession(None, 'route-old-cli')  # resident in owner memory
    with closing(db):
        yield SimpleNamespace(db=db, authority=authority, owner=owner, child=child, retired=retired,
                              path=tmp_path / 'state.db')


def test_prune_under_the_live_owner_is_applied_by_the_owner_not_refused(owned_store, monkeypatch, capsys):
    from hermes_cli import gateway_client
    from hermes_cli import gateway_runtime
    import hermes_state
    from hermes_cli.gateway_client import rpc_error
    store = owned_store
    holder = subprocess.Popen([sys.executable, '-c', _HOLDER, str(store.path)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == 'ready'
    opened = []
    real_db = hermes_state.SessionDB

    def recording_db(*args, **kwargs):
        opened.append(kwargs.get('read_only', False))
        return real_db(*args, **kwargs)

    @asynccontextmanager
    async def owner_socket(*_args, **_kwargs):
        connection = _connection(store.authority, operator=True)

        async def rpc(method, _timeout=None, **params):
            frame = await connection.dispatch({'id': 1, 'method': method, 'params': params})
            if 'error' in frame:
                raise rpc_error(frame['error'])
            return frame['result']
        try:
            yield SimpleNamespace(rpc=rpc)
        finally:
            await connection.close()

    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                        lambda home, **_: gateway_runtime.GatewayDiscovery('ready', endpoint=object()))
    monkeypatch.setattr(gateway_client, 'connect_gateway', owner_socket)
    monkeypatch.setattr(hermes_state, 'SessionDB', recording_db)
    args = Namespace(sessions_action='prune', dry_run=False, yes=True, force=False, never_active=False,
                     include_archived=False, include_pinned=False,
                     **{name: None for name in sessions_cmd._FILTER_ARGS})
    args.older_than = '90'
    try:
        assert sessions_cmd.cmd_sessions(args) in (None, 0)
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)
    out = capsys.readouterr().out
    assert 'Refusing' not in out and 'Pruned 2 session(s).' in out, out
    db = store.db
    # --older-than honoured; the retained reset child's logical owner and its history survive.
    assert db.get_session('old-cli') is None and db.get_session('old-cron') is None
    assert db.get_session('fresh-cli') is not None and db.get_session(store.owner) is not None
    assert [m['content'] for m in db.get_messages(store.child)] == ['kept turn']
    # The CLI never opened a writer; the owner retired the deleted ids from its own memory too.
    assert opened == [True], opened
    assert 'old-cli' not in store.authority.sessions and set(store.retired) == {'old-cli', 'old-cron'}


@pytest.mark.asyncio
@pytest.mark.parametrize('grant', ['viewer', 'remote_operator'])
async def test_owner_prune_is_local_operator_only(owned_store, grant):
    """Store-wide deletion across every principal's rows: only the native local operator ticket."""
    connection = _connection(owned_store.authority, operator=grant == 'remote_operator',
                             native=grant != 'remote_operator')
    try:
        refused = await connection.dispatch({'id': 1, 'method': 'session.prune',
                                             'params': {'filters': {'older_than_days': 0}}})
    finally:
        await connection.close()
    assert refused['error']['message'] == 'permission_denied', refused
    assert owned_store.db.get_session('old-cli') is not None


def _never_active_row(db, sid, *, age_days=45, **overrides):
    """A keyed gateway row that was opened and never used — the shape ``--never-active`` sweeps."""
    row = {'id': sid, 'source': 'telegram', 'user_id': 'user-1', 'session_key': f'agent:main:telegram:dm:{sid}',
           'chat_id': 'chat-1', 'chat_type': 'dm', 'started_at': time.time() - age_days * 86400,
           'message_count': 0, 'tool_call_count': 0, 'api_call_count': 0, 'input_tokens': 0,
           'output_tokens': 0, 'archived': 0, 'pinned': 0}
    row.update(overrides)
    db._execute_write(lambda conn: conn.execute(
        f"INSERT INTO sessions ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", list(row.values())))


def test_never_active_prune_under_the_live_owner_honours_include_archived(owned_store, monkeypatch, capsys):
    """``--never-active --include-archived`` travels through ``session.prune``: by default the owner keeps
    the never-active row ``sessions.auto_archive`` already archived; with the flag it goes, and a pinned
    row of the same shape survives either way."""
    from hermes_cli import gateway_client
    from hermes_cli import gateway_runtime
    from hermes_cli.gateway_client import rpc_error
    store = owned_store
    for sid, over in (('plain-junk', {}), ('archived-junk', {'archived': 1}), ('pinned-junk', {'pinned': 1})):
        _never_active_row(store.db, sid, **over)
    holder = subprocess.Popen([sys.executable, '-c', _HOLDER, str(store.path)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == 'ready'
    sent = []

    @asynccontextmanager
    async def owner_socket(*_args, **_kwargs):
        connection = _connection(store.authority, operator=True)

        async def rpc(method, _timeout=None, **params):
            sent.append((method, dict(params)))
            frame = await connection.dispatch({'id': 1, 'method': method, 'params': params})
            if 'error' in frame:
                raise rpc_error(frame['error'])
            return frame['result']
        try:
            yield SimpleNamespace(rpc=rpc)
        finally:
            await connection.close()

    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                        lambda home, **_: gateway_runtime.GatewayDiscovery('ready', endpoint=object()))
    monkeypatch.setattr(gateway_client, 'connect_gateway', owner_socket)
    args = Namespace(sessions_action='prune', dry_run=False, yes=True, force=False, never_active=True,
                     include_archived=False, include_pinned=False,
                     **{name: None for name in sessions_cmd._FILTER_ARGS})
    args.older_than = '30'
    try:
        assert sessions_cmd.cmd_sessions(args) in (None, 0)
        db = store.db
        assert db.get_session('plain-junk') is None and db.get_session('archived-junk') is not None
        assert sent == [('session.prune', {'never_active_days': 30.0})]  # default request unchanged on the wire

        args.include_archived = True
        assert sessions_cmd.cmd_sessions(args) in (None, 0)
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)
    assert db.get_session('archived-junk') is None and db.get_session('pinned-junk') is not None
    assert sent[-1] == ('session.prune', {'never_active_days': 30.0, 'include_archived': True})
    out = capsys.readouterr().out
    assert '(including archived)' in out and out.count('Deleted 1 never-active session(s)') == 2, out


@pytest.mark.asyncio
async def test_owner_prune_include_archived_belongs_to_the_never_active_sweep_only(owned_store):
    """``include_archived`` with ``filters`` (archived is already a filter there) or with a non-boolean
    value is the client's bug: ``invalid_params``, nothing deleted."""
    connection = _connection(owned_store.authority, operator=True)
    try:
        frames = [await connection.dispatch({'id': i, 'method': 'session.prune', 'params': params})
                  for i, params in enumerate((
                      {'filters': {'older_than_days': 0}, 'include_archived': True},
                      {'never_active_days': 30, 'include_archived': ['yes']},
                  ), start=1)]
    finally:
        await connection.close()
    assert [frame['error']['message'] for frame in frames] == ['invalid_params', 'invalid_params'], frames
    assert owned_store.db.get_session('old-cli') is not None
