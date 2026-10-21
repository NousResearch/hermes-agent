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
