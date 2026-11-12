"""`hermes sessions adopt` brings a profile's pre-gateway chats under its gateway, as the host.

The Webapp's browser reaches the gateway through its dashboard mount as the login it authenticated
(on a loopback bind, the session-token owner). Hermes is single-user, so that login is the machine
owner: like native Desktop it may claim an old chat by opening it, and `adopt` brings the rest under
the gateway in one pass, after which the browser login continues each.
"""
import argparse
import asyncio
from contextlib import closing
import json
import sqlite3

import pytest

from tests.gateway.fixtures.owner_process import running_owner

#: The owner's dashboard session token (``HERMES_DASHBOARD_SESSION_TOKEN``): what a loopback browser
#: login presents on the gateway's dashboard mount.
_BROWSER_TOKEN = 'b' * 43
_OWNER_ENV = {'HERMES_DASHBOARD_SESSION_TOKEN': _BROWSER_TOKEN}

_OLD_CHATS = (('old-desktop', 'desktop', None), ('old-desktop-nous', 'desktop', 'nous:account'),
              ('old-cli', 'cli', None), ('foreign-desktop', 'desktop', 'telegram:42'),
              ('api-row', 'api_server', None))


def _seed(home):
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / 'state.db')
    try:
        for sid, source, user_id in _OLD_CHATS:
            db.create_session(sid, source=source, model='old-model', cwd=str(home), user_id=user_id)
            db.append_message(sid, role='user', content=sid.upper())
            db.end_session(sid, 'session_switch')
    finally:
        db.close()


def _run_adopt(capsys, *argv):
    from hermes_cli import sessions_cmd
    from hermes_cli.subcommands.sessions import build_sessions_parser
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest='command'), cmd_sessions=sessions_cmd.cmd_sessions)
    args = parser.parse_args(['sessions', 'adopt', *argv])
    status = args.func(args)
    return status, capsys.readouterr()


def _adopt(capsys, *argv):
    status, captured = _run_adopt(capsys, *argv)
    return status, captured.out


async def _rpc(socket, method, **params):
    await socket.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
    async with asyncio.timeout(15):
        while True:
            reply = json.loads(await socket.recv())
            if reply.get('id') == method:
                return reply


async def _browser_resume(home, descriptor, sid):
    """What a browser opens on the gateway's own listener: ``/api/ws`` on its dashboard mount
    (``chat_ws.gateway_ws``), with an Origin, as the loopback session-token login; never a native grant."""
    from websockets.asyncio.client import connect
    from hermes_cli.web_server_boundary import _DESKTOP_ATTACHMENT_WS_MAX_BYTES
    origin = descriptor['api_origin']
    url = origin.replace('http:', 'ws:') + '/api/ws?token=' + _BROWSER_TOKEN
    async with connect(url, origin=origin, proxy=None, max_size=_DESKTOP_ATTACHMENT_WS_MAX_BYTES) as socket:
        return await _rpc(socket, 'session.resume', session_id=sid)


@pytest.mark.platforms("linux")
def test_adopting_old_chats_lets_the_browser_login_continue_them(tmp_path, model_peer, monkeypatch, capsys):
    with running_owner(tmp_path, model_peer, prepare=_seed, extra_env=_OWNER_ENV) as (home, descriptor, _env):
        monkeypatch.setenv('HERMES_HOME', str(home))
        # The browser login is the machine owner, exactly as native Desktop: it opens an old chat
        # itself, and that first claim is the owner's (the adopt pass no longer offers it).
        first = asyncio.run(_browser_resume(home, descriptor, 'old-desktop'))
        assert first.get('result', {}).get('session_id') == 'old-desktop', first

        status, out = _adopt(capsys, '--dry-run')
        assert status == 0 and '2 pre-gateway chat(s) to adopt (1 cli, 1 desktop)' in out, out

        status, out = _adopt(capsys)
        assert status == 0 and 'Adopted 2 chat(s).' in out, out
        for sid in ('old-desktop', 'old-desktop-nous', 'old-cli'):
            resumed = asyncio.run(_browser_resume(home, descriptor, sid))
            assert resumed.get('result', {}).get('session_id') == sid, resumed
            assert [m['content'] for m in resumed['result']['messages']] == [sid.upper()]
        # Another identity's row and a non-local row were never offered, so they stay as they were.
        assert asyncio.run(_browser_resume(home, descriptor, 'foreign-desktop'))['error']['message'] == 'not_found'

        status, out = _adopt(capsys)
        assert status == 0 and 'Nothing to adopt' in out, out


def _seed_long_and_moved(home):
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / 'state.db')
    try:
        # The owner's resume reply always carries the whole history: this one is over the 8 MiB a
        # default native client accepts in one frame.
        db.create_session('long-desktop', source='desktop', model='old-model', cwd=str(home))
        db.append_message('long-desktop', role='user', content='x' * (9 * 1024 * 1024))
        db.create_session('moved-desktop', source='desktop', model='old-model', cwd=str(home / 'deleted-worktree'))
        db.append_message('moved-desktop', role='user', content='MOVED')
        for sid in ('long-desktop', 'moved-desktop'):
            db.end_session(sid, 'session_switch')
    finally:
        db.close()


@pytest.mark.platforms("linux")
def test_adopt_survives_a_long_history_and_names_what_the_owner_refused(tmp_path, model_peer, monkeypatch, capsys):
    with running_owner(tmp_path, model_peer, prepare=_seed_long_and_moved, extra_env=_OWNER_ENV) as (home, descriptor, _env):
        monkeypatch.setenv('HERMES_HOME', str(home))
        status, out = _adopt(capsys)
        assert status == 0 and 'Adopted 1 chat(s).' in out, out
        assert 'Skipped 1: invalid_params (cwd: must be an absolute path to an existing directory).' in out, out
        resumed = asyncio.run(_browser_resume(home, descriptor, 'long-desktop'))
        assert resumed.get('result', {}).get('session_id') == 'long-desktop', str(resumed)[:300]


def _rollback_journal_store(home):
    """A seeded store in rollback-journal mode, where a writer's exclusive lock also blocks readers."""
    home.mkdir(parents=True)
    _seed(home)
    with closing(sqlite3.connect(home / 'state.db', isolation_level=None)) as conn:
        assert conn.execute('PRAGMA journal_mode=DELETE').fetchone()[0] == 'delete'
    return home / 'state.db'


def _store_files(home):
    return {path.name: path.read_bytes() for path in sorted(home.glob('state.db*'))}


def _refused_adopt(capsys, monkeypatch, home, *argv):
    """Runs adopt where no gateway is reachable: the store has to answer before any connection."""
    from hermes_cli import gateway_client

    def no_gateway(**_kwargs):
        raise AssertionError('adopt connected to the gateway')

    monkeypatch.setattr(gateway_client, 'connect_gateway', no_gateway)
    monkeypatch.setenv('HERMES_HOME', str(home))
    status, captured = _run_adopt(capsys, *argv)
    return status, captured.out, captured.err


@pytest.mark.parametrize('argv', [(), ('--dry-run',)])
def test_adopt_names_the_upgrade_for_a_store_no_newer_hermes_has_opened(tmp_path, monkeypatch, capsys, argv):
    """Adopt reads the store without upgrading it, so a store from an older release that lacks this
    release's tables gets one line naming the command that upgrades it, and stays as it was."""
    home = tmp_path / 'home'
    db = _rollback_journal_store(home)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute('DROP TABLE state_meta')
        conn.commit()
    before = _store_files(home)

    status, out, err = _refused_adopt(capsys, monkeypatch, home, *argv)

    assert status == 1 and out == '', out
    assert len(err.splitlines()) == 1 and 'older' in err and '`hermes sessions repair`' in err, err
    assert _store_files(home) == before


@pytest.mark.parametrize('argv', [(), ('--dry-run',)])
def test_adopt_says_to_close_the_holder_of_a_locked_store(tmp_path, monkeypatch, capsys, argv):
    """Another process's exclusive lock on a rollback-journal store blocks the read. Adopt says the
    store is locked and to retry once what holds it is closed, and changes nothing."""
    import hermes_state

    home = tmp_path / 'home'
    db = _rollback_journal_store(home)
    before = _store_files(home)
    monkeypatch.setattr(hermes_state, '_READ_BUSY_TIMEOUT_S', 0.1)
    with closing(sqlite3.connect(db, isolation_level=None)) as holder:
        holder.execute('BEGIN EXCLUSIVE')
        status, out, err = _refused_adopt(capsys, monkeypatch, home, *argv)
        holder.execute('ROLLBACK')

    assert status == 1 and out == '', out
    assert len(err.splitlines()) == 1 and 'locked' in err and 'close' in err.lower(), err
    assert _store_files(home) == before
