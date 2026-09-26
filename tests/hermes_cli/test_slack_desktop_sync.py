"""Desktop→Slack mirror safety and UI contract, with an entirely fake Slack transport."""
import json

import pytest

from hermes_state import SessionDB
from hermes_cli.slack_desktop_sync import availability, set_opt_in, mirror_row


def seeded(tmp_path, *, tokens='xoxb-one', scope='T-one', thread='1700000000.123', profile='default', receiving_bot_user_id='U-original'):
    home = tmp_path / profile
    home.mkdir(parents=True)
    if tokens is not None:
        (home / '.env').write_text('SLACK_BOT_TOKEN=' + tokens + '\n')
    db = SessionDB(db_path=home / 'state.db')
    key = f'agent:main:slack:group:{scope}:C-one:{thread}'
    origin = dict(platform='slack', chat_id='C-one', chat_type='group', scope_id=scope, thread_id=thread)
    db.create_session('s1', 'slack', session_key=key, chat_id='C-one', thread_id=thread,
                      origin_json=json.dumps(origin), profile_name=profile)
    db.save_gateway_routing_entry(key, json.dumps(dict(session_key=key, session_id='s1', origin=origin,
                                                        receiving_bot_user_id=receiving_bot_user_id)))
    return db, home


def auth(token):
    return {'ok': True, 'team_id': 'T-one', 'user_id': 'U-original', 'bot_id': 'B-original'}


def test_same_workspace_bot_swap_denied_but_original_and_rotated_tokens_deliver(tmp_path):
    db, home = seeded(tmp_path)
    sent = []
    def transport(token):
        return {'ok': True, 'team_id': 'T-one', 'user_id':
                'U-other' if token == 'xoxb-other' else 'U-original',
                'bot_id': 'B-other' if token == 'xoxb-other' else 'B-original'}
    def send(*args):
        sent.append(args)
        return {'ok': True}
    assert set_opt_in(db, 's1', True, auth_test=transport)
    row = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Hello',1)").lastrowid)
    (home / '.env').write_text('SLACK_BOT_TOKEN=xoxb-other\n')
    with pytest.raises(ValueError):
        set_opt_in(db, 's1', True, auth_test=transport)
    assert not mirror_row(db, 's1', row, 'user', send=send, auth_test=transport)
    assert sent == []
    (home / '.env').write_text('SLACK_BOT_TOKEN=xoxb-rotated\n')
    assert mirror_row(db, 's1', row, 'user', send=send, auth_test=transport)
    assert sent == [('xoxb-rotated', 'C-one', '1700000000.123', '[Desktop] Hello')]
    db.close()


def test_legacy_route_without_receiving_bot_identity_cannot_enable_or_deliver(tmp_path):
    db, home = seeded(tmp_path, receiving_bot_user_id=None)
    assert not availability(db, db.get_session('s1'))
    with pytest.raises(ValueError):
        set_opt_in(db, 's1', True, auth_test=auth)
    db._write_sql('UPDATE sessions SET slack_sync=1 WHERE id=?', ('s1',))
    row = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Hello',1)").lastrowid)
    sent = []
    assert not mirror_row(db, 's1', row, 'user', send=lambda *args: sent.append(args), auth_test=auth)
    assert not sent
    db.close()


def test_inbound_route_persists_receiving_adapter_identity_not_sender_or_token(tmp_path, monkeypatch):
    import asyncio
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.session import SessionStore
    from plugins.platforms.slack.adapter import SlackAdapter

    db = SessionDB(db_path=tmp_path / 'state.db')
    store = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    store._db = db
    adapter = SlackAdapter(PlatformConfig(enabled=True, token='xoxb-fake'))
    class FakeClient:
        async def auth_test(self):
            return auth('xoxb-fake')
    monkeypatch.setattr(adapter, '_new_web_client', lambda *args: FakeClient())
    asyncio.run(adapter._authenticate_workspace('xoxb-fake', None))
    assert adapter._team_bot_user_ids['T-one'] == 'U-original'
    source = adapter.build_source('C-one', chat_type='group', scope_id='T-one',
                                  thread_id='1700000000.123', user_id='U-sender')
    entry = store.get_or_create_session(source)
    persisted = db.load_gateway_routing_entries(scope=store._routing_scope())
    route = json.loads(persisted[entry.session_key])
    assert route['receiving_bot_user_id'] == 'U-original'
    assert route['receiving_bot_user_id'] != source.user_id
    assert 'xoxb-' not in json.dumps(route)
    restored = store._routing_entry_from_json(entry.session_key, persisted[entry.session_key])
    assert restored.receiving_bot_user_id == 'U-original'
    # A later lookup of the same route must not re-trust a replacement adapter.
    adapter._team_bot_user_ids['T-one'] = 'U-other'
    store.get_or_create_session(source)
    store.update_session(entry.session_key)
    assert json.loads(db.load_gateway_routing_entries(scope=store._routing_scope())[
        entry.session_key])['receiving_bot_user_id'] == 'U-original'
    # Startup restoration reloads the index without any transport reference.
    restarted = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    restarted._db = db
    assert restarted.get_or_create_session(source).receiving_bot_user_id == 'U-original'
    restarted.update_session(entry.session_key)
    assert json.loads(db.load_gateway_routing_entries(scope=store._routing_scope())[
        entry.session_key])['receiving_bot_user_id'] == 'U-original'
    db.close()


def test_auth_must_report_bot_id_and_matching_user_id(tmp_path):
    db, _ = seeded(tmp_path)
    for response in ({'ok': True, 'team_id': 'T-one', 'user_id': 'U-original'},
                     {'ok': True, 'team_id': 'T-one', 'user_id': 'U-other', 'bot_id': 'B-original'}):
        with pytest.raises(ValueError):
            set_opt_in(db, 's1', True, auth_test=lambda token: response)
    db.close()


def test_legacy_route_stays_unproven_after_inbound_resume_with_current_bot(tmp_path):
    from datetime import datetime
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.session import SessionEntry, SessionStore
    from plugins.platforms.slack.adapter import SlackAdapter

    db = SessionDB(db_path=tmp_path / 'state.db')
    store = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    store._db = db
    adapter = SlackAdapter(PlatformConfig(enabled=True, token='xoxb-current'))
    adapter._team_bot_user_ids['T-one'] = 'U-current'
    source = adapter.build_source('C-one', chat_type='group', scope_id='T-one',
                                  thread_id='1700000000.123')
    key = store._generate_session_key(source)
    legacy = SessionEntry(session_key=key, session_id='old', created_at=datetime.now(),
                          updated_at=datetime.now(), origin=source)
    db.save_gateway_routing_entry(key, json.dumps(legacy.to_dict()), scope=store._routing_scope())
    db.create_session('old', 'slack', session_key=key, chat_id='C-one', thread_id=source.thread_id,
                      origin_json=json.dumps(source.to_dict()))
    assert store.get_or_create_session(source).receiving_bot_user_id is None
    store.update_session(key)
    route = json.loads(db.load_gateway_routing_entries(scope=store._routing_scope())[key])
    assert 'receiving_bot_user_id' not in route
    db.close()


def test_slack_transport_uses_explicit_thread_and_no_unfurls(monkeypatch):
    from hermes_cli.slack_desktop_sync import _send
    from plugins.platforms.slack import adapter
    seen = []
    async def fake_post(session, token, method, payload, req_kw):
        seen.append((token, method, payload))
        return {'ok': True}
    monkeypatch.setattr(adapter, '_slack_json_post', fake_post)
    assert _send('xoxb-one', 'C-one', '1700000000.123', 'Answer')['ok']
    assert seen == [('xoxb-one', 'chat.postMessage', {
        'channel': 'C-one', 'thread_ts': '1700000000.123', 'text': 'Answer',
        'unfurl_links': False, 'unfurl_media': False,
    })]


def test_opt_in_and_only_one_delivery_per_durable_row(tmp_path):
    db, home = seeded(tmp_path)
    sent = []
    assert availability(db, db.get_session('s1'))
    assert set_opt_in(db, 's1', True, auth_test=auth)
    user = db._execute_write(lambda c: c.execute("INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Hello',1)").lastrowid)
    final = db._execute_write(lambda c: c.execute("INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','assistant','Answer',2)").lastrowid)
    def send(token, channel, thread, text):
        sent.append((token, channel, thread, text))
        return {'ok': True, 'ts': '1700000001.000'}
    assert mirror_row(db, 's1', user, 'user', send=send, auth_test=auth)
    assert mirror_row(db, 's1', final, 'assistant', send=send, auth_test=auth)
    assert not mirror_row(db, 's1', user, 'user', send=send, auth_test=auth)
    assert not mirror_row(db, 's1', final, 'assistant', send=send, auth_test=auth)
    assert sent == [('xoxb-one', 'C-one', '1700000000.123', '[Desktop] Hello'),
                    ('xoxb-one', 'C-one', '1700000000.123', 'Answer')]
    db.close()


@pytest.mark.parametrize('tokens', [None, '', 'xoxb-one,xoxb-two', 'xoxp-user'])
def test_no_or_multiple_tokens_refused(tmp_path, tokens):
    db, home = seeded(tmp_path, tokens=tokens)
    assert not availability(db, db.get_session('s1'))
    with pytest.raises(ValueError):
        set_opt_in(db, 's1', True, auth_test=auth)
    db.close()


def test_legacy_state_db_reconciles_opt_in_and_claim_schema(tmp_path):
    import sqlite3
    db, home = seeded(tmp_path)
    db.close()
    conn = sqlite3.connect(home / 'state.db')
    conn.execute('ALTER TABLE sessions DROP COLUMN slack_sync')
    conn.execute('DROP TABLE slack_desktop_mirror_claims')
    conn.commit()
    conn.close()
    reopened = SessionDB(db_path=home / 'state.db')
    assert reopened.get_session('s1')['slack_sync'] == 0
    assert not reopened._read_all('SELECT * FROM slack_desktop_mirror_claims')
    assert set_opt_in(reopened, 's1', True, auth_test=auth)
    reopened.close()


def test_real_profile_store_namespace_and_origin_are_required(tmp_path, monkeypatch):
    import hermes_constants
    root = tmp_path / 'hermes-root'
    home = root / 'profiles' / 'second'
    home.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: root)
    (home / '.env').write_text('SLACK_BOT_TOKEN=xoxb-second\n')
    db = SessionDB(db_path=home / 'state.db')
    origin = dict(platform='slack', scope_id='T-one', chat_id='C-one',
                  chat_type='group', thread_id='1700000000.123')
    key = 'agent:second:slack:group:T-one:C-one:1700000000.123'
    db.create_session('s1', 'slack', session_key=key, chat_id='C-one',
                      thread_id=origin['thread_id'], origin_json=json.dumps(origin))
    root_db = SessionDB(db_path=root / 'state.db')
    root_db.save_gateway_routing_entry(key, json.dumps(dict(session_key=key, session_id='s1', origin=origin,
                                                            receiving_bot_user_id='U-original')))
    assert db._own_profile_name() == 'second'
    assert set_opt_in(db, 's1', True, auth_test=auth)
    db._write_sql("UPDATE sessions SET session_key='agent:main:slack:group:T-one:C-one:1700000000.123' WHERE id='s1'")
    assert not availability(db, db.get_session('s1'))
    db.close()
    root_db.close()


def test_profile_named_main_uses_disambiguated_namespace(tmp_path, monkeypatch):
    import hermes_constants
    root = tmp_path / 'hermes-root'
    home = root / 'profiles' / 'main'
    home.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: root)
    (home / '.env').write_text('SLACK_BOT_TOKEN=xoxb-main\n')
    db = SessionDB(db_path=home / 'state.db')
    origin = dict(platform='slack', scope_id='T-one', chat_id='C-one',
                  chat_type='group', thread_id='1700000000.123')
    key = 'agent:main~:slack:group:T-one:C-one:1700000000.123'
    db.create_session('s1', 'slack', session_key=key, chat_id='C-one',
                      thread_id=origin['thread_id'], origin_json=json.dumps(origin))
    root_db = SessionDB(db_path=root / 'state.db')
    root_db.save_gateway_routing_entry(key, json.dumps(dict(session_key=key, session_id='s1', origin=origin,
                                                            receiving_bot_user_id='U-original')))
    assert db._own_profile_name() == 'main'
    assert availability(db, db.get_session('s1'))
    assert set_opt_in(db, 's1', True, auth_test=auth)
    db.close()
    root_db.close()


def test_wrong_workspace_and_missing_thread_refused(tmp_path):
    db, home = seeded(tmp_path)
    with pytest.raises(ValueError):
        set_opt_in(db, 's1', True, auth_test=lambda t: {'ok': True, 'team_id': 'T-other'})
    db.close()
    db, home = seeded(tmp_path / 'other', thread='')
    assert not availability(db, db.get_session('s1'))
    db.close()
    dm_home = tmp_path / 'dm'
    dm_home.mkdir()
    (dm_home / '.env').write_text('SLACK_BOT_TOKEN=xoxb-one\n')
    dm = SessionDB(db_path=dm_home / 'state.db')
    origin = {'platform': 'slack', 'scope_id': 'T-one', 'chat_id': 'D-one', 'chat_type': 'dm'}
    key = 'agent:main:slack:dm:T-one:D-one'
    dm.create_session('dm', 'slack', session_key=key, chat_id='D-one', chat_type='dm',
                      origin_json=json.dumps(origin))
    dm.save_gateway_routing_entry(key, json.dumps({'session_id': 'dm', 'session_key': key, 'origin': origin}))
    assert not availability(dm, dm.get_session('dm'))
    with pytest.raises(ValueError):
        set_opt_in(dm, 'dm', True, auth_test=auth)
    dm.close()


def test_cross_profile_token_never_borrowed(tmp_path, monkeypatch):
    db, home = seeded(tmp_path, tokens=None, profile='secondary')
    monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-launch')
    assert not availability(db, db.get_session('s1'))
    db.close()


def test_unknown_saved_token_entry_blocks_opt_in(tmp_path):
    db, home = seeded(tmp_path)
    (home / 'slack_tokens.json').write_text('{"T-other": "xoxb-other"}')
    assert not availability(db, db.get_session('s1'))
    with pytest.raises(ValueError):
        set_opt_in(db, 's1', True, auth_test=auth)
    db.close()


def test_two_profile_stores_have_independent_credentials(tmp_path):
    first, first_home = seeded(tmp_path, tokens='xoxb-first', profile='first')
    second, second_home = seeded(tmp_path, tokens='xoxb-second', profile='second')
    assert set_opt_in(first, 's1', True, auth_test=lambda t: {**auth(t), 'ok': t == 'xoxb-first'})
    assert set_opt_in(second, 's1', True, auth_test=lambda t: {**auth(t), 'ok': t == 'xoxb-second'})
    second_home.joinpath('slack_tokens.json').write_text('{"T-other": {"token": "xoxb-other"}}')
    assert availability(first, first.get_session('s1'))
    assert not availability(second, second.get_session('s1'))
    first.close()
    second.close()


def test_overridden_profile_secret_does_not_hide_second_token(tmp_path, monkeypatch):
    from agent import secret_scope
    db, home = seeded(tmp_path)
    monkeypatch.setattr(secret_scope, 'build_profile_secret_scope',
                        lambda path: {'SLACK_BOT_TOKEN': 'xoxb-external'})
    assert not availability(db, db.get_session('s1'))
    db.close()


def test_launch_env_disagreement_refuses_even_with_one_file_token(tmp_path, monkeypatch):
    from agent import secret_scope
    db, home = seeded(tmp_path)
    monkeypatch.setattr(secret_scope, '_is_process_home', lambda path: True)
    monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-other')
    assert not availability(db, db.get_session('s1'))
    db.close()


def test_row_profile_must_match_owning_store(tmp_path, monkeypatch):
    db, home = seeded(tmp_path)
    monkeypatch.setattr(db, '_own_profile_name', lambda: 'default')
    db._write_sql("UPDATE sessions SET profile_name='secondary' WHERE id='s1'")
    assert not availability(db, db.get_session('s1'))
    db.close()


def test_route_must_match_original_row_and_not_branch(tmp_path):
    db, home = seeded(tmp_path)
    db.save_gateway_routing_entry('agent:main:slack:group:T-one:C-other:1700000000.123',
        json.dumps(dict(session_id='s1', origin=dict(platform='slack', chat_id='C-other',
            scope_id='T-one', thread_id='1700000000.123'))))
    assert not availability(db, db.get_session('s1'))
    db.close()


def test_transport_exception_cannot_log_bot_token(tmp_path, caplog):
    db, home = seeded(tmp_path)
    token = 'xoxb-one'
    def auth_failure(_):
        raise RuntimeError(token)
    with pytest.raises(ValueError):
        set_opt_in(db, 's1', True, auth_test=auth_failure)
    assert token not in caplog.text
    caplog.clear()
    set_opt_in(db, 's1', True, auth_test=auth)
    row = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Hello',1)").lastrowid)
    def send_failure(*args):
        raise RuntimeError(token)
    assert not mirror_row(db, 's1', row, 'user', send=send_failure, auth_test=auth)
    assert token not in caplog.text
    db.close()


def test_claim_not_retried_after_ambiguous_send(tmp_path):
    db, home = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    row = db._execute_write(lambda c: c.execute("INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','One',1)").lastrowid)
    calls = []
    def fail(*args):
        calls.append(args)
        raise TimeoutError('ambiguous')
    assert not mirror_row(db, 's1', row, 'user', send=fail, auth_test=auth)
    assert not mirror_row(db, 's1', row, 'user', send=fail, auth_test=auth)
    assert len(calls) == 1
    db.close()
    reopened = SessionDB(db_path=home / 'state.db')
    assert not mirror_row(reopened, 's1', row, 'user', send=fail, auth_test=auth)
    assert len(calls) == 1
    reopened.close()

def test_commit_lock_retry_must_not_replay_network_send(tmp_path):
    import sqlite3
    db, _ = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    row_id = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Once',1)"
    ).lastrowid)
    sent = []
    original = db._conn
    class CommitFailure:
        failed = False
        def __getattr__(self, name):
            return getattr(original, name)
        def commit(self):
            if sent and not self.failed:
                self.failed = True
                raise sqlite3.OperationalError('database is locked')
            return original.commit()
    db._conn = CommitFailure()
    try:
        mirror_row(db, 's1', row_id, 'user',
                   send=lambda *a: sent.append(a) or {'ok': True}, auth_test=auth)
        assert len(sent) == 1
        assert not mirror_row(db, 's1', row_id, 'user',
                              send=lambda *a: sent.append(a) or {'ok': True}, auth_test=auth)
    finally:
        db._conn = original
        db.close()

def test_revoked_after_claim_is_not_sent(tmp_path, monkeypatch):
    db, _ = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    row_id = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Private',1)"
    ).lastrowid)
    original_write = db._execute_write
    calls = []
    writes = 0

    def intercept(fn, *args, **kwargs):
        nonlocal writes
        result = original_write(fn, *args, **kwargs)
        writes += 1
        if writes == 1:
            set_opt_in(db, 's1', False)
        return result

    monkeypatch.setattr(db, '_execute_write', intercept)
    assert not mirror_row(db, 's1', row_id, 'user', send=lambda *a: calls.append(a) or {'ok': True}, auth_test=auth)
    assert calls == []
    db.close()

def test_opt_out_waits_for_in_flight_send_before_reporting_success(tmp_path):
    from threading import Event, Thread

    db, home = seeded(tmp_path)
    peer = SessionDB(db_path=home / 'state.db')
    set_opt_in(db, 's1', True, auth_test=auth)
    row_id = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','In flight',1)"
    ).lastrowid)
    assert isinstance(row_id, int)
    entered, release, revoke_started, revoked = Event(), Event(), Event(), Event()
    result = []

    def send(*_args):
        entered.set()
        assert release.wait(5), 'test did not release fake transport'
        return {'ok': True}

    def mirror():
        result.append(mirror_row(db, 's1', row_id, 'user', send=send, auth_test=auth))

    def opt_out():
        revoke_started.set()
        set_opt_in(peer, 's1', False)
        revoked.set()

    sender = Thread(target=mirror, daemon=True)
    revoker = Thread(target=opt_out, daemon=True)
    try:
        sender.start()
        assert entered.wait(5)
        revoker.start()
        assert revoke_started.wait(5)
        assert not revoked.wait(0.1), 'opt-out completed while a post could still happen'
    finally:
        release.set()
        sender.join(5)
        revoker.join(5)
        peer.close()
        db.close()
    assert not sender.is_alive() and not revoker.is_alive()
    assert result == [True] and revoked.is_set()


def test_opt_out_request_revokes_future_sends_before_in_flight_post_finishes(tmp_path):
    from threading import Event, Thread
    import time
    db, home = seeded(tmp_path)
    peer = SessionDB(db_path=home / 'state.db')
    set_opt_in(db, 's1', True, auth_test=auth)
    row_id = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Held',1)"
    ).lastrowid)
    entered, release, started = Event(), Event(), Event()
    sender = Thread(target=lambda: mirror_row(db, 's1', row_id, 'user',
        send=lambda *a: entered.set() or release.wait(5) and {'ok': True}, auth_test=auth), daemon=True)
    revoker = Thread(target=lambda: (started.set(), set_opt_in(peer, 's1', False)), daemon=True)
    try:
        sender.start()
        assert entered.wait(5)
        revoker.start()
        assert started.wait(5)
        deadline = time.monotonic() + 1
        while db.get_session('s1')['slack_sync'] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert db.get_session('s1')['slack_sync'] == 0
        assert revoker.is_alive(), 'success cannot precede in-flight transport completion'
    finally:
        release.set()
        sender.join(5)
        revoker.join(5)
        peer.close()
        db.close()

def test_timed_out_opt_out_remains_fail_closed_until_retried(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import hermes_cli.slack_desktop_sync as mirror
    db, _ = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    original_lock = mirror._session_delivery_lock
    @contextmanager
    def stuck_lock(_db, _sid):
        raise TimeoutError('in-flight post did not finish')
        yield
    monkeypatch.setattr(mirror, '_session_delivery_lock', stuck_lock)
    with pytest.raises(TimeoutError):
        set_opt_in(db, 's1', False)
    assert db.get_session('s1')['slack_sync'] == 0
    monkeypatch.setattr(mirror, '_session_delivery_lock', original_lock)
    with pytest.raises(ValueError):
        set_opt_in(db, 's1', True, auth_test=auth)
    assert db.get_session('s1')['slack_sync'] == 0
    assert set_opt_in(db, 's1', False) is False
    assert set_opt_in(db, 's1', True, auth_test=auth)
    db.close()

def test_in_flight_post_does_not_hold_global_state_db_write_lock(tmp_path):
    from threading import Event, Thread
    db, home = seeded(tmp_path)
    peer = SessionDB(db_path=home / 'state.db')
    set_opt_in(db, 's1', True, auth_test=auth)
    row_id = db._execute_write(lambda c: c.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Held',1)"
    ).lastrowid)
    entered, release, finished = Event(), Event(), Event()
    sender = Thread(target=lambda: mirror_row(db, 's1', row_id, 'user',
        send=lambda *a: entered.set() or release.wait(5) and {'ok': True}, auth_test=auth), daemon=True)
    def unrelated_write():
        try:
            peer._write_sql("UPDATE sessions SET title='not blocked' WHERE id='s1'", patience_s=0.2)
            finished.set()
        except Exception:
            pass
    writer = Thread(target=unrelated_write, daemon=True)
    try:
        sender.start()
        assert entered.wait(5)
        writer.start()
        assert finished.wait(1), 'network held the global SQLite writer lock'
    finally:
        release.set()
        sender.join(5)
        writer.join(5)
        peer.close()
        db.close()

def test_slow_opt_in_cannot_resurrect_completed_opt_out(tmp_path):
    from threading import Event, Thread
    db, home = seeded(tmp_path)
    peer = SessionDB(db_path=home / 'state.db')
    started, release = Event(), Event()
    failures = []
    def slow_auth(token):
        started.set()
        assert release.wait(5)
        return auth(token)
    def opt_in():
        try:
            set_opt_in(db, 's1', True, auth_test=slow_auth)
        except ValueError as exc:
            failures.append(str(exc))
    worker = Thread(target=opt_in, daemon=True)
    try:
        worker.start()
        assert started.wait(5)
        assert set_opt_in(peer, 's1', False) is False
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert db.get_session('s1')['slack_sync'] == 0
        assert failures
    finally:
        release.set()
        worker.join(5)
        peer.close()
        db.close()

def test_patch_invalid_title_does_not_enable_mirror(tmp_path, monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from hermes_cli.web_routers import sessions as routes
    from hermes_cli.web_models import SessionRename
    db, home = seeded(tmp_path)
    monkeypatch.setattr(routes, '_open_session_db_for_profile', lambda profile, read_only: db)
    monkeypatch.setattr(db, 'close', lambda: None)
    monkeypatch.setattr('hermes_cli.slack_desktop_sync._auth_test', auth)
    with pytest.raises(HTTPException) as failure:
        asyncio.run(routes.rename_session_endpoint('s1', SessionRename(slack_sync=True, title='x' * 10000)))
    assert failure.value.status_code == 400
    assert db.get_session('s1')['slack_sync'] == 0


def test_patch_rejects_non_boolean_opt_in():
    from pydantic import ValidationError
    from hermes_cli.web_models import SessionRename
    with pytest.raises(ValidationError):
        SessionRename(slack_sync='yes')
    with pytest.raises(ValidationError):
        SessionRename(slack_sync=1)


def test_patch_targets_explicit_profile_store(tmp_path, monkeypatch):
    import asyncio
    from hermes_cli.web_routers import sessions as routes
    from hermes_cli.web_models import SessionRename
    first, _ = seeded(tmp_path, tokens=None, profile='first')
    second, _ = seeded(tmp_path, tokens='xoxb-second', profile='second')
    monkeypatch.setattr(routes, '_open_session_db_for_profile',
                        lambda profile, read_only: {'first': first, 'second': second}[profile])
    monkeypatch.setattr(first, 'close', lambda: None)
    monkeypatch.setattr(second, 'close', lambda: None)
    monkeypatch.setattr('hermes_cli.slack_desktop_sync._auth_test', auth)
    result = asyncio.run(routes.rename_session_endpoint('s1', SessionRename(profile='second', slack_sync=True)))
    assert result['slack_sync'] is True
    assert first.get_session('s1')['slack_sync'] == 0
    assert second.get_session('s1')['slack_sync'] == 1


def test_dashboard_list_detail_and_patch_contract(tmp_path, monkeypatch):
    import asyncio
    from hermes_cli.web_routers import sessions as routes
    from hermes_cli.web_models import SessionRename

    db, home = seeded(tmp_path)
    monkeypatch.setattr(routes, '_open_session_db_for_profile', lambda profile, read_only: db)
    monkeypatch.setattr(routes, '_serving_profile', lambda profile: 'default')
    monkeypatch.setattr(routes, '_maybe_auto_archive_for_profile', lambda profile: None)
    monkeypatch.setattr(routes, '_cron_default_profile', lambda: 'default')
    monkeypatch.setattr(routes, '_cron_profile_home', lambda profile: ('default', home))
    monkeypatch.setattr(db, 'close', lambda: None)
    listed = routes.get_sessions(limit=10, offset=0)
    assert listed['sessions'][0]['slack_sync'] is False
    assert listed['sessions'][0]['slack_sync_available'] is True
    detail = asyncio.run(routes.get_session_detail('s1'))
    assert detail['slack_sync_available'] is True
    assert detail['slack_sync'] is False
    monkeypatch.setattr('hermes_cli.slack_desktop_sync._auth_test', auth)
    updated = asyncio.run(routes.rename_session_endpoint('s1', SessionRename(slack_sync=True)))
    assert updated['slack_sync'] is True
    assert db.get_session('s1')['slack_sync'] == 1
    from hermes_cli.web_routers import profiles as profile_routes
    monkeypatch.setattr(profile_routes, '_profile_targets', lambda purpose: [('default', home)])
    monkeypatch.setattr(profile_routes, '_read_profile_db', lambda name, home, errors, fn: fn(db))
    unified = profile_routes.get_profiles_sessions(limit=10, offset=0)
    assert unified['sessions'][0]['slack_sync'] is True
    assert unified['sessions'][0]['slack_sync_available'] is True
    db._write_sql("UPDATE sessions SET message_count=1 WHERE id='s1'")
    monkeypatch.setattr(profile_routes, '_SIDEBAR_CACHE_TTL_SECONDS', 0)
    sidebar = profile_routes.get_profiles_sessions_sidebar(recents_limit=10, messaging_limit=10)
    assert sidebar['messaging']['sessions'][0]['slack_sync'] is True
    assert sidebar['messaging']['sessions'][0]['slack_sync_available'] is True
    (home / '.env').write_text('SLACK_BOT_TOKEN=xoxb-one,xoxb-two\n')
    refreshed = profile_routes.get_profiles_sessions_sidebar(recents_limit=10, messaging_limit=10)
    assert refreshed['messaging']['sessions'][0]['slack_sync_available'] is False


def test_only_visible_accepted_desktop_turn_and_completed_final_mirror(tmp_path):
    from tui_gateway.slack_mirror import accepted_user, completed_final
    db, home = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    user = db._execute_write(lambda c: c.execute("INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','user','Hello',1)").lastrowid)
    final = db._execute_write(lambda c: c.execute("INSERT INTO messages (session_id,role,content,timestamp) VALUES ('s1','assistant','Answer',2)").lastrowid)
    sends = []
    def send(*args):
        sends.append(args)
        return {'ok': True}
    opts = dict(send=send, auth_test=auth)
    assert not accepted_user(db, 's1', user, desktop=False, visible=True, **opts)
    assert not accepted_user(db, 's1', user, desktop=True, visible=False, **opts)
    assert accepted_user(db, 's1', user, desktop=True, visible=True, **opts)
    receipt = {'complete': True, 'final_assistant_row_id': final}
    assert not completed_final(db, 's1', receipt, status='error', text='Answer', desktop=True, **opts)
    assert not completed_final(db, 's1', receipt, status='complete', text='Answer', desktop=True, successful=False, **opts)
    assert not completed_final(db, 's1', receipt, status='complete', text='NO_REPLY', desktop=True, **opts)
    assert not completed_final(db, 's1', receipt, status='complete', text='', desktop=True, **opts)
    assert not completed_final(db, 's1', {'complete': False, 'final_assistant_row_id': final}, status='complete', text='Answer', desktop=True, **opts)
    assert completed_final(db, 's1', receipt, status='complete', text='Answer', desktop=True, **opts)
    assert len(sends) == 2
    db.close()


def test_compression_inherits_opt_in_but_branch_does_not(tmp_path):
    db, home = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    db.end_session('s1', 'compression')
    db.create_session('continued', 'slack', parent_session_id='s1')
    assert db.get_session('continued')['slack_sync'] == 1
    key = db.get_session('continued')['session_key']
    origin = json.loads(db.get_session('continued')['origin_json'])
    db.save_gateway_routing_entry(key, json.dumps(dict(session_key=key, session_id='continued', origin=origin,
                                                        receiving_bot_user_id='U-original')))
    projected = db.list_sessions_rich(limit=10, compact_rows=True)[0]
    assert projected['id'] == 'continued'
    assert availability(db, projected)
    db.create_session('branch', 'slack', parent_session_id='s1', model_config={'_branched_from': 's1'})
    assert db.get_session('branch')['slack_sync'] == 0
    db.create_session('reset', 'slack', parent_session_id='s1', model_config={'_reset_from': 's1'})
    assert db.get_session('reset')['slack_sync'] == 0
    db.close()


def test_atomic_compression_uses_original_route_without_rewriting_it(tmp_path):
    db, _ = seeded(tmp_path)
    assert set_opt_in(db, 's1', True, auth_test=auth)
    assert db.try_acquire_compression_lock('s1', 'holder', ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id='s1', child_session_id='tip', source='desktop',
        messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='holder')
    tip = db.get_session('tip')
    assert tip is not None and tip['slack_sync'] == 1
    # The gateway routing index may still point to the original session id.
    assert availability(db, tip)
    assert not availability(db, db.get_session('s1'))
    db.close()


def test_stale_parent_opt_out_revokes_live_compression_tip(tmp_path):
    db, _ = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    assert db.try_acquire_compression_lock('s1', 'holder', ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id='s1', child_session_id='tip', source='desktop',
        messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='holder')
    tip = db.get_session('tip')
    assert tip is not None and tip['slack_sync'] == 1
    assert set_opt_in(db, 's1', False) is False  # Stale Desktop row.
    root = db.get_session('s1')
    tip = db.get_session('tip')
    assert root is not None and root['slack_sync'] == 0
    assert tip is not None and tip['slack_sync'] == 0
    row_id = db._execute_write(lambda conn: conn.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES ('tip','user','Private',1)"
    ).lastrowid)
    assert isinstance(row_id, int)
    sent = []
    assert not mirror_row(db, 'tip', row_id, 'user', auth_test=auth,
                          send=lambda *args: sent.append(args) or {'ok': True})
    assert not sent
    db.close()


@pytest.mark.parametrize('atomic', [False, True])
def test_child_published_during_parent_revocation_cannot_reenable(tmp_path, monkeypatch, atomic):
    db, _ = seeded(tmp_path)
    set_opt_in(db, 's1', True, auth_test=auth)
    if atomic:
        assert db.try_acquire_compression_lock('s1', 'holder', ttl_seconds=60)
    original_write = db._execute_write
    published = False

    def intercept(fn, *args, **kwargs):
        nonlocal published
        result = original_write(fn, *args, **kwargs)
        # Interleave publication and an explicit child opt-in after revocation
        # intent commits, but before the parent's send leases are settled.
        parent = db.get_session('s1')
        if not published and parent is not None and parent['slack_sync_revoking']:
            published = True
            if atomic:
                db.publish_compression_child(
                    parent_session_id='s1', child_session_id='tip', source='desktop',
                    messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='holder')
            else:
                db.end_session('s1', 'compression')
                db.create_session('tip', 'desktop', parent_session_id='s1')
            with pytest.raises(ValueError):
                set_opt_in(db, 'tip', True, auth_test=auth)
        return result

    monkeypatch.setattr(db, '_execute_write', intercept)
    assert set_opt_in(db, 's1', False) is False
    assert published
    tip = db.get_session('tip')
    assert tip is not None and tip['slack_sync'] == 0
    assert tip['slack_sync_revoking'] == 0
    db.close()
