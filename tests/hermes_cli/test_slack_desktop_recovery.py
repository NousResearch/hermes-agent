"""Offline routing-index restart and standalone profile mirror regressions."""
import json


from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import SessionDB
from hermes_cli.slack_desktop_sync import availability, mirror_row, set_opt_in
from plugins.platforms.slack.adapter import SlackAdapter


TEAM, CHANNEL, THREAD, BOT = 'T-original', 'C-original', '1700000000.123', 'U-receiver'


def _auth(_token):
    return {'ok': True, 'team_id': TEAM, 'user_id': BOT, 'bot_id': 'B-receiver'}


def _message(db, session_id, content):
    row_id = db._execute_write(lambda conn: conn.execute(
        'INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, 1)',
        (session_id, 'user', content)).lastrowid)
    assert isinstance(row_id, int)
    return row_id


def test_restarted_gateway_keeps_original_bot_proof_through_compression(tmp_path, monkeypatch):
    import hermes_constants
    root = tmp_path / 'hermes'
    root.mkdir()
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: root)
    (root / '.env').write_text('SLACK_BOT_TOKEN=xoxb-original\n')
    db = SessionDB(db_path=root / 'state.db')
    adapter = SlackAdapter(PlatformConfig(enabled=True, token='xoxb-original'))
    adapter._team_bot_user_ids[TEAM] = BOT
    source = adapter.build_source(CHANNEL, chat_type='group', scope_id=TEAM, thread_id=THREAD)
    config = GatewayConfig()
    store = SessionStore(sessions_dir=root / 'sessions', config=config)
    store._db = db
    original = store.get_or_create_session(source)
    assert set_opt_in(db, original.session_id, True, auth_test=_auth)
    assert db.try_acquire_compression_lock(original.session_id, 'holder')
    db.publish_compression_child(parent_session_id=original.session_id, child_session_id='continued',
                                 source='desktop', messages=[{'role': 'user', 'content': 'summary'}],
                                 compression_lock_holder='holder')
    # Persisted route still names the ended parent when the gateway process disappears.
    before = json.loads(db.load_gateway_routing_entries(scope=store._routing_scope())[original.session_key])
    assert before['session_id'] == original.session_id and before['receiving_bot_user_id'] == BOT
    restarted = SessionStore(sessions_dir=root / 'sessions', config=config)
    restarted._db = db
    route = restarted.lookup_by_session_key(original.session_key)
    assert route is not None and route.session_id == 'continued'
    persisted = json.loads(db.load_gateway_routing_entries(scope=restarted._routing_scope())[original.session_key])
    assert persisted['session_id'] == 'continued'
    assert persisted['receiving_bot_user_id'] == BOT
    assert persisted['origin'] == source.to_dict()
    assert restarted.get_or_create_session(source).session_id == 'continued'
    restored = restarted.lookup_by_session_key(original.session_key)
    assert restored is not None and restored.receiving_bot_user_id == BOT
    tip = db.get_session('continued')
    assert availability(db, tip)
    sent = []
    row_id = _message(db, 'continued', 'After restart')
    assert mirror_row(db, 'continued', row_id, 'user', auth_test=_auth,
                      send=lambda *args: sent.append(args) or {'ok': True})
    assert sent == [('xoxb-original', CHANNEL, THREAD, '[Desktop] After restart')]
    # A new explicit route to a reset does not inherit the original opt-in/lineage.
    db.create_session('reset', 'slack', parent_session_id=original.session_id,
                      model_config={'_reset_from': original.session_id})
    db._write_sql("UPDATE sessions SET slack_sync=1 WHERE id='reset'")
    assert not availability(db, db.get_session('reset'))
    db.create_session('branch', 'slack', parent_session_id=original.session_id,
                      model_config={'_branched_from': original.session_id})
    db._write_sql("UPDATE sessions SET slack_sync=1 WHERE id='branch'")
    assert not availability(db, db.get_session('branch'))
    db.close()


def test_standalone_named_profile_uses_main_key_and_local_route(tmp_path, monkeypatch):
    import hermes_constants
    root = tmp_path / 'hermes'
    home = root / 'profiles' / 'second'
    home.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: root)
    (home / '.env').write_text('SLACK_BOT_TOKEN=xoxb-second\n')
    db = SessionDB(db_path=home / 'state.db')
    source = SessionSource(platform=Platform.SLACK, chat_id=CHANNEL, chat_type='group',
                           scope_id=TEAM, thread_id=THREAD)
    key = build_session_key(source)  # multiplexing disabled: agent:main, not agent:second
    origin = source.to_dict()
    db.create_session('standalone', 'slack', session_key=key, chat_id=CHANNEL,
                      thread_id=THREAD, origin_json=json.dumps(origin), profile_name='second')
    db.save_gateway_routing_entry(key, json.dumps(dict(session_key=key, session_id='standalone',
        origin=origin, receiving_bot_user_id=BOT, transport_profile='second')))
    assert db._own_profile_name() == 'second'
    assert set_opt_in(db, 'standalone', True, auth_test=_auth)
    row_id = _message(db, 'standalone', 'Standalone')
    sent = []
    assert mirror_row(db, 'standalone', row_id, 'user', auth_test=_auth,
                      send=lambda *args: sent.append(args) or {'ok': True})
    assert sent == [('xoxb-second', CHANNEL, THREAD, '[Desktop] Standalone')]
    denied_row = _message(db, 'standalone', 'Not from receiver')
    assert not mirror_row(db, 'standalone', denied_row, 'user', auth_test=lambda token: {
        **_auth(token), 'user_id': 'U-different'}, send=lambda *args: sent.append(args) or {'ok': True})
    assert len(sent) == 1
    db.create_session('fork', 'slack', parent_session_id='standalone',
                      model_config={'_branched_from': 'standalone'})
    assert not availability(db, db.get_session('fork'))
    db.close()
