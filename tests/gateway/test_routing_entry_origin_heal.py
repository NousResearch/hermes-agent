"""A routing entry persisted without an origin gets one back (#146696)."""
import json

import pytest


def _store(tmp_path, db, **config):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    store = SessionStore(config=GatewayConfig(**config), sessions_dir=tmp_path / 'sessions')
    store._db = db
    return store


def _strip(db, store, key, row_sql=None):
    """Persist *key*'s entry as the reporter's sessions had it: ``origin`` null, no platform."""
    data = store._entries[key].to_dict()
    data['origin'] = None
    data.pop('platform')
    db.save_gateway_routing_entry(key, json.dumps(data), scope=store._routing_scope())
    if row_sql:
        db._execute_write(lambda conn: conn.execute('UPDATE sessions SET ' + row_sql))


@pytest.fixture
def topic(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='-1001234567890', chat_type='group',
                           thread_id='600', user_id='111', chat_name='Forum')
    store = _store(tmp_path, db)
    entry = store.get_or_create_session(source)
    yield db, source, entry
    db.close()


@pytest.mark.parametrize('row_origin', [True, False], ids=['row-origin', 'row-columns'])
def test_load_restores_a_missing_origin_from_the_conversation_row(tmp_path, topic, row_origin):
    db, source, entry = topic
    store = _store(tmp_path, db)
    store._ensure_loaded()
    _strip(db, store, entry.session_key, None if row_origin else 'origin_json=NULL')

    reloaded = _store(tmp_path, db)
    origin = reloaded.lookup_by_session_key(entry.session_key).origin
    assert origin is not None
    assert (origin.platform, origin.chat_id, origin.thread_id, origin.user_id) == (
        source.platform, source.chat_id, source.thread_id, source.user_id)
    assert reloaded._generate_session_key(origin) == entry.session_key


def test_next_inbound_turn_fills_an_origin_nothing_else_can_restore(tmp_path, topic):
    db, source, entry = topic
    store = _store(tmp_path, db)
    store._ensure_loaded()
    _strip(db, store, entry.session_key, 'origin_json=NULL, chat_id=NULL')

    restored = _store(tmp_path, db)
    assert restored.lookup_by_session_key(entry.session_key).origin is None
    assert restored.get_or_create_session(source).session_id == entry.session_id

    persisted = _store(tmp_path, db).lookup_by_session_key(entry.session_key)
    assert persisted.origin is not None and persisted.origin.to_dict() == source.to_dict()


def test_named_profile_route_heals_from_row_columns_under_multiplex(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    store = _store(tmp_path, db, multiplex_profiles=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='-1001', chat_type='group',
                           thread_id='600', user_id='111', profile='work')
    entry = store.get_or_create_session(source)
    assert entry.session_key.startswith('agent:work:')
    _strip(db, store, entry.session_key, 'origin_json=NULL')

    reloaded = _store(tmp_path, db, multiplex_profiles=True)
    origin = reloaded.lookup_by_session_key(entry.session_key).origin
    assert origin is not None and origin.profile == 'work' and origin.thread_id == '600'
    db.close()

