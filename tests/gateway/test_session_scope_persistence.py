import json
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_recall_scope_key, build_session_key
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB


def test_session_store_persists_channel_recall_scope_without_user_id(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    db = SessionDB(tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store._db = db

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-a",
        chat_type="channel",
        user_id="user-1",
        guild_id="guild-1",
        chat_name="do-not-store-name-as-scope",
        chat_topic="do-not-store-topic",
        message_id="msg-1",
    )
    entry = store.get_or_create_session(source)

    row = db.get_session(entry.session_id)
    assert entry.session_key == build_session_key(source)
    assert row["scope_key"] == build_recall_scope_key(source)

    origin = json.loads(row["origin_json"])
    assert origin == {
        "platform": "discord",
        "chat_id": "channel-a",
        "chat_type": "channel",
        "user_id": "user-1",
        "guild_id": "guild-1",
    }

    second_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-a",
        chat_type="channel",
        user_id="user-2",
        guild_id="guild-1",
    )
    second = store.get_or_create_session(second_source)
    second_row = db.get_session(second.session_id)

    assert second.session_key == build_session_key(second_source)
    assert second.session_key != entry.session_key
    assert second_row["scope_key"] == row["scope_key"]


def test_compression_child_inherits_parent_scope_when_not_passed(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(
        "parent",
        source="discord",
        scope_key="agent:main:discord:channel:now",
        origin_json={"platform": "discord", "chat_id": "now", "chat_type": "channel"},
    )
    db.end_session("parent", "compression")

    db.create_session("child", source="discord", parent_session_id="parent")

    child = db.get_session("child")
    assert child["scope_key"] == "agent:main:discord:channel:now"
    assert json.loads(child["origin_json"]) == {
        "platform": "discord",
        "chat_id": "now",
        "chat_type": "channel",
    }


def test_session_reset_persists_existing_scope_key(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    db = SessionDB(tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store._db = db

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-a",
        chat_type="channel",
        user_id="user-1",
    )
    first = store.get_or_create_session(source)
    reset = store.reset_session(first.session_key)

    row = db.get_session(reset.session_id)
    assert row["scope_key"] == build_recall_scope_key(source)
    assert json.loads(row["origin_json"]) == {
        "platform": "discord",
        "chat_id": "channel-a",
        "chat_type": "channel",
        "user_id": "user-1",
    }


def test_agent_lazy_db_creation_backfills_gateway_scope(tmp_path):
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    db.create_session("api-session", source="api_server")

    agent = SimpleNamespace(
        _session_db_created=False,
        _session_db=db,
        session_id="api-session",
        platform="api_server",
        model="test-model",
        _session_init_model_config=None,
        _cached_system_prompt=None,
        _parent_session_id=None,
        _gateway_session_key="agent:main:api_server:client-a",
        _gateway_recall_scope_key="agent:main:api_server:client-a",
        _chat_id="client-a",
        _chat_type="dm",
        _thread_id=None,
        _user_id=None,
        _user_id_alt=None,
    )

    AIAgent._ensure_db_session(agent)

    row = db.get_session("api-session")
    assert row["scope_key"] == "agent:main:api_server:client-a"
    assert json.loads(row["origin_json"]) == {
        "platform": "api_server",
        "chat_id": "client-a",
        "chat_type": "dm",
    }


def test_agent_lazy_db_creation_uses_recall_scope_context(tmp_path):
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    db.create_session("api-session", source="api_server")

    agent = SimpleNamespace(
        _session_db_created=False,
        _session_db=db,
        session_id="api-session",
        platform="api_server",
        model="test-model",
        _session_init_model_config=None,
        _cached_system_prompt=None,
        _parent_session_id=None,
        _gateway_session_key=None,
        _gateway_recall_scope_key=None,
        _chat_id=None,
        _chat_type=None,
        _thread_id=None,
        _user_id=None,
        _user_id_alt=None,
    )

    tokens = set_session_vars(
        platform="api_server",
        chat_id="api-session",
        session_key="api-session",
        recall_scope_key="agent:main:api_server:api-session",
        session_id="api-session",
    )
    try:
        AIAgent._ensure_db_session(agent)
    finally:
        clear_session_vars(tokens)

    row = db.get_session("api-session")
    assert row["scope_key"] == "agent:main:api_server:api-session"
    assert json.loads(row["origin_json"]) == {
        "platform": "api_server",
        "chat_id": "api-session",
    }
