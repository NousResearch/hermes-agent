"""Heartbeat ownership follows compression, but ends at a conversation boundary."""

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_cli.heartbeat import (
    HeartbeatManager,
    HeartbeatState,
    load_heartbeat,
    migrate_heartbeat_to_session,
    save_heartbeat,
)


def test_reset_clears_only_the_departing_conversations_heartbeat(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="reset", user_id="owner")
    entry = store.get_or_create_session(source)
    key, parent = entry.session_key, entry.session_id
    child = parent + "-compressed"
    state = HeartbeatState(prompt="check deploy", interval_seconds=60, created_at=1)
    save_heartbeat(parent, state)
    save_heartbeat("unrelated", state)
    db = store._db_for_key(key)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="telegram",
        require_compression_lease=False, model="offline", model_config={},
        system_prompt="offline", messages=[{"role": "user", "content": "retained"}],
    )
    assert migrate_heartbeat_to_session(parent, child)
    assert store.advance_compression_session(key, parent, child)
    assert load_heartbeat(parent) is None
    assert HeartbeatManager(child).due_prompt(now=120)

    replacement = store.reset_session(key)

    assert replacement.session_id != child
    assert load_heartbeat(child) is None
    assert load_heartbeat(replacement.session_id) is None
    assert HeartbeatManager("unrelated").due_prompt(now=120)
    # Resuming the archived conversation must not resurrect its old schedule.
    store.switch_session(key, child)
    assert load_heartbeat(child) is None
