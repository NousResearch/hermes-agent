"""Elapsed time cannot replace a durable conversation; explicit boundaries still can."""
from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


@pytest.mark.parametrize("mode", ["idle", "daily", "both", "none"])
def test_old_reset_config_cannot_rotate_durable_conversation(tmp_path, mode):
    config = GatewayConfig.from_dict({
        "default_reset_policy": {"mode": mode, "idle_minutes": 1},
        "reset_by_type": {"dm": {"mode": mode, "idle_minutes": 1}},
        "reset_by_platform": {"telegram": {"mode": mode, "idle_minutes": 1}},
    })
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="time-invariant", user_id="test")
    old = store.get_or_create_session(source)
    messages = [{"role": "user", "content": "keep my conversation"},
                {"role": "assistant", "content": "including after a restart"}]
    for message in messages:
        store.append_to_transcript(old.session_id, message)
    old.updated_at = datetime.now() - timedelta(days=3)
    old.resume_pending = True
    old.last_resume_marked_at = old.updated_at
    store._save()
    routed = store.get_or_create_session(source)
    assert routed.session_id == old.session_id
    assert [{"role": m["role"], "content": m["content"]} for m in store.load_transcript(routed.session_id)] == messages
    assert store._db.get_session(old.session_id)["end_reason"] is None
    # Missing routing indexes must recover the same durable transcript too.
    store._db.replace_gateway_routing_entries({}, scope=store._routing_scope())
    store._entries.clear()
    recovered = store.get_or_create_session(source)
    assert recovered.session_id == old.session_id
    explicit = store.reset_session(recovered.session_key)
    assert explicit.session_id != old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] == "session_reset"
    assert [{"role": m["role"], "content": m["content"]} for m in store.load_transcript(old.session_id)] == messages
    store._db.close()
