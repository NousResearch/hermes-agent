"""Configured session reset boundaries rotate durable gateway conversations."""
from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.session import SessionSource, SessionStore


@pytest.mark.parametrize(
    ("mode", "age", "expected_reason"),
    [
        ("idle", timedelta(minutes=2), "idle"),
        ("daily", timedelta(days=1), "daily"),
        ("both", timedelta(minutes=2), "idle"),
    ],
)
def test_configured_reset_rotates_durable_conversation(tmp_path, mode, age, expected_reason):
    config = GatewayConfig.from_dict({
        "default_reset_policy": {"mode": mode, "idle_minutes": 1},
    })
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="time-invariant", user_id="test")
    old = store.get_or_create_session(source)
    messages = [{"role": "user", "content": "keep my conversation"},
                {"role": "assistant", "content": "including after a restart"}]
    for message in messages:
        store.append_to_transcript(old.session_id, message)
    old.updated_at = datetime.now() - age
    store._save()
    routed = store.get_or_create_session(source)
    assert routed.session_id != old.session_id
    assert routed.was_auto_reset is True
    assert routed.auto_reset_reason == expected_reason
    assert routed.prev_session_id == old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] == expected_reason
    assert [{"role": m["role"], "content": m["content"]} for m in store.load_transcript(old.session_id)] == messages
    store._db.close()


def test_none_mode_preserves_durable_conversation(tmp_path):
    config = GatewayConfig.from_dict({
        "default_reset_policy": {"mode": "none", "idle_minutes": 1},
    })
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="time-invariant", user_id="test")
    old = store.get_or_create_session(source)
    old.updated_at = datetime.now() - timedelta(days=3)
    store._save()

    routed = store.get_or_create_session(source)

    assert routed.session_id == old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] is None
    store._db.close()


def test_platform_reset_policy_overrides_default(tmp_path):
    config = GatewayConfig.from_dict({
        "default_reset_policy": {"mode": "none"},
        "reset_by_platform": {
            "telegram": {"mode": "idle", "idle_minutes": 1},
        },
    })
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="platform-policy", user_id="test")
    old = store.get_or_create_session(source)
    old.updated_at = datetime.now() - timedelta(minutes=2)
    store._save()

    routed = store.get_or_create_session(source)

    assert routed.session_id != old.session_id
    assert routed.auto_reset_reason == "idle"
    store._db.close()


def test_load_gateway_config_consumes_session_reset_yaml(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "session_reset:\n  mode: both\n  at_hour: 4\n  idle_minutes: 240\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_gateway_config()

    assert config.default_reset_policy.mode == "both"
    assert config.default_reset_policy.at_hour == 4
    assert config.default_reset_policy.idle_minutes == 240


def test_reset_policy_round_trip_keeps_overrides():
    config = GatewayConfig.from_dict({
        "default_reset_policy": {"mode": "daily", "at_hour": 4},
        "reset_by_type": {"group": {"mode": "none"}},
        "reset_by_platform": {"whatsapp": {"mode": "idle", "idle_minutes": 30}},
    })

    restored = GatewayConfig.from_dict(config.to_dict())

    assert restored.default_reset_policy.mode == "daily"
    assert restored.reset_by_type["group"].mode == "none"
    assert restored.reset_by_platform[Platform.WHATSAPP].idle_minutes == 30
