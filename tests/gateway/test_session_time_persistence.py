"""Focused routing coverage for configured automatic session resets."""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.session import SessionSource, SessionStore


NOW = datetime(2026, 9, 16, 9, 0, 0)


def _store(tmp_path, policy):
    config = GatewayConfig.from_dict({"default_reset_policy": policy})
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(
        platform=Platform.WHATSAPP, chat_id="reset-regression", user_id="test"
    )
    with patch("gateway.session_lifecycle._now", return_value=NOW):
        entry = store.get_or_create_session(source)
    return store, source, entry


def test_live_yaml_session_reset_reaches_gateway_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "session_reset:\n  mode: both\n  at_hour: 4\n  idle_minutes: 240\n",
        encoding="utf-8",
    )

    policy = load_gateway_config().default_reset_policy

    assert policy.mode == "both"
    assert policy.at_hour == 4
    assert policy.idle_minutes == 240


@pytest.mark.parametrize(
    ("policy", "updated_at", "reason"),
    [
        ({"mode": "idle", "idle_minutes": 240}, NOW - timedelta(minutes=241), "idle"),
        ({"mode": "daily", "at_hour": 4}, NOW.replace(hour=3), "daily"),
        ({"mode": "both", "at_hour": 4, "idle_minutes": 240}, NOW.replace(hour=3), "idle"),
    ],
)
def test_overdue_policy_rotates_routed_session(tmp_path, policy, updated_at, reason):
    store, source, old = _store(tmp_path, policy)
    old.updated_at = updated_at
    old.last_prompt_tokens = 1
    store._save()

    with patch("gateway.session_lifecycle._now", return_value=NOW):
        routed = store.get_or_create_session(source)

    assert routed.session_id != old.session_id
    assert routed.was_auto_reset is True
    assert routed.auto_reset_reason == reason
    assert routed.prev_session_id == old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] == reason
    store._db.close()


@pytest.mark.parametrize("mode", ["idle", "daily", "both"])
def test_fresh_policy_keeps_routed_session(tmp_path, mode):
    store, source, old = _store(
        tmp_path, {"mode": mode, "at_hour": 4, "idle_minutes": 240}
    )
    old.updated_at = NOW - timedelta(minutes=1)

    with patch("gateway.session_lifecycle._now", return_value=NOW):
        routed = store.get_or_create_session(source)

    assert routed.session_id == old.session_id
    store._db.close()


def test_mode_none_keeps_old_routed_session(tmp_path):
    store, source, old = _store(tmp_path, {"mode": "none"})
    old.updated_at = NOW - timedelta(days=30)

    with patch("gateway.session_lifecycle._now", return_value=NOW):
        routed = store.get_or_create_session(source)

    assert routed.session_id == old.session_id
    store._db.close()


def test_suspension_still_wins_over_time_policy(tmp_path):
    store, source, old = _store(
        tmp_path, {"mode": "both", "at_hour": 4, "idle_minutes": 240}
    )
    old.updated_at = NOW - timedelta(days=2)
    old.suspended = True

    with patch("gateway.session_lifecycle._now", return_value=NOW):
        routed = store.get_or_create_session(source)

    assert routed.session_id != old.session_id
    assert routed.auto_reset_reason == "suspended"
    assert store._db.get_session(old.session_id)["end_reason"] == "suspended"
    store._db.close()


def test_overdue_recovered_session_rotates_instead_of_reopening(tmp_path):
    store, source, old = _store(tmp_path, {"mode": "idle", "idle_minutes": 240})
    old.updated_at = NOW - timedelta(minutes=241)
    old.last_prompt_tokens = 1
    store._save()
    store._db.touch_session_activity(
        old.session_id, old.updated_at.timestamp(), description="last user message"
    )
    store._entries.clear()

    with patch("gateway.session_lifecycle._now", return_value=NOW):
        routed = store.get_or_create_session(source)

    assert routed.session_id != old.session_id
    assert routed.auto_reset_reason == "idle"
    assert routed.prev_session_id == old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] == "idle"
    store._db.close()