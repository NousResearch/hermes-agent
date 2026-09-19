"""Focused contracts for the lazy, opt-in gateway idle conversation boundary."""

from datetime import timedelta

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, SessionStore
from gateway.session_lifecycle import _now


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user")


def _store(tmp_path, *, minutes=30, active=lambda _key: False):
    return SessionStore(
        tmp_path,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
            idle_new_conversation_minutes=minutes,
        )}),
        has_active_processes_fn=active,
    )


def test_idle_boundary_is_disabled_without_positive_platform_setting(tmp_path):
    store = _store(tmp_path, minutes=None)
    entry = store.get_or_create_session(_source())
    entry.updated_at = _now() - timedelta(hours=1)

    assert not store.idle_boundary_due(_source())


def test_idle_boundary_only_crosses_after_configured_threshold(tmp_path):
    store = _store(tmp_path)
    entry = store.get_or_create_session(_source())
    # Keep clear of microsecond drift between the assignment and the check.
    entry.updated_at = _now() - timedelta(minutes=29)
    assert not store.idle_boundary_due(_source())

    entry.updated_at = _now() - timedelta(minutes=31)
    assert store.idle_boundary_due(_source())


def test_recent_activity_refreshes_the_idle_boundary(tmp_path):
    store = _store(tmp_path)
    entry = store.get_or_create_session(_source())
    entry.updated_at = _now() - timedelta(minutes=31)
    assert store.idle_boundary_due(_source())

    store.update_session(entry.session_key)
    assert not store.idle_boundary_due(_source())


def test_idle_boundary_preserves_sessions_with_work_in_flight(tmp_path):
    store = _store(tmp_path, active=lambda _key: True)
    entry = store.get_or_create_session(_source())
    entry.updated_at = _now() - timedelta(minutes=31)

    assert not store.idle_boundary_due(_source())


def test_idle_boundary_preserves_an_active_turn_marker(tmp_path):
    store = _store(tmp_path)
    entry = store.get_or_create_session(_source())
    entry.updated_at = _now() - timedelta(minutes=31)
    entry.active_turn_token = "turn-in-progress"

    assert not store.idle_boundary_due(_source())
