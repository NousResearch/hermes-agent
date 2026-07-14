from unittest.mock import MagicMock, patch

from gateway.mobile_notifications import (
    FCMNotifier,
    MobileDevice,
    MobileDeviceStore,
    build_fcm_message,
)


def _device(**overrides):
    values = {
        "installation_id": "install-1",
        "token": "secret-device-token",
        "host_profile_id": "mobile-host-1",
        "updated_at": 1.0,
    }
    values.update(overrides)
    return MobileDevice(**values)


def test_device_store_round_trips_and_deletes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MobileDeviceStore()

    stored = store.upsert({
        "installation_id": "install-1",
        "token": "token-1",
        "host_profile_id": "host-1",
        "app_version": "0.2.0",
        "capabilities": {"notifications": True, "bubbles": False, "overlay": True},
    })

    assert stored.installation_id == "install-1"
    assert store.list() == [stored]
    assert store.list()[0].overlay_enabled is True
    assert store.list()[0].bubbles_enabled is False
    assert store.delete("install-1") is True
    assert store.list() == []


def test_device_store_replaces_same_installation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MobileDeviceStore()
    base = {"installation_id": "same", "host_profile_id": "host-1"}
    store.upsert({**base, "token": "old"})
    store.upsert({**base, "token": "new"})

    assert [item.token for item in store.list()] == ["new"]


def test_fcm_payload_contains_status_only():
    payload = build_fcm_message(_device(), {
        "event": "session.completed",
        "session_id": "session-1",
        "run_id": "run-1",
        "title": "Release checks",
        "state": "completed",
        "active_count": 2,
        "output": "private assistant response must not leave the host",
    })

    data = payload["message"]["data"]
    assert payload["message"]["fid"] == "secret-device-token"
    assert data == {
        "event": "session.completed",
        "host_profile_id": "mobile-host-1",
        "session_id": "session-1",
        "run_id": "run-1",
        "title": "Release checks",
        "state": "completed",
        "active_count": "2",
    }
    assert "private assistant response" not in str(payload)


def test_notifier_is_noop_until_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MobileDeviceStore()
    store.upsert({"installation_id": "i", "token": "t", "host_profile_id": "h"})
    with patch("gateway.mobile_notifications.load_mobile_notification_config", return_value={}):
        assert FCMNotifier(store).send({"event": "session.started"}) == 0


def test_notifier_prunes_unregistered_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MobileDeviceStore()
    store.upsert({"installation_id": "i", "token": "expired", "host_profile_id": "h"})
    credentials = MagicMock(token="access-token")
    response = MagicMock(status_code=404, text="UNREGISTERED")
    with patch(
        "gateway.mobile_notifications.load_mobile_notification_config",
        return_value={"enabled": True, "project_id": "project"},
    ), patch("google.auth.default", return_value=(credentials, "project")), patch(
        "requests.post", return_value=response,
    ):
        assert FCMNotifier(store).send({"event": "session.started"}) == 0

    assert store.list() == []
