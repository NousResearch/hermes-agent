from unittest.mock import MagicMock, patch

from gateway.mobile_notifications import (
    FCMNotifier,
    MobileDevice,
    MobileDeviceStore,
    MobilePairingStore,
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


def test_fcm_payload_contains_bounded_status_v2_only():
    payload = build_fcm_message(_device(), {
        "event": "session.completed",
        "session_id": "session-1",
        "run_id": "run-1",
        "title": "Release checks",
        "state": "completed",
        "active_count": 2,
        "latest_status": "  Running   checks  " + "x" * 300,
        "timestamp": 123.5,
        "tasks_completed": 4,
        "tasks_total": 8,
        "active_subagents": 2,
        "error_category": "network",
        "output": "private assistant response must not leave the host",
    })

    data = payload["message"]["data"]
    assert payload["message"]["token"] == "secret-device-token"
    assert data == {
        "event": "session.completed",
        "host_profile_id": "mobile-host-1",
        "session_id": "session-1",
        "run_id": "run-1",
        "title": "Release checks",
        "state": "completed",
        "active_count": "2",
        "latest_status": "Running checks " + "x" * 165,
        "timestamp": "123.5",
        "tasks_completed": "4",
        "tasks_total": "8",
        "active_subagents": "2",
        "error_category": "network",
    }
    assert "private assistant response" not in str(payload)


def test_fcm_payload_drops_unapproved_error_categories():
    data = build_fcm_message(_device(), {"error_category": "raw traceback"})["message"]["data"]
    assert "error_category" not in data


def test_pairing_grant_is_single_use_and_tokens_are_hashed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MobilePairingStore()
    grant = store.create_grant(now=100)

    paired = store.exchange(
        grant.secret,
        installation_id="install-1",
        device_name="Pixel",
        now=101,
    )
    assert paired.scope == "mobile.full"
    assert store.authenticate(paired.token) is True
    assert paired.token not in (tmp_path / "runtime" / "mobile_pairing.json").read_text()
    assert store.exchange(grant.secret, installation_id="install-2", now=102) is None


def test_pairing_grant_expires_and_manual_code_works(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MobilePairingStore()
    grant = store.create_grant(now=100)

    assert store.exchange(grant.secret, installation_id="late", now=401) is None
    fresh = store.create_grant(now=500)
    paired = store.exchange(fresh.code, installation_id="manual", now=501)
    assert paired is not None
    assert store.revoke(paired.device_id) is True
    assert store.authenticate(paired.token) is False


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
