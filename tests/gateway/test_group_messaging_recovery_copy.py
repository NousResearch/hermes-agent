"""Explain uncertain work before offering a retry, without changing controls."""

import pytest

from gateway import hosted_room_messaging as messaging
from tests.gateway.test_hosted_room_messaging import _FakeService, _seed_rooms


WARNING = "A Bot’s last task could not be confirmed. Retrying may repeat its actions."
RECONNECT_HINT = (
    "Some Bots need to reconnect. Open this Group Chat in Hermes Desktop and choose Reconnect."
)


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("state", ["indeterminate", "deferred", "settled"])
def test_uncertain_task_has_context_next_to_retry(tmp_path, monkeypatch, remote, state):
    db, room, _ = _seed_rooms(tmp_path)
    room = {**room, "messaging_ref": 1}
    service = _FakeService(db)
    status = {"working": False, "blocked": state != "settled", "counts": {state: 1}}
    service.room_status = status
    if remote:
        room = {**room, "_room_mode": "remote", "_remote_member_id": "ops"}
        monkeypatch.setattr(messaging, "_remote_summary", lambda *_: {
            "room": room, "events": [], "status": status,
        })
        service.status = lambda *_: pytest.fail("remote detail used local status")
    result = messaging.format_room_detail(service, room)
    assert (WARNING in result) == (state == "indeterminate")
    assert ("Retry:" in result) == (state != "settled")
    if state == "indeterminate":
        assert result.index(WARNING) < result.index("Retry:")
    assert not service.sent and not service.stopped and not service.retried


def test_classic_command_failure_keeps_its_existing_explanation(tmp_path):
    db, room, _ = _seed_rooms(tmp_path)
    room = {**room, "messaging_ref": 1, "_room_mode": "desktop", "desktop_available": True,
            "desktop_failed_commands": 1, "desktop_command": {"state": "failed"}}
    result = messaging.format_room_detail(_FakeService(db), room)
    assert WARNING not in result
    assert "The latest command could not be applied" in result
    assert "Retry:" in result


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("needs_reauthorization", [False, True])
@pytest.mark.parametrize("working", [False, True])
def test_reconnect_hint_is_conditional_and_preserves_work_controls(
    tmp_path, monkeypatch, remote, needs_reauthorization, working,
):
    db, room, _ = _seed_rooms(tmp_path)
    room = {**room, "messaging_ref": 1}
    service = _FakeService(db)
    status = {
        "working": working, "blocked": False, "counts": {"deferred": 1},
        "peer_routes": [{"member_id": "peer", "status": "needs_reauthorization"}]
        if needs_reauthorization else [],
    }
    service.room_status = status
    if remote:
        room = {**room, "_room_mode": "remote", "_remote_member_id": "ops"}
        monkeypatch.setattr(messaging, "_remote_summary", lambda *_: {
            "room": room, "events": [], "status": status,
        })
        service.status = lambda *_: pytest.fail("remote detail used local status")
    result = messaging.format_room_detail(service, room)
    assert (RECONNECT_HINT in result) is needs_reauthorization
    assert "Retry: `/group 1 retry`" in result
    assert ("Stop: `/group 1 stop`" in result) is working
    assert ("work queued or running" in result) is working
    assert not service.sent and not service.stopped and not service.retried
