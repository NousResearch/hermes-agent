"""The SQLite backend's live settle of a hosted-room task past plain expiry of its own lease, with main's driver cases and assertions; only the store access goes through the backend's ``hosted_room_driver`` verbs."""

from __future__ import annotations

import pytest

from persistence.gateway.hosted_rooms import HostedTaskIdentity, StaleLeaseError


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _identity(task_id: str = "task-1", *, turn_id: str = "turn-1") -> HostedTaskIdentity:
    return HostedTaskIdentity(room_id="room-1", task_id=task_id, thread_id="thread-1", turn_id=turn_id)


def _payload() -> dict:
    return {"target_profile": "ops", "prompt": "Inspect the release candidate.", "source_event_seq": 1}


@pytest.fixture
def driver(backend, scopes):
    store = backend.open(scopes["a"])
    store.hosted_rooms.create_room(
        placement="profile", room_id="room-1", name="Release room", members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a", now=90)
    return store.hosted_room_driver


def _lease(driver, clock, *, process="process-a", ttl=30):
    return driver.acquire_lease(
        placement="profile", room_id="room-1", gateway_id="gateway-a", authority_epoch=1, process_generation=process,
        ttl_seconds=ttl, now=clock())


def _admit(driver, identity, clock):
    return driver.admit_task(identity, placement="profile", payload=_payload(), now=clock())


def test_live_settlement_commits_after_plain_lease_expiry(driver):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(driver, clock, ttl=5)
    _admit(driver, identity, clock)
    attempt = driver.start_task(identity, lease, placement="profile", expected_cancel_generation=0, now=clock())

    clock.advance(5)  # plain expiry of our own unreleased, un-reclaimed lease
    with pytest.raises(StaleLeaseError):
        driver.settle_task(
            attempt, placement="profile", settlement_id="settlement-strict", status="settled",
            result={"text": "late"}, now=clock())
    settled = driver.settle_task(
        attempt, placement="profile", settlement_id="settlement-live", status="settled", result={"text": "late"},
        now=clock(), allow_expired_lease=True)

    assert settled["status"] == "settled"
    assert driver.get_task(identity, placement="profile")["status"] == "settled"


def test_live_settlement_after_expiry_still_fences_on_reclaim(driver):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(driver, clock, ttl=5)
    _admit(driver, identity, clock)
    attempt = driver.start_task(identity, lease, placement="profile", expected_cancel_generation=0, now=clock())

    clock.advance(5)
    second = _lease(driver, clock, process="process-b")
    driver.recover_room(second, placement="profile", now=clock())

    with pytest.raises(StaleLeaseError):
        driver.settle_task(
            attempt, placement="profile", settlement_id="settlement-live", status="settled", result={"text": "late"},
            now=clock(), allow_expired_lease=True)
    assert driver.get_task(identity, placement="profile")["status"] == "indeterminate"
