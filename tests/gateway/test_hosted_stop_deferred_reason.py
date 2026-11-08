"""Stop publishes its own cancellation reason after a turn was deferred."""

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as state
from tests.tui_gateway.test_hosted_room_observation_lease import db as db_fixture, room as room_fixture, store as store_fixture
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, _admit, _identity,
)

db = db_fixture
room, store = room_fixture, store_fixture


def test_stop_of_a_deferred_turn_is_not_reported_as_member_unavailable(db):
    identity, clock = _identity(), lambda: 100.0
    _admit(db, identity)
    lease = state.acquire_lease(
        db, placement='profile', room_id=BINDING.room_id, gateway_id=BINDING.gateway_id, authority_epoch=1,
        process_generation="p1", ttl_seconds=30, clock=clock)
    attempt = state.start_task(db, identity, lease, placement='profile', expected_cancel_generation=0, clock=clock)
    state.mark_attempt_indeterminate(db, attempt, lease, placement='profile', clock=clock)
    state.defer_indeterminate_task(
        db, identity, lease, placement='profile', expected_execution_generation=1, expected_cancel_generation=0,
        reason="member_unavailable", clock=clock)

    cancelled = state.cancel_task(db, identity, placement='profile', cancel_id="room-stop:1", expected_cancel_generation=0,
                                  clock=clock)

    # The room's turn.cancelled takes its reason from the task result (plan_publication).
    payload, _effects = discussion._cancelled_effects(cancelled["result"], newer_same_thread=False)
    assert payload["reason"] != "member_unavailable"
