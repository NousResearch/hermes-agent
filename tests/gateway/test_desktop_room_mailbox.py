from __future__ import annotations

import hashlib

import pytest

from gateway import desktop_room_mailbox as mailbox


class Clock:
    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def authorities(*room_ids: str, token: str = "authority:one") -> list[dict]:
    return [{"room_id": room_id, "authority_token": token} for room_id in room_ids]


def authority_commitment(token: str = "authority:one") -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def test_enqueue_is_idempotent_and_rejects_key_reuse(tmp_path):
    db = tmp_path / "state.db"
    first = mailbox.enqueue_command(
        db,
        command_id="messaging:abc",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
    )
    replay = mailbox.enqueue_command(
        db,
        command_id="messaging:abc",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
    )

    assert first["state"] == "pending"
    assert replay["idempotent"] is True
    with pytest.raises(mailbox.DesktopRoomMailboxError, match="different room work"):
        mailbox.enqueue_command(
            db,
            command_id="messaging:abc",
            room_id="room-1",
            authority_hash=authority_commitment(),
            action="send",
            payload={"message": "changed"},
        )


def test_enqueue_moves_the_cross_process_pending_signal(tmp_path):
    db = tmp_path / "desktop_room_mailbox.db"
    signal = mailbox.pending_signal_path(db)

    mailbox.enqueue_command(
        db,
        command_id="messaging:first",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
    )
    first = signal.read_text(encoding="ascii")
    mailbox.enqueue_command(
        db,
        command_id="messaging:second",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "again"},
    )

    assert signal.read_text(encoding="ascii") != first
    assert signal.stat().st_mode & 0o777 == 0o600


def test_signal_failure_does_not_change_a_durable_enqueue(tmp_path, monkeypatch):
    db = tmp_path / "desktop_room_mailbox.db"
    mailbox.enqueue_command(
        db,
        command_id="messaging:seed",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "seed"},
    )
    monkeypatch.setattr(
        type(mailbox.pending_signal_path(db)),
        "write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read only")),
    )

    queued = mailbox.enqueue_command(
        db,
        command_id="messaging:still-durable",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
    )

    assert queued["state"] == "pending"


def test_one_desktop_claims_and_presence_is_room_scoped(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:one",
        room_id="name:Classic room",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
        clock=clock,
    )

    claimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("name:Classic room"),
        clock=clock,
    )
    duplicate = mailbox.claim_commands(
        db,
        consumer_id="desktop:second",
        room_authorities=authorities("name:Classic room", token="authority:two"),
        clock=clock,
    )

    assert [item["command_id"] for item in claimed] == ["messaging:one"]
    assert duplicate == []
    assert mailbox.room_available(db, "name:Classic room", clock=clock) is True
    assert mailbox.room_available(db, "another-room", clock=clock) is False


def test_expired_claim_is_recovered_by_the_registered_desktop(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:retry",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
        clock=clock,
    )
    first = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        claim_ttl=10,
        presence_ttl=10,
        clock=clock,
    )
    clock.value += 11
    second = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        clock=clock,
    )

    assert first[0]["attempts"] == 1
    assert second[0]["attempts"] == 2
    with pytest.raises(mailbox.DesktopRoomMailboxError, match="no longer owned"):
        mailbox.complete_command(
            db,
            consumer_id="desktop:first",
            command_id="messaging:retry",
            lease_token=first[0]["lease_token"],
            success=True,
            result={"thread_id": "old"},
            clock=clock,
        )

    clock.value += mailbox.CLAIM_TTL_SECONDS + 1
    assert (
        mailbox.claim_commands(
            db,
            consumer_id="desktop:first",
            room_authorities=authorities("room-1"),
            clock=clock,
        )
        == []
    )
    exhausted = mailbox.latest_command_states(db, ["room-1"])["room-1"]
    assert exhausted["state"] == "failed"
    assert exhausted["result"]["code"] == "automatic_attempts_exhausted"

    retried = mailbox.retry_failed_command(
        db,
        room_id="room-1",
        command_id="messaging:retry",
        clock=clock,
    )
    assert retried["attempts"] == 0
    reclaimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        clock=clock,
    )
    assert reclaimed[0]["attempts"] == 1


def test_completion_ack_retry_is_idempotent(tmp_path):
    db = tmp_path / "state.db"
    mailbox.enqueue_command(
        db,
        command_id="messaging:complete",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="stop",
        payload={"target_command_id": "messaging:send-1"},
    )
    claimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
    )
    first = mailbox.complete_command(
        db,
        consumer_id="desktop:first",
        command_id="messaging:complete",
        lease_token=claimed[0]["lease_token"],
        success=True,
        result={"stopped": True},
    )
    replay = mailbox.complete_command(
        db,
        consumer_id="desktop:first",
        command_id="messaging:complete",
        lease_token=claimed[0]["lease_token"],
        success=True,
        result={"stopped": True},
    )

    assert first["state"] == "completed"
    assert replay["idempotent"] is True


def test_explicit_retry_requeues_one_failed_command_idempotently(tmp_path):
    db = tmp_path / "state.db"
    mailbox.enqueue_command(
        db,
        command_id="messaging:failed",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
    )
    claimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
    )[0]
    failed = mailbox.complete_command(
        db,
        consumer_id="desktop:first",
        command_id=claimed["command_id"],
        lease_token=claimed["lease_token"],
        success=False,
        result={"error": "temporarily unavailable"},
    )

    assert failed["state"] == "failed"
    retried = mailbox.retry_failed_command(db, room_id="room-1")
    replay = mailbox.retry_failed_command(
        db,
        room_id="room-1",
        command_id=claimed["command_id"],
    )
    assert retried["state"] == "pending"
    assert retried["attempts"] == 0
    assert replay["idempotent"] is True
    reclaimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
    )
    assert reclaimed[0]["command_id"] == "messaging:failed"
    assert reclaimed[0]["attempts"] == 1


def test_reclaim_rotates_token_and_fences_a_stale_attempt(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:fenced",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
        clock=clock,
    )
    first = mailbox.claim_commands(
        db,
        consumer_id="desktop:same-install",
        room_authorities=authorities("room-1"),
        claim_ttl=5,
        clock=clock,
    )[0]
    clock.value += 6
    second = mailbox.claim_commands(
        db,
        consumer_id="desktop:same-install",
        room_authorities=authorities("room-1"),
        claim_ttl=5,
        clock=clock,
    )[0]

    assert first["lease_token"] != second["lease_token"]
    with pytest.raises(mailbox.DesktopRoomMailboxError, match="no longer owned"):
        mailbox.complete_command(
            db,
            consumer_id="desktop:same-install",
            command_id="messaging:fenced",
            lease_token=first["lease_token"],
            success=True,
            result={"thread_id": "old"},
            clock=clock,
        )


def test_live_claim_can_be_renewed(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:renew",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
        clock=clock,
    )
    claimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        claim_ttl=10,
        clock=clock,
    )[0]
    clock.value += 8
    renewed = mailbox.renew_command(
        db,
        consumer_id="desktop:first",
        command_id="messaging:renew",
        lease_token=claimed["lease_token"],
        claim_ttl=10,
        clock=clock,
    )
    clock.value += 5

    assert renewed["lease_token"] == claimed["lease_token"]
    assert (
        mailbox.complete_command(
            db,
            consumer_id="desktop:first",
            command_id="messaging:renew",
            lease_token=claimed["lease_token"],
            success=True,
            result={"thread_id": "thread-1"},
            clock=clock,
        )["state"]
        == "completed"
    )


def test_presence_expires_without_deleting_pending_work(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:later",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "later"},
        clock=clock,
    )
    mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        presence_ttl=5,
        claim_ttl=5,
        clock=clock,
    )
    clock.value += 6

    assert mailbox.room_available(db, "room-1", clock=clock) is False
    reclaimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        clock=clock,
    )
    assert reclaimed[0]["command_id"] == "messaging:later"


def test_authority_token_fences_a_cold_desktop_after_owner_expiry(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:fenced-room",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
        clock=clock,
    )
    mailbox.claim_commands(
        db,
        consumer_id="desktop:owner",
        room_authorities=authorities("room-1"),
        claim_ttl=5,
        presence_ttl=5,
        clock=clock,
    )
    clock.value += 6

    assert (
        mailbox.claim_commands(
            db,
            consumer_id="desktop:cold",
            room_authorities=authorities("room-1", token="authority:wrong"),
            clock=clock,
        )
        == []
    )
    reclaimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:owner",
        room_authorities=authorities("room-1"),
        clock=clock,
    )
    assert reclaimed[0]["command_id"] == "messaging:fenced-room"


def test_claim_cannot_establish_room_authority(tmp_path):
    db = tmp_path / "state.db"
    mailbox.enqueue_command(
        db,
        command_id="messaging:unregistered",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
    )

    assert (
        mailbox.claim_commands(
            db,
            consumer_id="desktop:untrusted",
            room_authorities=authorities("room-1", token="authority:guessed"),
        )
        == []
    )

    claimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:owner",
        room_authorities=authorities("room-1"),
    )
    assert [item["command_id"] for item in claimed] == ["messaging:unregistered"]


def test_projected_authority_commitment_is_idempotent_and_fenced(tmp_path):
    db = tmp_path / "state.db"
    assert mailbox.register_projected_authorities(
        db,
        [{"room_id": "room-1", "authority_hash": authority_commitment()}],
    ) == ["room-1"]
    assert mailbox.register_projected_authorities(
        db,
        [{"room_id": "room-1", "authority_hash": authority_commitment()}],
    ) == ["room-1"]

    assert (
        mailbox.register_projected_authorities(
            db,
            [
                {
                    "room_id": "room-1",
                    "authority_hash": authority_commitment("authority:other"),
                }
            ],
        )
        == []
    )

    assert mailbox.register_projected_authorities(
        db,
        [
            {
                "room_id": "room-1",
                "authority_hash": authority_commitment("authority:other"),
            },
            {"room_id": "room-2", "authority_hash": authority_commitment()},
        ],
    ) == ["room-2"]

    with pytest.raises(mailbox.DesktopRoomMailboxError, match="commitment"):
        mailbox.enqueue_command(
            db,
            command_id="messaging:conflict",
            room_id="room-1",
            authority_hash=authority_commitment("authority:other"),
            action="send",
            payload={"message": "must not replace owner"},
        )


def test_stop_supersedes_an_earlier_send_before_it_is_claimed(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    for action in ("send", "stop"):
        mailbox.enqueue_command(
            db,
            command_id=f"messaging:{action}",
            room_id="room-1",
            authority_hash=authority_commitment(),
            action=action,
            payload=(
                {"message": "hello"}
                if action == "send"
                else {"target_command_id": "messaging:send"}
            ),
            clock=clock,
        )

    stopped = mailbox.claim_commands(
        db,
        consumer_id="desktop:owner",
        room_authorities=authorities("room-1"),
        actions=["stop"],
        clock=clock,
    )
    assert [item["action"] for item in stopped] == ["stop"]
    assert stopped[0]["target_command_state"] == "failed"
    assert stopped[0]["target_result_code"] == "superseded_by_stop"
    sends = mailbox.claim_commands(
        db,
        consumer_id="desktop:owner",
        room_authorities=authorities("room-1"),
        actions=["send"],
        clock=clock,
    )
    assert sends == []


def test_renew_keeps_room_ownership_alive_for_long_turn(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:long",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "hello"},
        clock=clock,
    )
    command = mailbox.claim_commands(
        db,
        consumer_id="desktop:owner",
        room_authorities=authorities("room-1"),
        claim_ttl=45,
        presence_ttl=90,
        clock=clock,
    )[0]

    for _ in range(8):
        clock.value += 30
        mailbox.renew_command(
            db,
            consumer_id="desktop:owner",
            command_id=command["command_id"],
            lease_token=command["lease_token"],
            claim_ttl=45,
            presence_ttl=90,
            clock=clock,
        )

    assert mailbox.room_available(db, "room-1", clock=clock) is True
    assert (
        mailbox.claim_commands(
            db,
            consumer_id="desktop:other",
            room_authorities=authorities("room-1"),
            actions=["stop"],
            clock=clock,
        )
        == []
    )


def test_default_presence_overlaps_the_minute_desktop_backstop(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.register_projected_authorities(
        db,
        [{"room_id": "room-1", "authority_hash": authority_commitment()}],
        clock=clock,
    )
    mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        clock=clock,
    )

    clock.value += 61
    assert mailbox.room_available(db, "room-1", clock=clock) is True
    clock.value += 30
    assert mailbox.room_available(db, "room-1", clock=clock) is False


def test_latest_command_state_is_scoped_per_room(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:first",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "first"},
        clock=clock,
    )
    clock.value += 1
    mailbox.enqueue_command(
        db,
        command_id="messaging:second",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "second"},
        clock=clock,
    )
    mailbox.enqueue_command(
        db,
        command_id="messaging:other",
        room_id="room-2",
        authority_hash=authority_commitment(),
        action="stop",
        payload={"target_command_id": "messaging:send-1"},
        clock=clock,
    )

    states = mailbox.latest_command_states(db, ["room-1", "room-2"])

    assert states["room-1"]["command_id"] == "messaging:second"
    assert states["room-2"]["command_id"] == "messaging:other"


def test_paged_claims_preserve_presence_for_every_owned_room(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    rooms = [f"room-{index}" for index in range(260)]

    for index in range(0, len(rooms), mailbox.MAX_ROOM_IDS):
        batch = rooms[index : index + mailbox.MAX_ROOM_IDS]
        mailbox.register_projected_authorities(
            db,
            [
                {"room_id": room_id, "authority_hash": authority_commitment()}
                for room_id in batch
            ],
            clock=clock,
        )
        mailbox.claim_commands(
            db,
            consumer_id="desktop:first",
            room_authorities=authorities(*batch),
            clock=clock,
        )

    assert mailbox.available_room_ids(db, rooms, clock=clock) == set(rooms)


def test_presence_refresh_allows_secret_proven_takeover_after_expiry(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.register_projected_authorities(
        db,
        [{"room_id": "room-1", "authority_hash": authority_commitment()}],
        clock=clock,
    )

    assert mailbox.refresh_presence(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        presence_ttl=5,
        clock=clock,
    ) == ["room-1"]
    assert (
        mailbox.refresh_presence(
            db,
            consumer_id="desktop:second",
            room_authorities=authorities("room-1"),
            presence_ttl=5,
            clock=clock,
        )
        == []
    )

    clock.value += 6
    assert (
        mailbox.refresh_presence(
            db,
            consumer_id="desktop:second",
            room_authorities=authorities("room-1", token="authority:wrong"),
            clock=clock,
        )
        == []
    )
    assert mailbox.refresh_presence(
        db,
        consumer_id="desktop:second",
        room_authorities=authorities("room-1"),
        clock=clock,
    ) == ["room-1"]


def test_pending_command_expires_instead_of_running_days_later(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    mailbox.enqueue_command(
        db,
        command_id="messaging:stale",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="send",
        payload={"message": "old"},
        clock=clock,
    )

    clock.value += mailbox.PENDING_TTL_SECONDS + 1
    assert (
        mailbox.claim_commands(
            db,
            consumer_id="desktop:first",
            room_authorities=authorities("room-1"),
            clock=clock,
        )
        == []
    )
    state = mailbox.latest_command_states(db, ["room-1"])["room-1"]
    assert state["state"] == "failed"
    assert state["result"]["code"] == "command_expired"


def test_retry_requeues_all_bounded_expired_commands_oldest_first(tmp_path):
    db = tmp_path / "state.db"
    clock = Clock()
    for index in range(2):
        mailbox.enqueue_command(
            db,
            command_id=f"messaging:stale-{index}",
            room_id="room-1",
            authority_hash=authority_commitment(),
            action="send",
            payload={"message": str(index)},
            clock=clock,
        )
        clock.value += 1

    clock.value += mailbox.PENDING_TTL_SECONDS + 1
    assert (
        mailbox.claim_commands(
            db,
            consumer_id="desktop:first",
            room_authorities=authorities("room-1"),
            clock=clock,
        )
        == []
    )

    frozen = mailbox.retryable_command_ids(db, room_id="room-1")
    assert frozen == ("messaging:stale-0", "messaging:stale-1")
    retried = mailbox.retry_failed_commands(
        db,
        room_id="room-1",
        command_ids=frozen,
        clock=clock,
    )
    assert [command["command_id"] for command in retried] == list(frozen)

    claimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
        clock=clock,
    )
    assert [command["payload"]["message"] for command in claimed] == ["0", "1"]
    assert [command["attempts"] for command in claimed] == [1, 1]


def test_pending_queue_is_bounded_per_group_chat(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(mailbox, "MAX_PENDING_COMMANDS_PER_ROOM", 2)
    for index in range(2):
        mailbox.enqueue_command(
            db,
            command_id=f"messaging:{index}",
            room_id="room-1",
            authority_hash=authority_commitment(),
            action="send",
            payload={"message": str(index)},
        )

    with pytest.raises(mailbox.DesktopRoomMailboxError, match="too many commands"):
        mailbox.enqueue_command(
            db,
            command_id="messaging:overflow",
            room_id="room-1",
            authority_hash=authority_commitment(),
            action="send",
            payload={"message": "overflow"},
        )


def test_stop_supersedes_pending_sends_and_is_claimed_first(tmp_path):
    db = tmp_path / "state.db"
    for index in range(2):
        mailbox.enqueue_command(
            db,
            command_id=f"messaging:send-{index}",
            room_id="room-1",
            authority_hash=authority_commitment(),
            action="send",
            payload={"message": str(index)},
        )
    mailbox.enqueue_command(
        db,
        command_id="messaging:stop",
        room_id="room-1",
        authority_hash=authority_commitment(),
        action="stop",
        payload={"target_command_id": "messaging:send-1"},
    )

    with mailbox._transaction(db) as conn:
        send_states = {
            str(row["command_id"]): str(row["state"])
            for row in conn.execute(
                "SELECT command_id, state FROM desktop_room_commands WHERE action='send'"
            )
        }
    claimed = mailbox.claim_commands(
        db,
        consumer_id="desktop:first",
        room_authorities=authorities("room-1"),
    )

    assert send_states == {
        "messaging:send-0": "failed",
        "messaging:send-1": "failed",
    }
    assert [command["command_id"] for command in claimed] == ["messaging:stop"]
    assert claimed[0]["target_command_state"] == "failed"
    assert claimed[0]["target_result_code"] == "superseded_by_stop"
