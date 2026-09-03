"""Durable approval decisions from messaging Group Chat controls."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import time

import pytest

from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms


def _action(**overrides):
    return {
        "kind": "approval",
        "authority_gateway_id": "gateway-1",
        "authority_epoch": 1,
        "task_id": "task-1",
        "execution_generation": 2,
        "request_id": "request-1",
        "approval": {
            "description": "Run focused tests",
            "command": "pytest -q tests/focused",
            "choices": ["once", "deny"],
        },
        **overrides,
    }


def _pending(db: Path):
    return approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-1",
        action=_action(),
    )


def test_pending_approval_is_bounded_and_cleared_exactly(tmp_path):
    db = tmp_path / "state.db"
    action = _action(
        approval={
            "description": "x" * 1_000,
            "command": "pytest\n-q\ttests/focused",
            "choices": ["once", "session", "always", "deny"],
        }
    )
    pending = approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-1",
        action=action,
    )

    assert pending["approval"] == {
        "description": "x" * approvals.MAX_APPROVAL_TEXT_CHARS,
        "command": "pytest -q tests/focused",
        "choices": ["once", "deny"],
    }
    assert (
        approvals.clear_pending_approval(
            db,
            room_id="room-1",
            member_id="member-1",
            request_id="other-request",
        )
        == 0
    )
    assert len(approvals.list_pending_approvals(db, room_id="room-1")) == 1
    assert (
        approvals.clear_pending_approval(
            db,
            room_id="room-1",
            member_id="member-1",
            request_id="request-1",
        )
        == 1
    )


def test_existing_approval_table_migrates_observer_generation(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE hosted_room_pending_approvals (
               room_id TEXT NOT NULL,
               authority_gateway_id TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               member_id TEXT NOT NULL,
               task_id TEXT NOT NULL,
               execution_generation INTEGER NOT NULL,
               request_id TEXT NOT NULL,
               profile TEXT NOT NULL,
               session_id TEXT NOT NULL,
               description TEXT NOT NULL,
               command_text TEXT NOT NULL,
               updated_at REAL NOT NULL,
               PRIMARY KEY (room_id, member_id)
           )"""
    )
    conn.commit()
    conn.close()

    pending = _pending(db)

    assert pending["observer_generation"] == "legacy"
    assert (
        approvals.list_pending_approvals(db, room_id="room-1")[0]["observer_generation"]
        == "legacy"
    )


def test_existing_approval_table_migrates_concurrently(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE hosted_room_pending_approvals (
               room_id TEXT NOT NULL,
               authority_gateway_id TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               member_id TEXT NOT NULL,
               task_id TEXT NOT NULL,
               execution_generation INTEGER NOT NULL,
               request_id TEXT NOT NULL,
               profile TEXT NOT NULL,
               session_id TEXT NOT NULL,
               description TEXT NOT NULL,
               command_text TEXT NOT NULL,
               updated_at REAL NOT NULL,
               PRIMARY KEY (room_id, member_id)
           )"""
    )
    conn.commit()
    conn.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _index: approvals.list_all_pending_approvals(db),
                range(16),
            )
        )

    assert results == [[]] * 16
    conn = sqlite3.connect(db)
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(hosted_room_pending_approvals)")
    }
    conn.close()
    assert {"observer_generation", "observer_lease_generation"} <= columns


def test_observer_lease_is_checked_after_waiting_for_write_lock(tmp_path):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Lock race",
        members=[
            {"member_id": "default", "profile": "default"},
            {"member_id": "member-1", "profile": "member-1"},
        ],
        authority_gateway_id="gateway-1",
    )
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id="gateway-1",
        authority_epoch=int(room["authority_epoch"]),
        process_generation="worker-1",
        ttl_seconds=0.15,
        clock=time.time,
    )
    action = _action(
        observer_generation="worker-1",
        observer_lease_generation=lease.lease_generation,
    )
    approvals.list_all_pending_approvals(db)
    blocker = sqlite3.connect(db, timeout=10)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                approvals.persist_pending_approval,
                db,
                room_id="room-1",
                member_id="member-1",
                action=action,
            )
            time.sleep(0.25)
            blocker.commit()
            with pytest.raises(approvals.MessagingApprovalObservationStale):
                future.result(timeout=5)
    finally:
        if blocker.in_transaction:
            blocker.rollback()
        blocker.close()


def test_observer_lease_cannot_outlive_room_authority_epoch(tmp_path):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Authority race",
        members=[
            {"member_id": "default", "profile": "default"},
            {"member_id": "member-1", "profile": "member-1"},
        ],
        authority_gateway_id="gateway-1",
    )
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id="gateway-1",
        authority_epoch=int(room["authority_epoch"]),
        process_generation="worker-1",
        ttl_seconds=30,
        clock=time.time,
    )
    action = _action(
        observer_generation="worker-1",
        observer_lease_generation=lease.lease_generation,
    )
    hosted_rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-1",
        expected_epoch=int(room["authority_epoch"]),
        new_gateway_id="gateway-1",
        event_id="authority-epoch-2",
    )

    with pytest.raises(approvals.MessagingApprovalObservationStale):
        approvals.persist_pending_approval(
            db,
            room_id="room-1",
            member_id="member-1",
            action=action,
        )


def test_command_replay_is_idempotent_and_conflicting_reuse_fails(tmp_path):
    db = tmp_path / "state.db"
    pending = _pending(db)
    first = approvals.begin_approval_command(
        db,
        command_id="approval-command-1",
        pending=pending,
        choice="once",
    )
    replay = approvals.begin_approval_command(
        db,
        command_id="approval-command-1",
        pending=pending,
        choice="once",
    )

    assert first["state"] == "pending"
    assert replay["idempotent"] is True
    with pytest.raises(
        approvals.MessagingApprovalError,
        match="different content",
    ):
        approvals.begin_approval_command(
            db,
            command_id="approval-command-1",
            pending=pending,
            choice="deny",
        )


def test_cross_process_command_waits_then_resolves_once(tmp_path):
    class Service:
        def __init__(self):
            self.calls = []

        def approve_room_task(self, room_id, **kwargs):
            self.calls.append((room_id, kwargs))
            return {"resolved": 1}

    db = tmp_path / "state.db"
    pending = _pending(db)
    queued = approvals.submit_approval(
        db,
        service=None,
        command_id="approval-command-1",
        pending=pending,
        choice="deny",
    )
    service = Service()
    resolved = approvals.submit_approval(
        db,
        service=service,
        command_id="approval-command-1",
        pending=pending,
        choice="deny",
    )
    replay = approvals.submit_approval(
        db,
        service=service,
        command_id="approval-command-1",
        pending=pending,
        choice="deny",
    )

    assert queued["queued"] is True
    assert resolved["result"] == "Denied."
    assert replay["idempotent"] is True
    assert service.calls == [
        (
            "room-1",
            {
                "member_id": "member-1",
                "task_id": "task-1",
                "execution_generation": 2,
                "choice": "deny",
                "request_id": "request-1",
            },
        )
    ]


def test_post_journal_execution_error_is_reported_as_queued(tmp_path):
    class FailingService:
        def approve_room_task(self, _room_id, **_kwargs):
            raise OSError("worker transport restarted")

    db = tmp_path / "state.db"
    pending = _pending(db)
    result = approvals.submit_approval(
        db,
        service=FailingService(),
        command_id="approval-command-1",
        pending=pending,
        choice="once",
    )

    assert result["queued"] is True
    assert len(approvals.list_pending_approval_commands(db, room_id="room-1")) == 1


def test_post_resolution_receipt_error_is_not_reported_as_rejection(
    tmp_path,
    monkeypatch,
):
    class Service:
        def approve_room_task(self, _room_id, **_kwargs):
            return {"resolved": 1}

    db = tmp_path / "state.db"
    pending = _pending(db)
    monkeypatch.setattr(
        approvals,
        "complete_approval_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk busy")),
    )

    result = approvals.submit_approval(
        db,
        service=Service(),
        command_id="approval-command-1",
        pending=pending,
        choice="once",
    )

    assert result["queued"] is True
    assert result["applied"] is True


def test_one_pending_approval_cannot_queue_conflicting_decisions(tmp_path):
    db = tmp_path / "state.db"
    pending = _pending(db)
    approvals.begin_approval_command(
        db,
        command_id="approval-command-1",
        pending=pending,
        choice="once",
    )

    with pytest.raises(approvals.MessagingApprovalError, match="different decision"):
        approvals.begin_approval_command(
            db,
            command_id="approval-command-2",
            pending=pending,
            choice="deny",
        )


def test_stale_pending_approvals_and_commands_expire(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    now = [1_000_000.0]
    monkeypatch.setattr(approvals.time, "time", lambda: now[0])
    pending = _pending(db)
    approvals.begin_approval_command(
        db,
        command_id="approval-command-1",
        pending=pending,
        choice="once",
    )

    now[0] += approvals.PENDING_APPROVAL_TTL_SECONDS + 1

    assert approvals.list_pending_approvals(db, room_id="room-1") == []
    assert approvals.list_pending_approval_commands(db, room_id="room-1") == []
    assert (
        approvals.approval_command(
            db,
            command_id="approval-command-1",
        )
        is None
    )

    approvals.persist_pending_approval(
        db,
        room_id="room-2",
        member_id="member-2",
        action=_action(task_id="task-2", request_id="request-2"),
    )
    assert (
        approvals.approval_command(
            db,
            command_id="approval-command-1",
        )
        is None
    )


def test_pending_approval_journal_has_a_per_room_cap(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(approvals, "MAX_PENDING_COMMANDS_PER_ROOM", 2)
    for number in range(1, 3):
        pending = approvals.persist_pending_approval(
            db,
            room_id="room-1",
            member_id=f"member-{number}",
            action=_action(
                task_id=f"task-{number}",
                request_id=f"request-{number}",
            ),
        )
        approvals.begin_approval_command(
            db,
            command_id=f"approval-command-{number}",
            pending=pending,
            choice="once",
        )

    third = approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-3",
        action=_action(task_id="task-3", request_id="request-3"),
    )
    with pytest.raises(approvals.MessagingApprovalError, match="Too many"):
        approvals.begin_approval_command(
            db,
            command_id="approval-command-3",
            pending=third,
            choice="once",
        )


def test_completed_receipts_do_not_consume_the_pending_cap(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(approvals, "MAX_PENDING_COMMANDS_PER_ROOM", 2)
    for number in range(1, 4):
        pending = approvals.persist_pending_approval(
            db,
            room_id="room-1",
            member_id=f"member-{number}",
            action=_action(
                task_id=f"task-{number}",
                request_id=f"request-{number}",
            ),
        )
        command_id = f"approval-command-{number}"
        approvals.begin_approval_command(
            db,
            command_id=command_id,
            pending=pending,
            choice="once",
        )
        approvals.complete_approval_command(
            db,
            command_id=command_id,
            result="Approved once.",
        )

    assert (
        approvals.approval_command(
            db,
            command_id="approval-command-1",
        )["state"]
        == "completed"
    )
    assert (
        approvals.approval_command(
            db,
            command_id="approval-command-3",
        )["state"]
        == "completed"
    )


def test_multiple_approvals_require_an_explicit_number(tmp_path):
    db = tmp_path / "state.db"
    first = _pending(db)
    second = approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-2",
        action=_action(task_id="task-2", request_id="request-2"),
    )
    pending = approvals.list_pending_approvals(db, room_id="room-1")

    with pytest.raises(approvals.MessagingApprovalError, match="Choose"):
        approvals.select_pending_approval(pending)
    assert (
        approvals.select_pending_approval(
            pending,
            approvals.approval_reference(first),
        )[1]
        == first
    )
    assert (
        approvals.select_pending_approval(
            pending,
            approvals.approval_reference(second),
        )[1]
        == second
    )


def test_approval_code_does_not_retarget_after_list_reordering(tmp_path):
    db = tmp_path / "state.db"
    first = _pending(db)
    second = approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-2",
        action=_action(task_id="task-2", request_id="request-2"),
    )
    reference = approvals.approval_reference(second)
    approvals.clear_pending_approval(
        db,
        room_id="room-1",
        member_id=str(first["member_id"]),
    )
    reordered = approvals.list_pending_approvals(db, room_id="room-1")

    assert approvals.select_pending_approval(reordered, reference)[1] == second
    with pytest.raises(approvals.MessagingApprovalError, match="approval code"):
        approvals.select_pending_approval(reordered, "2")


def test_single_text_approval_also_requires_its_stable_code(tmp_path):
    db = tmp_path / "state.db"
    pending = _pending(db)

    with pytest.raises(approvals.MessagingApprovalError, match="approval code"):
        approvals.select_pending_approval([pending])
    assert (
        approvals.select_pending_approval(
            [pending],
            approvals.approval_reference(pending),
        )[1]
        == pending
    )


def test_desktop_hosted_approval_names_the_supported_surface(tmp_path):
    class Backend:
        db_path = tmp_path / "state.db"
        service = None

    with pytest.raises(
        approvals.MessagingApprovalError,
        match="Approve or deny the command there",
    ):
        approvals.submit_room_approval(
            Backend(),
            {"room_id": "classic-room", "_room_mode": "desktop"},
            command_id="approval-command-1",
            choice="once",
            installation_owner_authorized=True,
        )

    with pytest.raises(
        approvals.MessagingApprovalError,
        match="owner chat connected to the device running",
    ):
        approvals.submit_room_approval(
            Backend(),
            {"room_id": "remote-room", "_room_mode": "remote"},
            command_id="approval-command-2",
            choice="once",
            installation_owner_authorized=True,
        )


def test_secondary_profile_cannot_submit_room_approval(tmp_path):
    class Backend:
        db_path = tmp_path / "state.db"
        service = None

    with pytest.raises(
        approvals.MessagingApprovalError,
        match="installation owner",
    ):
        approvals.submit_room_approval(
            Backend(),
            {"room_id": "room-1"},
            command_id="approval-command-secondary",
            choice="once",
            installation_owner_authorized=False,
        )


def test_picker_tokens_are_short_and_bound_to_the_current_request(tmp_path):
    db = tmp_path / "state.db"
    pending = [_pending(db)]
    room = {"room_id": "room-1", "members": []}
    choices = approvals.approval_picker_choices(room, pending)

    assert all(len(choice["value"].encode()) <= 64 for choice in choices)
    index, choice, request_id = approvals.resolve_approval_picker_choice(
        room,
        pending,
        choices[0]["value"],
    )
    assert (index, choice, request_id) == (1, "once", "request-1")

    changed = [{**pending[0], "task_id": "task-2"}]
    with pytest.raises(approvals.MessagingApprovalError, match="changed"):
        approvals.resolve_approval_picker_choice(
            room,
            changed,
            choices[0]["value"],
        )


def test_approval_display_neutralizes_markup_shaped_bot_and_command_text(tmp_path):
    db = tmp_path / "state.db"
    pending = approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-1",
        action=_action(
            approval={
                "description": "**Approve** [open](https://example.test) @all",
                "command": "rm -rf /",
                "choices": ["once", "deny"],
            }
        ),
    )
    room = {
        "room_id": "room-1",
        "members": [{"member_id": "member-1", "display_name": "[Admin](url) @all"}],
    }

    rendered = approvals.format_pending_approvals(
        type(
            "Service", (), {"status": lambda *_args: {"pending_actions": [pending]}}
        )(),
        room,
        room_reference="1",
    )
    choices = approvals.approval_picker_choices(room, [pending])

    assert "[Admin](url)" not in rendered
    assert "@all" not in rendered
    assert "**Approve**" not in rendered
    assert "［Admin］(url) ＠all" in rendered
    assert "Command: rm -rf /" in rendered
    assert choices[0]["label"].startswith(
        "✓ 1. Approve once · ［Admin］(url) ＠all · member-1"
    )
    title = approvals.format_approval_picker_title(room, [pending])
    assert "＊＊Approve＊＊ ［open］(https://example.test) ＠all" in title


def test_native_picker_title_keeps_the_complete_bounded_action(tmp_path):
    db = tmp_path / "state.db"
    action = "Run harmless preview. " + ("check output; " * 15) + "DELETE ALL DATA"
    pending = approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-1",
        action=_action(
            approval={
                "description": action,
                "command": "",
                "choices": ["once", "deny"],
            }
        ),
    )

    title = approvals.format_approval_picker_title(
        {"room_id": "room-1", "members": []},
        [pending],
    )

    assert action in title
    rendered = approvals.format_pending_approvals(
        type(
            "Service", (), {"status": lambda *_args: {"pending_actions": [pending]}}
        )(),
        {"room_id": "room-1", "members": []},
        room_reference="1",
    )
    assert "DELETE ALL DATA" in rendered


def test_native_buttons_distinguish_duplicate_names_with_index_and_handle(tmp_path):
    db = tmp_path / "state.db"
    first = _pending(db)
    second = approvals.persist_pending_approval(
        db,
        room_id="room-1",
        member_id="member-2",
        action=_action(task_id="task-2", request_id="request-2"),
    )
    room = {
        "room_id": "room-1",
        "members": [
            {"member_id": "member-1", "display_name": "Reviewer", "handle": "alpha"},
            {"member_id": "member-2", "display_name": "Reviewer", "handle": "beta"},
        ],
    }

    labels = [
        choice["label"]
        for choice in approvals.approval_picker_choices(room, [first, second])
    ]

    assert labels == [
        "✓ 1. Approve once · Reviewer · ＠alpha",
        "✕ 1. Deny · Reviewer · ＠alpha",
        "✓ 2. Approve once · Reviewer · ＠beta",
        "✕ 2. Deny · Reviewer · ＠beta",
    ]

    long_room = {
        **room,
        "members": [
            {"member_id": "member-1", "display_name": "R" * 80, "handle": "alpha"},
            {"member_id": "member-2", "display_name": "R" * 80, "handle": "beta"},
        ],
    }
    long_labels = [
        choice["label"]
        for choice in approvals.approval_picker_choices(long_room, [first, second])
    ]
    assert long_labels[0].endswith("＠alpha")
    assert long_labels[2].endswith("＠beta")
    assert long_labels[0] != long_labels[2]
