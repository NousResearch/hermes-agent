"""Manual preflight binds saved evidence without acting or creating questions."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_manual_recovery as recovery
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from tests.gateway.test_api_server_room_replicas import setup  # noqa: F401
from tests.gateway.test_hosted_room_replica_ingress import (
    HOME, TARGET, SECRET, grant, ingest, pair as pair,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix_only", [False, True])
async def test_http_v2_evidence_keeps_unknown_tail_and_pending_lineage(setup, prefix_only):
    from aiohttp.test_utils import TestClient, TestServer
    from tests.gateway.test_work_record_v2_admission_closure import enrolled_work
    from tests.gateway.test_api_server_room_work_records import deliver
    from tests.gateway.test_api_server_room_replicas import TARGET as HTTP_TARGET
    source, target, app = setup
    async with TestClient(TestServer(app)) as http:
        client, token, record = await enrolled_work(http, source, prefix_only=prefix_only)
        if not prefix_only:
            record = {**record, "tasks": [], "receipts": []}
            record["digest"] = records.digest({k: v for k, v in record.items() if k not in {"revision", "digest"}})
            await deliver(client, token, record)
        result = recovery.prepare_recovery(target, room_id="room", target_gateway_id=HTTP_TARGET)
        assert result["reconciliation_required"] is True
        assert result["accepted_tail"] == "unverified"
        assert result["execution_authorized"] is False
        if prefix_only:
            assert "lineage_unverified" in result["blockers"]
            assert "work_records_unavailable" in result["blockers"]
        else:
            assert result["blockers"] == []
            assert result["work_records"]["tasks"] == []
            assert result["work_records"]["incompleteness"] == ["prior_authority_work_unknown"]


def preview(pair, target_id=TARGET):
    return recovery.prepare_recovery(pair[1], room_id="room", target_gateway_id=target_id)


def copy_records(pair, token):
    record = records.capture(pair[0], room_id="room", local_gateway_id=HOME)
    records.ingest(pair[1], record=record, token=token, secret=SECRET,
                   target_install_id=TARGET, target_profile="reviewer")
    return record


def test_archive_bytes_not_public_summary_bind_selection(pair):
    import sqlite3
    from gateway.hosted_room_work_storage import INVALID_TABLE
    token, _ = grant(pair[1], permissions=("replicate", "work_records"))
    ingest(pair, token)
    copy_records(pair, token)
    first = preview(pair)
    with sqlite3.connect(pair[1]) as conn:
        conn.execute("DROP TRIGGER trg_work_invalid_insert")
        conn.execute(f"INSERT INTO {INVALID_TABLE}(source_table,room_id,revision,digest,record_json,disposition) VALUES(?, 'room',1,'opaque','PRIVATE_INVALID','invalid')", (records.TARGET_TABLE,))
    second = preview(pair)
    with sqlite3.connect(pair[1]) as conn:
        conn.execute("DROP TRIGGER trg_work_invalid_update")
        conn.execute(f"UPDATE {INVALID_TABLE} SET record_json='OTHER_PRIVATE_INVALID'")
    third = preview(pair)
    assert first["snapshot_id"] != second["snapshot_id"] != third["snapshot_id"]
    assert "PRIVATE_INVALID" not in json.dumps(third)
    assert third["reconciliation_required"] is True


def test_missing_work_is_unknown_not_an_empty_execution_clearance(pair):
    token, _ = grant(pair[1])
    ingest(pair, token)
    before = replicas.replica_state(pair[1], room_id="room")
    result = preview(pair)
    assert result["source_authority"] == before["authority"]
    assert result["execution_authorized"] is False
    assert result["accepted_tail"] == "unverified"
    assert result["reconciliation_required"] is True
    assert result["blockers"] == ["work_records_unavailable"]
    assert result["candidate_member_ids"] == ["reviewer"]
    assert result["previous_host_member_ids"] == ["writer"]
    assert replicas.replica_state(pair[1], room_id="room") == before
    with rooms._transaction(pair[1]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms WHERE room_id='room'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_events WHERE room_id='room'").fetchone()[0] == 0
        assert conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"


def test_five_minute_copy_bookkeeping_does_not_change_preview_identity(pair):
    token, _ = grant(pair[1])
    ingest(pair, token)
    first = preview(pair)
    # A five-minute update/reconnect is not a new owner decision.
    with rooms._transaction(pair[1], immediate=True) as conn:
        conn.execute("UPDATE hosted_room_replicas SET updated_at=updated_at+300 WHERE room_id='room'")
    later = preview(pair)
    assert later["copy_updated_at"] > first["copy_updated_at"]
    assert later["snapshot_id"] == first["snapshot_id"]
    assert later["execution_authorized"] is False
    ingest(pair, token)
    assert preview(pair)["snapshot_id"] == first["snapshot_id"]


def test_new_history_changes_the_confirmation_binding_without_exposing_content(pair):
    token, _ = grant(pair[1])
    ingest(pair, token)
    first = preview(pair)
    rooms.append_event(pair[0], room_id="room", event_id="new-input", kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": "PRIVATE_RECOVERY_CONTENT"},
        authority_gateway_id=HOME, authority_epoch=1)
    ingest(pair, token)
    newer = preview(pair)
    assert newer["snapshot_id"] != first["snapshot_id"]
    assert newer["saved_through_seq"] == first["saved_through_seq"] + 1
    assert "PRIVATE_RECOVERY_CONTENT" not in json.dumps(newer)


def test_task_only_changes_invalidate_the_preview_and_keep_exact_evidence(pair):
    token, _ = grant(pair[1], permissions=("replicate", "work_records"))
    ingest(pair, token)
    task = driver.TaskIdentity("room", "task", "thread", "turn")
    source_seq = rooms.read_events(pair[0], room_id="room")["latest_seq"]
    driver.admit_task(pair[0], task, payload={"prompt": "PRIVATE_PROMPT", "target_profile": "default",
        "target_member_id": "writer", "source_event_seq": source_seq}, clock=lambda: 100)
    copied = copy_records(pair, token)
    first = preview(pair)
    assert first["blockers"] == []
    assert first["work_records"]["digest"] == copied["digest"]
    assert first["reconciliation_required"] is True
    lease = driver.acquire_lease(pair[0], room_id="room", gateway_id=HOME, authority_epoch=1,
        process_generation="process", ttl_seconds=30, clock=lambda: 100)
    driver.start_task(pair[0], task, lease, expected_cancel_generation=0, clock=lambda: 100)
    copy_records(pair, token)
    second = preview(pair)
    assert second["saved_through_seq"] == first["saved_through_seq"]
    assert second["snapshot_id"] != first["snapshot_id"]
    assert second["work_records"]["phases"] == {"running": 1}
    assert "PRIVATE_PROMPT" not in json.dumps(second)
    assert second["execution_authorized"] is False


@pytest.mark.parametrize("change,expected", [
    ("partial", "copy_incomplete"), ("quarantine", "copy_quarantined"),
    ("disband", "group_disbanded"), ("other_target", "target_not_a_participant"),
])
def test_unsafe_or_unavailable_candidates_do_not_gain_authority(pair, change, expected):
    token, _ = grant(pair[1])
    if change == "partial":
        rooms.append_event(pair[0], room_id="room", event_id="second", kind="message.user",
            actor={"kind": "user", "id": "owner"}, payload={"text": "second"},
            authority_gateway_id=HOME, authority_epoch=1)
        ingest(pair, token, page=rooms.read_events(pair[0], room_id="room", limit=1))
    elif change == "disband":
        rooms.disband_room(pair[0], room_id="room", expected_gateway_id=HOME, expected_epoch=1)
        ingest(pair, token, page=rooms.read_events(pair[0], room_id="room", include_disbanded=True))
    else:
        ingest(pair, token)
    if change == "quarantine":
        with rooms._transaction(pair[1], immediate=True) as conn:
            conn.execute("""UPDATE hosted_room_replica_events
                SET seq=(SELECT MAX(seq)+1 FROM hosted_room_replica_events WHERE room_id='room')
                WHERE room_id='room' AND seq=(SELECT MIN(seq) FROM hosted_room_replica_events WHERE room_id='room')""")
    result = preview(pair, "install:other" if change == "other_target" else TARGET)
    assert expected in result["blockers"]
    assert result["execution_authorized"] is False
    with rooms._transaction(pair[1]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms WHERE room_id='room'").fetchone()[0] == 0


def test_rpc_uses_its_own_target_identity_and_does_not_activate_a_worker(pair, monkeypatch):
    import tui_gateway.server as server
    from tui_gateway import methods_groups

    token, _ = grant(pair[1])
    ingest(pair, token)
    monkeypatch.setattr(rooms, "default_db_path", lambda: pair[1])
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: TARGET)
    monkeypatch.setattr(methods_groups, "_service", None)
    response = server._methods["groups.recovery.prepare"](1, {"room_id": "room", "target_gateway_id": HOME})
    assert "error" not in response, response
    assert response["result"]["target_gateway_id"] == TARGET
    assert response["result"]["execution_authorized"] is False
    assert methods_groups._service is None
    assert "error" in server._methods["groups.promote"](2, {"room_id": "room", "confirm": True})


def test_preview_uses_one_consistent_view_during_copy_progress(pair, monkeypatch):
    token, _ = grant(pair[1])
    ingest(pair, token)
    baseline = preview(pair)
    entered, release, writer_started = threading.Event(), threading.Event(), threading.Event()
    original = recovery._history_digest

    def paused(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(recovery, "_history_digest", paused)
    rooms.append_event(pair[0], room_id="room", event_id="later", kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": "later"},
        authority_gateway_id=HOME, authority_epoch=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(preview, pair)
        assert entered.wait(3)

        def update():
            writer_started.set()
            return ingest(pair, token)

        writing = pool.submit(update)
        assert writer_started.wait(3)
        release.set()
        assert reading.result(5)["snapshot_id"] == baseline["snapshot_id"]
        writing.result(5)
    assert preview(pair)["snapshot_id"] != baseline["snapshot_id"]
