from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3

import pytest

from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.store import DispatchStore
from tests.plugins.test_builder_adapter_schema import (
    FakeKanban,
    make_adapter,
    request_payload,
)


def test_concurrent_identical_reservations_create_one_dispatch(tmp_path):
    store = DispatchStore(tmp_path / "journal.db")

    def reserve(_):
        return store.reserve(
            "00000000-0000-0000-0000-000000000001",
            "k" * 32,
            "a" * 64,
            "CYCLE_001",
            "principal",
        )[1]

    with ThreadPoolExecutor(max_workers=8) as pool:
        created = list(pool.map(reserve, range(20)))
    assert created.count(True) == 1
    assert created.count(False) == 19


def test_same_key_different_hash_conflicts(tmp_path):
    store = DispatchStore(tmp_path / "journal.db")
    store.reserve("00000000-0000-0000-0000-000000000001", "k" * 32, "a" * 64, "CYCLE_001", "p")
    with pytest.raises(AdapterError) as raised:
        store.reserve("00000000-0000-0000-0000-000000000002", "k" * 32, "b" * 64, "CYCLE_001", "p")
    assert raised.value.code == "IDEMPOTENCY_CONFLICT"


def test_same_key_is_bound_to_authenticated_principal(tmp_path):
    store = DispatchStore(tmp_path / "journal.db")
    store.reserve(
        "00000000-0000-0000-0000-000000000001",
        "k" * 32,
        "a" * 64,
        "CYCLE_001",
        "principal-a",
    )
    with pytest.raises(AdapterError) as raised:
        store.reserve(
            "00000000-0000-0000-0000-000000000001",
            "k" * 32,
            "a" * 64,
            "CYCLE_001",
            "principal-b",
        )
    assert raised.value.code == "IDEMPOTENCY_CONFLICT"


def test_cross_principal_dispatch_id_collision_preserves_owner_record(tmp_path):
    store = DispatchStore(tmp_path / "dispatch.db")
    dispatch_id = "00000000-0000-0000-0000-000000000001"
    store.reserve(dispatch_id, "a" * 32, "1" * 64, "CYCLE_A", "principal-a")
    before = store.get(dispatch_id)
    with pytest.raises(AdapterError) as raised:
        store.reserve(
            dispatch_id, "b" * 32, "2" * 64, "CYCLE_B", "principal-b"
        )
    assert raised.value.code == "IDEMPOTENCY_CONFLICT"
    assert store.get(dispatch_id) == before


def test_reservation_atomically_binds_packet_bytes_hash_and_audit(tmp_path):
    store = DispatchStore(tmp_path / "dispatch.db")
    dispatch_id = "00000000-0000-0000-0000-000000000001"
    packet = {"packet": {"cycle": "CYCLE_A"}, "sha256": "f" * 64}
    record, created = store.reserve(
        dispatch_id,
        "a" * 32,
        "1" * 64,
        "CYCLE_A",
        "principal-a",
        {"request": True},
        packet,
    )
    assert created
    assert record["reservation_event_id"]
    assert record["packet_sha256"] == hashlib.sha256(
        record["packet_json"].encode()
    ).hexdigest()
    with sqlite3.connect(store.path) as conn:
        row = conn.execute(
            "SELECT kind,payload_json FROM audit_events WHERE event_id=?",
            (record["reservation_event_id"],),
        ).fetchone()
    assert row[0] == "DISPATCH_RESERVED"
    assert json.loads(row[1])["packet_bytes"] == record["packet_json"]


def test_packet_identity_conflict_blocks_dispatch_and_is_audited(tmp_path):
    store = DispatchStore(tmp_path / "dispatch.db")
    dispatch_id = "00000000-0000-0000-0000-000000000001"
    record, _ = store.reserve(
        dispatch_id,
        "a" * 32,
        "1" * 64,
        "CYCLE_A",
        "principal-a",
        packet={"packet": {"cycle": "CYCLE_A"}, "sha256": "f" * 64},
    )
    with pytest.raises(AdapterError) as raised:
        store.assert_packet_identity(dispatch_id, record["packet_json"] + " ")
    assert raised.value.code == "CONTRACT_MISMATCH"
    assert store.get(dispatch_id)["phase"] == "PACKET_CONFLICT"
    with sqlite3.connect(store.path) as conn:
        assert conn.execute(
            "SELECT count(*) FROM audit_events "
            "WHERE dispatch_id=? AND kind='EXECUTION_PACKET_CONFLICT'",
            (dispatch_id,),
        ).fetchone()[0] == 1


def test_dual_column_packet_tampering_cannot_replace_reservation_identity(tmp_path):
    store = DispatchStore(tmp_path / "dispatch.db")
    dispatch_id = "00000000-0000-0000-0000-000000000001"
    store.reserve(
        dispatch_id,
        "a" * 32,
        "1" * 64,
        "CYCLE_A",
        "principal-a",
        packet={"packet": {"cycle": "CYCLE_A"}, "sha256": "f" * 64},
    )
    replacement = json.dumps(
        {"packet": {"cycle": "ATTACKER"}, "sha256": "0" * 64},
        sort_keys=True,
        separators=(",", ":"),
    )
    replacement_hash = hashlib.sha256(replacement.encode()).hexdigest()
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE dispatches SET packet_json=?,packet_sha256=? WHERE dispatch_id=?",
            (replacement, replacement_hash, dispatch_id),
        )
    with pytest.raises(AdapterError) as raised:
        store.assert_packet_identity(dispatch_id)
    assert raised.value.code == "CONTRACT_MISMATCH"
    assert store.get(dispatch_id)["phase"] == "PACKET_CONFLICT"


def test_immutable_reservation_rejects_update_and_delete(tmp_path):
    store = DispatchStore(tmp_path / "dispatch.db")
    dispatch_id = "00000000-0000-0000-0000-000000000001"
    store.reserve(
        dispatch_id,
        "a" * 32,
        "1" * 64,
        "CYCLE_A",
        "principal-a",
        packet={"packet": {"cycle": "CYCLE_A"}, "sha256": "f" * 64},
    )
    with sqlite3.connect(store.path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE dispatch_reservations SET packet_sha256=? "
                "WHERE dispatch_id=?",
                ("0" * 64, dispatch_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "DELETE FROM dispatch_reservations WHERE dispatch_id=?",
                (dispatch_id,),
            )


def test_preflight_rejection_does_not_poison_idempotency_key(tmp_path):
    kanban = FakeKanban()
    adapter = make_adapter(tmp_path, kanban=kanban)
    payload = request_payload(tmp_path)
    rejected = dict(payload, unexpected=True)
    assert adapter.dispatch("principal", rejected)["status"] == "REJECTED"
    assert adapter.dispatch("principal", payload)["status"] == "ACCEPTED"
    assert kanban.created == 1


def test_reserved_boundary_reconciles_through_native_idempotency(tmp_path):
    kanban = FakeKanban()
    adapter = make_adapter(tmp_path, kanban=kanban)
    payload = request_payload(tmp_path)
    from plugins.builder_adapter.canonical import canonical_sha256

    adapter.store.reserve(
        payload["dispatch_id"],
        payload["idempotency_key"],
        canonical_sha256(payload),
        payload["cycle_id"],
        "principal",
        payload,
    )
    result = adapter.dispatch("principal", payload)
    assert result["status"] == "ACCEPTED"
    assert adapter.store.get(payload["dispatch_id"])["phase"] == "TASK_CREATED"


def test_audit_chain_is_tamper_evident_and_recursively_redacted(tmp_path):
    store = DispatchStore(tmp_path / "journal.db")
    first = store.audit(
        "FIRST",
        None,
        {"nested": [{"api_key": "never-write-me"}], "safe": "value"},
    )
    second = store.audit("SECOND", None, {"token_value": "also-secret"})
    with sqlite3.connect(store.path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM audit_events ORDER BY sequence").fetchall()
    assert [row["event_id"] for row in rows] == [first, second]
    previous = "0" * 64
    for row in rows:
        payload = json.loads(row["payload_json"])
        assert "never-write-me" not in row["payload_json"]
        assert "also-secret" not in row["payload_json"]
        material = json.dumps(
            {
                "event_id": row["event_id"],
                "dispatch_id": row["dispatch_id"],
                "kind": row["kind"],
                "payload": payload,
                "created_at": row["created_at"],
                "previous_hash": previous,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert row["previous_hash"] == previous
        assert row["event_hash"] == hashlib.sha256(material).hexdigest()
        previous = row["event_hash"]


def test_failed_audit_insert_rolls_back_state_transition(tmp_path):
    store = DispatchStore(tmp_path / "journal.db")
    dispatch_id = "00000000-0000-0000-0000-000000000001"
    store.reserve(dispatch_id, "k" * 32, "a" * 64, "CYCLE", "principal")
    duplicate = store.audit("EXISTING", dispatch_id, {})
    with pytest.raises(sqlite3.IntegrityError):
        store.transition_with_audit(
            dispatch_id,
            phase="TASK_CREATED",
            task_id="task",
            result={"status": "ACCEPTED"},
            event_id=duplicate,
            kind="TASK_CREATED",
            payload={},
        )
    assert store.get(dispatch_id)["phase"] == "RESERVED"


def test_task_creation_transition_crash_recovers_without_replacement_task(
    tmp_path, monkeypatch
):
    kanban = FakeKanban()
    adapter = make_adapter(tmp_path, kanban=kanban)
    payload = request_payload(tmp_path)
    original = adapter.store.transition_with_audit
    calls = {"count": 0}

    def crash_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected transition crash")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter.store, "transition_with_audit", crash_once)
    unknown = adapter.dispatch("principal", payload)
    assert unknown["status"] == "UNKNOWN"
    assert unknown["kanban_task_id"] == "t_12345678"
    assert adapter.store.get(payload["dispatch_id"])["phase"] == "RESERVED"

    recovered = adapter.dispatch("principal", payload)
    assert recovered["status"] == "ACCEPTED"
    assert recovered["kanban_task_id"] == "t_12345678"
    assert adapter.store.get(payload["dispatch_id"])["phase"] == "TASK_CREATED"


def test_completion_transition_crash_returns_unknown_then_reconciles(
    tmp_path, monkeypatch
):
    from plugins.builder_adapter.completion import CompletionAttestor

    kanban = FakeKanban()
    adapter = make_adapter(tmp_path, kanban=kanban)
    payload = request_payload(tmp_path)
    assert adapter.dispatch("principal", payload)["status"] == "ACCEPTED"
    kanban.status = "done"
    monkeypatch.setattr(
        CompletionAttestor,
        "complete",
        lambda *args, **kwargs: {
            "git": {"resulting_sha": "9" * 40},
            "audit_event_refs": [],
        },
    )
    original = adapter.store.transition_with_audit
    calls = {"count": 0}

    def crash_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected completion journal crash")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter.store, "transition_with_audit", crash_once)
    unknown = adapter.dispatch("principal", payload)
    assert unknown["status"] == "UNKNOWN"
    assert unknown["terminal"] is False
    recovered = adapter.dispatch("principal", payload)
    assert recovered["status"] == "SUCCEEDED"
    assert recovered["terminal"] is True
