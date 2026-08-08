"""SQL-layer digest UDF guards for ORCH V4 (route/request/result/event/payload)."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_bridge import OrchBridge, init_sidecar_db
from hermes_cli.kanban_orch_canonical import digest, event_key, request_digest, result_digest
from hermes_cli.kanban_orch_digest_udf import (
    build_payload_json_and_digest,
    build_result_json_and_digest,
    build_route_json_and_digest,
)
from hermes_cli.kanban_orch_schema_sidecar import apply_sidecar_schema
from hermes_cli.kanban_orch_schema_v4 import apply_schema


def _conn_inplace():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    apply_schema(conn, test_open_capability=True)
    return conn, path


def test_sql_route_digest_mismatch_rejected():
    conn, path = _conn_inplace()
    try:
        b = "board_0123456789abcdef"
        sk = "a" * 64
        route_json, route_d = build_route_json_and_digest(
            origin_kind="board_only",
            platform="local",
            adapter_instance_id="ad",
            account_id="ac",
            conversation_id="cv",
            required_ack_family="none",
            required_ack_strength="none",
            route_revision=1,
        )
        conn.execute(
            "INSERT INTO kanban_board_identity(singleton,board_instance_id,canonical_board_key,created_at)"
            " VALUES (1, ?, 'k', 1)",
            (b,),
        )
        conn.execute(
            "INSERT INTO orch_replay_selectors"
            " (selector_key,board_instance_id,tenant_scope,selector_kind,selector_value,"
            "  adapter_instance_id,conversation_id,lineage_id,current_generation,"
            "  ledger_revision,created_at,updated_at)"
            " VALUES (?,?, '', 'event','v','ad','cv','lin',0,0,1,1)",
            (sk, b),
        )
        with pytest.raises(sqlite3.IntegrityError, match="route_digest_mismatch"):
            conn.execute(
                "INSERT INTO orch_origins"
                " (board_instance_id,tenant_scope,origin_id,schema_version,selector_key,"
                "  origin_kind,platform,adapter_instance_id,account_id,conversation_id,"
                "  selector_kind,selector_value,thread_id,reply_to_id,session_id,"
                "  notifier_profile,route_revision,route_json,route_digest,"
                "  required_ack_family,required_ack_strength,created_at)"
                " VALUES (?,?, 'o1',4,?,'board_only','local','ad','ac','cv',"
                "  'event','v','','','','',1,?,?,'none','none',1)",
                (b, "", sk, route_json, "f" * 64),
            )
        conn.execute(
            "INSERT INTO orch_origins"
            " (board_instance_id,tenant_scope,origin_id,schema_version,selector_key,"
            "  origin_kind,platform,adapter_instance_id,account_id,conversation_id,"
            "  selector_kind,selector_value,thread_id,reply_to_id,session_id,"
            "  notifier_profile,route_revision,route_json,route_digest,"
            "  required_ack_family,required_ack_strength,created_at)"
            " VALUES (?,?, 'o1',4,?,'board_only','local','ad','ac','cv',"
            "  'event','v','','','','',1,?,?,'none','none',1)",
            (b, "", sk, route_json, route_d),
        )
        conn.commit()
    finally:
        conn.close()
        os.unlink(path)


def test_sql_request_digest_mismatch_rejected_on_sidecar():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    try:
        apply_sidecar_schema(conn, test_open_capability=True)
        b = "board_0123456789abcdef"
        sk = "a" * 64
        route_json, route_d = build_route_json_and_digest(
            origin_kind="board_only",
            platform="local",
            adapter_instance_id="ad",
            account_id="ac",
            conversation_id="cv",
            required_ack_family="none",
            required_ack_strength="none",
            route_revision=1,
        )
        req = {
            "schema_version": 4,
            "kind": "orch_request",
            "selector_key": sk,
            "request_key": sk,
            "origin_id": "oid1",
            "lineage_id": "lin1",
            "generation": 1,
            "title": "t",
            "synthesis_strategy": "parent_owned",
            "completion_policy": "board_only",
            "requirements": [],
        }
        req_json = json.dumps(req, sort_keys=True, separators=(",", ":"))
        req_d = request_digest(req)
        conn.execute(
            "INSERT INTO kanban_board_identity(singleton,board_instance_id,canonical_board_key,created_at)"
            " VALUES (1, ?, 'k', 1)",
            (b,),
        )
        conn.execute(
            "INSERT INTO orch_replay_selectors"
            " (selector_key,board_instance_id,tenant_scope,selector_kind,selector_value,"
            "  adapter_instance_id,conversation_id,lineage_id,current_generation,"
            "  ledger_revision,created_at,updated_at)"
            " VALUES (?,?, '', 'event','v','ad','cv','lin1',0,0,1,1)",
            (sk, b),
        )
        conn.execute(
            "INSERT INTO orch_origins"
            " (board_instance_id,tenant_scope,origin_id,schema_version,selector_key,"
            "  origin_kind,platform,adapter_instance_id,account_id,conversation_id,"
            "  selector_kind,selector_value,thread_id,reply_to_id,session_id,"
            "  notifier_profile,route_revision,route_json,route_digest,"
            "  required_ack_family,required_ack_strength,created_at)"
            " VALUES (?,?, 'oid1',4,?,'board_only','local','ad','ac','cv',"
            "  'event','v','','','','',1,?,?,'none','none',1)",
            (b, "", sk, route_json, route_d),
        )
        conn.execute("INSERT INTO _soft_fk_tasks(id,title,status,created_at) VALUES ('p1','t','pending',1)")
        with pytest.raises(sqlite3.IntegrityError, match="request_digest_mismatch"):
            conn.execute(
                "INSERT INTO orch_requests"
                " (board_instance_id,tenant_scope,orch_id,lineage_id,generation,"
                "  selector_key,selector_ledger_revision,request_key,request_schema_version,"
                "  request_json,request_digest,origin_id,parent_task_id,"
                "  lifecycle_state,lifecycle_revision,cancel_epoch,"
                "  delivery_epoch_revision,plan_epoch_revision,plan_version,"
                "  synthesis_strategy,max_retries,created_at,updated_at)"
                " VALUES (?,?, 'orch1','lin1',1,?,0,?,4,?,?,'oid1','p1',"
                "  'submitted',0,0,0,0,0,'parent_owned',0,1,1)",
                (b, "", sk, sk, req_json, "e" * 64),
            )
        conn.execute(
            "INSERT INTO orch_requests"
            " (board_instance_id,tenant_scope,orch_id,lineage_id,generation,"
            "  selector_key,selector_ledger_revision,request_key,request_schema_version,"
            "  request_json,request_digest,origin_id,parent_task_id,"
            "  lifecycle_state,lifecycle_revision,cancel_epoch,"
            "  delivery_epoch_revision,plan_epoch_revision,plan_version,"
            "  synthesis_strategy,max_retries,created_at,updated_at)"
            " VALUES (?,?, 'orch1','lin1',1,?,0,?,4,?,?,'oid1','p1',"
            "  'submitted',0,0,0,0,0,'parent_owned',0,1,1)",
            (b, "", sk, sk, req_json, req_d),
        )
        conn.commit()
    finally:
        conn.close()
        os.unlink(path)


def test_result_event_payload_receipt_guards(tmp_path):
    native = tmp_path / "n.db"
    side = tmp_path / "s.db"
    nc = sqlite3.connect(native)
    nc.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, "
        "created_at INTEGER NOT NULL, tenant TEXT)"
    )
    nc.execute("INSERT INTO tasks VALUES ('task-1','t','pending',1,'')")
    nc.commit()
    nc.close()
    init_sidecar_db(str(side))
    br = OrchBridge(str(native), str(side), board_instance_id="board_0123456789abcdef", tenant_scope="")
    bound = br.bind_parent_task("board_0123456789abcdef", "", "orch-evt-1", "task-1")
    conn = br._sidecar

    # commit clock + capability required by existing event insert authority guard
    from hermes_cli.kanban_orch_db import grant
    from hermes_cli.kanban_orch_capability import CapabilityGrant

    grant(
        conn,
        CapabilityGrant(
            kind="commit_clock",
            board=bound.board_instance_id,
            target_key="*",
        ),
    )
    conn.execute(
        "INSERT INTO kanban_commit_clock(singleton, commit_seq, last_txn_id, updated_at)"
        " VALUES (1, 0, 'txn-0', 1)"
    )
    conn.execute(
        "UPDATE kanban_commit_clock SET commit_seq=1, last_txn_id='txn-1', updated_at=2 WHERE singleton=1"
    )
    grant(
        conn,
        CapabilityGrant(
            kind="event_insert",
            board=bound.board_instance_id,
            tenant=bound.tenant_scope,
            object_id=bound.orch_id,
            revision=0,
            epoch=0,
            target_key="*",
        ),
    )

    # --- events payload + event_key ---
    payload = {"hello": "world", "n": 1}
    payload_json, payload_d = build_payload_json_and_digest(payload)
    ek = event_key(
        {
            "board_instance_id": bound.board_instance_id,
            "tenant_scope": bound.tenant_scope,
            "orch_id": bound.orch_id,
            "lifecycle_revision": 0,
            "cancel_epoch": 0,
            "event_kind": "request_transition",
            "target_key": bound.orch_id,
            "payload_digest": payload_d,
        }
    )
    with pytest.raises(sqlite3.IntegrityError, match="payload_digest_mismatch"):
        conn.execute(
            "INSERT INTO orch_events"
            " (board_instance_id,tenant_scope,orch_id,lifecycle_revision,cancel_epoch,"
            "  commit_seq,event_kind,target_key,event_key,payload_json,payload_digest,created_at)"
            " VALUES (?,?,?,0,0,1,'request_transition',?,?,?,?,1)",
            (
                bound.board_instance_id,
                bound.tenant_scope,
                bound.orch_id,
                bound.orch_id,
                ek,
                payload_json,
                "a" * 64,
            ),
        )
    with pytest.raises(sqlite3.IntegrityError, match="event_key_mismatch"):
        conn.execute(
            "INSERT INTO orch_events"
            " (board_instance_id,tenant_scope,orch_id,lifecycle_revision,cancel_epoch,"
            "  commit_seq,event_kind,target_key,event_key,payload_json,payload_digest,created_at)"
            " VALUES (?,?,?,0,0,1,'request_transition',?,?,?,?,1)",
            (
                bound.board_instance_id,
                bound.tenant_scope,
                bound.orch_id,
                bound.orch_id,
                "b" * 64,
                payload_json,
                payload_d,
            ),
        )
    conn.execute(
        "INSERT INTO orch_events"
        " (board_instance_id,tenant_scope,orch_id,lifecycle_revision,cancel_epoch,"
        "  commit_seq,event_kind,target_key,event_key,payload_json,payload_digest,created_at)"
        " VALUES (?,?,?,0,0,1,'request_transition',?,?,?,?,1)",
        (
            bound.board_instance_id,
            bound.tenant_scope,
            bound.orch_id,
            bound.orch_id,
            ek,
            payload_json,
            payload_d,
        ),
    )

    # --- receipt_json digest (needs obligation/attempt chain is heavy; test UDF path via empty-ish insert fails earlier)
    # Direct UDF check for receipt/result formulas:
    res_json, res_d = build_result_json_and_digest(
        request_digest_hex=bound.request_digest,
        plan_digest_hex="c" * 64,
        accepted_lane_set=["lane-1", "lane-2"],
        synthesis={"text": "ok"},
    )
    assert result_digest(json.loads(res_json)) == res_d
    assert conn.execute("SELECT orch_result_digest_eq(?, ?)", (res_json, res_d)).fetchone()[0] == 1
    assert conn.execute("SELECT orch_result_digest_eq(?, ?)", (res_json, "d" * 64)).fetchone()[0] == 0

    receipt_obj = {"provider_message_id": "m1", "ok": True}
    receipt_json = json.dumps(receipt_obj, sort_keys=True, separators=(",", ":"))
    receipt_d = digest(receipt_obj)
    assert conn.execute("SELECT orch_canonical_digest_eq(?, ?)", (receipt_json, receipt_d)).fetchone()[0] == 1
    assert conn.execute("SELECT orch_canonical_digest_eq(?, ?)", (receipt_json, "e" * 64)).fetchone()[0] == 0

    # attempt evidence digest guard via direct insert into attempts requires obligation FK.
    # Prove UDF + trigger presence:
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'orch_v4_%digest%'"
        )
    }
    for need in [
        "orch_v4_results_result_digest_guard",
        "orch_v4_events_payload_digest_guard",
        "orch_v4_receipts_receipt_digest_guard",
        "orch_v4_attempt_events_event_digest_guard",
        "orch_v4_attempts_evidence_digest_guard",
    ]:
        assert need in names

    conn.commit()
    br.close()


def test_result_digest_formula_stable():
    obj = {
        "schema_version": 4,
        "kind": "orch_result",
        "request_digest": "a" * 64,
        "plan_digest": "b" * 64,
        "accepted_lane_set": ["x", "y"],
        "synthesis": {"v": 1},
    }
    d1 = result_digest(obj)
    d2 = result_digest(obj)
    assert d1 == d2
    assert len(d1) == 64
