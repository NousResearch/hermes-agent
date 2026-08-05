"""SQL-layer digest UDF guards for ORCH V4."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_canonical import request_digest
from hermes_cli.kanban_orch_digest_udf import build_route_json_and_digest
from hermes_cli.kanban_orch_schema_v4 import apply_schema
from hermes_cli.kanban_orch_schema_sidecar import apply_sidecar_schema


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
        # matching digest passes
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
