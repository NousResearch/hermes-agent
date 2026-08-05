"""ORCH V4 deeper multi-lane: plan materialization + orch_nodes + lane accept.

Board_only sidecar path (N1–N4):
- Native RO observation of parent + task_links children
- All control-plane writes to sidecar orch_v4.db only
- Requires ≥2 linked children to materialize a real plan graph
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from hermes_cli.kanban_orch_api import OrchAPIError, apply_lifecycle_transition_db
from hermes_cli.kanban_orch_bridge import BridgeError, OrchBridge
from hermes_cli.kanban_orch_canonical import digest
from hermes_cli.kanban_orch_cmin import children_progress, read_native_children
from hermes_cli.kanban_orch_db import begin_immediate, grant
from hermes_cli.kanban_orch_capability import CapabilityGrant
from hermes_cli.kanban_orch_lifecycle import LifecycleError
from hermes_cli.kanban_orch_multilane_schema import apply_multilane_soft_fk_patch
from hermes_cli.kanban_orch_plan import validate_plan
from hermes_cli.kanban_orch_writer_switch import open_live_bridge

DONE_NATIVE = frozenset({"done", "completed"})


class MultiLaneError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _now() -> int:
    return int(time.time())


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass
class MultiLaneStep:
    action: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiLaneResult:
    orch_id: str
    parent_task_id: str
    before_state: str
    after_state: str
    plan_version: int
    steps: list[MultiLaneStep]
    skipped: bool = False
    reason: str | None = None
    accepted_required_lanes: int = 0
    children_total: int = 0
    children_done: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_request(conn: sqlite3.Connection, *, board: str, tenant: str, orch_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT r.*, o.origin_kind FROM orch_requests r "
        "JOIN orch_origins o ON o.board_instance_id=r.board_instance_id "
        " AND o.tenant_scope=r.tenant_scope AND o.origin_id=r.origin_id "
        "WHERE r.board_instance_id=? AND r.tenant_scope=? AND r.orch_id=?",
        (board, tenant, orch_id),
    ).fetchone()
    if row is None:
        raise MultiLaneError("request_not_found")
    return row


def _ensure_commit_clock(conn: sqlite3.Connection, *, board: str) -> int:
    """Ensure commit clock exists and return a commit_seq > 0 for materialization."""
    row = conn.execute("SELECT commit_seq FROM kanban_commit_clock WHERE singleton=1").fetchone()
    if row is None:
        grant(conn, CapabilityGrant(kind="commit_clock", board=board, revision=0, target_key="bootstrap"))
        conn.execute(
            "INSERT INTO kanban_commit_clock(singleton, commit_seq, last_txn_id, updated_at) VALUES (1,0,?,?)",
            ("bootstrap", _now()),
        )
        seq = 0
    else:
        seq = int(row[0])
    if seq < 1:
        grant(conn, CapabilityGrant(kind="commit_clock", board=board, revision=1, target_key="materialize"))
        conn.execute(
            "UPDATE kanban_commit_clock SET commit_seq=1, last_txn_id=?, updated_at=? WHERE singleton=1",
            ("materialize", _now()),
        )
        return 1
    return seq


def _ensure_decomposition_lease(
    conn: sqlite3.Connection,
    *,
    board: str,
    tenant: str,
    orch_id: str,
    owner_run_id: int,
    lifecycle_revision: int,
    cancel_epoch: int,
) -> tuple[int, int]:
    """Return (owner_run_id, lease_epoch)."""
    existing = conn.execute(
        "SELECT owner_run_id, epoch, lease_state, expires_at FROM orch_stage_leases "
        "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND stage='decomposition'",
        (board, tenant, orch_id),
    ).fetchone()
    if existing and existing["lease_state"] == "active" and int(existing["expires_at"]) > _now():
        return int(existing["owner_run_id"]), int(existing["epoch"])
    grant(
        conn,
        CapabilityGrant(
            kind="lease_claim",
            board=board,
            tenant=tenant,
            object_id=orch_id,
            revision=lifecycle_revision,
            epoch=cancel_epoch,
            target_key="decomposition",
        ),
    )
    token = _sha("lease", board, tenant, orch_id, "decomposition", str(owner_run_id))
    exp = _now() + 3600
    if existing is None:
        conn.execute(
            "INSERT INTO orch_stage_leases("
            "board_instance_id,tenant_scope,orch_id,stage,owner_run_id,owner_profile,"
            "token_hash,epoch,expires_at,lease_state,updated_at"
            ") VALUES (?,?,?,'decomposition',?,?,?,?,?,'active',?)",
            (board, tenant, orch_id, owner_run_id, "cmin-multilane", token, 1, exp, _now()),
        )
        return owner_run_id, 1
    # refresh via delete+insert not allowed; update may be restricted — try update path
    # stage lease update triggers may exist; if blocked, raise
    conn.execute(
        "UPDATE orch_stage_leases SET owner_run_id=?, token_hash=?, expires_at=?, "
        "lease_state='active', updated_at=? "
        "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND stage='decomposition'",
        (owner_run_id, token, exp, _now(), board, tenant, orch_id),
    )
    return owner_run_id, int(existing["epoch"])


def _ensure_reconciliation_lease(
    conn: sqlite3.Connection,
    *,
    board: str,
    tenant: str,
    orch_id: str,
    owner_run_id: int,
    lifecycle_revision: int,
    cancel_epoch: int,
) -> tuple[int, int]:
    existing = conn.execute(
        "SELECT owner_run_id, epoch, lease_state, expires_at FROM orch_stage_leases "
        "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND stage='reconciliation'",
        (board, tenant, orch_id),
    ).fetchone()
    if existing and existing["lease_state"] == "active" and int(existing["expires_at"]) > _now():
        return int(existing["owner_run_id"]), int(existing["epoch"])
    grant(
        conn,
        CapabilityGrant(
            kind="lease_claim",
            board=board,
            tenant=tenant,
            object_id=orch_id,
            revision=lifecycle_revision,
            epoch=cancel_epoch,
            target_key="reconciliation",
        ),
    )
    token = _sha("lease", board, tenant, orch_id, "reconciliation", str(owner_run_id))
    exp = _now() + 3600
    if existing is None:
        conn.execute(
            "INSERT INTO orch_stage_leases("
            "board_instance_id,tenant_scope,orch_id,stage,owner_run_id,owner_profile,"
            "token_hash,epoch,expires_at,lease_state,updated_at"
            ") VALUES (?,?,?,'reconciliation',?,?,?,?,?,'active',?)",
            (board, tenant, orch_id, owner_run_id, "cmin-multilane", token, 1, exp, _now()),
        )
        return owner_run_id, 1
    conn.execute(
        "UPDATE orch_stage_leases SET owner_run_id=?, token_hash=?, expires_at=?, "
        "lease_state='active', updated_at=? "
        "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND stage='reconciliation'",
        (owner_run_id, token, exp, _now(), board, tenant, orch_id),
    )
    return owner_run_id, int(existing["epoch"])


def _mirror_task(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    title: str,
    status: str,
    board: str | None = None,
    tenant: str | None = None,
    orch_id: str | None = None,
    plan_version: int | None = None,
    node_key: str | None = None,
    binding_revision: int | None = None,
    cancel_epoch: int | None = None,
) -> None:
    exists = conn.execute("SELECT id FROM _soft_fk_tasks WHERE id=?", (task_id,)).fetchone()
    if exists is None:
        if board is not None:
            grant(
                conn,
                CapabilityGrant(
                    kind="task_bind",
                    board=board,
                    tenant=tenant or "",
                    object_id=orch_id or "",
                    revision=binding_revision or 0,
                    epoch=cancel_epoch or 0,
                    target_key=task_id,
                ),
            )
        conn.execute(
            "INSERT INTO _soft_fk_tasks("
            "id,title,status,created_at,orch_board_instance_id,orch_tenant_scope,orch_id,"
            "orch_plan_version,orch_node_key,orch_binding_revision,orch_cancel_epoch"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                task_id,
                title,
                status,
                _now(),
                board,
                tenant,
                orch_id,
                plan_version,
                node_key,
                binding_revision,
                cancel_epoch,
            ),
        )
        return
    if board is None:
        return
    grant(
        conn,
        CapabilityGrant(
            kind="task_bind",
            board=board,
            tenant=tenant or "",
            object_id=orch_id or "",
            revision=binding_revision or 0,
            epoch=cancel_epoch or 0,
            target_key=task_id,
        ),
    )
    conn.execute(
        "UPDATE _soft_fk_tasks SET title=?, status=?, "
        "orch_board_instance_id=?, orch_tenant_scope=?, orch_id=?, "
        "orch_plan_version=?, orch_node_key=?, orch_binding_revision=?, orch_cancel_epoch=? "
        "WHERE id=?",
        (
            title,
            status,
            board,
            tenant,
            orch_id,
            plan_version,
            node_key,
            binding_revision,
            cancel_epoch,
            task_id,
        ),
    )


def materialize_plan_from_children(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    parent_task_id: str,
    parent_title: str,
    parent_status: str,
    children: list[dict[str, str]],
) -> dict[str, Any]:
    """Build requirements/plan/nodes/edges/coverage + materialize → waiting_lanes."""
    apply_multilane_soft_fk_patch(conn)
    tenant = "" if tenant_scope is None else str(tenant_scope)
    if len(children) < 2:
        raise MultiLaneError("need_at_least_two_children")

    begin_immediate(conn)
    try:
        req = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
        if req["origin_kind"] != "board_only":
            raise MultiLaneError("not_board_only")
        if req["lifecycle_state"] != "decomposing":
            raise MultiLaneError(f"bad_state:{req['lifecycle_state']}")
        if int(req["plan_version"]) != 0:
            raise MultiLaneError("plan_already_set")

        plan_version = 1
        life_rev = int(req["lifecycle_revision"])
        cancel_epoch = int(req["cancel_epoch"])
        owner_run = 900001
        commit_seq = _ensure_commit_clock(conn, board=board_instance_id)
        owner_run, lease_epoch = _ensure_decomposition_lease(
            conn,
            board=board_instance_id,
            tenant=tenant,
            orch_id=orch_id,
            owner_run_id=owner_run,
            lifecycle_revision=life_rev,
            cancel_epoch=cancel_epoch,
        )

        # requirements + plan graph
        lane_specs = []
        for i, ch in enumerate(children, start=1):
            rid = f"req-{i}"
            rdigest = _sha("req", orch_id, ch["child_id"], str(i))
            lane_specs.append(
                {
                    "requirement_id": rid,
                    "ordinal": i,
                    "requirement_digest": rdigest,
                    "node_key": f"lane-{i}",
                    "lane_label": f"L{i}",
                    "task_id": ch["child_id"],
                    "title": ch.get("title") or f"lane-{i}",
                    "status": ch.get("status") or "ready",
                }
            )

        # insert requirements
        for ls in lane_specs:
            grant(
                conn,
                CapabilityGrant(
                    kind="plan_build",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=ls["requirement_digest"],
                ),
            )
            rjson = json.dumps(
                {"requirement_id": ls["requirement_id"], "task_id": ls["task_id"]},
                separators=(",", ":"),
                sort_keys=True,
            )
            conn.execute(
                "INSERT INTO orch_request_requirements("
                "board_instance_id,tenant_scope,orch_id,requirement_id,ordinal,"
                "requirement_json,requirement_digest,required"
                ") VALUES (?,?,?,?,?,?,?,1)",
                (
                    board_instance_id,
                    tenant,
                    orch_id,
                    ls["requirement_id"],
                    ls["ordinal"],
                    rjson,
                    ls["requirement_digest"],
                ),
            )

        plan_obj = {
            "schema_version": 4,
            "kind": "orch_plan",
            "orch_id": orch_id,
            "plan_version": plan_version,
            "synthesis_strategy": "parent_owned",
            "lanes": [
                {"node_key": ls["node_key"], "lane_label": ls["lane_label"], "task_id": ls["task_id"], "required": True}
                for ls in lane_specs
            ],
        }
        plan_json = json.dumps(plan_obj, separators=(",", ":"), sort_keys=True)
        plan_digest = digest(plan_obj)
        grant(
            conn,
            CapabilityGrant(
                kind="plan_build",
                board=board_instance_id,
                tenant=tenant,
                object_id=orch_id,
                revision=life_rev,
                epoch=cancel_epoch,
                target_key=plan_digest,
            ),
        )
        conn.execute(
            "INSERT INTO orch_plans("
            "board_instance_id,tenant_scope,orch_id,plan_version,schema_version,"
            "lineage_id,generation,request_key,request_digest,origin_id,parent_task_id,"
            "synthesis_strategy,plan_json,plan_digest,created_by_run_id,created_at"
            ") VALUES (?,?,?,?,4,?,?,?,?,?,?,'parent_owned',?,?,?,?)",
            (
                board_instance_id,
                tenant,
                orch_id,
                plan_version,
                req["lineage_id"],
                int(req["generation"]),
                req["request_key"],
                req["request_digest"],
                req["origin_id"],
                parent_task_id,
                plan_json,
                plan_digest,
                owner_run,
                _now(),
            ),
        )

        # parent node ordinal 1, lanes 2..
        route_json = "{}"
        route_digest = _sha("route", board_instance_id, orch_id, "parent")
        nodes = [
            {
                "node_key": "__parent__",
                "lane_lineage_key": _sha("ll", orch_id, "__parent__"),
                "role": "parent",
                "lane_label": "",
                "required": 1,
                "ordinal": 1,
                "task_id": parent_task_id,
                "title": parent_title,
                "status": parent_status,
            }
        ]
        for i, ls in enumerate(lane_specs, start=2):
            nodes.append(
                {
                    "node_key": ls["node_key"],
                    "lane_lineage_key": _sha("ll", orch_id, ls["node_key"]),
                    "role": "lane",
                    "lane_label": ls["lane_label"],
                    "required": 1,
                    "ordinal": i,
                    "task_id": ls["task_id"],
                    "title": ls["title"],
                    "status": ls["status"],
                }
            )

        for n in nodes:
            grant(
                conn,
                CapabilityGrant(
                    kind="plan_build",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=n["node_key"],
                ),
            )
            conn.execute(
                "INSERT INTO orch_plan_nodes("
                "board_instance_id,tenant_scope,orch_id,plan_version,node_key,lane_lineage_key,"
                "role,lane_label,normalized_goal,normalized_done_when,required,route_json,"
                "route_digest,ordinal"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    board_instance_id,
                    tenant,
                    orch_id,
                    plan_version,
                    n["node_key"],
                    n["lane_lineage_key"],
                    n["role"],
                    n["lane_label"],
                    f"goal:{n['node_key']}",
                    f"done:{n['node_key']}",
                    n["required"],
                    route_json,
                    route_digest if n["role"] == "parent" else _sha("route", orch_id, n["node_key"]),
                    n["ordinal"],
                ),
            )

        # edges lane -> parent
        for ls in lane_specs:
            ekey = _sha("edge", orch_id, ls["node_key"], "__parent__")[:32]
            # edge_key may need longer - use full hex
            ekey = _sha("edge", orch_id, ls["node_key"], "__parent__")
            grant(
                conn,
                CapabilityGrant(
                    kind="plan_build",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=ekey,
                ),
            )
            conn.execute(
                "INSERT INTO orch_plan_edges("
                "board_instance_id,tenant_scope,orch_id,plan_version,edge_key,"
                "parent_node_key,child_node_key,edge_kind"
                ") VALUES (?,?,?,?,?,?,?,'orch_required_for_synthesis')",
                (
                    board_instance_id,
                    tenant,
                    orch_id,
                    plan_version,
                    ekey,
                    ls["node_key"],
                    "__parent__",
                ),
            )
            grant(
                conn,
                CapabilityGrant(
                    kind="plan_build",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=f"{ls['requirement_id']}:{ls['node_key']}",
                ),
            )
            conn.execute(
                "INSERT INTO orch_plan_coverage("
                "board_instance_id,tenant_scope,orch_id,plan_version,requirement_id,node_key"
                ") VALUES (?,?,?,?,?,?)",
                (
                    board_instance_id,
                    tenant,
                    orch_id,
                    plan_version,
                    ls["requirement_id"],
                    ls["node_key"],
                ),
            )

        validate_plan(conn, board=board_instance_id, tenant=tenant, orch=orch_id, plan_version=plan_version)

        # bind soft tasks + orch_nodes BEFORE materialization row
        # Order: insert/ensure unbound soft tasks → orch_nodes → bind orch columns on soft tasks
        for n in nodes:
            exists = conn.execute("SELECT id, orch_id FROM _soft_fk_tasks WHERE id=?", (n["task_id"],)).fetchone()
            if exists is None:
                conn.execute(
                    "INSERT INTO _soft_fk_tasks(id,title,status,created_at) VALUES (?,?,?,?)",
                    (n["task_id"], n["title"], n["status"], _now()),
                )
            grant(
                conn,
                CapabilityGrant(
                    kind="materialize",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=n["node_key"],
                ),
            )
            conn.execute(
                "INSERT INTO orch_nodes("
                "board_instance_id,tenant_scope,orch_id,plan_version,node_key,task_id,"
                "node_state,current_route_digest,route_revision,created_by_run_id,created_at,updated_at"
                ") VALUES (?,?,?,?,?,?,'planned',?,1,?,?,?)",
                (
                    board_instance_id,
                    tenant,
                    orch_id,
                    plan_version,
                    n["node_key"],
                    n["task_id"],
                    _sha("route", orch_id, n["node_key"]),
                    owner_run,
                    _now(),
                    _now(),
                ),
            )
            # bind orch identity onto soft task (trigger allows via orch_nodes existence)
            grant(
                conn,
                CapabilityGrant(
                    kind="task_bind",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=n["task_id"],
                ),
            )
            conn.execute(
                "UPDATE _soft_fk_tasks SET title=?, status=?, "
                "orch_board_instance_id=?, orch_tenant_scope=?, orch_id=?, "
                "orch_plan_version=?, orch_node_key=?, orch_binding_revision=?, orch_cancel_epoch=? "
                "WHERE id=?",
                (
                    n["title"],
                    n["status"],
                    board_instance_id,
                    tenant,
                    orch_id,
                    plan_version,
                    n["node_key"],
                    life_rev,
                    cancel_epoch,
                    n["task_id"],
                ),
            )

        # soft links: edge direction lane(parent_id) -> __parent__(child_id)
        for ls in lane_specs:
            ekey = _sha("edge", orch_id, ls["node_key"], "__parent__")
            grant(
                conn,
                CapabilityGrant(
                    kind="link_bind",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=ekey,
                ),
            )
            conn.execute(
                "INSERT INTO _soft_fk_task_links("
                "parent_id,child_id,kind,orch_board_instance_id,orch_tenant_scope,orch_id,"
                "orch_plan_version,orch_edge_key,orch_binding_revision,orch_cancel_epoch"
                ") VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    ls["task_id"],
                    parent_task_id,
                    "orch_required_for_synthesis",
                    board_instance_id,
                    tenant,
                    orch_id,
                    plan_version,
                    ekey,
                    life_rev,
                    cancel_epoch,
                ),
            )

        node_count = conn.execute(
            "SELECT count(*) FROM orch_nodes WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND plan_version=?",
            (board_instance_id, tenant, orch_id, plan_version),
        ).fetchone()[0]
        edge_count = conn.execute(
            "SELECT count(*) FROM _soft_fk_task_links WHERE orch_board_instance_id=? AND orch_tenant_scope=? "
            "AND orch_id=? AND orch_plan_version=?",
            (board_instance_id, tenant, orch_id, plan_version),
        ).fetchone()[0]
        graph_digest = _sha("graph", orch_id, str(plan_version), str(node_count), str(edge_count))
        grant(
            conn,
            CapabilityGrant(
                kind="materialize",
                board=board_instance_id,
                tenant=tenant,
                object_id=orch_id,
                revision=life_rev,
                epoch=cancel_epoch,
                target_key=graph_digest,
            ),
        )
        conn.execute(
            "INSERT INTO orch_plan_materializations("
            "board_instance_id,tenant_scope,orch_id,plan_version,request_lifecycle_revision,"
            "cancel_epoch,lease_epoch,plan_digest,observed_graph_digest,observed_node_count,"
            "observed_edge_count,materialized_by_run_id,commit_seq,materialized_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                board_instance_id,
                tenant,
                orch_id,
                plan_version,
                life_rev,
                cancel_epoch,
                lease_epoch,
                plan_digest,
                graph_digest,
                int(node_count),
                int(edge_count),
                owner_run,
                commit_seq,
                _now(),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    # lifecycle CAS outside previous txn (own transaction)
    step = apply_lifecycle_transition_db(
        conn,
        board_instance_id=board_instance_id,
        tenant_scope=tenant,
        orch_id=orch_id,
        event="valid_plan_materialized",
        to_state="waiting_lanes",
    )
    return {
        "plan_version": plan_version,
        "plan_digest": plan_digest,
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "transition": step,
        "lanes": [{"node_key": ls["node_key"], "task_id": ls["task_id"]} for ls in lane_specs],
    }


def accept_done_required_lanes(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    child_status_by_id: dict[str, str],
) -> dict[str, Any]:
    """Accept required lane nodes whose native/child status is done."""
    apply_multilane_soft_fk_patch(conn)
    tenant = "" if tenant_scope is None else str(tenant_scope)
    begin_immediate(conn)
    accepted: list[str] = []
    try:
        req = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
        if req["lifecycle_state"] != "waiting_lanes":
            raise MultiLaneError(f"bad_state:{req['lifecycle_state']}")
        plan_version = int(req["plan_version"])
        life_rev = int(req["lifecycle_revision"])
        cancel_epoch = int(req["cancel_epoch"])
        plan_epoch = int(req["plan_epoch_revision"])
        owner_run = 900002
        owner_run, lease_epoch = _ensure_reconciliation_lease(
            conn,
            board=board_instance_id,
            tenant=tenant,
            orch_id=orch_id,
            owner_run_id=owner_run,
            lifecycle_revision=life_rev,
            cancel_epoch=cancel_epoch,
        )

        lanes = conn.execute(
            "SELECT n.node_key, n.task_id, n.node_state, pn.required "
            "FROM orch_nodes n "
            "JOIN orch_plan_nodes pn ON pn.board_instance_id=n.board_instance_id "
            " AND pn.tenant_scope=n.tenant_scope AND pn.orch_id=n.orch_id "
            " AND pn.plan_version=n.plan_version AND pn.node_key=n.node_key "
            "WHERE n.board_instance_id=? AND n.tenant_scope=? AND n.orch_id=? "
            "AND n.plan_version=? AND pn.role='lane' AND pn.required=1",
            (board_instance_id, tenant, orch_id, plan_version),
        ).fetchall()

        for lane in lanes:
            st = (child_status_by_id.get(lane["task_id"]) or "").lower()
            if st not in DONE_NATIVE:
                continue
            if lane["node_state"] == "accepted":
                accepted.append(lane["node_key"])
                continue
            # move node planned/ready/running -> running -> accepted
            # first ensure running
            if lane["node_state"] in {"planned", "ready"}:
                grant(
                    conn,
                    CapabilityGrant(
                        kind="node_transition",
                        board=board_instance_id,
                        tenant=tenant,
                        object_id=orch_id,
                        revision=life_rev,
                        epoch=cancel_epoch,
                        target_key=lane["node_key"],
                    ),
                )
                nxt = "ready" if lane["node_state"] == "planned" else "running"
                conn.execute(
                    "UPDATE orch_nodes SET node_state=?, updated_at=? "
                    "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND node_key=?",
                    (nxt, _now(), board_instance_id, tenant, orch_id, lane["node_key"]),
                )
                if nxt == "ready":
                    grant(
                        conn,
                        CapabilityGrant(
                            kind="node_transition",
                            board=board_instance_id,
                            tenant=tenant,
                            object_id=orch_id,
                            revision=life_rev,
                            epoch=cancel_epoch,
                            target_key=lane["node_key"],
                        ),
                    )
                    conn.execute(
                        "UPDATE orch_nodes SET node_state='running', updated_at=? "
                        "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND node_key=?",
                        (_now(), board_instance_id, tenant, orch_id, lane["node_key"]),
                    )
            elif lane["node_state"] != "running":
                continue

            outcome = _sha("outcome", orch_id, lane["node_key"], lane["task_id"], "done")
            # task run — capability target may be task_id (NEW.id null on BEFORE INSERT)
            grant(
                conn,
                CapabilityGrant(
                    kind="task_run_start",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key="*",
                ),
            )
            grant(
                conn,
                CapabilityGrant(
                    kind="task_run_write",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key="*",
                ),
            )
            cur = conn.execute(
                "INSERT INTO _soft_fk_task_runs(task_id,status,started_at,ended_at,outcome,outcome_digest,cancellation_epoch)"
                " VALUES (?, 'done', ?, ?, 'completed', ?, ?)",
                (lane["task_id"], _now() - 1, _now(), outcome, cancel_epoch),
            )
            run_id = int(cur.lastrowid)
            grant(
                conn,
                CapabilityGrant(
                    kind="accept_lane",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=plan_epoch,
                    epoch=cancel_epoch,
                    target_key=str(run_id),
                ),
            )
            conn.execute(
                "INSERT INTO orch_node_acceptances("
                "board_instance_id,tenant_scope,orch_id,plan_version,node_key,task_id,"
                "accepted_run_id,outcome_digest,accepted_by_run_id,acceptance_lease_epoch,"
                "plan_epoch_revision,cancel_epoch,accepted_at"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    board_instance_id,
                    tenant,
                    orch_id,
                    plan_version,
                    lane["node_key"],
                    lane["task_id"],
                    run_id,
                    outcome,
                    owner_run,
                    lease_epoch,
                    plan_epoch,
                    cancel_epoch,
                    _now(),
                ),
            )
            grant(
                conn,
                CapabilityGrant(
                    kind="node_transition",
                    board=board_instance_id,
                    tenant=tenant,
                    object_id=orch_id,
                    revision=life_rev,
                    epoch=cancel_epoch,
                    target_key=lane["node_key"],
                ),
            )
            conn.execute(
                "UPDATE orch_nodes SET node_state='accepted', updated_at=? "
                "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND node_key=?",
                (_now(), board_instance_id, tenant, orch_id, lane["node_key"]),
            )
            accepted.append(lane["node_key"])

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {"accepted_node_keys": accepted, "accepted_count": len(accepted)}


def advance_after_acceptances(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    origin_kind: str = "board_only",
) -> list[dict[str, Any]]:
    """waiting_lanes → synthesizing → completed (board_only short path)."""
    tenant = "" if tenant_scope is None else str(tenant_scope)
    steps: list[dict[str, Any]] = []
    req = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
    if req["lifecycle_state"] != "waiting_lanes":
        return steps
    plan_version = int(req["plan_version"])
    required = conn.execute(
        "SELECT count(*) FROM orch_plan_nodes "
        "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND plan_version=? "
        "AND role='lane' AND required=1",
        (board_instance_id, tenant, orch_id, plan_version),
    ).fetchone()[0]
    accepted = conn.execute(
        "SELECT count(*) FROM orch_node_acceptances "
        "WHERE board_instance_id=? AND tenant_scope=? AND orch_id=? AND plan_version=?",
        (board_instance_id, tenant, orch_id, plan_version),
    ).fetchone()[0]
    if int(accepted) < 2 or int(accepted) < int(required):
        return steps
    steps.append(
        apply_lifecycle_transition_db(
            conn,
            board_instance_id=board_instance_id,
            tenant_scope=tenant,
            orch_id=orch_id,
            event="required_set_accepted",
            to_state="synthesizing",
            accepted_required_lanes=int(accepted),
        )
    )
    # board_only: skip full delivery — complete from synthesizing
    steps.append(
        apply_lifecycle_transition_db(
            conn,
            board_instance_id=board_instance_id,
            tenant_scope=tenant,
            orch_id=orch_id,
            event="board_only_parent_done",
            to_state="completed",
            origin_kind=origin_kind,
            children_all_done=True,
        )
    )
    return steps


def run_multilane_once(
    conn: sqlite3.Connection,
    *,
    board_instance_id: str,
    tenant_scope: str,
    orch_id: str,
    parent_task_id: str,
    parent_title: str,
    parent_status: str,
    children: list[dict[str, str]],
) -> MultiLaneResult:
    """One progressive multi-lane step based on native children truth."""
    tenant = "" if tenant_scope is None else str(tenant_scope)
    req = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
    before = req["lifecycle_state"]
    total, done, all_done = children_progress(children)
    steps: list[MultiLaneStep] = []

    base = MultiLaneResult(
        orch_id=orch_id,
        parent_task_id=parent_task_id,
        before_state=before,
        after_state=before,
        plan_version=int(req["plan_version"]),
        steps=[],
        children_total=total,
        children_done=done,
    )

    if req["origin_kind"] != "board_only":
        base.skipped = True
        base.reason = "not_board_only"
        return base
    if before in {"completed", "failed", "cancelled"}:
        base.skipped = True
        base.reason = "already_terminal"
        return base

    try:
        if before == "decomposing" and total >= 2 and int(req["plan_version"]) == 0:
            mat = materialize_plan_from_children(
                conn,
                board_instance_id=board_instance_id,
                tenant_scope=tenant,
                orch_id=orch_id,
                parent_task_id=parent_task_id,
                parent_title=parent_title,
                parent_status=parent_status,
                children=children,
            )
            steps.append(MultiLaneStep(action="materialize", detail=mat))
        elif before == "waiting_lanes" and total >= 2:
            status_map = {c["child_id"]: c["status"] for c in children}
            acc = accept_done_required_lanes(
                conn,
                board_instance_id=board_instance_id,
                tenant_scope=tenant,
                orch_id=orch_id,
                child_status_by_id=status_map,
            )
            steps.append(MultiLaneStep(action="accept_lanes", detail=acc))
            if int(acc.get("accepted_count") or 0) >= 2:
                adv = advance_after_acceptances(
                    conn,
                    board_instance_id=board_instance_id,
                    tenant_scope=tenant,
                    orch_id=orch_id,
                )
                for a in adv:
                    steps.append(MultiLaneStep(action="lifecycle", detail=a))
        else:
            base.skipped = True
            base.reason = f"no_multilane_rule:{before}:children={done}/{total}"
            return base
    except (MultiLaneError, OrchAPIError, LifecycleError, sqlite3.IntegrityError, ValueError) as exc:
        code = getattr(exc, "code", None) or str(exc)
        raise MultiLaneError(f"multilane_failed:{code}") from exc

    req2 = _load_request(conn, board=board_instance_id, tenant=tenant, orch_id=orch_id)
    accepted_n = conn.execute(
        "SELECT count(*) FROM orch_node_acceptances WHERE board_instance_id=? AND tenant_scope=? AND orch_id=?",
        (board_instance_id, tenant, orch_id),
    ).fetchone()[0]
    return MultiLaneResult(
        orch_id=orch_id,
        parent_task_id=parent_task_id,
        before_state=before,
        after_state=req2["lifecycle_state"],
        plan_version=int(req2["plan_version"]),
        steps=steps,
        skipped=False,
        accepted_required_lanes=int(accepted_n),
        children_total=total,
        children_done=done,
    )


def live_multilane_parent(parent_task_id: str) -> MultiLaneResult:
    br: OrchBridge | None = None
    try:
        br = open_live_bridge()
        apply_multilane_soft_fk_patch(br._sidecar)
        ref = br.read_native_task(parent_task_id)
        if ref is None:
            raise MultiLaneError("native_parent_missing")
        children = read_native_children(br._native, parent_task_id)
        # enrich titles
        enriched = []
        for c in children:
            crow = br._native.execute("SELECT title, status FROM tasks WHERE id=?", (c["child_id"],)).fetchone()
            enriched.append(
                {
                    "child_id": c["child_id"],
                    "status": (crow["status"] if crow else c["status"] or "").lower(),
                    "title": (crow["title"] if crow else c["child_id"]),
                }
            )
        row = br._sidecar.execute(
            "SELECT board_instance_id, tenant_scope, orch_id FROM orch_requests WHERE parent_task_id=? "
            "ORDER BY updated_at DESC LIMIT 1",
            (parent_task_id,),
        ).fetchone()
        if row is None:
            raise MultiLaneError("request_not_found_for_parent")
        return run_multilane_once(
            br._sidecar,
            board_instance_id=row["board_instance_id"],
            tenant_scope=row["tenant_scope"] or "",
            orch_id=row["orch_id"],
            parent_task_id=parent_task_id,
            parent_title=ref.title or parent_task_id,
            parent_status=ref.status,
            children=enriched,
        )
    except BridgeError as exc:
        raise MultiLaneError(f"bridge:{exc.code}") from exc
    finally:
        if br is not None:
            br.close()


__all__ = [
    "MultiLaneError",
    "MultiLaneResult",
    "MultiLaneStep",
    "materialize_plan_from_children",
    "accept_done_required_lanes",
    "advance_after_acceptances",
    "run_multilane_once",
    "live_multilane_parent",
]
