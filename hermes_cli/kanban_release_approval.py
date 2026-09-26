"""Task/release-bound, persisted and idempotent release-approval saga.

The SQLite transactions claim/finalize a local saga; they do not pretend the
external promotion is globally ACID.  The adapter is retried with one stable
operation key and must be idempotent.  Every subprocess call is argv-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from hermes_cli.kanban_db_connect import write_txn

_APPROVAL_TEXT = "freigegeben"
_REVIEW_STATUS = "Auf Dev zur Prüfung"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ACTOR_SUBJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,255}")
_LEASE_SECONDS = 7700
_ADAPTER_CONTRACT_VERSION = "cuto-hermes-release/v1"
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_RECEIPT_FIELDS = {
    "ok", "contract_version", "operation_key", "task_id", "release_id",
    "manifest_sha256", "manual_test_cases_sha256", "source_commit",
    "workflow_run_id", "bundle_sha256", "artifacts", "dev", "test",
    "production", "previous_release_id", "rollback_available",
    "database_restore_attempted",
}


@dataclass(frozen=True)
class ApprovalContext:
    platform: str
    chat_id: str
    thread_id: str
    actor_id: str
    reply_to_message_id: str


@dataclass(frozen=True)
class PresentedRelease:
    gate_id: str
    task_id: str
    release_id: str
    manifest_sha256: str


@dataclass(frozen=True)
class ApprovalResult:
    ok: bool
    classification: str
    release_id: str | None = None
    dev_result: str = "unknown"
    test_result: str = "unknown"
    production_result: str = "unknown"
    active_release_id: str | None = None
    previous_release_id: str | None = None
    rollback_available: bool | None = None
    detail: str = ""

    def user_message(self) -> str:
        release = self.release_id or "unbekannt"
        active = self.active_release_id or "unbekannt"
        dev = "unbekannt" if self.dev_result == "unknown" else self.dev_result
        test = "unbekannt" if self.test_result == "unknown" else self.test_result
        production = (
            "unbekannt" if self.production_result == "unknown" else self.production_result
        )
        rollback = (
            "unbekannt" if self.rollback_available is None
            else "ja" if self.rollback_available else "nein"
        )
        prefix = "Freigabe verarbeitet" if self.ok else "Fehler: Freigabe konnte nicht verarbeitet werden"
        return (
            f"{prefix}. Release-ID: {release}. Dev: {dev}. "
            f"Test: {test}. Prod: {production}. "
            f"Aktiv: {active}. Vorgänger: {self.previous_release_id or 'unbekannt'}. "
            f"Rollback verfügbar: {rollback}."
        )


def _required(value: str, name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _artifact_digests(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("artifact_digests must be a non-empty mapping")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or _ARTIFACT_NAME.fullmatch(name) is None:
            raise ValueError("artifact_digests contains an invalid artifact name")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError("artifact_digests must contain lowercase SHA-256 values")
        result[name] = digest
    return dict(sorted(result.items()))


def present_release(
    conn,
    *,
    task_id: str,
    release_id: str,
    manifest_sha256: str,
    manual_test_cases_digest: str,
    source_commit: str,
    bundle_sha256: str,
    artifact_digests: Mapping[str, str],
    workflow_status: str,
    active_dev_release_id: str,
    platform: str,
    chat_id: str,
    thread_id: str,
    actor_id: str,
    presented_message_id: str,
    previous_release_id: str | None,
    rollback_available: bool,
    presented_at: int | None = None,
) -> PresentedRelease:
    """Persist the exact visible approval target and supersede older route gates."""
    fields = {
        "task_id": task_id, "release_id": release_id, "platform": platform,
        "chat_id": chat_id, "actor_id": actor_id, "presented_message_id": presented_message_id,
    }
    fields = {name: _required(value, name) for name, value in fields.items()}
    if _ACTOR_SUBJECT.fullmatch(fields["actor_id"]) is None:
        raise ValueError("actor_id must be an authenticated opaque subject")
    manifest_sha256 = str(manifest_sha256 or "").lower()
    manual_test_cases_digest = str(manual_test_cases_digest or "").lower()
    if not _DIGEST.fullmatch(manifest_sha256) or not _DIGEST.fullmatch(manual_test_cases_digest):
        raise ValueError("manifest and manual-test digests must be lowercase SHA-256 values")
    source_commit = str(source_commit or "").lower()
    bundle_sha256 = str(bundle_sha256 or "").lower()
    if _SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a lowercase 40-character commit SHA")
    if _DIGEST.fullmatch(bundle_sha256) is None:
        raise ValueError("bundle_sha256 must be a lowercase SHA-256 value")
    artifacts_json = json.dumps(
        _artifact_digests(artifact_digests), sort_keys=True, separators=(",", ":")
    )
    if previous_release_id is not None:
        previous_release_id = _required(previous_release_id, "previous_release_id")
        if previous_release_id == fields["release_id"]:
            raise ValueError("previous release must differ from the presented release")
    if bool(rollback_available) != (previous_release_id is not None):
        raise ValueError("rollback availability must match the verified previous release")
    if workflow_status != _REVIEW_STATUS:
        raise ValueError(f"workflow_status must be {_REVIEW_STATUS!r}")
    if active_dev_release_id != fields["release_id"]:
        raise ValueError("active Dev release must equal the presented release")
    gate_id = "rg_" + secrets.token_hex(8)
    now = int(time.time()) if presented_at is None else int(presented_at)
    thread_id = str(thread_id or "")
    with write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id=?", (fields["task_id"],)).fetchone() is None:
            raise ValueError("task does not exist")
        if conn.execute(
            "SELECT 1 FROM kanban_release_gates WHERE gate_status='consuming' AND "
            "(task_id=? OR (platform=? AND chat_id=? AND thread_id=? AND actor_id=?))",
            (fields["task_id"], fields["platform"], fields["chat_id"], thread_id,
             fields["actor_id"]),
        ).fetchone() is not None:
            raise RuntimeError("release promotion is in progress for this task or approval route")
        conn.execute(
            "INSERT INTO kanban_release_state "
            "(task_id,workflow_status,active_dev_release_id,manifest_sha256,updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
            "workflow_status=excluded.workflow_status,active_dev_release_id=excluded.active_dev_release_id,"
            "manifest_sha256=excluded.manifest_sha256,updated_at=excluded.updated_at",
            (fields["task_id"], workflow_status, active_dev_release_id, manifest_sha256, now),
        )
        conn.execute(
            "UPDATE kanban_release_gates SET gate_status='superseded' "
            "WHERE platform=? AND chat_id=? AND thread_id=? AND actor_id=? "
            "AND gate_status IN ('active','consuming')",
            (fields["platform"], fields["chat_id"], thread_id, fields["actor_id"]),
        )
        conn.execute(
            "INSERT INTO kanban_release_gates "
            "(id,task_id,release_id,manifest_sha256,manual_test_cases_digest,source_commit,"
            "bundle_sha256,artifact_digests_json,workflow_status,"
            "active_dev_release_id,platform,chat_id,thread_id,actor_id,presented_message_id,"
            "presented_at,previous_release_id,rollback_available,gate_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active')",
            (gate_id, fields["task_id"], fields["release_id"], manifest_sha256,
             manual_test_cases_digest, source_commit, bundle_sha256, artifacts_json,
             workflow_status, active_dev_release_id,
             fields["platform"], fields["chat_id"], thread_id, fields["actor_id"],
             fields["presented_message_id"], now, previous_release_id,
             int(bool(rollback_available))),
        )
    return PresentedRelease(gate_id, fields["task_id"], fields["release_id"], manifest_sha256)


def update_release_state(
    conn, *, task_id: str, workflow_status: str, active_dev_release_id: str,
    manifest_sha256: str, updated_at: int | None = None,
) -> None:
    """Update approval-time release truth without mutating the visible gate snapshot."""
    task_id = _required(task_id, "task_id")
    active_dev_release_id = _required(active_dev_release_id, "active_dev_release_id")
    manifest_sha256 = str(manifest_sha256 or "").lower()
    if not _DIGEST.fullmatch(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 value")
    now = int(time.time()) if updated_at is None else int(updated_at)
    with write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone() is None:
            raise ValueError("task does not exist")
        if conn.execute(
            "SELECT 1 FROM kanban_release_gates WHERE task_id=? AND gate_status='consuming'",
            (task_id,),
        ).fetchone() is not None:
            raise RuntimeError("release promotion is in progress for this task")
        conn.execute(
            "INSERT INTO kanban_release_state "
            "(task_id,workflow_status,active_dev_release_id,manifest_sha256,updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
            "workflow_status=excluded.workflow_status,active_dev_release_id=excluded.active_dev_release_id,"
            "manifest_sha256=excluded.manifest_sha256,updated_at=excluded.updated_at",
            (task_id, str(workflow_status or ""), active_dev_release_id, manifest_sha256, now),
        )


def _result(classification: str, gate=None, **kwargs) -> ApprovalResult:
    active_release_id = kwargs.pop(
        "active_release_id",
        gate["current_active_dev_release_id"] if gate is not None else None,
    )
    return ApprovalResult(
        ok=False,
        classification=classification,
        release_id=gate["release_id"] if gate is not None else None,
        active_release_id=active_release_id,
        **kwargs,
    )


def _operation_key(gate) -> str:
    bound = "\0".join((
        gate["id"], gate["task_id"], gate["release_id"], gate["manifest_sha256"],
        gate["manual_test_cases_digest"], gate["source_commit"], gate["bundle_sha256"],
        gate["artifact_digests_json"], gate["actor_id"], gate["presented_message_id"],
    ))
    return "release-approval:" + hashlib.sha256(bound.encode()).hexdigest()


def _approval_id(gate) -> str:
    """Return a stable dispatcher-owned ID without exposing route identifiers."""
    return "hermes:" + hashlib.sha256(gate["id"].encode()).hexdigest()


def _claim(conn, text: str, context: ApprovalContext, now: int):
    if text != _APPROVAL_TEXT:
        return None, _result("not_approval")
    with write_txn(conn):
        rows = conn.execute(
            "SELECT g.*, t.status AS task_status, s.workflow_status AS current_workflow_status, "
            "s.active_dev_release_id AS current_active_dev_release_id, "
            "s.manifest_sha256 AS current_manifest_sha256 FROM kanban_release_gates g "
            "JOIN tasks t ON t.id=g.task_id "
            "LEFT JOIN kanban_release_state s ON s.task_id=g.task_id "
            "WHERE g.platform=? AND g.chat_id=? "
            "AND g.thread_id=? AND g.actor_id=? AND g.presented_message_id=? "
            "ORDER BY g.presented_at DESC, g.id DESC",
            (context.platform, context.chat_id, context.thread_id or "", context.actor_id,
             context.reply_to_message_id),
        ).fetchall()
        route_state = conn.execute(
            "SELECT s.active_dev_release_id FROM kanban_release_gates g "
            "JOIN kanban_release_state s ON s.task_id=g.task_id "
            "WHERE g.platform=? AND g.chat_id=? AND g.thread_id=? AND g.actor_id=? "
            "AND g.gate_status IN ('active','consuming') "
            "AND s.active_dev_release_id=g.release_id "
            "ORDER BY g.presented_at DESC,g.id DESC LIMIT 1",
            (context.platform, context.chat_id, context.thread_id or "", context.actor_id),
        ).fetchone()
        route_active_release_id = (
            route_state["active_dev_release_id"] if route_state is not None else None
        )
        if len(rows) != 1:
            return None, _result(
                "stale_approval", active_release_id=route_active_release_id
            )
        gate = rows[0]
        if gate["gate_status"] == "approved":
            return None, _result(
                "replay", gate, active_release_id=route_active_release_id
            )
        if (gate["gate_status"] not in {"active", "consuming"}
                or gate["workflow_status"] != _REVIEW_STATUS
                or gate["active_dev_release_id"] != gate["release_id"]
                or gate["current_workflow_status"] != _REVIEW_STATUS
                or gate["current_active_dev_release_id"] != gate["release_id"]
                or gate["current_manifest_sha256"] != gate["manifest_sha256"]
                or gate["task_status"] in {"done", "archived"}
                or not _DIGEST.fullmatch(gate["manifest_sha256"])
                or not _DIGEST.fullmatch(gate["manual_test_cases_digest"])):
            return None, _result(
                "stale_approval",
                gate,
                active_release_id=(
                    route_active_release_id or gate["current_active_dev_release_id"]
                ),
            )
        op_key = _operation_key(gate)
        saga = conn.execute("SELECT * FROM kanban_release_sagas WHERE gate_id=?", (gate["id"],)).fetchone()
        if saga is not None and saga["state"] == "succeeded":
            return None, _result("replay", gate)
        if (saga is not None and saga["state"] == "promoting"
                and saga["lease_expires"] is not None and saga["lease_expires"] > now):
            # The current owner may still be mutating the external targets.
            # Once its lease expires, retry enters the adapter's explicit
            # resume/reconcile action with this same operation key.
            return None, _result("in_progress", gate)
        owner = secrets.token_hex(16)
        action = "promote" if saga is None else "resume"
        if saga is None:
            conn.execute(
                "INSERT INTO kanban_release_sagas "
                "(operation_key,gate_id,state,owner_token,lease_expires,created_at,updated_at) "
                "VALUES (?,?,'promoting',?,?,?,?)",
                (op_key, gate["id"], owner, now + _LEASE_SECONDS, now, now),
            )
        else:
            conn.execute(
                "UPDATE kanban_release_sagas SET state='promoting',owner_token=?,lease_expires=?,"
                "error_class=NULL,updated_at=? WHERE operation_key=?",
                (owner, now + _LEASE_SECONDS, now, op_key),
            )
        conn.execute("UPDATE kanban_release_gates SET gate_status='consuming' WHERE id=?", (gate["id"],))
        return (gate, op_key, owner, action), None


def _validate_receipt(receipt: object, gate, operation_key: str) -> dict:
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("promotion adapter returned an invalid release receipt schema")
    expected_identity = {
        "ok": True,
        "contract_version": _ADAPTER_CONTRACT_VERSION,
        "operation_key": operation_key,
        "task_id": gate["task_id"],
        "release_id": gate["release_id"],
        "manifest_sha256": gate["manifest_sha256"],
        "manual_test_cases_sha256": gate["manual_test_cases_digest"],
        "source_commit": gate["source_commit"],
        "bundle_sha256": gate["bundle_sha256"],
        "database_restore_attempted": False,
    }
    if any(receipt.get(field) != value for field, value in expected_identity.items()):
        raise ValueError("promotion adapter receipt identity differs from the approval gate")
    workflow_run_id = receipt.get("workflow_run_id")
    if not isinstance(workflow_run_id, str) or not workflow_run_id.isdigit():
        raise ValueError("promotion adapter returned an invalid workflow run ID")

    raw_artifacts = receipt.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ValueError("promotion adapter omitted artifact digests")
    if any(
        not isinstance(name, str)
        or _ARTIFACT_NAME.fullmatch(name) is None
        or not isinstance(value, dict)
        or set(value) != {"sha256"}
        or not isinstance(value.get("sha256"), str)
        or _DIGEST.fullmatch(value["sha256"]) is None
        for name, value in raw_artifacts.items()
    ):
        raise ValueError("promotion adapter returned invalid artifact digests")
    actual_artifacts = {name: value["sha256"] for name, value in raw_artifacts.items()}
    if actual_artifacts != json.loads(gate["artifact_digests_json"]):
        raise ValueError("promotion adapter substituted an unapproved artifact")

    states = {}
    for target in ("dev", "test", "production"):
        state = receipt.get(target)
        if (not isinstance(state, dict) or set(state) != {"result", "active_release_id"}
                or state.get("result") not in {"success", "failed", "not_started"}):
            raise ValueError(f"promotion adapter returned invalid {target} state")
        active = state.get("active_release_id")
        if active is not None and (not isinstance(active, str) or not active or active == "unknown"):
            raise ValueError(f"promotion adapter returned unknown {target} active state")
        states[target] = (state["result"], active)

    release_id = gate["release_id"]
    if states["dev"] != ("success", release_id):
        raise ValueError("promotion adapter did not read back the approved Dev release")
    test_result, test_active = states["test"]
    production_result, production_active = states["production"]
    if test_result == "success":
        if test_active != release_id or production_result not in {"success", "failed"}:
            raise ValueError("promotion adapter returned an invalid Test-to-Production transition")
        if production_result == "success" and production_active != release_id:
            raise ValueError("promotion adapter did not read back the Production release")
        if production_result == "failed" and production_active == release_id:
            raise ValueError("Production failure did not prove candidate compensation")
    elif test_result == "failed":
        if test_active == release_id or states["production"] != ("not_started", test_active):
            raise ValueError("Test failure did not prove Production was left unchanged")
    else:
        raise ValueError("promotion adapter did not attempt the approved Test release")

    previous = receipt.get("previous_release_id")
    rollback_available = receipt.get("rollback_available")
    if (previous is not None and (not isinstance(previous, str) or not previous
                                  or previous == release_id)
            or not isinstance(rollback_available, bool)
            or rollback_available != (previous is not None)
            or previous != gate["previous_release_id"]
            or rollback_available != bool(gate["rollback_available"])):
        raise ValueError("promotion adapter returned unverified previous-release rollback state")
    if production_active != release_id and production_active != previous:
        raise ValueError("promotion adapter previous release differs from Production read-back")
    return receipt


def _adapter_receipt(
    promotion_argv: Sequence[str], gate, operation_key: str, *, action: str = "promote",
) -> dict:
    if not promotion_argv or not all(isinstance(value, str) and value for value in promotion_argv):
        raise ValueError("promotion adapter argv is not configured")
    command = [*promotion_argv, action, "--task-id", gate["task_id"],
               "--release-id", gate["release_id"], "--manifest-sha256", gate["manifest_sha256"],
               "--operation-key", operation_key]
    trusted_input = json.dumps({
        "approval_id": _approval_id(gate),
        "actor_subject": gate["actor_id"],
        "contract_version": _ADAPTER_CONTRACT_VERSION,
        "manual_test_cases_sha256": gate["manual_test_cases_digest"],
    }, sort_keys=True, separators=(",", ":")) + "\n"
    adapter_env = {
        name: os.environ[name]
        for name in ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT")
        if name in os.environ
    }
    adapter_env.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    completed = subprocess.run(command, input=trusted_input, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=7600, check=True,
                               env=adapter_env)
    return _validate_receipt(json.loads(completed.stdout), gate, operation_key)


def _from_receipt(ok: bool, classification: str, gate, receipt: dict, detail: str = "") -> ApprovalResult:
    return ApprovalResult(
        ok=ok, classification=classification, release_id=gate["release_id"],
        dev_result=receipt["dev"]["result"], test_result=receipt["test"]["result"],
        production_result=receipt["production"]["result"],
        active_release_id=receipt["production"].get("active_release_id")
                          or receipt["test"].get("active_release_id")
                          or receipt["dev"].get("active_release_id"),
        previous_release_id=receipt.get("previous_release_id"),
        rollback_available=(
            receipt.get("rollback_available")
            if isinstance(receipt.get("rollback_available"), bool)
            else None
        ),
        detail=detail,
    )


def process_approval(
    conn,
    *,
    text: str,
    context: ApprovalContext,
    promotion_argv: Sequence[str],
    now: int | None = None,
) -> ApprovalResult:
    """Claim, invoke and finalize one exact approval without holding a DB lock over I/O."""
    timestamp = int(time.time()) if now is None else int(now)
    claim, early = _claim(conn, text, context, timestamp)
    if early is not None:
        return early
    if claim is None:  # defensive: _claim returns either a claim or a result
        return _result("stale_approval")
    gate, operation_key, owner, action = claim
    try:
        receipt = _adapter_receipt(promotion_argv, gate, operation_key, action=action)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        with write_txn(conn):
            conn.execute(
                "UPDATE kanban_release_sagas SET error_class='adapter',lease_expires=?,"
                "updated_at=? WHERE operation_key=? AND owner_token=?",
                (timestamp, timestamp, operation_key, owner),
            )
        return ApprovalResult(
            ok=False,
            classification="adapter_failed",
            release_id=gate["release_id"],
            detail="Promotion outcome is unknown; retry resumes with the same operation key.",
        )

    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    promotion_succeeded = all(
        receipt[target]["result"] == "success"
        and receipt[target].get("active_release_id") == gate["release_id"]
        for target in ("dev", "test", "production")
    )
    with write_txn(conn):
        current = conn.execute(
            "SELECT g.*, t.status AS task_status, s.workflow_status AS current_workflow_status, "
            "s.active_dev_release_id AS current_active_dev_release_id, "
            "s.manifest_sha256 AS current_manifest_sha256 FROM kanban_release_gates g "
            "JOIN tasks t ON t.id=g.task_id LEFT JOIN kanban_release_state s ON s.task_id=g.task_id "
            "WHERE g.id=?",
            (gate["id"],),
        ).fetchone()
        saga = conn.execute("SELECT * FROM kanban_release_sagas WHERE operation_key=?", (operation_key,)).fetchone()
        if (current is None or current["gate_status"] != "consuming"
                or current["current_workflow_status"] != _REVIEW_STATUS
                or current["current_active_dev_release_id"] != gate["release_id"]
                or current["current_manifest_sha256"] != gate["manifest_sha256"]
                or current["task_status"] in {"done", "archived"} or saga is None
                or saga["owner_token"] != owner or saga["state"] != "promoting"):
            if saga is not None and saga["owner_token"] == owner:
                conn.execute("UPDATE kanban_release_sagas SET state='failed',error_class='stale_after_promotion',"
                             "adapter_receipt=?,lease_expires=NULL,updated_at=? WHERE operation_key=?",
                             (encoded, timestamp, operation_key))
            return _from_receipt(False, "stale_after_promotion", gate, receipt)
        if not promotion_succeeded:
            conn.execute("UPDATE kanban_release_sagas SET state='failed',error_class='promotion_failed',"
                         "adapter_receipt=?,lease_expires=NULL,updated_at=? "
                         "WHERE operation_key=? AND owner_token=?",
                         (encoded, timestamp, operation_key, owner))
            return _from_receipt(False, "promotion_failed", gate, receipt)
        conn.execute("UPDATE kanban_release_sagas SET state='succeeded',adapter_receipt=?,"
                     "lease_expires=NULL,updated_at=? WHERE operation_key=? AND owner_token=?",
                     (encoded, timestamp, operation_key, owner))
        conn.execute("UPDATE kanban_release_gates SET gate_status='approved' WHERE id=?", (gate["id"],))
        from hermes_cli.kanban_db import _append_event
        _append_event(conn, gate["task_id"], "release_approved", {
            "operation_key": operation_key, "release_id": gate["release_id"],
            "manifest_sha256": gate["manifest_sha256"], "result": receipt,
        })
    return _from_receipt(True, "success", gate, receipt)


def process_current_board_approval(
    *, text: str, context: ApprovalContext, promotion_argv: Sequence[str], now: int | None = None,
) -> ApprovalResult:
    """Thread-safe gateway entry point: open and close the current board connection."""
    from hermes_cli.kanban_db_connect import connect_closing
    with connect_closing() as conn:
        return process_approval(conn, text=text, context=context,
                                promotion_argv=promotion_argv, now=now)
