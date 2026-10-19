"""Typed Issue 22 controls and pending intake on Hermes' existing control socket.

The gateway process owns the Kanban connection and calls the typed owner API. The
caller supplies only an Issue number and comment ID; no comment body crosses IPC.
"""
from __future__ import annotations

import logging
import dataclasses
import threading
import sqlite3
from pathlib import Path
from typing import Callable

from hermes_cli.issue_ingress import (
    AdmissionRefused,
    admit_github_comment,
    bind_next_pending_task,
    publish_control_result,
    publish_pending_task_result,
    record_control_result,
    issue22_task_is_active,
    validate_ingress_config,
)
from github_bridge.config import load_config
from github_bridge.auth import GitHubAppAuth
import json
import time

logger = logging.getLogger(__name__)
ISSUE22_VERB = "issue22-admit"
MAX_COMMENT_ID = (1 << 63) - 1


def build_issue22_admit_handler(
    *,
    connect_owner_db: Callable[[], sqlite3.Connection],
    config_path: Path,
    admit: Callable[..., object] = admit_github_comment,
    validate_config: Callable[[Path], object] = validate_ingress_config,
    publish_result: Callable[..., int] = publish_control_result,
    bind_task: Callable[..., dict | None] = bind_next_pending_task,
    record_result: Callable[..., None] = record_control_result,
    enabled: bool = False,
    issue_number: int = 22,
):
    """Build a GatewayControlServer ``verb(params)`` handler.

    ``connect_owner_db`` runs inside the Gateway control-socket worker thread.
    Production wiring must supply the Hermes-owned missions-board connector.
    """
    pinned_config = Path(config_path)
    owner_lock = threading.Lock()

    def _handle(params: dict) -> dict:
        if enabled is not True:
            return {"verdict": "REJECT", "reason": "INGRESS_DISABLED"}
        if (type(params) is not dict or set(params) != {"issue_number", "comment_id"}
                or type(params.get("issue_number")) is not int
                or params["issue_number"] != issue_number
                or type(params.get("comment_id")) is not int
                or not (0 < params["comment_id"] <= MAX_COMMENT_ID)):
            return {"verdict": "REJECT", "reason": "REQUEST_SCHEMA"}

        conn = None
        try:
            pins = validate_config(pinned_config)
            if getattr(pins, "issue_number", issue_number) != issue_number:
                raise AdmissionRefused("ISSUE_CONFIG_PIN_MISMATCH")
            conn = connect_owner_db()
            receipt = admit(conn, config_path=pinned_config, comment_id=params["comment_id"])
            if getattr(receipt, "verdict", None) != "ACK" or getattr(receipt, "kind", None) not in {"PING", "STATUS", "TASK"}:
                return {"verdict": "REJECT", "reason": "OWNER_RECEIPT_INVALID"}
            state = None
            if receipt.kind == "TASK":
                try:
                    dispatch = bind_task(conn, config_path=pinned_config)
                    state = dispatch.get("state") if isinstance(dispatch, dict) else None
                except AdmissionRefused as exc:
                    logger.info("Issue 22 TASK remains pending: %s", exc)
                if state not in {"queued", "awaiting_binding"}:
                    state = "awaiting_binding"
            result_comment_id = publish_result(
                pinned_config, comment_id=params["comment_id"], kind=receipt.kind,
                **({"state": state} if state is not None else {}),
            )
            if type(result_comment_id) is not int or result_comment_id <= 0:
                return {"verdict": "REJECT", "reason": "RESULT_UNCONFIRMED"}
            record_result(
                conn, repository_id=receipt.repository_id,
                issue_number=receipt.issue_number, nonce=receipt.nonce,
                result_comment_id=result_comment_id,
            )
            response = {"verdict": "ACK", "kind": receipt.kind, "issue_number": issue_number}
            if receipt.kind == "TASK":
                response["state"] = state
                if state == "queued" and isinstance(dispatch, dict) and dispatch.get("task_id"):
                    response["task_id"] = dispatch["task_id"]
                    response["session_id"] = conn.execute("SELECT session_id FROM tasks WHERE id=?", (dispatch["task_id"],)).fetchone()[0]
            return response
        except AdmissionRefused as exc:
            return {"verdict": "REJECT", "reason": str(exc)}
        except Exception:
            logger.exception("Issue 22 control-socket admission failed")
            return {"verdict": "REJECT", "reason": "OWNER_ERROR"}
        finally:
            if conn is not None:
                conn.close()

    def handle(params: dict) -> dict:
        if not owner_lock.acquire(blocking=False):
            return {"verdict": "REJECT", "reason": "INGRESS_BUSY"}
        try:
            return _handle(params)
        finally:
            owner_lock.release()

    return handle


def build_issue22_poll_handler(
    *, config_path: Path, enabled: bool = False, admit_handler=None,
    publish_task_result: Callable[[], dict | None] | None = None,
    task_active: Callable[[], bool] | None = None,
    issue_number: int = 22,
    is_processed: Callable[[int], bool] | None = None,
):
    """Find one fresh, exact-shape control or bounded task comment per poll.

    The comment body is parsed only as a bounded JSON selector. It never crosses
    the local socket; admission re-fetches and independently validates by ID.
    """
    pinned_config = Path(config_path)
    accept = admit_handler or build_issue22_admit_handler(
        config_path=pinned_config, enabled=enabled,
    )

    def handle(params):
        if enabled is not True:
            return {"verdict": "REJECT", "reason": "INGRESS_DISABLED"}
        if params not in (None, {}) or type(params) not in (dict, type(None)):
            return {"verdict": "REJECT", "reason": "REQUEST_SCHEMA"}
        try:
            if publish_task_result is not None:
                result = publish_task_result()
                if result is not None:
                    return result
            if task_active is not None and task_active():
                return {"verdict": "BUSY", "reason": "TASK_EXECUTOR_OCCUPIED", "issue_number": issue_number}
            cfg = load_config(pinned_config)
            if cfg.issue_ingress_issue_number != issue_number:
                raise AdmissionRefused("ISSUE_CONFIG_PIN_MISMATCH")
            cfg = dataclasses.replace(
                cfg, request_timeout=min(cfg.request_timeout, 5),
            )
            auth = GitHubAppAuth(cfg)
            token = auth.fetch_installation_token()
            if token.permissions.get("issues") not in {"read", "write"}:
                return {"verdict": "BLOCKED", "reason": "ISSUES_READ_NOT_GRANTED"}
            auth.get_token = lambda: token
            comments = auth.list_issue_comments(
                cfg.owner, cfg.repo, issue_number, per_page=100,
            )
            actor_id = cfg.issue_ingress_actor_id
            now = int(time.time())
            for row in comments if isinstance(comments, list) else []:
                if not isinstance(row, dict) or type(row.get("id")) is not int:
                    continue
                if is_processed is not None and is_processed(row["id"]):
                    continue
                user = row.get("user") or {}
                if (type(user) is not dict or user.get("id") != actor_id
                        or user.get("type") != "User"
                        or row.get("created_at") != row.get("updated_at")):
                    continue
                body = row.get("body")
                if not isinstance(body, str) or len(body.encode("utf-8")) > cfg.issue_ingress_max_body_bytes:
                    continue
                try:
                    command = json.loads(body)
                except (ValueError, TypeError):
                    continue
                if (type(command) is not dict
                        or command.get("kind") not in {"PING", "STATUS", "TASK"}
                        or type(command.get("expires_at")) is not int
                        or not now < command["expires_at"] <= now + 900):
                    continue
                expected = {"kind", "nonce", "expires_at"}
                if command["kind"] == "TASK":
                    expected.add("task")
                    task_text = command.get("task")
                    if (not isinstance(task_text, str) or not task_text.strip()
                            or len(task_text.encode("utf-8")) > cfg.issue_ingress_max_body_bytes - 128):
                        continue
                if set(command) != expected:
                    continue
                # Revalidation by the admission owner is authoritative.
                return accept({"issue_number": issue_number, "comment_id": row["id"]})
            return {"verdict": "NOOP", "issue_number": issue_number}
        except Exception:
            logger.exception("Issue 22 polling failed")
            return {"verdict": "REJECT", "reason": "POLL_ERROR"}

    return handle
