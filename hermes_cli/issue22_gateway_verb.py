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
    publish_control_result,
    record_control_result,
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
    record_result: Callable[..., None] = record_control_result,
    enabled: bool = False,
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
                or params["issue_number"] != 22
                or type(params.get("comment_id")) is not int
                or not (0 < params["comment_id"] <= MAX_COMMENT_ID)):
            return {"verdict": "REJECT", "reason": "REQUEST_SCHEMA"}

        conn = None
        try:
            validate_config(pinned_config)
            conn = connect_owner_db()
            receipt = admit(conn, config_path=pinned_config, comment_id=params["comment_id"])
            if getattr(receipt, "verdict", None) != "ACK" or getattr(receipt, "kind", None) not in {"PING", "STATUS", "TASK"}:
                return {"verdict": "REJECT", "reason": "OWNER_RECEIPT_INVALID"}
            result_comment_id = publish_result(
                pinned_config, comment_id=params["comment_id"], kind=receipt.kind,
            )
            if type(result_comment_id) is not int or result_comment_id <= 0:
                return {"verdict": "REJECT", "reason": "RESULT_UNCONFIRMED"}
            record_result(
                conn, repository_id=receipt.repository_id,
                issue_number=receipt.issue_number, nonce=receipt.nonce,
                result_comment_id=result_comment_id,
            )
            response = {"verdict": "ACK", "kind": receipt.kind, "issue_number": 22}
            if receipt.kind == "TASK":
                response["state"] = "awaiting_binding"
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


def build_issue22_poll_handler(*, config_path: Path, enabled: bool = False,
                               admit_handler=None):
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
            cfg = load_config(pinned_config)
            cfg = dataclasses.replace(
                cfg, request_timeout=min(cfg.request_timeout, 5),
            )
            auth = GitHubAppAuth(cfg)
            token = auth.fetch_installation_token()
            if token.permissions.get("issues") not in {"read", "write"}:
                return {"verdict": "BLOCKED", "reason": "ISSUES_READ_NOT_GRANTED"}
            auth.get_token = lambda: token
            comments = auth.list_issue_comments(
                cfg.owner, cfg.repo, 22, per_page=100,
            )
            actor_id = cfg.issue_ingress_actor_id
            now = int(time.time())
            for row in comments if isinstance(comments, list) else []:
                if not isinstance(row, dict) or type(row.get("id")) is not int:
                    continue
                user = row.get("user") or {}
                if (type(user) is not dict or user.get("id") != actor_id
                        or user.get("type") != "User"
                        or row.get("created_at") != row.get("updated_at")):
                    continue
                body = row.get("body")
                if not isinstance(body, str) or len(body.encode("utf-8")) > 1024:
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
                            or len(task_text.encode("utf-8")) > 896):
                        continue
                if set(command) != expected:
                    continue
                # Revalidation by the admission owner is authoritative.
                return accept({"issue_number": 22, "comment_id": row["id"]})
            return {"verdict": "NOOP", "issue_number": 22}
        except Exception:
            logger.exception("Issue 22 polling failed")
            return {"verdict": "REJECT", "reason": "POLL_ERROR"}

    return handle
