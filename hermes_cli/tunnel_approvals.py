"""JSONL hold-request / approval store for ``hermes tunnel`` hold-open.

Records live at ``~/.hermes/tunnel/hold_requests.jsonl`` (append-only).
Admin-gated approve/deny: a caller whose identity is not in ``admin_ids``
gets ``PermissionError``. Status transitions are validated
(``pending -> approved|denied`` only); any other transition raises
``ValueError``. Unknown ids raise ``KeyError``.
"""

from __future__ import annotations

import json
import os
import time
import uuid


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _read_all(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return out


def _write_all(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def file_request(path, *, user, subdomains, reason, requested_until) -> str:
    rid = new_id()
    rec = {"id": rid, "user": user, "subdomains": list(subdomains), "reason": reason,
           "requested_until": requested_until, "status": "pending",
           "approved_until": None, "decided_by": None,
           "created_at": time.time(), "decided_at": None}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rid


def _find(path, rid) -> tuple[list[dict], int]:
    records = _read_all(path)
    for i, r in enumerate(records):
        if r["id"] == rid:
            return records, i
    raise KeyError(rid)


def get(path, rid) -> dict | None:
    try:
        records, i = _find(path, rid)
    except KeyError:
        return None
    return records[i]


def list_pending(path) -> list[dict]:
    return [r for r in _read_all(path) if r["status"] == "pending"]


def _require_admin(by, admin_ids) -> None:
    if by not in admin_ids:
        raise PermissionError(f"{by!r} is not a tunnel admin")


def _resolve(path, rid, *, new_status, by, admin_ids, approved_until=None, reason=None) -> dict:
    _require_admin(by, admin_ids)
    records, i = _find(path, rid)
    rec = records[i]
    if rec["status"] != "pending":
        raise ValueError(f"hold request {rid} already {rec['status']}")
    rec["status"] = new_status
    rec["decided_by"] = by
    rec["decided_at"] = time.time()
    if new_status == "approved":
        rec["approved_until"] = approved_until
    elif new_status == "denied":
        rec["deny_reason"] = reason
    records[i] = rec
    _write_all(path, records)
    return rec


def approve(path, rid, *, until, by, admin_ids) -> dict:
    return _resolve(path, rid, new_status="approved", by=by, admin_ids=admin_ids, approved_until=until)


def deny(path, rid, *, reason, by, admin_ids) -> dict:
    return _resolve(path, rid, new_status="denied", by=by, admin_ids=admin_ids, reason=reason)


def is_approved(path, rid) -> bool:
    rec = get(path, rid)
    return bool(rec and rec["status"] == "approved")


def approved_until(path, rid):
    rec = get(path, rid)
    return rec["approved_until"] if rec and rec["status"] == "approved" else None
