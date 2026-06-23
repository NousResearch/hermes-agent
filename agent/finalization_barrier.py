"""Finalization barrier receipts for mutation-boundary claims."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover
    def get_hermes_home() -> Path:  # type: ignore
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

RECEIPT_SCHEMA = "loaw.finalization-barrier-receipt.v1"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _receipt_dir() -> Path:
    return Path(get_hermes_home()).expanduser() / "finalization-receipts"


def _write(kind: str, payload: Dict[str, Any]) -> Path:
    payload = dict(payload)
    payload.setdefault("schema", RECEIPT_SCHEMA)
    payload.setdefault("kind", kind)
    payload.setdefault("created_at", _utc())
    payload.setdefault("receipt_id", f"{kind}-{uuid.uuid4().hex[:12]}")
    d = _receipt_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{payload['receipt_id']}.json"
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


def hash_protected_files(paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists() or not p.is_file():
            out[str(p)] = {"exists": False}
            continue
        out[str(p)] = {"exists": True, "sha256": _sha(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
    return out


def changed_paths(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[str]:
    keys = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))


def wait_for_background_review(timeout: float = 30.0) -> Dict[str, Any]:
    joined = []
    deadline = time.time() + timeout
    for t in list(threading.enumerate()):
        if t is threading.current_thread():
            continue
        if t.name == "bg-review":
            remaining = max(0.0, deadline - time.time())
            t.join(remaining)
            joined.append({"name": t.name, "alive_after_join": t.is_alive()})
    return {"threads": joined, "all_completed": not any(x["alive_after_join"] for x in joined)}


def run_finalization_barrier(
    *,
    protected_paths: Optional[Iterable[str]] = None,
    authorized_changed_paths: Optional[Iterable[str]] = None,
    before_hashes: Optional[Dict[str, Dict[str, Any]]] = None,
    quiescence_seconds: float = 0.25,
    wait_timeout: float = 30.0,
) -> Dict[str, Any]:
    protected_paths = list(protected_paths or [])
    authorized = sorted(set(authorized_changed_paths or []))
    before = before_hashes or hash_protected_files(protected_paths)
    pending = _write("finalization_pending", {"protected_path_count": len(protected_paths)})
    post_hook_state = wait_for_background_review(wait_timeout)
    hook_receipt = _write("post_hook_state", post_hook_state)
    time.sleep(quiescence_seconds)
    after_quiet = hash_protected_files(protected_paths)
    quiet_receipt = _write("post_quiescence_sentinel", {"protected_hashes": after_quiet})
    changed = changed_paths(before, after_quiet)
    unauthorized = sorted(set(changed) - set(authorized))
    unreceipted = unauthorized  # receipts can be wired by callers; unauthorized is blocking by default.
    reconciliation = {
        "authorized_changed_paths": authorized,
        "actual_changed_paths": changed,
        "unauthorized_changed_paths": unauthorized,
        "unreceipted_changed_paths": unreceipted,
        "gate": "green" if post_hook_state["all_completed"] and not unauthorized and not unreceipted else "red",
    }
    recon_receipt = _write("changed_path_reconciliation", reconciliation)
    # Any write after this receipt would create a mismatch in a later barrier call;
    # callers that detect it must emit incident/corrected-gate receipts.
    complete = dict(reconciliation)
    complete.update({
        "finalization_pending_receipt": str(pending),
        "post_hook_state_receipt": str(hook_receipt),
        "post_quiescence_sentinel_receipt": str(quiet_receipt),
        "changed_path_reconciliation_receipt": str(recon_receipt),
    })
    complete_receipt = _write("finalization_complete", complete)
    complete["finalization_complete_receipt"] = str(complete_receipt)
    return complete
