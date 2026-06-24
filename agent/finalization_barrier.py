"""Finalization barrier receipts for mutation-boundary claims."""
from __future__ import annotations

import hashlib
import json
import os
import glob
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

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


def _canonical(path: Path) -> str:
    """Return a symlink-safe canonical path string without requiring existence."""
    try:
        return str(path.expanduser().resolve(strict=False))
    except Exception:
        return str(path.expanduser())


def default_protected_paths() -> List[str]:
    """Discover active personalization/rule/config paths that must be watched.

    The set is deliberately derived from the live Hermes home at agent startup,
    not from a static fixture, so a real invocation cannot accidentally get a
    green barrier while watching zero active files.  Missing optional paths are
    ignored here; unreadable or deleted paths after startup are caught by the
    before/after reconciliation below.
    """
    home = Path(get_hermes_home()).expanduser()
    candidates: Set[str] = set()
    patterns = [
        home / "skills" / "**" / "SKILL.md",
        home / "skills" / "**" / "references" / "**" / "*",
        home / "skills" / "**" / "templates" / "**" / "*",
        home / "skills" / "**" / "scripts" / "**" / "*",
        home / "profiles" / "**" / "*",
        home / "memories" / "**" / "*",
        home / "memory*",
        home / "*.yaml",
        home / "*.yml",
        home / "*.json",
        Path.home() / ".cursor" / "mcp.json",
        Path.home() / ".openclaw" / "config.yaml",
        Path.home() / ".openclaw" / "mcp.json",
        Path.home() / ".openclaw" / "profiles" / "**" / "*",
        Path.home() / ".openclaw" / "workspace-arta-personal" / "registries" / "MOSES-SINGLE-SOURCE-OF-TRUTH.md",
    ]
    for pat in patterns:
        for raw in glob.glob(str(pat), recursive=True):
            p = Path(raw)
            if p.is_file() and not p.is_symlink():
                candidates.add(_canonical(p))
    return sorted(candidates)


def initialize_protected_finalization_state(agent: Any, *, paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Attach fail-closed finalization watch state to an agent instance."""
    protected = sorted({_canonical(Path(p)) for p in (paths or default_protected_paths())})
    before = hash_protected_files(protected)
    agent._protected_finalization_paths = protected
    agent._protected_finalization_before_hashes = before
    agent._authorized_finalization_changed_paths = []
    incomplete = sorted(p for p, rec in before.items() if not rec.get("exists") or rec.get("error"))
    return {
        "protected_path_count": len(protected),
        "before_hash_count": len(before),
        "incomplete_paths": incomplete,
        "gate": "green" if protected and len(before) == len(protected) and not incomplete else "red",
    }


def hash_protected_files(paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw in paths:
        p = Path(raw).expanduser()
        key = _canonical(p)
        if p.is_symlink():
            out[key] = {"exists": True, "error": "symlink_escape_forbidden", "is_symlink": True}
            continue
        if not p.exists() or not p.is_file():
            out[key] = {"exists": False}
            continue
        try:
            out[key] = {"exists": True, "sha256": _sha(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        except OSError as exc:
            out[key] = {"exists": True, "error": str(exc)}
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
    post_hooks_completed: bool = True,
) -> Dict[str, Any]:
    protected_paths = sorted({_canonical(Path(p)) for p in (protected_paths or [])})
    authorized = sorted({_canonical(Path(p)) for p in (authorized_changed_paths or [])})
    before = before_hashes or {}
    pending = _write("finalization_pending", {"protected_path_count": len(protected_paths)})
    post_hook_state = wait_for_background_review(wait_timeout)
    hook_receipt = _write("post_hook_state", post_hook_state)
    time.sleep(quiescence_seconds)
    after_quiet = hash_protected_files(protected_paths)
    quiet_receipt = _write("post_quiescence_sentinel", {"protected_hashes": after_quiet})
    changed = changed_paths(before, after_quiet)
    missing_before = sorted(set(protected_paths) - set(before))
    missing_after = sorted(set(protected_paths) - set(after_quiet))
    unreadable = sorted(
        p for p in protected_paths
        if before.get(p, {}).get("error") or after_quiet.get(p, {}).get("error")
        or before.get(p, {}).get("exists") is False or after_quiet.get(p, {}).get("exists") is False
    )
    unauthorized = sorted(set(changed) - set(authorized))
    unreceipted = unauthorized  # receipts can be wired by callers; unauthorized is blocking by default.
    fail_closed_reasons = []
    if not protected_paths:
        fail_closed_reasons.append("empty_protected_path_set")
    if not before:
        fail_closed_reasons.append("empty_before_hash_map")
    if missing_before:
        fail_closed_reasons.append("incomplete_before_hash_map")
    if missing_after:
        fail_closed_reasons.append("incomplete_after_hash_map")
    if unreadable:
        fail_closed_reasons.append("unreadable_or_missing_protected_path")
    if not post_hook_state["all_completed"]:
        fail_closed_reasons.append("background_thread_alive")
    if not post_hooks_completed:
        fail_closed_reasons.append("post_run_hook_incomplete")
    if unauthorized:
        fail_closed_reasons.append("unauthorized_changed_path")
    if unreceipted:
        fail_closed_reasons.append("unreceipted_write")
    reconciliation = {
        "authorized_changed_paths": authorized,
        "actual_changed_paths": changed,
        "unauthorized_changed_paths": unauthorized,
        "unreceipted_changed_paths": unreceipted,
        "missing_before_hash_paths": missing_before,
        "missing_after_hash_paths": missing_after,
        "unreadable_or_missing_paths": unreadable,
        "fail_closed_reasons": fail_closed_reasons,
        "post_hooks_completed": post_hooks_completed,
        "gate": "green" if not fail_closed_reasons else "red",
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
