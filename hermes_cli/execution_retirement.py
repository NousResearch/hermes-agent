"""Redo receipts for physically settled execution references, never timed eviction."""

import json
import logging
import threading

from hermes_cli import active_sessions as registry

logger = logging.getLogger(__name__)
_pending = {}
_pending_lock = threading.Lock()
_retry_thread = None
_retry_delay = threading.Event()


def _receipt_path(state_path, lease_id):
    return state_path.parent / "execution_retirements" / f"{lease_id}.json"


def record_settlement(permit):
    """Persist exact ownership before a possibly contended registry release."""
    registry.atomic_json_write(
        _receipt_path(permit.lease.state_path, permit.lease.lease_id), permit.identity,
        indent=None, sort_keys=True)


def reconcile_locked(state_path, entries):
    """Caller holds the budget lock. Unprovable receipts leave capacity fail closed."""
    receipts = list((state_path.parent / "execution_retirements").glob("*.json"))
    by_id = {entry["lease_id"]: entry for entry in entries}
    retired = set()
    for path in receipts:
        receipt = json.loads(path.read_text(encoding="utf-8-sig"))
        lease_id = path.stem
        if not isinstance(receipt, dict) or receipt.get("lease_id") != lease_id:
            raise registry.ActiveSessionRegistryError("invalid execution retirement receipt")
        entry = by_id.get(lease_id)
        if entry is not None and entry != receipt:
            raise registry.ActiveSessionRegistryError("execution retirement ownership mismatch")
        if entry is not None:
            retired.add(lease_id)
    kept = [entry for entry in entries if entry["lease_id"] not in retired]
    if retired:
        registry._write_entries(state_path, kept)
    for path in receipts:
        path.unlink(missing_ok=True)
    return kept, len(retired)


def _retry_settlements():
    global _retry_thread
    while True:
        with _pending_lock:
            if not _pending:
                _retry_thread = None
                return
            pending = list(_pending.items())
        for key, permit in pending:
            try:
                permit.release()
            except (OSError, RuntimeError, ValueError):
                continue
            with _pending_lock:
                _pending.pop(key, None)
        # No provider work or ambient profile reads; pinned references survive a
        # temporary lock/filesystem failure while this owner process remains live.
        _retry_delay.wait(1.0)


def retry_settlement(permit):
    """Keep retrying only a caller-declared physically completed reference."""
    from agent.memory_provider import spawn_context_thread

    global _retry_thread
    key = (str(permit.lease.state_path), permit.lease.lease_id)
    with _pending_lock:
        _pending[key] = permit
        if _retry_thread is None:
            _retry_thread = spawn_context_thread(_retry_settlements, name="execution-settlement")
            _retry_thread.start()
    logger.warning("Completed execution reference queued for automatic settlement recovery")
