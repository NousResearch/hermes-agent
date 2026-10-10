"""Carry verified completion ownership across asynchronous adapter admission.

The durable acknowledgement still means queued, not model execution. Prove the owner before
that acknowledgement so a transient ownership lookup cannot silently consume the completion.
"""

from dataclasses import dataclass
import logging


logger = logging.getLogger("gateway.run")


@dataclass(frozen=True)
class CompletionOwnerReceipt:
    session_key: str
    session_id: str
    pinned_session_id: str


async def prepare_completion_owner(runner, event) -> bool:
    """Resolve a pinned push completion before scheduling it; storage errors propagate."""
    metadata = event.metadata or {}
    pin = str(metadata.get("gateway_session_id") or "").strip()
    key = str(metadata.get("gateway_session_key") or "").strip()
    if not pin:
        return False
    if not key or runner._session_key_for_source(event.source) != key:
        return False
    entry = await runner.async_session_store.lookup_by_session_key(key)
    if entry is None and not getattr(runner.session_store, "_routing_db_loaded", True):
        raise RuntimeError("Completion routing database unavailable")
    if entry is None or entry.suspended:
        return False
    # A legacy route can itself name a delegate. The child completion must repair its
    # proven coordinator before ancestry resolution; otherwise it is terminally dropped
    # merely because no human inbound has repaired the old route yet.
    entry = await runner.async_session_store._reconcile_poisoned_delegate_route(
        key, entry, event.source, quarantine_invalid=False,
    )
    if entry is None:
        return False
    resolved = await runner._resolve_async_delegation_session(entry, pin, raise_lookup_errors=True)
    if resolved is None:
        return False
    event._completion_owner_receipt = CompletionOwnerReceipt(key, resolved.session_id, pin)
    return True


async def resolve_prepared_completion_owner(runner, event, receipt):
    """Reuse the proof for an unchanged route, and preserve queued compression continuations.

Only a changed route needs another ownership lookup. If that lookup is unavailable, the
already-admitted event returns to the adapter's existing FIFO/backoff path, never a human
message merge slot. This does not wait for the previous turn or mark model execution complete.
"""
    try:
        current = await runner.async_session_store.lookup_by_session_key(receipt.session_key)
        if current is None and not getattr(runner.session_store, "_routing_db_loaded", True):
            raise RuntimeError("Completion routing database unavailable")
        if current is None or current.suspended:
            return None
        if current.session_id == receipt.session_id:
            return current
        resolved = await runner._resolve_async_delegation_session(
            current, receipt.session_id, raise_lookup_errors=True,
        )
    except Exception:
        adapter = runner._delivery_adapter_for(event.source)
        if adapter is None:
            raise
        runner._enqueue_fifo(receipt.session_key, event, adapter)
        logger.debug("Deferring admitted completion while owner lookup is unavailable for %s",
                     receipt.session_key, exc_info=True)
        return None
    if resolved is not None:
        event._completion_owner_receipt = CompletionOwnerReceipt(
            receipt.session_key, resolved.session_id, receipt.pinned_session_id,
        )
    return resolved
