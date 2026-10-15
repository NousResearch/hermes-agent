"""Compression-attempt snapshots and ownership locks (facade-owned state)."""
from __future__ import annotations

import contextlib
import copy
import threading
from typing import Any


def _snapshot_compressor_attempt_state(compressor: Any) -> dict[str, Any]:
    """Copy only the mutable bookkeeping owned by one compression attempt.
    The allow-list avoids copying clients, DB handles, locks and plugin resources; missing fields are ignored
    so legacy/third-party compressors keep working."""
    from agent import conversation_compression as owner
    try:
        values = vars(compressor)
    except TypeError:
        return {}
    selected = {name: values[name] for name in owner._COMPRESSOR_ATTEMPT_STATE_FIELDS if name in values}
    # Copy the collection as one object so aliases between fields (notably
    # _active_compression_telemetry and _last_compression_telemetry) survive.
    return copy.deepcopy(selected)


def _summary_adoption_window() -> Any:
    """The calling attempt's fence adoption window; a no-op outside a fenced summary dispatch."""
    from agent import conversation_compression as owner
    window = getattr(owner._SUMMARY_ADOPTION_FENCE.get(), "adoption_window", None)
    return window() if callable(window) else contextlib.nullcontext()


def _compressor_attempt_serial_lock(compressor: Any) -> Any:
    """Per-compressor lock serializing the durable cooldown rollback against claims. The process-wide
    claim lock must stay cheap (every compressor in a gateway shares it), so the slow SQLite write in
    ``_restore_compressor_attempt_state`` is fenced by THIS lock instead; ``_claim_compressor_attempt``
    takes it first so a claim on that compressor waits for the restore while other compressors proceed.
    A slotted/frozen compressor that cannot hold the attribute gets a no-op (its guard is off anyway)."""
    from agent import conversation_compression as owner
    with owner._COMPRESSOR_ATTEMPT_LOCK:
        lock = getattr(compressor, "_compression_attempt_serial_lock", None)
        if lock is None:
            lock = threading.Lock()
            try:
                compressor._compression_attempt_serial_lock = lock
            except Exception:
                return contextlib.nullcontext()
        return lock


def _working_attempt_is_current(compressor: Any, generation: Any) -> bool:
    """True when *generation* is still the last attempt that began summary work.

    Without a published marker (attribute-less compressor, or the attempt never reached
    dispatch) supersession falls back to the entry-generation ownership check; a compressor
    that no attempt has ever claimed cannot have been superseded."""
    from agent import conversation_compression as owner
    if not generation:
        return True
    with owner._COMPRESSOR_ATTEMPT_LOCK:
        marker = getattr(compressor, "_compression_working_attempt_generation", None)
        if marker is None:
            entry_generation = int(getattr(compressor, "_compression_attempt_generation", 0) or 0)
            return not entry_generation or entry_generation == generation
        return int(marker) == int(generation)
