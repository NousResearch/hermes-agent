"""W1 / F-010: typed error for transactional-boundary violations in
`AIAgent._compress_context`.

Pre-refactor, `_compress_context` mutated `self.session_id`,
`self._cached_system_prompt`, and the SQLite `sessions` row non-atomically.
A failure mid-sequence (SQLite lock, disk full, title-lookup exception)
left the agent with a cached prompt mismatched to the persisted session.
The caller was told to `/new`, but the DB already carried a compressed
system prompt against uncompressed history; the next resume replayed the
mismatch.

The compression path now captures a snapshot of the agent's in-memory
state before any mutation and restores it on any known-failure raise
below. The failure surfaces as `CompressionFailed` so the outer loop can
emit the existing "try /new" message cleanly — without publishing the
partial state.
"""
from __future__ import annotations


class CompressionFailed(Exception):
    """Raised from `_compress_context` when the transaction cannot complete.

    Attributes:
        stage: short identifier for where the failure occurred — one of
            ``"no_progress"``, ``"compress"``, ``"prompt_rebuild"``,
            ``"db_split"``, ``"state_carry"``, ``"memory_flush"``,
            or ``"unknown"``.
        original_message_count: number of messages in the pre-compression
            history, preserved for logging / UX messaging.
    """

    def __init__(self, stage: str, original_message_count: int, reason: str = ""):
        self.stage = stage
        self.original_message_count = original_message_count
        self.reason = reason
        suffix = f": {reason}" if reason else ""
        super().__init__(f"compression failed at stage={stage}{suffix}")
