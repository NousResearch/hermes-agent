"""Backend capability declaration.

Backends advertise facts about themselves; application code branches on these declared facts, never on a backend's name. Each True flag carries a behavioral floor defined by the owning capability contract (for search: what query forms and visibility rules must hold), not a promise of identical scores or internals.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PersistenceCapabilities"]


@dataclass(frozen=True, slots=True)
class PersistenceCapabilities:
    """Declared facts about one assembled backend. No defaults: every assembly states its capabilities explicitly."""

    full_text_search: bool
    substring_search: bool
    transactional_claims: bool
    cross_process: bool
    remote: bool
    reopen_durability: bool
    """Committed writes survive closing and reopening the assembled backend over the same store. Declaring backends run the reopen-durability contract cases through ``reopen()``; non-declaring backends (the in-memory reference) skip them via the capability gate and refuse ``reopen()`` with :class:`persistence.CapabilityUnsupported`."""
    authority_fences: bool
    """The backend enforces authority epoch, session revision, and execution generation fences with the validation order defined by each authority operation, refusing stale fences with :class:`persistence.FenceStale`."""
    durable_admissions: bool
    """The backend persists the fail-closed admission lifecycle (admit/claim/settle/cancel/recover/resolve-unknown with digest dedup, per-session FIFO, and unknown barriers)."""
    receipted_writes: bool
    """Composite mutations commit their retry receipt in the same atomic transition as the domain write (worker sequenced writes, atomic bindings); an exact duplicate replays its stored result instead of re-executing."""
    physical_admin: bool
    """The assembly offers this backend's physical admin capability (doctor, journal mode, schema repair, index optimize, VACUUM, physical backup and restore); a backend without it answers the admin doors with an unsupported result."""
