"""Wave-2 cell stubs — contract clauses named, implementations interlocked.

These are deliberate ``pytest.skip`` placeholders, not TODOs: each names the
exact contract clause it will pin and the reason it is not implemented in the
skeleton PR. See the tracking issue (#80921) for the full ~39-cell matrix.
"""

from __future__ import annotations

import pytest


def test_cell4_fork_determinism_on_rewind_stub():
    """Cell 4 — fork determinism on the edit/rewind paths.

    Contract clause: after an edit/rewind fork, recovery yields exactly the
    chosen prefix (no resurrection of superseded turns, no archived-copy
    duplication), deterministically across independent recovery passes.

    Deliberately deferred: the rewind/archive semantics this cell would pin
    are being actively redesigned in #82956–#82959 (durable row-id
    addressing, archive-mode dedup, /retry archive_drop plumbing). Writing
    the cell against today's behavior would test a moving target; it lands
    as that cluster's conformance check once the contract settles.
    """
    pytest.skip(
        "wave 2: interlocked with the rewind/archive redesign "
        "(#82956-#82959) — cell lands as that cluster's conformance check"
    )


def test_cell5_delivery_outbox_exactly_once_stub():
    """Cell 5 — effect exactly-once across the delivery outbox boundary.

    Contract clause: a crash between provider send and durable record must
    not double-deliver on catch-up (cron ticker restart, gateway reboot) —
    the cell the paper (arXiv:2608.03836) probes as 'effect exactly-once',
    distinct from cell 2's consume-once (claim vs side-effect).

    Deliberately deferred: requires a deterministic fake-transport seam for
    the delivery path (the send must be observable without a live adapter);
    the cron delivery-scope interplay is also in flight (#83197/#83557).
    """
    pytest.skip(
        "wave 2: needs a deterministic fake-transport seam; cron delivery "
        "scope fixes in flight (#83197/#83557)"
    )
