"""Shared timeout budgets for ``hermes gateway migrate --backends``.

Leaf module imported by the plan, host, and apply engine so all three read the same timeout
budgets without cyclic imports.
"""

from __future__ import annotations

#: Upper bound for the new gateway to answer on every address. A good run passes as soon as it answers
#: (the whole outage was 3.4 to 8.5 s on the Linux and macOS test guests); only a failing run waits the
#: whole budget, with the old address dark, before it rolls back. It is generous because this is the
#: floor for the whole boot: a busy multi-profile home prints its startup banner 56-92 s before it
#: finishes, and a first start on a small ARM board or a cold disk is slower still.
VERIFY_SECONDS = 300.0
#: Upper bound for ``start_gateway`` to hand over to a replacement process (see ``_await_replacement``).
#: Advisory: past it the verify step still decides, so a slow hand-off costs time, never correctness.
RESTART_HANDOFF_SECONDS = 240.0
#: Upper bound for the old backend to answer again during rollback or ``--restore``.
ROLLBACK_VERIFY_SECONDS = 60.0
