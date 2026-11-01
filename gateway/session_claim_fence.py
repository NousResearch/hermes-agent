"""The claim that starts a queued turn, behind the process's update/retirement fence.

An in-app update proves the backend idle and holds ``hermes_cli.backend_retirement``'s permit
until the hand-off stops this process. Messaging, API and Desktop turns all start here, so a claim
that cannot enter the fence leaves its head queued (waiters keep waiting): the drain retries until
the permit ends (cancelled, expired) and runs it, or the restarted owner drains it. The fence is
held across the claim until its execution stamp is set, so the idle proof that grants a permit
either sees this claim's admission or its running turn — never neither.
"""

from functools import partial

UPDATE_HOLD = 'update_hold'


async def claim_head(authority, ref, live, first):
    """``(row, first)`` as ``SessionAuthority._claim_next`` answers it, or ``(UPDATE_HOLD, first)``."""
    from hermes_cli.backend_retirement import retirement
    from gateway.session_authority import claim_session_input  # the facade is the patch seam
    from gateway.session_runtime_workers import tracked_write

    with retirement.work() as admitted:
        if not admitted:
            return UPDATE_HOLD, first
        authority._require_admission_open()
        live.rescan = False  # a _schedule from here on may not be visible to this claim's read
        # The row these checks validated, never whichever is first when the write runs: a
        # cancellation commits from a worker thread and can advance the FIFO at any point.
        row = await tracked_write(authority, partial(claim_session_input, authority.db, epoch=authority.epoch,
                                                     session_id=ref.session_id, _guard=authority._admission_gate(),
                                                     head=first['admission_id'] if first is not None else None),
                                  then=partial(authority._stamp_claim, ref, live))
    # Input scheduled while this claim's transaction ran may postdate its read: re-read
    # instead of letting the drain exit idle over a committed admission.
    return (False if row is None and live.rescan else row), first
