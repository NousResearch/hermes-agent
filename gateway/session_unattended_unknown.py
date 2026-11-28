"""An ``unknown`` turn nobody is waiting on settles as interrupted (#134889 GW-10, #139532).

A turn lost across an owner restart is fenced ``unknown`` and pauses its session's FIFO, because
only the person who sent it knows whether to resend it. An automation delivery has no such person:
a cron run (``CRON_PRINCIPAL_ID``), a process completion or heartbeat (``automation:<route>``,
``local_automation_v1``, a native envelope's ``automation`` block) or a webhook delivery
(``webhook_delivery``) or a Kanban card attempt (``kanban_card_v1``, set by the dispatcher). Pausing on it stalls every message queued behind it until an operator
notices (a Bot Chat sat ten hours behind one).

Such a row settles through the same fenced resolution a person's Discard uses
(``admissions.resolve_unknown``): terminal ``interrupted``, never requeued or replayed, with a
Hermes transcript notice committed in the same transaction. No second recovery store.
"""
from __future__ import annotations

import logging

from persistence.authority.vocabulary import CRON_PRINCIPAL_ID

logger = logging.getLogger(__name__)


def is_unattended(row) -> bool:
    """True for an admission no person submitted (so nobody can be asked to resend it)."""
    principal = row.get('principal_id') or ''
    if principal == CRON_PRINCIPAL_ID or principal.startswith('automation:'):
        return True
    payload = row.get('payload') or {}
    native = payload.get('native_text_v1') or {}
    # A webhook delivery: the provider already got its 202 and nobody can be asked to resend it.
    # A Kanban card attempt (#134889 KB-8): the dispatcher's marker, committed only by
    # ``SessionAuthority.submit(_kanban_card=...)``; a card row main minted without it still pauses.
    return ('local_automation_v1' in payload or 'kanban_card_v1' in payload
            or 'automation' in native or 'webhook_delivery' in native)


def _unattended_unknown(records):
    from gateway.session_authority import admission_row
    return [row for row in map(admission_row, records) if row['status'] == 'unknown' and is_unattended(row)]


def settle_recovered_unattended(store, epoch) -> int:
    """Boot recovery, right after ``admissions.recover`` fenced lost turns ``unknown`` and before
    any FIFO is scheduled: settle every unattended one. Synchronous; runs off the owner loop.
    *store* is the profile's storage interface handle."""
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from gateway.session_ingress_media import release_admission_media
    from persistence.core.errors import AuthorityRefusal
    settled = 0
    for row in _unattended_unknown(store.admissions.list_unknown()):
        try:
            store.admissions.resolve_unknown(epoch=epoch, admission_id=row['admission_id'],
                                             generation=row['generation'], notice=AUTOMATION_INTERRUPTED_NOTICE)
        except AuthorityRefusal as exc:
            logger.warning('Lost automation admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        release_admission_media(store, row['admission_id'])
        settled += 1
        logger.info('Lost automation admission %s on session %s settled interrupted (not replayed)',
                    row['admission_id'], row['target_session_id'])
    return settled


async def settle_unattended_unknown(authority, ref) -> bool:
    """The drain found an ``unknown`` head (fenced in this owner, e.g. a lost managed worker):
    settle the unattended ones through the owner's resolution. True when any settled, so the
    drain re-reads its FIFO instead of pausing; a person's ``unknown`` row still pauses it."""
    import asyncio

    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import AuthorityRefusal
    rows = await asyncio.to_thread(authority_persistence(authority).admissions.list, session_id=ref.session_id)
    settled = False
    for row in _unattended_unknown(rows):
        try:
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'],
                                                     notice=AUTOMATION_INTERRUPTED_NOTICE)
        except AuthorityRefusal as exc:
            logger.warning('Lost automation admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        settled = True
    return settled


async def settle_dead_owner_unknown(authority, ref) -> bool:
    """Settle ``unknown`` admissions no process can still complete, a person's included.

    A person's lost turn pauses the FIFO so they can choose to resend it (``/discard``), but that
    choice presumes somebody can still complete the fenced turn. On a messaging surface the
    pause notice may never be seen: a Discord thread sat a full day behind one dead worker
    (Hermes 0.23.x, 2026-11-21) with every later message queued behind it. The transcript's write
    guards make a live turn hold a live turn lease, so when the conversation has lease rows and
    every one of them is expired or names a provably dead pid, nobody can complete the fenced
    turn: settle it through the same resolution a Discard uses — terminal ``interrupted``, never
    replayed — with the person's failed-turn notice, and the queued followers run. No lease rows
    is no proof (the owner may simply never have reached its lease write), so the pause keeps its
    meaning; any live (or unprobeable) lease keeps the pause unchanged.
    """
    import asyncio
    import time

    from agent.turn_failure_copy import FAILED_TURN_NOTICE
    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import AuthorityRefusal
    from process_identity import holder_process_is_dead

    store = authority_persistence(authority)
    rows = await asyncio.to_thread(store.admissions.list, session_id=ref.session_id)
    from gateway.session_authority import admission_row
    unknown = [row for row in map(admission_row, rows) if row['status'] == 'unknown']
    if not unknown:
        return False
    leases = await asyncio.to_thread(store.coordination.session_turn_leases, session_id=ref.session_id)
    if not leases:
        return False  # no lease rows: no proof the owner is gone, the person's pause keeps its meaning
    now = time.time()
    if any(float(expires_at) > now and not holder_process_is_dead(holder)
           for holder, expires_at in leases):
        return False  # a live owner may still complete or explicitly resolve the turn
    settled = False
    for row in unknown:
        try:
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'],
                                                     notice=FAILED_TURN_NOTICE)
        except AuthorityRefusal as exc:
            logger.warning('Dead-owner admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        settled = True
        logger.info("Lost admission %s on session %s settled interrupted (owner gone; not replayed)",
                    row['admission_id'], ref.session_id)
    return settled
