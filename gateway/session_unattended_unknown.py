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

A person's lost turn pauses the FIFO, but only while nobody has typed past it: their own later
input is the signal that the turn will not be replayed (its side effects are unprovable either
way), so it settles the same way and their newer input runs instead of waiting behind a turn that
can never resume. An ``unknown`` row with no follower is still left for the person to resolve.
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


def _superseded_unknown(records):
    """Attended ``unknown`` rows a LATER attended input of the same session already answers.

    Only the person who sent a lost turn may decide to send it again, so an ``unknown`` row with
    nothing behind it stays theirs to resolve. A row with a person's own input queued after it is
    already answered by that input: they typed past it, and holding the FIFO in front of the newer
    message only costs them a reply. An unattended follower (a cron run, a process completion) is
    not that signal and never ends a person's pause."""
    from gateway.session_authority import admission_row
    rows = [admission_row(record) for record in records]
    followers = [row['seq'] for row in rows if row['status'] == 'queued' and not is_unattended(row)]
    if not followers:
        return []
    newest = max(followers)
    return [row for row in rows
            if row['status'] == 'unknown' and row['seq'] < newest and not is_unattended(row)]


async def settle_superseded_unknown(authority, ref) -> bool:
    """The drain found an ``unknown`` head this session's own newer input already supersedes: settle
    it through the owner's fenced resolution, the transition ``/discard`` commits, so the input the
    person sent after the lost turn runs instead. True when any settled, so the drain re-reads its
    FIFO instead of pausing."""
    import asyncio

    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import AuthorityRefusal
    rows = await asyncio.to_thread(authority_persistence(authority).admissions.list, session_id=ref.session_id)
    settled = False
    for row in _superseded_unknown(rows):
        try:
            # No ``notice``: a person gets the store's discarded-turn copy (this turn did not
            # complete, verify the effects before resending), and nothing restarted on this path.
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'])
        except AuthorityRefusal as exc:
            logger.warning('Superseded admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        settled = True
    return settled
