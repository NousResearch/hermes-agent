"""In-process completion ancestry for the existing admission FIFO.

The producer's private registry holds the reservation. Durable completion IDs are
delivery identities only; replay after owner restart carries no consent evidence.
"""
from contextlib import contextmanager, nullcontext
import json


def _producer_reservation_for_event(event, route):
    """Read a live producer record, never a consent assertion in its event payload."""
    if event.get('session_key') != route:
        return None
    reservation = None
    if event.get('type') == 'completion':
        from tools.process_registry import process_registry
        session = process_registry.get(event.get('session_id'))
        if (session is not None and session.id == event.get('session_id')
                and session.started_at == event.get('started_at') and session.exited
                and session.notify_on_complete and session.session_key == route):
            reservation = getattr(session, 'approval_task_reservation', None)
    elif event.get('type') == 'async_delegation' and not event.get('task_failure_notice'):
        from tools.async_delegation import approval_reservation_for_completion
        found = approval_reservation_for_completion(event.get('delegation_id'))
        if found is not None and found[0] == route:
            reservation = found[1]
    return (reservation if reservation is not None and reservation.evidence.session == route
            and reservation.evidence.task_id else None)


def completion_reservation_for_event(event, route, profile):
    reservation = _producer_reservation_for_event(event, route)
    return reservation if reservation is not None and reservation.evidence.profile == profile else None


def completion_batch_origin_for_event(event, route):
    """Group only live producer siblings; admission checks the profile again."""
    reservation = _producer_reservation_for_event(event, route)
    if reservation is None or not reservation.active() or not reservation.lease.live():
        return None
    return reservation.lease, reservation.evidence


def release_completion_event_reservation(event):
    if event.get('type') == 'completion':
        release_completion_reservation(('completion', event.get('session_id'), event.get('started_at')))
    elif event.get('type') == 'async_delegation' and not event.get('task_failure_notice'):
        release_completion_reservation(('async_delegation', event.get('delegation_id'), ''))


def release_completion_reservation(identity):
    """Settle a producer that will never execute a continuation turn."""
    from tools.approval_context import release_background_approval_task
    kind, producer_id, incarnation = identity
    if kind == 'completion':
        from tools.process_registry import process_registry
        session = process_registry.get(producer_id)
        if (session is not None and session.id == producer_id and session.started_at == incarnation
                and session.exited):
            reservation = getattr(session, 'approval_task_reservation', None)
            session.approval_task_reservation = None
            release_background_approval_task(reservation)
    elif kind == 'async_delegation' and incarnation == '':
        from tools.async_delegation import release_completion_approval_reservation
        release_completion_approval_reservation(producer_id)


def release_admitted_completion(row):
    descriptor = row['payload'].get('local_automation_v1') or (row['payload'].get('native_text_v1') or {}).get('automation') or {}
    for value in descriptor.get('identities') or [descriptor.get('identity')]:
        try:
            identity = json.loads(value)
        except (TypeError, ValueError):
            continue
        if isinstance(identity, list) and len(identity) == 3:
            release_completion_reservation(identity)


@contextmanager
def completion_task_scope(authority, row, live):
    payload = row['payload']
    descriptor = payload.get('local_automation_v1') or (payload.get('native_text_v1') or {}).get('automation') or {}
    raw = descriptor.get('identities') or [descriptor.get('identity')]
    identities = []
    for value in raw:
        try:
            identity = json.loads(value)
        except (TypeError, ValueError):
            identity = None
        identities.append(identity)
    completion_kinds = ('completion', 'async_delegation')
    if not any(isinstance(identity, list) and identity and identity[0] in completion_kinds
               for identity in identities):
        with nullcontext():
            yield
        return
    if any(not isinstance(identity, list) or len(identity) != 3 or identity[0] not in completion_kinds
           for identity in identities):
        from tools.approval_context import completion_approval_task
        try:
            with completion_approval_task((None,), 'completion:' + row['admission_id']):
                yield
        finally:
            for identity in identities:
                if isinstance(identity, list) and len(identity) == 3 and identity[0] in completion_kinds:
                    release_completion_reservation(identity)
        return
    from tools.approval_context import completion_approval_task, release_background_approval_task
    reservations = []
    releases = []
    for kind, producer_id, incarnation in identities:
        reservation = None
        if kind == 'completion':
            from tools.process_registry import process_registry
            session = process_registry.get(producer_id)
            if (session is not None and session.id == producer_id and session.started_at == incarnation
                    and session.exited
                    and session.session_key == live.route and session.notify_on_complete):
                reservation = getattr(session, 'approval_task_reservation', None)
                releases.append(lambda s=session: setattr(s, 'approval_task_reservation', None))
        elif incarnation == '':
            from tools.async_delegation import (approval_reservation_for_completion,
                                                release_completion_approval_reservation)
            found = approval_reservation_for_completion(producer_id)
            if found is not None and found[0] == live.route:
                reservation = found[1]
                releases.append(lambda p=producer_id: release_completion_approval_reservation(p))
        if (reservation is not None and (reservation.evidence.profile != str(authority.profile_id)
                or reservation.evidence.session != live.route or not reservation.evidence.task_id)):
            reservation = None
        reservations.append(reservation)
    try:
        with completion_approval_task(reservations, 'completion:' + row['admission_id']):
            yield
    finally:
        for release in releases:
            release()
        for reservation in reservations:
            release_background_approval_task(reservation)
