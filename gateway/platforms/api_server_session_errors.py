"""Bounded HTTP refusals for unreadable or contested session ownership."""

from hermes_state_lineage import SessionLineageInventoryError


def lineage_response(exc, error_response):
    unavailable = isinstance(exc, SessionLineageInventoryError)
    return error_response('Session ownership could not be verified; retry later.' if unavailable
                          else 'Session ID is already owned or has conflicting owners.',
                          503 if unavailable else 409, code=getattr(exc, 'code', 'session_lineage_conflict'))
