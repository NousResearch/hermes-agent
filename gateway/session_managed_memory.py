"""Authority-claimed turns in the gateway's memory-aware admission (``gateway/admission_memory.py``).

A held claim leaves the queue once a Stop is latched for it (a user's Stop through the session,
or a gateway stop) or its generation stops executing: :func:`claim_still_wanted`.

A managed session (a bypass/config-only session, a kanban card) starts a worker process instead of
running in the gateway, so it never passes the in-process message handler's memory hold. It waits
here, before the worker spawns: a kanban card as background work, any other managed session as a
person. A Stop while it waits falls through to the spawn path, which records the interruption.
"""

from __future__ import annotations

from typing import Optional


def claim_still_wanted(authority, session_id, generation) -> bool:
    """No Stop latched for this claim and the authority still executes its generation. Before the
    turn has an agent, a Stop (and a gateway stop) only latches ``pending_stops``."""
    stops = getattr(authority, 'pending_stops', {})
    return stops.get(session_id) != generation and authority.executing.get(session_id) == generation


async def hold_managed_for_memory(authority, ref, row, policy) -> Optional[str]:
    """None to spawn the worker; else the refusal that answers the turn (memory never freed)."""
    from gateway.admission_memory import BACKGROUND, PERSON, get_gate, wait_async
    from gateway.session_context_refs import refused_turn_result
    from agent.i18n import t

    generation = row.get('generation')
    kind = BACKGROUND if policy.kanban_json is not None else PERSON
    verdict = await wait_async(kind, on_hold=lambda: None,
                               still_wanted=lambda: claim_still_wanted(authority, ref.session_id, generation))
    if verdict != 'timeout':
        return None
    refusal = t('gateway.busy.memory_gave_up', minutes=max(1, round(get_gate().settings.max_wait_seconds / 60)))
    authority.pending_results[row['admission_id']] = {'result': refused_turn_result(row, refusal), 'usage': {}}
    return refusal
