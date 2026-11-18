"""Seed rows through the storage interface: a store handle from ``open_persistence`` (or ``storage_handle``).

``seed_session`` creates a session row with its title, flags and end; ``seed_messages`` appends message dicts;
``settle_claim`` settles a claimed admission; ``seed_admission`` brings one admission to a status. The other seeding operations are plain calls on the handle:
``sessions.reopen``, ``sessions.delete``, ``messages.append_batch``, ``messages.compact``,
``messages.replace_transcript``, ``runtime.begin_epoch``, ``admissions.admit``, ``admissions.claim``,
``admissions.recover``, ``admissions.cancel`` and ``lineage.publish_compression_child``.
"""
import json

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


def seed_session(store, session_id, source, *, title=None, pinned=None, archived=None, hidden=None, end_reason=None,
                 origin_json=None, **fields):
    """Create the session row (``fields`` are ``NewSession`` fields; ``origin_json`` text is decoded into ``origin``),
    then set its title, its pinned, archived and hidden flags, and end it with ``end_reason``, each only when given.
    Pinning unhides a row, so ``hidden`` is set after it. Returns the session id."""
    if origin_json is not None:
        fields["origin"] = json.loads(origin_json)
    store.sessions.create(NewSession(session_id=session_id, source=source, **fields))
    if title is not None:
        store.sessions.set_title(session_id, title)
    if pinned is not None:
        store.sessions.set_pinned(session_id, pinned)
    if archived is not None:
        store.sessions.set_archived(session_id, archived)
    if hidden is not None:
        store.sessions.set_hidden(session_id, hidden)
    if end_reason is not None:
        store.sessions.end(session_id, end_reason)
    return session_id


def seed_messages(store, session_id, *messages):
    """Append each message dict in order, one row each, through ``messages.append``. A dict carries the live message
    keys ``NewMessage.from_live`` reads (``role``, ``content``, ``tool_calls``, ``_compressed_summary``...). Returns the
    row ids."""
    return [store.messages.append(session_id, NewMessage.from_live(message)) for message in messages]


def settle_claim(store, claimed, outcome="completed", result=None):
    """Settle the admission ``admissions.claim`` returned, under the epoch that claimed it; returns the settled record."""
    return store.admissions.settle(epoch=claimed.owner_epoch, admission_id=claimed.admission_id,
                                   generation=claimed.generation, outcome=outcome, result=result)


def seed_admission(store, session_id, request_id, status="queued", outcome=None, *, principal_id="seed"):
    """Admit *request_id* on the existing session *session_id* under a new epoch and bring it to *status*: ``queued``
    as admitted, ``started`` claimed, ``terminal`` settled with *outcome*, ``unknown`` claimed and then marked by a
    later epoch's boot recovery. The session holds no other queued, started or unknown admission. Returns the
    admission id."""
    epoch = store.runtime.begin_epoch(instance_id=f"seed:{request_id}")
    record, _created = store.admissions.admit(epoch=epoch, principal_id=principal_id, session_id=session_id,
                                               request_id=request_id, payload={"text": request_id})
    if status != "queued":
        claimed = store.admissions.claim(epoch=epoch, session_id=session_id)
        if status == "terminal":
            settle_claim(store, claimed, outcome)
        elif status == "unknown":
            store.admissions.recover(epoch=store.runtime.begin_epoch(instance_id=f"seed:{request_id}:recover"))
    return record.admission_id


def seed_usage(store, session_id, **counts):
    """Account one API call through ``usage.queue_usage``, then drain the queue. ``counts`` are flat keywords: the
    ``TokenCounts`` counters, ``model`` with the ``billing_*`` route fields, and the other ``UsageDelta`` fields; they
    map onto the delta as the agent's turn accounting maps them (a route only with a model)."""
    from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
    tokens = TokenCounts(**{key: counts.pop(key, 0) for key in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")})
    route = {key: counts.pop(key, None) for key in ("model", "billing_provider", "billing_base_url", "billing_mode")}
    store.usage.queue_usage(session_id, UsageDelta(tokens=tokens, route=BillingRoute(**route) if route["model"] else None,
                                                   **counts))
    store.usage.flush_usage()
