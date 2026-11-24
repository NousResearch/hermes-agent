"""Which attached clients were used recently: per connection and session, in memory.

A notification plugin asks, at the end of a turn, whether someone is at a client that shows the
session, so a phone can stay quiet. Each connection keeps, per session, the newest "last input"
time of two sources: input the owner itself sees on that connection (``INPUT_METHODS`` via
``touch`` from ``AuthorityConnection.dispatch``, and a result frame answering a server request a
reconnecting viewer is handed again, a skip or cancel included, via ``seen``) and the client's own
``client.activity`` reports (``report``), which say how long ago the user last touched the device or
the Hermes window. Idle time is simply ``now - last input``: no timer ages a record out, so it grows
until new input. A report is the connection's whole statement: it replaces the last report of each
session it lists, and every record of a session it does not list is dropped, owner-seen input
included (``sessions: []`` drops them all). ``away`` (screen locked, machine suspending) clears the
connection's records at once, and each source keeps at most ``MAX_SESSIONS`` records, dropping the
least recently touched.

One socket can serve several profiles, and session ids are unique only per profile. Owner-seen
input is kept for the profile whose authority saw it. A report names sessions by id alone, so it is
kept unresolved and resolved at read time: it counts for the one profile this connection is attached
to that id in, and for none while the id names attached sessions in more than one (no evidence is
made up). Resolving at read time also counts a report that raced ahead of its attach. Once the
connection is attached to no session of an id, its report for that id is dropped (``detached``), so
it never counts for another profile's same-id session attached later.

A record is read only while its connection is attached to that session (``viewers``), so a detached
or closed client is never a viewer. Nothing is persisted: a restarted owner knows nobody. No
permission, prompt or toolset reads any of it.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

MAX_SESSIONS = 16
# The longest session id presence keeps: a report or an input naming a longer one is no evidence.
MAX_SESSION_ID_CHARS = 256
# Requests that are the user's own hand on this connection for ``params['session_id']``.
# ``session.resume`` / attach is not: opening a window is not being at it. ``clarify.lock`` would
# be, but it names a request id, not a session, so ``touch`` cannot place it.
INPUT_METHODS = frozenset({'prompt.submit', 'prompt.steer_queued', 'prompt.replace_queued', 'approval.respond',
                           'clarify.respond', 'session.interrupt', 'session.steer', 'session.redirect'})
_BASES = ('system', 'page')


# The key of a reported record: a report names no profile (see the module docstring).
_REPORTED = object()


@dataclass
class _Record:
    last_input: float  # time.monotonic()
    basis: str  # 'input' (seen by the owner) or the reported 'system' / 'page'
    can_alert: bool | None = None  # None for owner-seen input; a report always carries one


class ClientPresence:
    """One connection's records, newest-touched last, keyed ``(profile_id, session_id)`` for
    owner-seen input and ``(_REPORTED, session_id)`` for the last report, at most ``MAX_SESSIONS`` of
    each. Written from the socket's reader (touched input, reports) and read from turn threads, so
    every access holds the lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: OrderedDict[tuple, _Record] = OrderedDict()

    def seen(self, profile_id, session_id, at):
        """Owner-seen input for *session_id* of the profile *profile_id* at *at*."""
        key = (profile_id, session_id)
        with self._lock:
            record = self._records.get(key)
            # An older time than the record's never moves its clock back.
            self._keep(key, record if record is not None and at < record.last_input else _Record(at, 'input'))

    def reported(self, session_ids, at, basis, can_alert):
        """A report: input at *at* for each of *session_ids*, and nothing any more for any other session.
        It replaces the last report outright; only ``read`` weighs it against owner-seen input."""
        listed = set(session_ids)
        with self._lock:
            for key in [key for key in self._records if key[1] not in listed]:
                del self._records[key]
            for session_id in session_ids:
                self._keep((_REPORTED, session_id), _Record(at, basis, can_alert))

    def _keep(self, key, record):
        self._records[key] = record
        self._records.move_to_end(key)
        # The bound is per source: a report may name MAX_SESSIONS sessions, and input seen on them
        # must not push those reports (their ``can_alert``) out, nor a report the newer input.
        reported = key[0] is _REPORTED
        same = [other for other in self._records if (other[0] is _REPORTED) is reported]
        for oldest in same[:-MAX_SESSIONS]:
            del self._records[oldest]

    def clear(self):
        with self._lock:
            self._records.clear()

    def forget_report(self, session_id):
        with self._lock:
            self._records.pop((_REPORTED, session_id), None)

    def read(self, profile_id, session_id, now, *, reported=True):
        """``(idle_seconds, basis, can_alert)`` for *session_id* of *profile_id*, or None when nothing
        was seen: the newer of its owner-seen input and, when *reported*, the last report naming it."""
        with self._lock:
            seen = self._records.get((profile_id, session_id))
            report = self._records.get((_REPORTED, session_id)) if reported else None
            if seen is None and report is None:
                return None
            newest = seen if report is None or (seen is not None and seen.last_input > report.last_input) else report
            return (max(0.0, now - newest.last_input), newest.basis,
                    report.can_alert if report is not None else None)


def touch(connection, method, params):
    """Record the owner-seen input *method* is for its session on *connection*'s transport."""
    session_id = params.get('session_id')
    if method in INPUT_METHODS and _session_id(session_id):
        seen(connection, session_id)


def seen(connection, session_id):
    """Owner-seen input for *session_id* on *connection*, kept for the profile of the authority it
    was dispatched on. Reads attributes with defaults only, whatever connection it is handed; both
    callers log a failure and carry on, so observing input never fails (or changes the answer of)
    the request it runs in."""
    presence = getattr(getattr(connection, 'transport', None), 'client_presence', None)
    if presence is not None:
        profile_id = getattr(getattr(connection, 'authority', None), 'profile_id', None)
        presence.seen(profile_id, session_id, time.monotonic())


def _session_id(value):
    return isinstance(value, str) and 0 < len(value) <= MAX_SESSION_ID_CHARS


def _idle_problem(idle):
    if isinstance(idle, bool) or not isinstance(idle, (int, float)):
        return True
    try:
        return not math.isfinite(idle) or idle < 0
    except OverflowError:  # an int no float can hold
        return True


def activity_problem(params) -> str | None:
    """Why *params* is not a ``client.activity`` report (None when it is one)."""
    sessions = params.get('sessions')
    if (not isinstance(sessions, list) or len(sessions) > MAX_SESSIONS
            or not all(_session_id(sid) for sid in sessions)):
        return (f'sessions must be a list of at most {MAX_SESSIONS} session ids '
                f'of at most {MAX_SESSION_ID_CHARS} characters')
    if _idle_problem(params.get('idle_seconds')):
        return 'idle_seconds must be a number >= 0'
    if params.get('basis') not in _BASES:
        return 'basis must be "system" or "page"'
    if any(key in params and not isinstance(params[key], bool) for key in ('can_alert', 'away')):
        return 'can_alert and away must be booleans'
    return None


def report(transport, params) -> str | None:
    """Apply one ``client.activity`` call to *transport*'s presence; why it was refused, else None.
    Without ``away`` the report replaces the connection's whole statement (``ClientPresence.reported``).
    A transport that keeps no presence (stdio) accepts and ignores it."""
    presence = getattr(transport, 'client_presence', None)
    if presence is None:
        return None
    if problem := activity_problem(params):
        return problem
    if params.get('away', False):
        presence.clear()
        return None
    presence.reported(params['sessions'], time.monotonic() - params['idle_seconds'], params['basis'],
                      params.get('can_alert', True))
    return None


def _attached(authority, transport, session_id):
    live = authority.sessions.get(session_id)
    return live is not None and any(authority.events.get(actor.transport_id) is transport
                                    for actor in list(live.subscribers.values()))


def _attached_elsewhere(authority, transport, session_id):
    """Whether *transport* is also attached to a same-id session of another profile in this process."""
    from gateway.session_authorities import all_authorities
    return any(_attached(other, transport, session_id)
               for other in all_authorities(getattr(authority, 'runner', None)) if other is not authority)


def detached(authority, transport, session_id):
    """*transport* left *session_id* on *authority*. Once it is attached to no session of that id in
    this process its report for the id is dropped, so the report never counts for another profile's
    same-id session it attaches later; a report sent before the first attach still counts."""
    presence = getattr(transport, 'client_presence', None)
    if presence is not None and not (_attached(authority, transport, session_id)
                                     or _attached_elsewhere(authority, transport, session_id)):
        presence.forget_report(session_id)


def viewers(authority, session_id) -> list[dict]:
    """One entry per connection attached to the live *session_id* on *authority* right now:
    ``{name, kind, idle_seconds, basis, can_alert}`` (``name``/``kind`` None without a hello,
    ``idle_seconds``/``basis`` None when nothing was seen for this session, ``can_alert`` None while no
    report of this connection counts for it; a report that leaves ``can_alert`` out says True).
    ``[]`` for a session that is not live here."""
    live = authority.sessions.get(session_id)
    if live is None:
        return []
    now = time.monotonic()
    entries = []
    for actor in list(live.subscribers.values()):
        transport = authority.events.get(actor.transport_id)
        if transport is None:
            continue
        identity = getattr(transport, 'client_identity', None) or {}
        presence = getattr(transport, 'client_presence', None)
        seen_here = None if presence is None else presence.read(
            authority.profile_id, session_id, now, reported=not _attached_elsewhere(authority, transport, session_id))
        idle, basis, can_alert = seen_here or (None,) * 3
        entries.append({'name': identity.get('name'), 'kind': identity.get('kind'), 'idle_seconds': idle,
                        'basis': basis, 'can_alert': can_alert})
    return entries
