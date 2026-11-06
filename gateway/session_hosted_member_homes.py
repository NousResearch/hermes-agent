"""Whether a local member another profile's daemon served can still run anything on this host.

A turn deferred behind that daemon's unconfirmed Stop settles once the daemon confirms
(``HostedRoomRuntime.deferred_stop_confirmed``). A profile this host no longer serves (gone from
the group's roster, or its home deleted) never confirms: its daemon's source attest refuses the
member, so waiting would hold Disband forever. The turn then settles only on proof that no agent
can run it: every home the profile was served from has no live gateway owner. With neither a
roster entry nor a recorded home (a turn dispatched before homes were recorded) that home is
unknown, so the turn is kept. Discovery answers
``absent`` only with no control socket, no multiplexer serving the home, and no holder of its
``gateway.lock``, which a daemon keeps until its process exits. A daemon that is alive but does
not answer keeps the turn, and so does a profile still served here: only its daemon's Stop decides.

Each attempt also records the home it was admitted at (``record_attempt_home``, when that home's
daemon attests the submit). Once its Stop has begun (stopping or deferred), the Stop is asked
there, not wherever ``hosted_rooms.profiles`` points now: a home the profile was moved to has no
such session, and its "nothing here" would settle a turn the old home still holds. That old home's
daemon may still confirm the Stop; once it is gone, the attempt settles. A running attempt still
follows the mapping (a moved profile's next reads go to its new home).
"""
import json
from pathlib import Path

from gateway.hosted_rooms import RoomStopPendingError

_HOMES = 'gateway.hosted.member_homes.v1:'


class MemberHomeUnknownError(RoomStopPendingError):
    """Disband waits on a turn whose member profile is no longer configured and whose home was
    never recorded: nothing here can reach its daemon, so only the user can unblock it."""
    reason = 'member_home_unknown'

    def __init__(self, profile):
        super().__init__(
            f"Group chat work for '{profile}' may still be running, but that profile is no longer "
            "configured here. Add it back to hosted_rooms.profiles so its Stop can be confirmed, "
            "or Discard the turn, then retry deletion.")


def _key(room_id, member_id):
    return _HOMES + json.dumps([room_id, member_id], separators=(',', ':'))


def record_member_home(db, room_id, member_id, home):
    """Remember each home a member's daemon was reached at, so the proof still covers it once the
    roster no longer names it (a ``hosted_rooms.profiles`` entry removed, then a restart)."""
    key, home = _key(room_id, member_id), str(home)
    def write(conn):
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        homes = json.loads(row[0]) if row is not None else []
        if home not in homes:
            conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                         (key, json.dumps([*homes, home])))
    db._execute_write(write)


_LIVE_ATTEMPT = frozenset({'running', 'stopping', 'deferred', 'indeterminate'})
_STOPPING = frozenset({'stopping', 'deferred'})


def _attempt_prefix(room_id, member_id):
    return _HOMES + json.dumps([room_id, member_id], separators=(',', ':'))[:-1] + ','


def _attempt_homes(conn, room_id, member_id, statuses=_LIVE_ATTEMPT):
    """``{key: (home, live)}`` for every attempt home recorded for this member; live while its
    task row is still that attempt (same generation) in one of *statuses*."""
    prefix, found = _attempt_prefix(room_id, member_id), {}
    for key, home in conn.execute('SELECT key, value FROM state_meta WHERE substr(key, 1, length(?))=?',
                                  (prefix, prefix)).fetchall():
        task_id, generation = json.loads(key[len(_HOMES):])[2:]
        row = conn.execute('SELECT status, execution_generation FROM hosted_room_driver_tasks '
                           'WHERE room_id=? AND task_id=?', (room_id, task_id)).fetchone()
        found[key] = (home, row is not None and row[0] in statuses and row[1] == generation)
    return found


def record_attempt_home(db, room_id, member_id, task_id, generation, home):
    """Remember the home that admitted one attempt (the first one wins), and drop the records of
    this member's attempts that are over, so the store holds only its live ones."""
    key = _HOMES + json.dumps([room_id, member_id, task_id, int(generation)], separators=(',', ':'))
    def write(conn):
        for old, (_home, live) in _attempt_homes(conn, room_id, member_id).items():
            if not live and old != key:
                conn.execute('DELETE FROM state_meta WHERE key=?', (old,))
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', (key, str(home)))
    db._execute_write(write)


def stopping_attempt_home(db, room_id, member_id, task):
    """The home that admitted *task*'s current attempt once its Stop has begun, or None (not
    stopping, or admitted before attempt homes were recorded)."""
    if task.get('status') not in _STOPPING:
        return None
    key = _HOMES + json.dumps([room_id, member_id, task['identity'].task_id, int(task['execution_generation'])],
                              separators=(',', ':'))
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    return None if row is None else Path(row[0])


def stopping_attempt_homes(db, room_id, member_id):
    """Homes holding an attempt of this member whose Stop is not yet confirmed: each may still
    answer for it."""
    with db._read_ctx() as conn:
        return {home for home, live in _attempt_homes(conn, room_id, member_id, _STOPPING).values() if live}


def forget_room_member_homes_locked(conn, room_id):
    """Drop a disbanded room's recorded homes, in its tombstone's transaction: no turn is left."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'").fetchone():
        prefix = _HOMES + json.dumps([room_id], separators=(',', ':'))[:-1] + ','
        conn.execute('DELETE FROM state_meta WHERE substr(key, 1, length(?))=?', (prefix, prefix))


def _recorded_homes(db, room_id, member_id):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_key(room_id, member_id),)).fetchone()
    return [Path(home) for home in json.loads(row[0])] if row is not None else []


def _served_here(home):
    from hermes_constants import named_profile_home, named_profile_is_deleted
    return home.is_dir() and not (named_profile_home(home) is not None and named_profile_is_deleted(home))


def _home_unknown(service, room_id, task, served):
    from tui_gateway.hosted_room_service import task_member_id
    member = task_member_id(task)
    return (served is None and not _recorded_homes(service.authority.db, room_id, member)
            and stopping_attempt_home(service.authority.db, room_id, member, task) is None)


def unknown_home_refusal(service, room_id, deferred):
    """The refusal naming the first *deferred* turn whose member's home is unknown, or None."""
    for task in deferred:
        profile = task['payload']['target_profile']
        if _home_unknown(service, room_id, task, service.profile_homes().get(profile)):
            return MemberHomeUnknownError(profile)
    return None


def unserved_member_stopped(service, room_id, task):
    """True when *task*'s member profile is no longer served on this host and no gateway owns any
    home it was served from (its conventional home, each recorded one, and the roster's)."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    from tui_gateway.hosted_room_service import task_member_id
    profile = task['payload']['target_profile']
    served = service.profile_homes().get(profile)
    admitted = stopping_attempt_home(service.authority.db, room_id, task_member_id(task), task)
    if admitted is not None and served is not None and Path(served) != admitted:
        # Moved to another home, whose daemon has no such session: only the home that admitted
        # this attempt can hold it, and the new one being up must not hold Disband forever.
        return discover_gateway_endpoint(admitted).state == 'absent'
    if served is not None and _served_here(Path(served)):
        return False
    if _home_unknown(service, room_id, task, served):
        return False  # the home its daemon ran from is unknown: an absent default proves nothing
    own = Path(service.authority.profile_id)
    root = multiplexer_root_for(own) or own
    recorded = _recorded_homes(service.authority.db, room_id, task_member_id(task))
    homes = {root if profile == 'default' else root / 'profiles' / profile, *recorded}
    if served is not None:
        homes.add(Path(served))
    return all(discover_gateway_endpoint(home).state == 'absent' for home in homes)
