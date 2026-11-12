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

from hermes_agent.gateway.hosted_rooms import RoomStopPendingError

_HOMES = 'gateway.hosted.member_homes.v1:'


class MemberHomeUnknownError(RoomStopPendingError):
    """Disband waits on a turn whose member profile is no longer configured and whose home was
    never recorded: nothing here can reach its daemon, so only the user can unblock it."""
    reason = 'member_home_unknown'

    def __init__(self, profile):
        super().__init__(
            f"Group chat work for '{profile}' may still be running, but that profile is no longer "
            "configured here. Add it back to hosted_rooms.profiles so its Stop can be confirmed, "
            "then retry deletion.")


def _key(room_id, member_id):
    return _HOMES + json.dumps([room_id, member_id], separators=(',', ':'))


def record_member_home(db, room_id, member_id, home):
    """Remember each home a member's daemon was reached at, so the proof still covers it once the
    roster no longer names it (a ``hosted_rooms.profiles`` entry removed, then a restart)."""
    key, home = _key(room_id, member_id), str(home)
    if Path(home) in _recorded_homes(db, room_id, member_id):
        return  # every resolve calls this: a read keeps the common case off the writer
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
    this member's attempts that are over, so the store holds only its live ones. A resubmit after
    proven non-admission is a new generation (``start_task``), so it never reuses this record."""
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
    from hermes_agent.tui_gateway.hosted_room_service import task_member_id
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


def moved_attempt_gone(service, room_id, task):
    """True when *task*'s profile moved and the home that admitted its attempt has no gateway, so
    nothing can still run it and Retry may start a new generation at the current home."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_agent.tui_gateway.hosted_room_service import task_member_id
    served = service.profile_homes().get(task['payload']['target_profile'])
    admitted = stopping_attempt_home(service.authority.db, room_id, task_member_id(task), task)
    return (admitted is not None and served is not None and Path(served) != admitted
            and discover_gateway_endpoint(admitted).state == 'absent')


def legacy_stop_homes(db, room_id, member_id, task_id=None):
    """Homes whose Stop reads attest accepts for a stopping or deferred turn of *member_id*
    admitted before attempt homes were recorded (``task_id``, when the read names one, must be
    such a turn): any home the member was served from may hold it."""
    from hermes_agent.gateway.hosted_room_driver import list_tasks
    from hermes_agent.tui_gateway.hosted_room_service import task_member_id
    if not any(task_member_id(t) == member_id and task_id in (None, t['identity'].task_id)
               and stopping_attempt_home(db, room_id, member_id, t) is None
               for t in list_tasks(db.db_path, room_id=room_id) if t['status'] in _STOPPING):
        return set()
    return {str(home) for home in _recorded_homes(db, room_id, member_id)}


def _legacy_home_released(service, room_id, member, home, task):
    """Ask one older home whether its room session holds nothing; one that holds this turn is sent
    its Stop, and the turn is kept until a later read finds it gone. Unreachable keeps it."""
    from hermes_agent.gateway.session_hosted_transport import HostedRoomOwnerRPC
    from hermes_agent.tui_gateway.hosted_room_driver import room_session_title
    profile = task['payload']['target_profile']
    rpc = HostedRoomOwnerRPC(home=home, source_home=service.authority.profile_id, room_id=room_id,
                             member_id=member, profile=profile)
    kw = {'profile': profile, 'source': 'bot_room'}
    try:
        if rpc.resolve_exact(title=room_session_title(room_id), **kw) is None:
            return True
        info = rpc.info(session_id=rpc.ref.session_id, **kw)
        if not info.get('active'):
            # Nothing queued or running there. An input left 'unknown' by a restart reads inactive:
            # its agent died with the old process, the same rule as a live Stop's acknowledgement.
            return True
        if info.get('task_id') != task['identity'].task_id:
            return False  # another turn holds that session
        rpc.interrupt(session_id=rpc.ref.session_id, expected_task_id=task['identity'].task_id, **kw)
        # A queued input is cancelled at once; a running one stays until its run ends.
        return not rpc.info(session_id=rpc.ref.session_id, **kw).get('active')
    except Exception:  # health: allow BLE001 -- an unreachable or refusing home keeps the turn
        return False


def unsettled_legacy_homes(service, room_id, task):
    """For a turn admitted before attempt homes were recorded, every other home its member was
    served from whose gateway is up and has not answered that it holds nothing for the room: the
    current home's answer says nothing about those. Empty when the turn has an attempt home."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_agent.tui_gateway.hosted_room_service import task_member_id
    member, db = task_member_id(task), service.authority.db
    if stopping_attempt_home(db, room_id, member, task) is not None:
        return []
    served = service.profile_homes().get(task['payload']['target_profile'])
    return [home for home in _recorded_homes(db, room_id, member)
            if (served is None or home != Path(served))
            and discover_gateway_endpoint(home).state != 'absent'
            and not _legacy_home_released(service, room_id, member, home, task)]


def unserved_member_stopped(service, room_id, task):
    """True when *task*'s member profile is no longer served on this host and no gateway owns any
    home it was served from (its conventional home, each recorded one, and the roster's)."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    from hermes_agent.tui_gateway.hosted_room_service import task_member_id
    profile = task['payload']['target_profile']
    served = service.profile_homes().get(profile)
    admitted = stopping_attempt_home(service.authority.db, room_id, task_member_id(task), task)
    if admitted is not None and (served is None or Path(served) != admitted):
        # Moved or removed: only the home that admitted this attempt can hold it, and another home
        # the member was served from being up must not hold Disband forever.
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
