"""Whether a local member another profile's daemon served can still run anything on this host.

A turn deferred behind that daemon's unconfirmed Stop settles once the daemon confirms
(``HostedRoomRuntime.deferred_stop_confirmed``). A profile this host no longer serves (gone from
the group's roster, or its home deleted) never confirms: its daemon's source attest refuses the
member, so waiting would hold Disband forever. The turn then settles only on proof that no agent
can run it: every home the profile was served from has no live gateway owner. Discovery answers
``absent`` only with no control socket, no multiplexer serving the home, and no holder of its
``gateway.lock``, which a daemon keeps until its process exits. A daemon that is alive but does
not answer keeps the turn, and so does a profile still served here: only its daemon's Stop decides.
"""
import json
from pathlib import Path

_HOMES = 'gateway.hosted.member_homes.v1:'


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


def _recorded_homes(db, room_id, member_id):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_key(room_id, member_id),)).fetchone()
    return [Path(home) for home in json.loads(row[0])] if row is not None else []


def _served_here(home):
    from hermes_constants import named_profile_home, named_profile_is_deleted
    return home.is_dir() and not (named_profile_home(home) is not None and named_profile_is_deleted(home))


def unserved_member_stopped(service, room_id, task):
    """True when *task*'s member profile is no longer served on this host and no gateway owns any
    home it was served from (its conventional home, each recorded one, and the roster's)."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    from tui_gateway.hosted_room_service import task_member_id
    profile = task['payload']['target_profile']
    served = service.profile_homes().get(profile)
    if served is not None and _served_here(Path(served)):
        return False
    own = Path(service.authority.profile_id)
    root = multiplexer_root_for(own) or own
    homes = {root if profile == 'default' else root / 'profiles' / profile,
             *_recorded_homes(service.authority.db, room_id, task_member_id(task))}
    if served is not None:
        homes.add(Path(served))
    return all(discover_gateway_endpoint(home).state == 'absent' for home in homes)
