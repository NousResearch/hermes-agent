"""Connection-owned groups.state reads, not NEW-work admission or recovery."""
from collections import Counter
from contextlib import contextmanager, nullcontext
from typing import Any
from dataclasses import dataclass, replace
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError, _epoch


@dataclass(frozen=True)
class GroupStateOwner:
    authority: Any
    db: Any
    epoch: int
    instance_id: str
    profile_id: str
    service: Any
    runtime: Any

    @classmethod
    def capture(cls, authority):
        service = getattr(authority, 'hosted_room_service', None)
        return cls(authority, authority.db, authority.epoch, authority.instance_id,
                   authority.profile_id, service, getattr(service, 'runtime', None))

    def current(self, conn):
        a, db = self.authority, self.db
        if (conn is None or db._read_conns_closed or conn is not db._conn
                or a.db is not db or (a.epoch, a.instance_id, a.profile_id) !=
                (self.epoch, self.instance_id, self.profile_id)
                or db._db_replaced or db._db_file_was_replaced()
                or db._db_wal_generation_lost or db._wal_generation_was_lost()):
            raise RuntimeStoreError('group_state_unavailable')
        db._raise_if_db_corrupt()
        _epoch(conn, self.epoch)

    @contextmanager
    def read(self):
        # Close and writer replacement share this non-reentrant lifetime lock.
        # This Route substrate predates SessionDB.live_read_connection. Borrow
        # its existing connection under the same actual close/writer lock; do
        # not add a Runtime prerequisite or fall back to recovering _read_ctx.
        with self.db._lock:
            conn = None if self.db._read_conns_closed else self.db._conn
            self.current(conn)
            yield conn
            self.current(conn)


@dataclass(frozen=True)
class GroupStateRead:
    owner: GroupStateOwner
    service: Any
    actor: Any
    room: dict
    conn: Any

    def authorize(self):
        owner, service, actor = self.owner, self.service, self.actor
        owner.current(self.conn)
        if actor.profile_id != owner.profile_id or 'session:read' not in actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        if service is not None:
            current = getattr(service, 'authority', None)
            if (getattr(current, 'db', None) is not owner.db
                    or getattr(current, 'profile_id', None) != owner.profile_id
                    or Path(service.db_path).resolve() != Path(owner.db.db_path).resolve()):
                raise RuntimeStoreError('group_state_unavailable')
        row = self.conn.execute('SELECT value FROM state_meta WHERE key=?',
            ('gateway.hosted.owner.v1:' + self.room['room_id'],)).fetchone()
        if row is None or row[0] != actor.subject:
            raise RuntimeStoreError('permission_denied')
        if self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='hosted_room_quarantine'").fetchone():
            if self.conn.execute('SELECT 1 FROM hosted_room_quarantine WHERE room_id=?',
                                 (self.room['room_id'],)).fetchone():
                raise RuntimeStoreError('group_state_unavailable')

    def actions_available(self):
        from gateway.session_hosted_peer_retry import _require_retry_owner
        try:
            _require_retry_owner(self.service, self.owner.authority, self.owner.runtime)
            return True
        except RuntimeStoreError:
            return False

    def action_permitted(self, kind):
        capability = {'approval': 'session:approve', 'retry': 'session:control',
                      'discard': 'session:control'}.get(kind)
        return capability in self.actor.capabilities


def read_group_state(owner, authority, actor, params):
    from gateway import hosted_rooms as rooms
    if owner.authority is not authority or actor.profile_id != owner.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    room_id = rooms._room_id(params.get('room_id'))
    installed = getattr(authority, 'hosted_room_service', None)
    # A cold service explicitly installed on this SAME pinned authority may
    # resume its durable rights. A substituted service.authority never may.
    if (installed is not None and installed is not owner.service
            and getattr(installed, 'authority', None) is authority
            and Path(installed.db_path).resolve() == Path(owner.db.db_path).resolve()):
        owner = replace(owner, service=installed, runtime=installed.runtime)
    service = owner.service
    # Policy precedes the owner SQL lock everywhere that both are required.
    with service._policy_lock if service is not None else nullcontext():
        with owner.read() as conn:
            conn.execute('BEGIN')
            try:
                scope = GroupStateRead(owner, service, actor, {'room_id': room_id}, conn)
                scope.authorize()
                room = rooms.room_state(owner.db.db_path, **params, conn=conn)
                scope = GroupStateRead(owner, service, actor, room, conn)
                result = {'room': room}
                if service is not None and room.get('disbanded_at') is None:
                    # A replaced S.runtime withdraws controls, not the original
                    # owner's informational snapshot. Only installation of a
                    # new service above may bind a new request runtime.
                    runtime = owner.runtime.status()
                    if runtime['running'] and not runtime['stopping']:
                        result['driver_status'] = service.status(room_id, state_read=scope)
                scope.authorize()
            finally:
                conn.rollback()
            # Recheck lifetime/room authorization outside the historical view.
            scope.authorize()
            if 'driver_status' in result:
                available = scope.actions_available()
                result['driver_status']['pending_actions'] = [
                    a for a in result['driver_status']['pending_actions']
                    if a['kind'] in {'output_retry', 'output_cleanup'}
                    or (available and scope.action_permitted(a['kind']))]
            return result


def driver_status(service, room_id, scope):
    """Canonical status projection from the authorized, already-held connection."""
    from gateway import hosted_room_driver as tasks
    from gateway.session_hosted_peer_retry import _validate
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    if scope.service is not service or scope.room['room_id'] != room_id:
        raise RuntimeStoreError('permission_denied')
    scope.authorize()
    conn = scope.conn
    rows = tasks._tasks_in_order(conn, room_id, None)
    pending = [tasks._task_from_row(row) for row in rows]
    counts = Counter(task['status'] for task in pending)
    runtime_owner = scope.owner.runtime
    runtime = runtime_owner.status()
    actions = []
    available = scope.actions_available()
    if available:
        actions.extend(dict(action) for (room, _), action in service._pending_actions.items()
                       if room == room_id and action['kind'] != 'retry'
                       and scope.action_permitted(action['kind']))
    if available and scope.action_permitted('retry'):
        binding = HostedRoomBinding(room_id, scope.room['authority_gateway_id'], scope.room['authority_epoch'])
        peers = {m['member_id'] for m in scope.room['members'] if m.get('target', {}).get('kind') == 'peer'}
        for task in pending:
            if task['status'] not in {'indeterminate', 'deferred'}:
                continue
            member = task['payload'].get('target_member_id') or task['payload']['target_profile']
            if member in peers:
                try:
                    _validate(service, task, binding, conn)
                    lease = runtime_owner._leases.get(room_id)
                    if lease is None:
                        continue
                    tasks._require_active_lease(conn, lease, now=runtime_owner.clock())
                except (RuntimeStoreError, ValueError):
                    continue
            actions.append(dict(kind='discard' if task['status'] == 'indeterminate' else 'retry',
                member_id=member, task_id=task['identity'].task_id,
                execution_generation=task['execution_generation']))
    return dict(running=runtime['running'], working=any(counts.get(s) for s in ('queued', 'running', 'stopping')),
        blocked=room_id in runtime['blocked_rooms'] or bool(counts.get('indeterminate') or counts.get('stopping')),
        counts=dict(counts), pending_actions=actions, peer_routes=service._route_statuses(room_id))
