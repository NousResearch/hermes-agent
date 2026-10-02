"""Authority-bound hosted service; no legacy session server is constructed."""
import asyncio
from contextlib import nullcontext
from pathlib import Path
import threading

from gateway.session_contract import Principal
from gateway.session_authorities import active_authority, all_authorities, owner_scope
from gateway.session_hosted_controls import HostedControls
from hermes_state_runtime import RuntimeStoreError, _epoch
from tui_gateway.hosted_room_service import HostedRoomService

_OWNER = 'gateway.hosted.owner.v1:'


class CanonicalHostedRoomService(HostedControls, HostedRoomService):
    def __init__(self, authority, loop):
        self.authority, self.loop = authority, loop
        self.member_rpcs = {}
        # Serializes every peer route publication (registration, renewal) with Disband.
        self.peer_route_lock = threading.RLock()
        self._disband_upkeep_lock = threading.Lock()
        self._disband_retry_at = {}
        self._peer_cleanup_inflight = set()
        self._peer_renewals, self._peer_renewal_scans = {}, {}  # session_group_peer_routes
        super().__init__(None, db_path=authority.db.db_path)

    def _load_stored_links(self):
        super()._load_stored_links()
        for key, client in self.peer_clients.items():
            client.proof_install_id = self.peer_routes[key].target_install_id

    def _make_rpc(self, server):
        # Member-specific canonical transports retain exact durable history. They
        # intentionally use the runtime's receipt-capable (non-legacy) recovery path.
        return self

    def register_peer_route(self, *, room_id, member_id, route, client, target_url=None, catalog=None,
                            expected_grant=None, authorize=None):
        # Persisted, then published; the grant it replaces is retired (session_group_peer_routes).
        from gateway.session_group_peer_routes import publish_route
        publish_route(self, room_id=room_id, member_id=member_id, route=route, client=client,
                      target_url=target_url, catalog=catalog, expected_grant=expected_grant,
                      authorize=authorize)

    def _track_peer_client(self, binding, key, route, client):
        from gateway.session_group_peer_routes import CanonicalPeerClient
        return CanonicalPeerClient(self, binding, key, route, client)

    def _runtime_options(self):
        # A peer turn its gateway never received is deferred with that proof, so the room's
        # next turn runs; Retry (HostedControls) requeues it. Peer grants renew in the room's cycle.
        return {'defer_not_admitted_members': True, 'maintain_leased_room': self._maintain_peer_grants,
                'maintain_service': self._maintain_peer_lifecycle}

    def _maintain_peer_grants(self, binding, lease):
        from gateway.session_group_peer_routes import maintain_peer_grants
        maintain_peer_grants(self, binding, lease)

    def profile_homes(self):
        from gateway.run import _load_gateway_config
        from gateway.hosted_rooms_common import IDENTIFIER_RE
        from gateway.session_authorities import served_profile_name
        home = Path(self.authority.profile_id)
        own = served_profile_name(home)
        # The coordinator's own threads do not inherit the caller's ContextVars.
        with owner_scope(self.authority):
            configured = _load_gateway_config().get('hosted_rooms', {}).get('profiles', {})
        result = {own: home}
        if not isinstance(configured, dict):
            raise RuntimeStoreError('invalid_params')
        for name, value in configured.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise RuntimeStoreError('invalid_params')
            target = Path(value)
            if not IDENTIFIER_RE.fullmatch(name) or not target.is_absolute() or target != target.resolve():
                raise RuntimeStoreError('invalid_params')
            if name == own and target != home:
                raise RuntimeStoreError('permission_denied')
            result[name] = target
        return result

    def local_profiles(self):
        return tuple(self.profile_homes())

    def attest(self, selector, operation, params):
        from dataclasses import asdict
        from gateway.hosted_room_driver import list_tasks
        if set(selector) != {'room_id', 'member_id', 'profile'}:
            raise RuntimeStoreError('invalid_params')
        room_id, member, profile = (selector[k] for k in ('room_id', 'member_id', 'profile'))
        owner = self._owner(room_id)
        self._owned_authority(room_id)
        room = self._room(room_id)
        if not any(m['member_id'] == member and m['profile'] == profile
                   and m.get('target', {}).get('kind', 'local') == 'local' for m in room['members']):
            raise RuntimeStoreError('permission_denied')
        target_home = self.profile_homes().get(profile)
        if target_home is None or params.get('_target_home') != str(target_home):
            raise RuntimeStoreError('permission_denied')
        result = {'owner': owner, 'target_home': str(target_home)}
        if operation == 'approve':
            from gateway.hosted_room_approval import require_current_approval
            task = params.get('task')
            if not isinstance(task, dict) or task.get('room_id') != room_id:
                raise RuntimeStoreError('permission_denied')
            current = require_current_approval(self, room_id, member, task.get('task_id'),
                                               params.get('execution_generation'))
            if asdict(current['identity']) != task:
                raise RuntimeStoreError('permission_denied')
        if operation in {'submit', 'execute', 'attachment'}:
            matches = [t for t in list_tasks(self.db_path, room_id=room_id)
                       if asdict(t['identity']) == params.get('task')
                       and t['execution_generation'] == params.get('execution_generation')
                       and t['status'] == 'running'
                       and t['payload'].get('target_member_id', t['payload']['target_profile']) == member
                       and t['payload']['target_profile'] == profile
                       and (operation == 'execute' or (
                           t['payload']['prompt'] == params.get('prompt')
                           and t['payload'].get('attachments', []) == (params.get('attachments') or [])))]
            if len(matches) != 1:
                raise RuntimeStoreError('permission_denied')
            payload = matches[0]['payload']
            if operation == 'attachment':
                # Bytes only: the prompt/manifest already attested for the submit would
                # otherwise compete with the chunk for the bounded response line.
                from gateway.session_hosted_transport import source_attachment_chunk
                result.update(source_attachment_chunk(self, member, room_id, payload.get('attachments', []), params))
            else:
                from gateway.session_hosted_transport import source_attachment_digests
                manifest = payload.get('attachments', [])
                result.update(prompt=payload['prompt'], attachments=manifest,
                              attachment_digests=source_attachment_digests(self, member, room_id, manifest))
        return result


    def begin_disband(self, room_id):
        """Persist the admission fence before waiting for any remote Stop or cleanup."""
        import json
        gateway_id, epoch = self._owned_authority(room_id)
        self.authority.db._execute_write(lambda conn: conn.execute(
            'INSERT OR IGNORE INTO state_meta(key,value) VALUES (?,?)',
            ('gateway.peer.retiring.v1:' + room_id, json.dumps([gateway_id, epoch]))))

    def is_retiring(self, room_id):
        with self.authority.db._read_ctx() as conn:
            return conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                                ('gateway.peer.retiring.v1:' + room_id,)).fetchone() is not None

    def send(self, *, room_id, **kwargs):
        with self.peer_route_lock:
            if self.is_retiring(room_id):
                raise RuntimeStoreError('room_retiring')
            return super().send(room_id=room_id, **kwargs)

    def _resume_disbands(self):
        import json
        import time
        from gateway import hosted_rooms
        from tui_gateway.hosted_room_peer_http import room_grant_request_budget, room_grant_request_budget_remaining
        if not self._disband_upkeep_lock.acquire(blocking=False):
            return  # Stop resolves its binding through this provider too.
        try:
            with self.authority.db._read_ctx() as conn:
                pending = conn.execute('SELECT key,value FROM state_meta WHERE key LIKE ?',
                                       ('gateway.peer.retiring.v1:%',)).fetchall()
            now = time.time()
            with room_grant_request_budget(2):
                for row in sorted(pending, key=lambda row: self._disband_retry_at.get(row['key'], 0)):
                    if room_grant_request_budget_remaining() <= 0:
                        break
                    if self._disband_retry_at.get(row['key'], 0) > now:
                        continue
                    self._disband_retry_at[row['key']] = now + 5
                    room_id = row['key'][len('gateway.peer.retiring.v1:'):]
                    try:
                        with self.peer_route_lock:
                            room = hosted_rooms.room_state(self.db_path, room_id=room_id, include_disbanded=True)
                            if room.get('disbanded_at') is not None:
                                continue
                            gateway_id, epoch = json.loads(row['value'])
                            if (room['authority_gateway_id'], room['authority_epoch']) != (gateway_id, epoch):
                                continue
                            self.stop_room(room_id, cancel_id='room-disbanded', require_acknowledged=True)
                            self.revoke_room_routes(room_id)
                            hosted_rooms.disband_room(self.db_path, room_id=room_id,
                                                      expected_gateway_id=gateway_id, expected_epoch=epoch)
                    except Exception:
                        # The durable fence remains; a later cycle resumes exact Stop.
                        continue
        finally:
            self._disband_upkeep_lock.release()


    def revoke_room_routes(self, room_id):
        from gateway import hosted_room_links, hosted_rooms
        from gateway import session_group_peer_cleanup as cleanup
        with self.peer_route_lock:
            routes = [link for link in hosted_room_links.load_room_links(self.db_path) if link.room_id == room_id]
            # Journal all grants before contacting any target or removing any route.
            with hosted_rooms._transaction(self.db_path) as conn:
                for link in routes:
                    cleanup.retain(self.db_path, link, mode='scope', conn=conn)
            cleanup.drain(self, force=True, room_id=room_id)
            hosted_rooms.delete_room_link_records(self.db_path, room_id=room_id)
            with self._policy_lock:
                for link in routes:
                    key = (room_id, link.member_id)
                    for table in (self.peer_routes, self._peer_route_status, self.peer_clients):
                        table.pop(key, None)
            return len(routes)

    def status(self, room_id=None):
        from gateway import session_group_peer_cleanup as cleanup
        return {**super().status(room_id), 'peer_cleanup': cleanup.status(self.db_path, room_id),
                'retiring': self.is_retiring(room_id) if room_id is not None else False}

    def _maintain_peer_lifecycle(self):
        from gateway import session_group_peer_cleanup as cleanup
        cleanup.drain(self)
        self._resume_disbands()

    def bindings(self):
        with self.authority.db._read_ctx() as conn:
            owned = {r[0][len(_OWNER):] for r in conn.execute(
                'SELECT key FROM state_meta WHERE key LIKE ?', (_OWNER + '%',))}
        return tuple(b for b in super().bindings() if b.room_id in owned)

    def _turn_lock(self, profile):
        return nullcontext()

    def authorize_room(self, actor_subject, room_id, *, create=False):
        from gateway.hosted_rooms_common import IDENTIFIER_RE
        if (not isinstance(actor_subject, str) or not actor_subject
                or not isinstance(room_id, str) or len(room_id) > 128
                or not IDENTIFIER_RE.fullmatch(room_id)):
            raise RuntimeStoreError('invalid_params')
        def write(conn):
            _epoch(conn, self.authority.epoch)
            key = _OWNER + room_id
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            if row is None and create:
                historical = conn.execute(
                    'SELECT 1 FROM hosted_rooms WHERE room_id=? UNION ALL '
                    'SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?',
                    (room_id, room_id)).fetchone()
                if historical is None:
                    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, actor_subject))
                    return True
            if row is None or row[0] != actor_subject:
                raise RuntimeStoreError('permission_denied')
            return True
        return self.authority.db._execute_write(write)

    def _owner(self, room_id):
        with self.authority.db._read_ctx() as conn:
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_OWNER + room_id,)).fetchone()
        if row is None:
            raise RuntimeStoreError('permission_denied')
        return row[0]

    def _resolve_member_transport(self, binding, task):
        if self._member_is_peer(binding.room_id, str(task['payload'].get('target_member_id') or task['payload'].get('target_profile'))):
            from gateway.session_group_peers import refused_peer_turn
            refused = refused_peer_turn(self, binding.room_id, task)
            if refused is not None:
                return refused
            transport = super()._resolve_member_transport(binding, task)
            if task.get('status') == 'queued':
                # Bound before the dispatch is sent: Retry later requires this same authority.
                from gateway.session_group_peer_controls import capture_retry_binding
                transport.nonadmission_retry_binding = capture_retry_binding(
                    self, binding, task, transport.route, transport.client)
            return transport
        from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
        payload = task['payload']
        member = str(payload.get('target_member_id') or payload.get('target_profile'))
        profile = payload['target_profile']
        owner = self._owner(binding.room_id)
        home = self.profile_homes().get(profile)
        if home is None:
            raise RuntimeStoreError('permission_denied')
        key = binding.room_id, member, profile, owner, str(home)
        if key not in self.member_rpcs:
            if home != Path(self.authority.profile_id):
                from gateway.session_hosted_transport import HostedRoomOwnerRPC
                self.member_rpcs[key] = HostedRoomOwnerRPC(home=home,
                    source_home=self.authority.profile_id, room_id=binding.room_id,
                    member_id=member, profile=profile)
                return self.member_rpcs[key]
            def authorized(conn, operation, identity, generation):
                # Admission checks must share the FIFO writer's snapshot. Opening
                # another transaction here would reintroduce the revocation race.
                from gateway.hosted_rooms import _room_from_row
                from gateway.hosted_room_driver import (
                    _task_from_row, _require_room_authority, RoomUnavailableError, StaleLeaseError)
                _epoch(conn, self.authority.epoch)
                if operation in {'submit', 'execute'}:
                    try:
                        _require_room_authority(conn, binding.room_id, binding.gateway_id, binding.authority_epoch)
                    except (RoomUnavailableError, StaleLeaseError):
                        return False
                owned = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                     (_OWNER + binding.room_id,)).fetchone()
                if owned is None or owned[0] != owner:
                    return False
                stored = conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?',
                                      (binding.room_id,)).fetchone()
                if stored is None or stored['disbanded_at'] is not None:
                    return False
                room = _room_from_row(stored)
                if (room['authority_gateway_id'], room['authority_epoch']) != (binding.gateway_id, binding.authority_epoch):
                    return False
                members = room['members']
                if not any(m.get('member_id') == member and m.get('profile') == profile
                           and (m.get('target') is None or (
                               isinstance(m.get('target'), dict)
                               and m['target'].get('kind', 'local') == 'local'))
                           for m in members):
                    return False
                if self.profile_homes().get(profile) != home:
                    return False
                if identity is not None:
                    stored = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                                          (binding.room_id, identity.task_id)).fetchone()
                    if stored is None:
                        return False
                    current = _task_from_row(stored)
                    return (current['identity'] == identity and current['execution_generation'] == generation
                            and current['payload'].get('target_profile') == profile
                            and current['payload'].get('target_member_id', profile) == member
                            and current['status'] in ({'running'} if operation in {'submit', 'execute'}
                                                       else {'running', 'stopping'}))
                return True
            def authorize(operation, identity, generation):
                with self.authority.db._read_ctx() as conn:
                    if not authorized(conn, operation, identity, generation):
                        return False
                if operation == 'approve':
                    # Approval owns its short transaction; never open it inside
                    # the admission writer or while retaining this read context.
                    from gateway.hosted_room_approval import require_current_approval
                    if identity is None:
                        return False
                    current = require_current_approval(self, binding.room_id, member, identity.task_id, generation)
                    return current['identity'] == identity
                return True
            def authorize_write(conn, identity, generation):
                # Raise to refuse rather than return False: a guard that only returns False
                # is ignored wherever the admission hook signals refusal by raising.
                if not authorized(conn, 'submit', identity, generation):
                    raise RuntimeStoreError('permission_denied')
                return True
            principal = Principal(owner, self.authority.profile_id,
                frozenset({'session:create', 'session:read', 'session:submit', 'session:control', 'session:approve'}),
                'hosted:' + binding.room_id + ':' + member)
            self.member_rpcs[key] = HostedRoomAuthorityRPC(self.authority, self.loop,
                room_id=binding.room_id, member_id=member, profile=profile, principal=principal, authorize=authorize,
                authorize_write=authorize_write)
        return self.member_rpcs[key]

    def check_admission(self, ref, row):
        """Reconstruct the private producer from durable task state before claim."""
        import json
        from gateway.hosted_room_driver import TaskIdentity, list_tasks
        from gateway.session_hosted_attachments import committed_submission_payload
        from tui_gateway.hosted_room_driver import HostedRoomBinding
        try:
            if not row['request_id'].startswith('hosted:'):
                raise ValueError('not a hosted admission')
            identity, generation = json.loads(row['request_id'][7:])
            identity = TaskIdentity(**identity)
            if type(generation) is not int or generation < 1:
                raise ValueError('invalid generation')
            owner = self._owner(identity.room_id)
            if owner != row['principal_id']:
                raise ValueError('foreign owner')
            room = self._room(identity.room_id)
            task = next(t for t in list_tasks(self.db_path, room_id=identity.room_id)
                        if t['identity'] == identity and t['execution_generation'] == generation)
            rpc = self._resolve_member_transport(HostedRoomBinding(identity.room_id,
                room['authority_gateway_id'], room['authority_epoch']), task)
            if (getattr(rpc, 'ref', None) != ref or task['status'] != 'running'
                    or row['payload'] != committed_submission_payload(rpc, task['payload']['prompt'], task['payload'].get('attachments'))
                    or rpc.authorizer('execute', identity, generation) is not True):
                raise ValueError('changed hosted binding')
            return task
        except (ValueError, TypeError, KeyError, StopIteration) as exc:
            raise RuntimeStoreError('permission_denied') from exc

    def approve(self, *, session_id, request_id, choice, expected_task_id, expected_execution_generation):
        rpc = next((r for r in self.member_rpcs.values() if r.ref.session_id == session_id), None)
        if rpc is None:
            raise RuntimeStoreError('permission_denied')
        return rpc.approve(session_id=session_id, request_id=request_id, choice=choice,
            expected_task_id=expected_task_id, expected_execution_generation=expected_execution_generation)


async def ensure_hosted_service(runner):
    """Prepare every transport before readiness can release any coordinator."""
    active = active_authority(runner)
    if active is None:
        raise RuntimeStoreError('profile_mismatch')
    for authority in all_authorities(runner):
        await _ensure_hosted_service(runner, authority)
    start_ready_hosted_services(runner)
    return active.hosted_room_service


async def _ensure_hosted_service(runner, authority):
    with owner_scope(authority):
        service = getattr(authority, 'hosted_room_service', None)
        if service is None:
            loop = asyncio.get_running_loop()
            service = await asyncio.to_thread(CanonicalHostedRoomService, authority, loop)
            authority.hosted_room_service = service
        if not getattr(service, '_transport_installed', False):
            from gateway.session_hosted_transport import install_hosted_transport
            install_hosted_transport(runner.session_control_server, authority, asyncio.get_running_loop(),
                                     attest=service.attest)
            service._transport_installed = True


def start_ready_hosted_services(runner):
    """Start only a fully prepared served set behind the published ready gate."""
    if (getattr(runner, 'session_runtime_descriptor', {}).get('state') != 'ready'
            or getattr(runner, '_draining', False)):
        return
    services = [getattr(authority, 'hosted_room_service', None) for authority in all_authorities(runner)]
    if any(service is None or not getattr(service, '_transport_installed', False) for service in services):
        return
    for service in services:
        # start() only releases its thread; preparation and disk access happened above.
        service.start()


async def stop_hosted_service(runner, timeout=5):
    loop = asyncio.get_running_loop()
    deadline, stopped = loop.time() + max(0, timeout), True
    for authority in all_authorities(runner):
        service = getattr(authority, 'hosted_room_service', None)
        if service is not None:
            result = await asyncio.to_thread(service.stop, timeout=max(0, deadline - loop.time()))
            stopped = result and stopped
    return stopped
