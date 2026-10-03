"""Authority-bound hosted service; no legacy session server is constructed."""
import asyncio
from contextlib import nullcontext
from pathlib import Path
import threading
import time

from gateway.session_contract import Principal
from gateway.session_authorities import active_authority, all_authorities, owner_scope
from gateway.session_hosted_controls import HostedControls
from gateway.session_hosted_output_publication import CanonicalHostedOutput
from hermes_state_runtime import RuntimeStoreError, _epoch
from tui_gateway.hosted_room_service import HostedRoomService

_OWNER = 'gateway.hosted.owner.v1:'


def _held_as_copy(conn, room_id):
    """Whether this installation keeps ``room_id`` as a copy of another host's room."""
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_replicas'").fetchone()
    return exists is not None and conn.execute(
        'SELECT 1 FROM hosted_room_replicas WHERE room_id=?', (room_id,)).fetchone() is not None


class CanonicalHostedRoomService(CanonicalHostedOutput, HostedControls, HostedRoomService):
    def __init__(self, authority, loop):
        self._remembered_attempts = {}  # tries per request a chat's remembered approval may answer
        self.authority, self.loop = authority, loop
        self.member_rpcs = {}
        # Serializes every peer route publication (registration, renewal) with Disband.
        self.peer_route_lock = threading.RLock()
        self._disband_upkeep_lock = threading.Lock()
        self._disband_retry_at = {}
        self._peer_cleanup_inflight = set()
        self._peer_renewals, self._peer_renewal_scans = {}, {}  # session_group_peer_routes
        super().__init__(None, db_path=authority.db.db_path)
        self.runtime.retire_stale_output = self.retire_stale_output
        from gateway.hosted_room_replication import HostedRoomReplicationPublisher
        # Copies history to the room's custodians; idle until such a route exists.
        self.replication = HostedRoomReplicationPublisher(self.db_path)
        # A task held back until its admission is copied runs as soon as a custodian acknowledges it.
        self.replication.on_acknowledged = lambda room_id: self.runtime.wakeup()

    def start(self):
        super().start()
        self.replication.start()
        self.succession.start()

    def stop(self, *, timeout=5.0):
        deadline = time.monotonic() + max(0.0, timeout)
        self.replication.stop(timeout=0)  # Signal first; the runtime's stop may take the budget.
        self.succession.stop(timeout=0)
        runtime_stopped = super().stop(timeout=max(0.0, deadline - time.monotonic()))
        succession_stopped = self.succession.stop(timeout=max(0.0, deadline - time.monotonic()))
        return (self.replication.stop(timeout=max(0.0, deadline - time.monotonic())) and runtime_stopped
                and succession_stopped)

    @property
    def succession(self):
        """Host-loss upkeep for the groups this computer hosts or keeps a copy of (default profile only)."""
        upkeep = self.__dict__.get('_succession_upkeep')
        if upkeep is None:
            from gateway.hosted_room_succession_status import SuccessionUpkeep
            upkeep = self.__dict__['_succession_upkeep'] = SuccessionUpkeep(
                self.succession_context, lease_context=self.lease_context,
                runner=getattr(self.authority, 'runner', None), loop=self.loop)
        return upkeep

    def publish_settled(self, room_id):
        """Publish the outcomes of this room's turns that have settled, and nothing else: no new turn is
        planned (a host handing its group over drains its running turns this way)."""
        from gateway import hosted_rooms
        with self._policy_lock:
            room = self._room(room_id)
            self._policy_snapshot(room)  # sync() feeds the publication
            try:
                self._publish_terminal_tasks(room)
            except hosted_rooms.EventCursorConflictError:
                pass  # published on the next look

    def lease_context(self):
        """What holding this service's own groups' leases needs: its room store, and its fence store
        when it has one. A host without the API server or a durable Runs store still holds its leases."""
        try:
            context = self.succession_context()
        except Exception:
            context = None  # its API server isn't up yet: the room store alone still holds its leases
        if context is not None:
            return context
        from gateway.hosted_room_succession_move import MoveContext
        return MoveContext(db_path=Path(self.db_path), runs_store=None, service=self, timeout=2.0)

    def succession_context(self, actor=None):
        """The succession context of this computer, or ``None`` where groups cannot move here."""
        from gateway.session_authorities import served_profile_name
        from gateway.session_group_peers import _api_server
        if served_profile_name(Path(self.authority.profile_id)) != 'default':
            return None
        adapter = _api_server(self.authority)
        store = getattr(adapter, '_run_idempotency_store', None)
        if store is None or store.durable is not True:
            return None
        from gateway.hosted_room_succession_move import MoveContext
        from gateway.platforms.api_server_room_succession import continuation_minter
        return MoveContext(db_path=Path(self.db_path), runs_store=store, service=self, timeout=2.0,
                           mint_grants=continuation_minter(adapter, Path(self.db_path), replace_same_epoch=False))

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
        from gateway.hosted_room_replica_retirement import bind_home_proof
        bind_home_proof(self.db_path, room_id=room_id)
        self.replication.wakeup()

    def _track_peer_client(self, binding, key, route, client):
        from gateway.session_group_peer_routes import CanonicalPeerClient
        return CanonicalPeerClient(self, binding, key, route, client)

    def _runtime_options(self):
        # A peer turn its gateway never received is deferred with that proof, so the room's
        # next turn runs; Retry (HostedControls) requeues it. Peer grants renew in the room's cycle.
        return {'defer_not_admitted_members': True, 'maintain_leased_room': self._maintain_peer_grants,
                'maintain_service': self._maintain_peer_lifecycle, 'dispatch_ready': self._dispatch_ready}

    def _dispatch_ready(self, binding, task):
        # In majority mode a task runs only once a majority of voters stored its task.admitted.
        from gateway.hosted_room_custody import dispatch_ready
        return dispatch_ready(self.db_path, binding.room_id, task['identity'].task_id, task['execution_generation'])

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
        if operation == 'output_scope':
            from gateway.session_hosted_output_owner import attest_output_scope
            result['scope'] = attest_output_scope(self, room_id, member, profile, params)
            return result
        from gateway.session_hosted_output_owner import OUTPUT_OPERATIONS
        if operation in OUTPUT_OPERATIONS:
            from gateway.session_hosted_output_owner import attest_output_action
            result.update(attest_output_action(self, room_id, member, profile, operation, params))
            return result
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
        """Persist the admission fence and copy proof before remote Stop or route cleanup."""
        from gateway.hosted_room_replica_retirement import bind_home_proof
        bind_home_proof(self.db_path, room_id=room_id)
        import json
        gateway_id, epoch = self._owned_authority(room_id)
        self.authority.db._execute_write(lambda conn: conn.execute(
            'INSERT OR IGNORE INTO state_meta(key,value) VALUES (?,?)',
            ('gateway.peer.retiring.v1:' + room_id, json.dumps([gateway_id, epoch]))))

    def is_retiring(self, room_id):
        with self.authority.db._read_ctx() as conn:
            return conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                                ('gateway.peer.retiring.v1:' + room_id,)).fetchone() is not None

    def _send_unpaused(self, *, room_id, **kwargs):
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
        result = {**super().status(room_id), 'peer_cleanup': cleanup.status(self.db_path, room_id),
                  'retiring': self.is_retiring(room_id) if room_id is not None else False,
                  'replication': self.replication.status(room_id)}
        if room_id is not None:
            from gateway.hosted_room_succession import waiting_tasks
            result['tasks'] = waiting_tasks(self.db_path, room_id)
        return result

    def _maintain_peer_lifecycle(self):
        from gateway import session_group_peer_cleanup as cleanup
        cleanup.drain(self)
        self._resume_disbands()
        self._maintain_succession()

    def _maintain_succession(self):
        """A peer that refused a hosted room's authority makes upkeep ask the group at once."""
        with self._policy_lock:
            suspect = {room for (room, _member), status in self._peer_route_status.items()
                       if status in {'needs_reauthorization', 'unavailable'}}
        if suspect:
            self.succession.suspect.update(suspect)
            self.succession.wakeup()

    def bindings(self):
        from gateway.hosted_room_succession_backup import serving_locked
        with self.authority.db._read_ctx() as conn:
            owned = {r[0][len(_OWNER):] for r in conn.execute(
                'SELECT key FROM state_meta WHERE key LIKE ?', (_OWNER + '%',))}
        # A host that paused (its epoch promised elsewhere, or continued on two) executes nothing.
        from contextlib import closing
        from gateway import hosted_rooms
        with closing(hosted_rooms._read_connection(Path(self.db_path))) as conn:
            return tuple(b for b in super().bindings() if b.room_id in owned and serving_locked(conn, b.room_id))

    def _room(self, room_id):
        room = super()._room(room_id)
        from gateway.hosted_room_succession import authority_lineage
        lineage = authority_lineage(self.db_path, room_id)
        return {**room, 'authority_lineage': lineage} if lineage else room

    def send(self, *, room_id, **kwargs):
        from gateway.hosted_room_succession import paused_reason
        reason = paused_reason(self.db_path, room_id)
        if reason is not None:
            raise RuntimeStoreError(reason)
        return self._send_unpaused(room_id=room_id, **kwargs)

    def announce_restart(self, until):
        """A planned restart is not loss: tell every group this computer hosts when it will be back."""
        from gateway.hosted_room_succession_move import append_state
        from gateway.hosted_room_succession_status import held_rooms
        context = self.succession_context()
        if context is None:
            return []
        rooms = held_rooms(context.db_path)['hosted']
        # In majority mode the voters keep this host's lease through the window, so no one takes over.
        from gateway.hosted_room_succession_automatic import extend_for_restart
        extend_for_restart(context, rooms, float(until))
        for room_id in rooms:
            append_state(context.db_path, room_id, 'host_restarting', until=float(until))
        self.replication.wakeup()
        return rooms

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
            if create and _held_as_copy(conn, room_id):
                # Its owner row may exist (consent to keep a copy): a hosted room never reuses the id.
                raise RuntimeStoreError('permission_denied')
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

    def stop_work(self, room_id, *, cancel_id):
        """Cancel this room's work without writing to its log, as a host stepping down must: unlike
        ``stop_room`` it appends no ``room.stop_requested``. Returns how many tasks it cancelled."""
        from tui_gateway.hosted_room_service import _STOPPABLE_STATUSES
        cancelled = 0
        with self._policy_lock:
            for task in self._list_tasks(room_id, _STOPPABLE_STATUSES):
                try:
                    self.runtime.cancel(task['identity'], cancel_id=cancel_id)
                    cancelled += 1
                except Exception:
                    continue  # settled meanwhile, or already stopping
        self.runtime.wakeup()
        return cancelled

    def _owner(self, room_id):
        with self.authority.db._read_ctx() as conn:
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_OWNER + room_id,)).fetchone()
        if row is None:
            raise RuntimeStoreError('permission_denied')
        return row[0]

    def _policy_profiles(self, room):
        # A room this gateway holds as a successor keeps its original home's local Bots there.
        from gateway.hosted_room_succession import inherited_origin
        from gateway.hosted_room_succession_move import policy_profiles
        if inherited_origin(self.db_path, str(room['room_id'])) is not None:
            return policy_profiles(room)
        return super()._policy_profiles(room)

    def _resolve_member_transport(self, binding, task):
        member_id = str(task['payload'].get('target_member_id') or task['payload'].get('target_profile'))
        if not self._member_is_peer(binding.room_id, member_id):
            from gateway.hosted_room_succession import inherited_origin
            origin = inherited_origin(self.db_path, binding.room_id)
            if origin is not None:
                # Never run the original home's Bot as a same-named profile of this successor:
                # its turn waits for that computer.
                return WaitingForHost(self.db_path, binding.room_id, origin)
        if self._member_is_peer(binding.room_id, member_id):
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
            def authorized(conn, operation, identity, generation, *, new_write=False):
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
                execute = new_write or operation == 'execute'
                if execute and conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                        ('gateway.peer.retiring.v1:' + binding.room_id,)).fetchone():
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
                    if execute and conn.execute("SELECT 1 FROM hosted_room_events WHERE room_id=? AND seq>? "
                            "AND kind='room.stop_requested' LIMIT 1",
                            (binding.room_id, current['payload']['source_event_seq'])).fetchone():
                        return False
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
                    # Approval owns its short transaction, outside the admission writer/read context.
                    from gateway.hosted_room_approval import require_current_approval
                    if identity is None:
                        return False
                    current = require_current_approval(self, binding.room_id, member, identity.task_id, generation)
                    return current['identity'] == identity
                return True
            def authorize_write(conn, identity, generation):
                # Raise to refuse rather than return False: a guard that only returns False
                # is ignored wherever the admission hook signals refusal by raising.
                if not authorized(conn, 'submit', identity, generation, new_write=True):
                    raise RuntimeStoreError('permission_denied')
                return True
            principal = Principal(owner, self.authority.profile_id,
                frozenset({'session:create', 'session:read', 'session:submit', 'session:control', 'session:approve'}),
                'hosted:' + binding.room_id + ':' + member)
            self.member_rpcs[key] = HostedRoomAuthorityRPC(self.authority, self.loop,
                room_id=binding.room_id, member_id=member, profile=profile, principal=principal, authorize=authorize,
                authorize_write=authorize_write)
        return self.member_rpcs[key]

    def check_admission(self, ref, row, *, _for_claim=False):
        """Validate the producer; the drain also requests its exact writer guard."""
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
                    or row['payload'] != committed_submission_payload(rpc, task['payload']['prompt'], task['payload'].get('attachments'), admission=row)
                    or rpc.authorizer('execute', identity, generation) is not True):
                raise ValueError('changed hosted binding')
            if _for_claim:
                from copy import deepcopy
                if not callable(rpc.authorize_write):
                    raise ValueError('same-home claim has no writer authority')
                expected = deepcopy(row)

                def authorize_claim(conn, selected):
                    if selected['admission_id'] != expected['admission_id']:
                        raise RuntimeStoreError('admission_changed')
                    if selected != expected:
                        raise RuntimeStoreError('permission_denied')
                    rpc.authorize_write(conn, identity, generation)

                return authorize_claim
            return task
        except (ValueError, TypeError, KeyError, StopIteration) as exc:
            raise RuntimeStoreError('permission_denied') from exc

    def approve_room_task(self, room_id, *, member_id, task_id, execution_generation, choice,
                          request_id=None, remember=None):
        """``always`` approves once and remembers the exact operation for the chat in ``remember``."""
        if choice == 'always':
            from gateway.group_chat_rules import remember_approval
            return remember_approval(self, room_id, member_id=member_id, task_id=task_id,
                                     execution_generation=execution_generation, request_id=request_id,
                                     remember=remember)
        if remember is not None:
            raise RuntimeStoreError('invalid_params')
        return super().approve_room_task(room_id, member_id=member_id, task_id=task_id,
                                         execution_generation=execution_generation, choice=choice,
                                         request_id=request_id)

    def _set_pending_action(self, room_id, member_id, action):
        super()._set_pending_action(room_id, member_id, action)
        if action is not None:
            # A chat's "Always allow in this chat" answers a matching request as soon as it is seen.
            from gateway.group_chat_rules import apply_remembered
            apply_remembered(self, room_id, member_id, action)

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
        # Finish any shared-file cleanup a restart interrupted before new turns run.
        from gateway.session_hosted_output import replay_output_cleanups
        await asyncio.to_thread(replay_output_cleanups, authority)


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


class WaitingForHost:
    """The transport of a Bot that runs only on the group's original host: its turns wait for it.

    Nothing is sent anywhere. Each attempt is deferred with proof it never ran, reason
    ``waiting_for_host`` and the missing resource, so the room's next turn proceeds.
    """

    def __init__(self, db_path, room_id, origin):
        from types import SimpleNamespace
        from gateway import hosted_room_succession as succession
        from contextlib import closing
        from gateway import hosted_rooms
        with closing(hosted_rooms._read_connection(Path(db_path))) as conn:
            configuration = succession.configuration_locked(conn, room_id)
        self.host_name = succession.label(configuration, origin)
        self.ref = SimpleNamespace(session_id='waiting-for-host')

    def _session(self, **_):
        return {'session_id': 'waiting-for-host'}

    resolve_exact = create = resume = _session

    def submit(self, **_):
        from gateway.hosted_room_succession import WaitingForHostError
        raise WaitingForHostError(resource='bot', host_name=self.host_name)

    def info(self, **_):
        return {'active': False, 'task_id': None}

    def history(self, **_):
        return ()

    def interrupt(self, **_):
        return None
