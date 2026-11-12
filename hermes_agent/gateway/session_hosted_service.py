"""Authority-bound hosted service; no legacy session server is constructed."""
import asyncio
import hashlib
from contextlib import nullcontext, suppress
from pathlib import Path

from gateway.session_contract import SessionActor
from gateway.session_authorities import active_authority, all_authorities, owner_scope, serving_authorities
from gateway.session_hosted_controls import HostedControls
from gateway.owner_identity import same_owner
from hermes_agent.hermes_state_runtime import RuntimeStoreError, _epoch
from hermes_agent.tui_gateway.hosted_room_service import HostedRoomService

_OWNER = 'gateway.hosted.owner.v1:'
# What the Stop of an attempt asks its admitting home after the profile moved: never create, submit or approve.
_STOP_READS = frozenset({'resolve_exact', 'resume', 'info', 'interrupt', 'history', 'results', 'discard'})
# A hosted attempt whose room task has moved past it: its queued member input can never run.
_RETIRED_ATTEMPT_STATUSES = frozenset({'queued', 'deferred', 'settled', 'failed', 'cancelled'})


def attempt_retired(task, generation):
    """Whether *task* has conclusively left attempt *generation*: deferred, finished or requeued at it,
    or restarted under a newer one. Running, stopping and indeterminate attempts may still run. Only
    a member on another gateway reopens a deferred generation on Retry; this checks members admitted on
    this host, whose Retry mints a new generation (``requeue_deferred_task``)."""
    if type(generation) is not int:
        return False
    current = int(task['execution_generation'])
    return current > generation or (current == generation and task['status'] in _RETIRED_ATTEMPT_STATUSES)


def retired_attempt_conflict():
    return RuntimeStoreError('admission_conflict', field='task',
                             detail='the room task no longer runs this attempt')


def probe_native_room_owner(db_path, *, room_id) -> bool:
    """Non-blocking, read-only: does the gateway owning this session store serve ``room_id``? Only rooms
    with an owner entry are in ``CanonicalHostedRoomService.bindings``."""
    from gateway.hosted_rooms import _probe
    return _probe(Path(db_path), 'state_meta', 'SELECT 1 FROM state_meta WHERE key=? LIMIT 1',
                  (_OWNER + str(room_id),), 'native room ownership is temporarily unavailable')


class CanonicalHostedRoomService(HostedControls, HostedRoomService):
    def __init__(self, authority, loop):
        self.authority, self.loop = authority, loop
        self.member_rpcs = {}
        super().__init__(None, db_path=authority.db.db_path)

    def _make_rpc(self, server):
        # Member-specific canonical transports retain exact durable history. They
        # intentionally use the runtime's receipt-capable (non-legacy) recovery path.
        return self

    def profile_homes(self):
        from gateway.session_hosted_members import member_roster
        return member_roster(self.authority)[0]

    def local_profiles(self):
        return tuple(self.profile_homes())

    def room_profiles(self, room):
        """A saved room keeps every local member it seated: unserving one bot must not stop the
        others. That member's own turns still fail (``_resolve_member_transport`` and ``attest``
        need a live home), and a new group cannot seat it (``local_profiles``)."""
        seated = {m.get('profile') for m in room.get('members') or ()
                  if isinstance(m, dict) and isinstance(m.get('profile'), str)
                  and (m.get('target') or {}).get('kind', 'local') == 'local'}
        return tuple(sorted(set(self.local_profiles()) | seated))

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
        claimed = params.get('_target_home')
        if (operation in _STOP_READS and claimed != str(target_home)
                and claimed in self._stopping_attempt_homes(room_id, member, params.get('expected_task_id'))):
            # The home that admitted a stopping attempt still answers for it; so does every home
            # a turn from before attempt homes were recorded may have been admitted at.
            target_home = claimed
        if target_home is None or claimed != str(target_home):
            raise RuntimeStoreError('permission_denied')
        result = {'owner': owner, 'target_home': str(target_home)}
        if operation == 'execute':
            # The exact attempt is over (deferred, finished, or retried as a newer generation): its
            # queued input can never run. Say so with the committed input, so the target still
            # hard-denies a changed row or owner before letting the member's queue move on.
            from gateway.hosted_room_driver import get_task, TaskIdentity
            with suppress(KeyError, TypeError, ValueError):
                stored = get_task(self.db_path, TaskIdentity(**params.get('task')))
                if (stored['identity'].room_id == room_id
                        and stored['payload'].get('target_member_id', stored['payload']['target_profile']) == member
                        and stored['payload']['target_profile'] == profile
                        and attempt_retired(stored, params.get('execution_generation'))):
                    from gateway.session_hosted_transport import source_attachment_digests
                    manifest = stored['payload'].get('attachments', [])
                    return {**result, 'retired': True, 'prompt': stored['payload']['prompt'], 'attachments': manifest,
                            'attachment_digests': source_attachment_digests(self, member, room_id, manifest)}
        if operation in {'submit', 'execute', 'attachment'}:
            matches = [t for t in list_tasks(self.db_path, room_id=room_id)
                       if asdict(t['identity']) == params.get('task')
                       and t['execution_generation'] == params.get('execution_generation')
                       and t['status'] == 'running'
                       and t['payload'].get('target_member_id', t['payload']['target_profile']) == member
                       and t['payload']['target_profile'] == profile
                       and (operation == 'execute' or (
                           (hashlib.sha256(t['payload']['prompt'].encode('utf-8')).hexdigest()
                            == params['prompt_sha256'] if 'prompt_sha256' in params
                            else t['payload']['prompt'] == params.get('prompt'))
                           and t['payload'].get('attachments', []) == (params.get('attachments') or [])))]
            if len(matches) != 1:
                raise RuntimeStoreError('permission_denied')
            if operation == 'submit':
                from gateway.session_hosted_member_homes import record_attempt_home
                record_attempt_home(self.authority.db, room_id, member, matches[0]['identity'].task_id,
                                    matches[0]['execution_generation'], target_home)
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


    def bindings(self):
        with self.authority.db._read_ctx() as conn:
            owned = {r[0][len(_OWNER):] for r in conn.execute(
                'SELECT key FROM state_meta WHERE key LIKE ?', (_OWNER + '%',))}
        return tuple(b for b in super().bindings() if b.room_id in owned)

    def _turn_lock(self, profile):
        return nullcontext()

    def authorize_room(self, actor_subject, room_id, *, create=False, conn=None):
        """Check (or, with ``create``, claim) room ownership. A ``conn`` runs the claim inside
        the caller's room-creation transaction, so owner and room commit or roll back together."""
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
            if row is None or not same_owner(row[0], actor_subject):
                raise RuntimeStoreError('permission_denied')
            return True
        return write(conn) if conn is not None else self.authority.db._execute_write(write)

    def _owner(self, room_id):
        with self.authority.db._read_ctx() as conn:
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_OWNER + room_id,)).fetchone()
        if row is None:
            raise RuntimeStoreError('permission_denied')
        return row[0]

    def _resolve_member_transport(self, binding, task):
        if self._member_is_peer(binding.room_id, str(task['payload'].get('target_member_id') or task['payload'].get('target_profile'))):
            return super()._resolve_member_transport(binding, task)
        from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
        payload = task['payload']
        member = str(payload.get('target_member_id') or payload.get('target_profile'))
        profile = payload['target_profile']
        owner = self._owner(binding.room_id)
        from gateway.session_hosted_member_homes import stopping_attempt_home
        # A Stop is asked where its attempt was admitted, wherever the profile points now.
        home = (stopping_attempt_home(self.authority.db, binding.room_id, member, task)
                or self.profile_homes().get(profile))
        if home is None:
            raise RuntimeStoreError('permission_denied')
        key = binding.room_id, member, profile, owner, str(home)
        if home != Path(self.authority.profile_id):
            from gateway.session_hosted_member_homes import record_member_home
            # Every resolve, not only a new transport: a profile re-added to hosted_rooms.profiles
            # reuses its cached transport, and its home must still be recorded.
            record_member_home(self.authority.db, binding.room_id, member, home)
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
                from gateway.hosted_room_driver import _task_from_row
                _epoch(conn, self.authority.epoch)
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
                    return authorized(conn, operation, identity, generation)
            def authorize_write(conn, identity, generation):
                # Raise to refuse rather than return False: a guard that only returns False
                # is ignored wherever the admission hook signals refusal by raising.
                if not authorized(conn, 'submit', identity, generation):
                    raise RuntimeStoreError('permission_denied')
                return True
            principal = SessionActor(owner, self.authority.profile_id,
                frozenset({'session:create', 'session:read', 'session:submit', 'session:control', 'session:approve'}),
                'hosted:' + binding.room_id + ':' + member)
            self.member_rpcs[key] = HostedRoomAuthorityRPC(self.authority, self.loop,
                room_id=binding.room_id, member_id=member, profile=profile, principal=principal, authorize=authorize,
                authorize_write=authorize_write)
        return self.member_rpcs[key]

    def _stopping_attempt_homes(self, room_id, member, task_id=None):
        from gateway.session_hosted_member_homes import legacy_stop_homes, stopping_attempt_homes
        db = self.authority.db
        return stopping_attempt_homes(db, room_id, member) | legacy_stop_homes(db, room_id, member, task_id)

    def _stop_pending_refusal(self, room_id, deferred):
        """Name an unconfigured member profile whose unknown home keeps a turn deferred."""
        from gateway.session_hosted_member_homes import unknown_home_refusal
        from hermes_agent.tui_gateway.hosted_room_service import task_member_id
        return unknown_home_refusal(self, room_id, [
            task for task in deferred if not self._member_is_peer(room_id, task_member_id(task))])

    def _stop_deferred_peer(self, room_id, task):
        """Also settle a local member's deferred turn once its profile is no longer served here and
        nothing on this host can run it (``unserved_member_stopped``): its daemon never answers."""
        from gateway.session_hosted_member_homes import unsettled_legacy_homes, unserved_member_stopped
        if super()._stop_deferred_peer(room_id, task):
            return not unsettled_legacy_homes(self, room_id, task)
        from hermes_agent.tui_gateway.hosted_room_service import task_member_id
        return (not self._member_is_peer(room_id, task_member_id(task))
                and unserved_member_stopped(self, room_id, task))

    def check_admission(self, ref, row):
        """Reconstruct the private producer from durable task state before claim."""
        import json
        from gateway.hosted_room_driver import TaskIdentity, list_tasks
        from gateway.session_hosted_attachments import committed_submission_payload
        from hermes_agent.tui_gateway.hosted_room_driver import HostedRoomBinding
        try:
            if not row['request_id'].startswith('hosted:'):
                raise ValueError('not a hosted admission')
            identity, generation = json.loads(row['request_id'][7:])
            identity = TaskIdentity(**identity)
            if type(generation) is not int or generation < 1:
                raise ValueError('invalid generation')
            owner = self._owner(identity.room_id)
            if not same_owner(owner, row['principal_id']):
                raise ValueError('foreign owner')
            room = self._room(identity.room_id)
            task = next(t for t in list_tasks(self.db_path, room_id=identity.room_id) if t['identity'] == identity)
            rpc = self._resolve_member_transport(HostedRoomBinding(identity.room_id,
                room['authority_gateway_id'], room['authority_epoch']), task)
            if (getattr(rpc, 'ref', None) != ref
                    or row['payload'] != committed_submission_payload(rpc, task['payload']['prompt'], task['payload'].get('attachments'))):
                raise ValueError('changed hosted binding')
            retired = attempt_retired(task, generation)
            # A retired attempt still needs a seated, served member: only the attempt check is skipped.
            if rpc.authorizer('execute', None if retired else identity, None if retired else generation) is not True:
                raise ValueError('changed hosted binding')
        except (ValueError, TypeError, KeyError, StopIteration) as exc:
            raise RuntimeStoreError('permission_denied') from exc
        if retired:
            raise retired_attempt_conflict()
        return task

    def approve(self, *, session_id, request_id, choice):
        rpc = next((r for r in self.member_rpcs.values() if r.ref.session_id == session_id), None)
        if rpc is None:
            raise RuntimeStoreError('permission_denied')
        return rpc.approve(session_id=session_id, request_id=request_id, choice=choice)


async def ensure_hosted_service(runner):
    """Prepare every transport before readiness can release any coordinator."""
    active = active_authority(runner)
    if active is None:
        raise RuntimeStoreError('profile_mismatch')
    for authority in serving_authorities(runner):
        await _ensure_hosted_service(runner, authority)
    start_ready_hosted_services(runner)
    return active.hosted_room_service


async def _ensure_hosted_service(runner, authority):
    with owner_scope(authority):
        service = getattr(authority, 'hosted_room_service', None)
        if service is None:
            loop = asyncio.get_running_loop()
            from gateway.hosted_rooms_gateway_import import import_standalone_rooms
            await asyncio.to_thread(import_standalone_rooms, Path(authority.db.db_path), runtime_epoch=authority.epoch)
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
    services = [getattr(authority, 'hosted_room_service', None) for authority in serving_authorities(runner)]
    if any(service is None or not getattr(service, '_transport_installed', False) for service in services):
        return
    for service in services:
        # start() only releases its thread; preparation and disk access happened above.
        service.start()
    from gateway.hosted_rooms_gateway_import import import_after_ready
    import_after_ready(serving_authorities(runner))


async def stop_hosted_service(runner, timeout=5):
    loop = asyncio.get_running_loop()
    deadline, stopped = loop.time() + max(0, timeout), True
    for authority in all_authorities(runner):
        service = getattr(authority, 'hosted_room_service', None)
        if service is not None:
            result = await asyncio.to_thread(service.stop, timeout=max(0, deadline - loop.time()))
            stopped = result and stopped
    return stopped
