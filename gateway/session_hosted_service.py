"""Authority-bound hosted service; no legacy session server is constructed."""
import asyncio
import hashlib
from contextlib import nullcontext, suppress
from pathlib import Path

from gateway.session_contract import SessionActor
from gateway.session_authorities import active_authority, all_authorities, owner_scope, serving_authorities
from gateway.session_hosted_controls import HostedControls
from gateway.owner_identity import same_owner
from gateway.session_persistence import authority_persistence, owner_rule
from persistence.core.errors import AuthorityRefusal
from persistence.gateway.hosted_rooms import HostedOwnerClaim, HostedTaskIdentity
from tui_gateway.hosted_room_service import HostedRoomService

# What the Stop of an attempt asks its admitting home after the profile moved: never create, submit or approve.
_STOP_READS = frozenset({'resolve_exact', 'resume', 'info', 'interrupt', 'history', 'results', 'discard'})
# A hosted attempt whose room task has moved past it: its queued member input can never run.
_RETIRED_ATTEMPT_STATUSES = frozenset({'queued', 'deferred', 'settled', 'failed', 'cancelled'})


def _check_claim(actor_subject, room_id):
    """Refuse ``invalid_params`` for an owner claim on a malformed actor or room id, as sent."""
    from gateway.hosted_rooms_common import IDENTIFIER_RE
    if (not isinstance(actor_subject, str) or not actor_subject
            or not isinstance(room_id, str) or len(room_id) > 128
            or not IDENTIFIER_RE.fullmatch(room_id)):
        raise AuthorityRefusal('invalid_params')


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
    return AuthorityRefusal('admission_conflict', field='task',
                            detail='the room task no longer runs this attempt')


def probe_native_room_owner(store, *, room_id) -> bool:
    """Non-blocking, read-only: does the gateway owning this session store serve ``room_id``? Only rooms
    with an owner entry are in ``CanonicalHostedRoomService.bindings``."""
    return store.hosted_rooms.probe_room_owner(str(room_id))


class CanonicalHostedRoomService(HostedControls, HostedRoomService):
    def __init__(self, authority, loop):
        self.authority, self.loop = authority, loop
        self.member_rpcs = {}
        super().__init__(None, store=authority_persistence(authority), placement='profile')

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
            raise AuthorityRefusal('invalid_params')
        room_id, member, profile = (selector[k] for k in ('room_id', 'member_id', 'profile'))
        owner = self._owner(room_id)
        self._owned_authority(room_id)
        room = self._room(room_id)
        if not any(m['member_id'] == member and m['profile'] == profile
                   and m.get('target', {}).get('kind', 'local') == 'local' for m in room['members']):
            raise AuthorityRefusal('permission_denied')
        target_home = self.profile_homes().get(profile)
        claimed = params.get('_target_home')
        if (operation in _STOP_READS and claimed != str(target_home)
                and claimed in self._stopping_attempt_homes(room_id, member, params.get('expected_task_id'))):
            # The home that admitted a stopping attempt still answers for it; so does every home
            # a turn from before attempt homes were recorded may have been admitted at.
            target_home = claimed
        if target_home is None or claimed != str(target_home):
            raise AuthorityRefusal('permission_denied')
        result = {'owner': owner, 'target_home': str(target_home)}
        if operation == 'execute':
            # The exact attempt is over (deferred, finished, or retried as a newer generation): its
            # queued input can never run. Say so with the committed input, so the target still
            # hard-denies a changed row or owner before letting the member's queue move on.
            from gateway.hosted_room_driver import get_task
            with suppress(KeyError, TypeError, ValueError):
                stored = get_task(self.store, HostedTaskIdentity(**params.get('task')), placement=self.placement)
                if (stored['identity'].room_id == room_id
                        and stored['payload'].get('target_member_id', stored['payload']['target_profile']) == member
                        and stored['payload']['target_profile'] == profile
                        and attempt_retired(stored, params.get('execution_generation'))):
                    from gateway.session_hosted_transport import source_attachment_digests
                    manifest = stored['payload'].get('attachments', [])
                    return {**result, 'retired': True, 'prompt': stored['payload']['prompt'], 'attachments': manifest,
                            'attachment_digests': source_attachment_digests(self, member, room_id, manifest)}
        if operation in {'submit', 'execute', 'attachment'}:
            matches = [t for t in list_tasks(self.store, placement=self.placement, room_id=room_id)
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
                raise AuthorityRefusal('permission_denied')
            if operation == 'submit':
                from gateway.session_hosted_member_homes import record_attempt_home
                record_attempt_home(authority_persistence(self.authority), room_id, member,
                                    matches[0]['identity'].task_id, matches[0]['execution_generation'], target_home)
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
        owned = authority_persistence(self.authority).hosted_rooms.owned_rooms()
        return tuple(b for b in super().bindings() if b.room_id in owned)

    def _turn_lock(self, profile):
        return nullcontext()

    def authorize_room(self, actor_subject, room_id):
        """Check room ownership."""
        _check_claim(actor_subject, room_id)
        authority_persistence(self.authority).hosted_rooms.claim_owner(
            epoch=self.authority.epoch, room_id=room_id, actor=actor_subject, owner=owner_rule())
        return True

    def insert_room(self, *, room_id, name, members, owner=None):
        """Insert the room on the authority's ``state.db``. *owner*'s claim runs first in the
        same write, so owner and room commit or roll back together."""
        from gateway.hosted_rooms import local_authority_gateway_id
        claim = None
        if owner is not None:
            _check_claim(owner, room_id)
            claim = HostedOwnerClaim(epoch=self.authority.epoch, actor=owner, rule=owner_rule())
        return authority_persistence(self.authority).hosted_rooms.create_room(
            placement='profile', room_id=room_id, name=name, members=members,
            authority_gateway_id=local_authority_gateway_id(), owner=claim)

    def create_room(self, *, room_id, name, members, owner=None):
        from gateway.hosted_room_discussion import validate_roster
        normalized = validate_roster(members, local_profiles=self.local_profiles())
        room = self.insert_room(room_id=room_id, name=name, owner=owner, members=[
            {'member_id': member.member_id, 'profile': member.profile, 'handle': member.handle,
             'target': dict(member.target or {}),
             **({'display_name': member.display_name} if member.display_name else {})}
            for member in normalized])
        self.runtime.wakeup()
        return room

    def _owner(self, room_id):
        owner = authority_persistence(self.authority).hosted_rooms.owner(room_id)
        if owner is None:
            raise AuthorityRefusal('permission_denied')
        return owner

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
        home = (stopping_attempt_home(authority_persistence(self.authority), binding.room_id, member, task)
                or self.profile_homes().get(profile))
        if home is None:
            raise AuthorityRefusal('permission_denied')
        key = binding.room_id, member, profile, owner, str(home)
        if home != Path(self.authority.profile_id):
            from gateway.session_hosted_member_homes import record_member_home
            # Every resolve, not only a new transport: a profile re-added to hosted_rooms.profiles
            # reuses its cached transport, and its home must still be recorded.
            record_member_home(authority_persistence(self.authority), binding.room_id, member, home)
        if key not in self.member_rpcs:
            if home != Path(self.authority.profile_id):
                from gateway.session_hosted_transport import HostedRoomOwnerRPC
                self.member_rpcs[key] = HostedRoomOwnerRPC(home=home,
                    source_home=self.authority.profile_id, room_id=binding.room_id,
                    member_id=member, profile=profile)
                return self.member_rpcs[key]
            def guard(identity, generation):
                # The member check the admission write runs (``admit(hosted=)``) and the reads run.
                from dataclasses import asdict
                from persistence.gateway.hosted_rooms import HostedMemberGuard, HostedTaskIdentity
                return HostedMemberGuard(
                    binding.room_id, owner, binding.gateway_id, binding.authority_epoch, member, profile,
                    home_current=self.profile_homes().get(profile) == home,
                    task=None if identity is None else HostedTaskIdentity(**asdict(identity)), generation=generation)
            def authorize(operation, identity, generation):
                return authority_persistence(self.authority).hosted_rooms.member_authorized(
                    guard(identity, generation), epoch=self.authority.epoch, operation=operation)
            principal = SessionActor(owner, self.authority.profile_id,
                frozenset({'session:create', 'session:read', 'session:submit', 'session:control', 'session:approve'}),
                'hosted:' + binding.room_id + ':' + member)
            self.member_rpcs[key] = HostedRoomAuthorityRPC(self.authority, self.loop,
                room_id=binding.room_id, member_id=member, profile=profile, principal=principal, authorize=authorize,
                hosted_guard=guard)
        return self.member_rpcs[key]

    def _stopping_attempt_homes(self, room_id, member, task_id=None):
        from gateway.session_hosted_member_homes import legacy_stop_homes, stopping_attempt_homes
        return (stopping_attempt_homes(authority_persistence(self.authority), room_id, member)
                | legacy_stop_homes(self, room_id, member, task_id))

    def _stop_pending_refusal(self, room_id, deferred):
        """Name an unconfigured member profile whose unknown home keeps a turn deferred."""
        from gateway.session_hosted_member_homes import unknown_home_refusal
        from tui_gateway.hosted_room_service import task_member_id
        return unknown_home_refusal(self, room_id, [
            task for task in deferred if not self._member_is_peer(room_id, task_member_id(task))])

    def _stop_deferred_peer(self, room_id, task):
        """Also settle a local member's deferred turn once its profile is no longer served here and
        nothing on this host can run it (``unserved_member_stopped``): its daemon never answers."""
        from gateway.session_hosted_member_homes import unsettled_legacy_homes, unserved_member_stopped
        if super()._stop_deferred_peer(room_id, task):
            return not unsettled_legacy_homes(self, room_id, task)
        from tui_gateway.hosted_room_service import task_member_id
        return (not self._member_is_peer(room_id, task_member_id(task))
                and unserved_member_stopped(self, room_id, task))

    def check_admission(self, ref, row):
        """Reconstruct the private producer from durable task state before claim."""
        import json
        from gateway.hosted_room_driver import list_tasks
        from gateway.session_hosted_attachments import committed_submission_payload
        from tui_gateway.hosted_room_driver import HostedRoomBinding
        try:
            if not row['request_id'].startswith('hosted:'):
                raise ValueError('not a hosted admission')
            identity, generation = json.loads(row['request_id'][7:])
            identity = HostedTaskIdentity(**identity)
            if type(generation) is not int or generation < 1:
                raise ValueError('invalid generation')
            owner = self._owner(identity.room_id)
            if not same_owner(owner, row['principal_id']):
                raise ValueError('foreign owner')
            room = self._room(identity.room_id)
            task = next(t for t in list_tasks(self.store, placement=self.placement, room_id=identity.room_id)
                        if t['identity'] == identity)
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
            raise AuthorityRefusal('permission_denied') from exc
        if retired:
            raise retired_attempt_conflict()
        return task

    def approve(self, *, session_id, request_id, choice):
        rpc = next((r for r in self.member_rpcs.values() if r.ref.session_id == session_id), None)
        if rpc is None:
            raise AuthorityRefusal('permission_denied')
        return rpc.approve(session_id=session_id, request_id=request_id, choice=choice)


async def ensure_hosted_service(runner):
    """Prepare every transport before readiness can release any coordinator."""
    active = active_authority(runner)
    if active is None:
        raise AuthorityRefusal('profile_mismatch')
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
            from persistence_assembly import store_location
            await asyncio.to_thread(import_standalone_rooms, store_location(authority_persistence(authority).scope),
                                    runtime_epoch=authority.epoch)
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
