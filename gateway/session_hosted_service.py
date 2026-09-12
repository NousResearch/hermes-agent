"""Authority-bound hosted service; no legacy session server is constructed."""
import asyncio
from contextlib import nullcontext
from pathlib import Path

from gateway.session_contract import Principal
from gateway.session_hosted_controls import HostedControls
from hermes_state_runtime import RuntimeStoreError, _epoch
from tui_gateway.hosted_room_service import HostedRoomService

_OWNER = 'gateway.hosted.owner.v1:'


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
        from gateway.run import _load_gateway_config
        from gateway.hosted_rooms_common import IDENTIFIER_RE
        from gateway.session_authorities import served_profile_name
        home = Path(self.authority.profile_id)
        own = served_profile_name(home)
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
            result.update(prompt=payload['prompt'], attachments=payload.get('attachments', []))
            if operation == 'attachment':
                from gateway.session_hosted_transport import source_attachment_chunk
                result.update(source_attachment_chunk(self, member, room_id, result['attachments'], params))
        return result


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
            return super()._resolve_member_transport(binding, task)
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
            def authorize(operation, identity, generation):
                self.authorize_room(owner, binding.room_id)
                room = self._room(binding.room_id)
                if (room['authority_gateway_id'], room['authority_epoch']) != (binding.gateway_id, binding.authority_epoch):
                    return False
                members = room['members']
                if not any(m.get('member_id') == member and m.get('profile') == profile for m in members):
                    return False
                if self.profile_homes().get(profile) != home:
                    return False
                if identity is not None:
                    from gateway.hosted_room_driver import list_tasks
                    return any(t['identity'] == identity and t['execution_generation'] == generation
                               and t['payload'].get('target_profile') == profile
                               and t['status'] in {'running', 'stopping'}
                               for t in list_tasks(self.db_path, room_id=binding.room_id))
                return True
            principal = Principal(owner, self.authority.profile_id,
                frozenset({'session:create', 'session:read', 'session:submit', 'session:control', 'session:approve'}),
                'hosted:' + binding.room_id + ':' + member)
            self.member_rpcs[key] = HostedRoomAuthorityRPC(self.authority, self.loop,
                room_id=binding.room_id, member_id=member, profile=profile, principal=principal, authorize=authorize)
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

    def approve(self, *, session_id, request_id, choice):
        rpc = next((r for r in self.member_rpcs.values() if r.ref.session_id == session_id), None)
        if rpc is None:
            raise RuntimeStoreError('permission_denied')
        return rpc.approve(session_id=session_id, request_id=request_id, choice=choice)


async def ensure_hosted_service(runner):
    authority = runner.session_authority
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
    await asyncio.to_thread(service.start)
    return service


async def stop_hosted_service(runner, timeout=5):
    service = getattr(runner.session_authority, 'hosted_room_service', None)
    return True if service is None else await asyncio.to_thread(service.stop, timeout=timeout)
