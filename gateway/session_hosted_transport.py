"""Hosted producers between independently owned local profile daemons.

Only the private OS-authenticated owner socket exposes these verbs. The source
owner attests durable membership/task state; the target never opens its database.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict
from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

from gateway.hosted_room_driver import TaskIdentity
from gateway.session_contract import Principal
from gateway.session_authorities import owner_scope
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from hermes_state_runtime import RuntimeStoreError, _epoch

_BINDING = 'gateway.hosted.transport.v1:'
_OPERATIONS = frozenset({'resolve_exact', 'create', 'resume', 'submit', 'history',
                         'info', 'interrupt', 'discard', 'approve'})
_CHUNK_BYTES = 24576
_CAPS = frozenset({'session:create', 'session:read', 'session:submit',
                   'session:control', 'session:approve'})


def owner_request(home, verb, params, *, timeout=30):
    """Authenticated private socket exchange; no owner-start or storage fallback."""
    home = Path(home)
    if home != home.resolve():
        raise RuntimeStoreError('permission_denied')
    from hermes_cli.gateway_runtime import discover_gateway_endpoint, control_home_for
    deadline = time.monotonic() + timeout
    discovered = discover_gateway_endpoint(home, timeout=timeout)
    if discovered.state != 'ready' or discovered.endpoint is None:
        raise RuntimeStoreError('runtime_draining')
    from hermes_constants import hermes_home_key
    if hermes_home_key(discovered.endpoint.profile_id) != hermes_home_key(home):
        raise RuntimeStoreError('profile_mismatch')
    # Socket identity and logical authority identity are distinct under multiplex.
    params = {**params, 'profile_id': discovered.endpoint.profile_id}
    home = control_home_for(home, discovered.endpoint)
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise RuntimeStoreError('runtime_draining')
    request = json.dumps({'protocol': 1, 'id': 1, 'verb': verb, 'params': params}).encode() + b'\n'
    if len(request) > 65536:
        raise RuntimeStoreError('invalid_params')
    if os.name == 'nt':
        from gateway.runtime_bootstrap_windows import query_runtime_control
        raw = query_runtime_control(home, request, timeout)
    else:
        from hermes_cli.gateway_runtime_discovery import _socket_path
        from gateway.control_socket import _read_response_line
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(timeout)
            peer.connect(str(_socket_path(home)))
            peer.sendall(request)
            def read():
                peer.settimeout(max(0.001, deadline - time.monotonic()))
                return peer.recv(65536)
            raw = _read_response_line(read, deadline)
    if not raw:
        raise RuntimeStoreError('runtime_draining')
    response = json.loads(raw)
    if response.get('ok') is not True:
        reason = str(response.get('error', '')).split(': ')[-1]
        if reason not in {'permission_denied', 'profile_mismatch', 'invalid_params',
                          'unknown_execution', 'stale_generation', 'admission_conflict'}:
            reason = 'runtime_draining'
        raise RuntimeStoreError(reason)
    if response.get('protocol') != 1 or response.get('id') != 1:
        raise RuntimeStoreError('runtime_draining')
    return response['result']


def _attest(binding, operation, params):
    try:
        result = owner_request(binding['source_home'], 'hosted-attest', {
            'selector': binding['selector'], 'operation': operation,
            'params': {**params, '_target_home': binding['target_home']}})
    except RuntimeStoreError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('runtime_draining') from exc
    if not isinstance(result, dict) or not isinstance(result.get('owner'), str) or not result['owner']:
        raise RuntimeStoreError('permission_denied')
    if result.get('target_home') != binding['target_home']:
        raise RuntimeStoreError('permission_denied')
    return result


def source_attachment_chunk(service, member, room_id, manifest, params):
    """Read scoped source bytes only on the source owner's authenticated handler."""
    index, offset = params.get('index'), params.get('offset')
    if (type(index) is not int or not 0 <= index < len(manifest)
            or type(offset) is not int or not 0 <= offset < manifest[index]['size']):
        raise RuntimeStoreError('permission_denied')
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    item = manifest[index]
    saved = HostedRoomAttachmentStore(service.db_path).read(room_id=room_id,
        attachment_id=item['attachment_id'], event_id=item['event_id'], recipient_member_id=member)
    if any(saved.attachment[key] != item[key] for key in ('kind', 'name', 'mime', 'size')):
        raise RuntimeStoreError('permission_denied')
    data = saved.data
    return {'data_base64': base64.b64encode(data[offset:offset + _CHUNK_BYTES]).decode('ascii'),
            'sha256': hashlib.sha256(data).hexdigest()}


def _attachment_data(binding, attested, params):
    """Transfer bytes, not foreign filenames, with a task fence on every chunk."""
    from gateway.hosted_room_driver import validate_bound_task_manifest
    manifest = attested.get('attachments', [])
    if not manifest:
        return []
    manifest = validate_bound_task_manifest(manifest)
    result = []
    for index, item in enumerate(manifest):
        data = bytearray()
        digest = None
        while len(data) < item['size']:
            chunk = _attest(binding, 'attachment', {
                'task': params['task'], 'execution_generation': params['execution_generation'],
                'prompt': attested['prompt'], 'attachments': manifest, 'index': index, 'offset': len(data)})
            raw = base64.b64decode(chunk['data_base64'], validate=True)
            expected = min(_CHUNK_BYTES, item['size'] - len(data))
            if (chunk['owner'] != attested['owner'] or len(raw) != expected
                    or (digest is not None and digest != chunk['sha256'])):
                raise RuntimeStoreError('permission_denied')
            digest = chunk['sha256']
            data.extend(raw)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeStoreError('permission_denied')
        result.append((item, bytes(data)))
    return result


def _principal(authority, binding):
    # Source-profile namespace prevents equal room owner strings on other owners
    # from aliasing a target's principal. Never widen Principal.profile_id.
    identity = json.dumps([binding['source_home'], binding['owner']], separators=(',', ':'))
    subject = 'hosted-owner:' + hashlib.sha256(identity.encode()).hexdigest()
    return Principal(subject, authority.profile_id, _CAPS, 'hosted-owner-transport')


def install_hosted_transport(server, authority, loop, *, attest):
    """Install profile-selected private routing over the existing authority registry.

    The callback must validate current room authority/member and, for submit and
    execute, exact TaskIdentity, generation and prompt against durable task data.
    It returns {'owner': durable_room_owner_subject}, never a client actor.
    Every wire envelope names the logical profile_id, not its control socket home.
    Reinstallation for a secondary preserves routing to all other served owners.
    """
    def select(envelope):
        from hermes_constants import assert_named_profile_home_live
        params = dict(envelope)
        profile_id = params.pop('profile_id', None)
        if not isinstance(profile_id, str) or not Path(profile_id).is_absolute():
            raise RuntimeStoreError('profile_mismatch')
        home = Path(profile_id)
        if home != home.resolve():
            raise RuntimeStoreError('profile_mismatch')
        registry = getattr(getattr(authority, 'runner', None), 'session_authorities', None)
        selected = registry.for_home(home) if registry is not None else authority
        if selected is None or selected.profile_id != profile_id:
            raise RuntimeStoreError('profile_mismatch')
        assert_named_profile_home_live(home)
        return selected, params

    def source(envelope, peer):
        selected, params = select(envelope)
        if set(params) != {'selector', 'operation', 'params'}:
            raise RuntimeStoreError('invalid_params')
        if params['operation'] not in _OPERATIONS | {'execute', 'attachment'}:
            raise RuntimeStoreError('invalid_params')
        callback = attest if selected is authority else getattr(
            getattr(selected, 'hosted_room_service', None), 'attest', None)
        if callback is None:
            raise RuntimeStoreError('runtime_draining')
        with owner_scope(selected):
            return callback(params['selector'], params['operation'], params['params'])

    def target(envelope, peer):
        selected, params = select(envelope)
        with owner_scope(selected):
            return produce(selected, params)

    def produce(authority, envelope):
        if set(envelope) != {'source_home', 'selector', 'operation', 'params'}:
            raise RuntimeStoreError('invalid_params')
        operation, params = envelope['operation'], dict(envelope['params'])
        selector = envelope['selector']
        if operation not in _OPERATIONS or set(selector) != {'room_id', 'member_id', 'profile'}:
            raise RuntimeStoreError('invalid_params')
        from gateway.session_authorities import served_profile_name
        profile = served_profile_name(Path(authority.profile_id))
        if selector['profile'] != profile:
            raise RuntimeStoreError('profile_mismatch')
        binding = {'source_home': envelope['source_home'], 'selector': selector,
                   'target_home': authority.profile_id}
        attested = _attest(binding, operation, params)
        binding['owner'] = attested['owner']
        principal = _principal(authority, binding)
        rpc = HostedRoomAuthorityRPC(authority, loop, **selector, principal=principal,
                                    authorize=lambda *args: True)
        key = _BINDING + rpc.ref.session_id
        encoded = json.dumps(binding, sort_keys=True)
        def persist(conn):
            _epoch(conn, authority.epoch)
            old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            if old is not None and old[0] != encoded:
                raise RuntimeStoreError('permission_denied')
            conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', (key, encoded))
        authority.db._execute_write(persist)
        if operation == 'submit':
            rpc.hosted_attachment_data = _attachment_data(binding, attested, params)
            params['attachments'] = attested['attachments'] or None
            params['task'] = TaskIdentity(**params['task'])
            params['on_terminal'] = lambda value: None
        result = rpc._call(operation, **params)
        return result

    server.private_handlers.update({'hosted-attest': source, 'hosted-producer': target})


def check_remote_hosted_admission(authority, ref, row):
    """Before claim, reauthorize durable target binding at its source owner.

    Returns False only for a non-transport session. A known transport binding
    fails closed on missing source, revoked membership or altered task payload.
    Call off the owner's event loop (the reverse RPC is synchronous).
    """
    with owner_scope(authority):
        return _check_remote_hosted_admission(authority, ref, row)


def _check_remote_hosted_admission(authority, ref, row):
    with authority.db._read_ctx() as conn:
        stored = conn.execute('SELECT value FROM state_meta WHERE key=?',
                              (_BINDING + ref.session_id,)).fetchone()
    if stored is None:
        return False
    try:
        binding = json.loads(stored[0])
        identity, generation = json.loads(row['request_id'][7:])
        if (not row['request_id'].startswith('hosted:')
                or row['principal_id'] != _principal(authority, binding).subject
                or ref.profile_id != authority.profile_id
                or binding.get('target_home') != authority.profile_id):
            raise ValueError('binding mismatch')
        params = {'task': identity, 'execution_generation': generation}
        attested = _attest(binding, 'execute', params)
        if attested['owner'] != binding['owner']:
            raise ValueError('owner changed')
        from gateway.session_hosted_attachments import committed_submission_payload
        rpc = SimpleNamespace(authority=authority, **binding['selector'],
            hosted_attachment_data=_attachment_data(binding, attested, params))
        if row['payload'] != committed_submission_payload(rpc, attested['prompt'], attested['attachments'], admission=row):
            raise ValueError('input changed')
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeStoreError('permission_denied') from exc
    return True


class HostedRoomOwnerRPC(HostedRoomAuthorityRPC):
    """Driver-compatible producer; carries data, never a caller principal."""
    def __init__(self, *, home, source_home, room_id, member_id, profile):
        self.home = Path(home)
        self.binding = {'source_home': str(Path(source_home)),
                        'selector': dict(room_id=room_id, member_id=member_id, profile=profile)}
        self.callbacks = {}
        self._lock = threading.Lock()
        self._monitor = None

    def _call(self, operation, **params):
        callback = params.pop('on_terminal', None)
        if isinstance(params.get('task'), TaskIdentity):
            params['task'] = asdict(params['task'])
        result = owner_request(self.home, 'hosted-producer', {
            **self.binding, 'operation': operation, 'params': params})
        if operation in {'create', 'resume', 'resolve_exact'} and result is not None:
            from gateway.session_contract import SessionRef
            self.ref = SessionRef(str(self.home), result['session_id'])
        if operation == 'submit' and callback is not None:
            with self._lock:
                self.callbacks[result['admission_id']] = callback
                if self._monitor is None or not self._monitor.is_alive():
                    self._monitor = threading.Thread(target=self._watch,
                        args=(params['session_id'],), daemon=True)
                    self._monitor.start()
        if operation == 'history':
            self._deliver(result)
        return result

    def _deliver(self, history):
        for row in history:
            with self._lock:
                callback = self.callbacks.pop(row.get('settlement_id'), None)
            if callback is not None:
                callback(row)

    def _watch(self, session_id):
        try:
            while True:
                with self._lock:
                    if not self.callbacks:
                        return
                self.history(profile=self.binding['selector']['profile'],
                             session_id=session_id, source='bot_room')
                time.sleep(0.25)
        except (OSError, ValueError, RuntimeStoreError):
            # Retain callbacks/input for normal driver history recovery; no retry
            # admission and no assertion that a timed-out request was unaccepted.
            return
