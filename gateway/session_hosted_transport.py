"""Hosted producers between independently owned local profile daemons.

Only the private OS-authenticated owner socket exposes these verbs. The source
owner attests durable membership/task state; the target never opens its database.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

from gateway.hosted_room_driver import TaskIdentity
from gateway.session_contract import Principal
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from hermes_state_runtime import RuntimeStoreError, _epoch

_BINDING = 'gateway.hosted.transport.v1:'
_OPERATIONS = frozenset({'resolve_exact', 'create', 'resume', 'submit', 'history',
                         'info', 'interrupt', 'discard', 'approve'})
_CAPS = frozenset({'session:create', 'session:read', 'session:submit',
                   'session:control', 'session:approve'})


def owner_request(home, verb, params, *, timeout=30):
    """Authenticated private socket exchange; no owner-start or storage fallback."""
    home = Path(home)
    if home != home.resolve():
        raise RuntimeStoreError('permission_denied')
    request = json.dumps({'protocol': 1, 'id': 1, 'verb': verb, 'params': params}).encode() + b'\n'
    if len(request) > 65536:
        raise RuntimeStoreError('invalid_params')
    if os.name == 'nt':
        from gateway.runtime_bootstrap_windows import query_runtime_control
        raw = query_runtime_control(home, request, timeout)
    else:
        from hermes_cli.gateway_runtime_discovery import _socket_path
        from gateway.control_socket import _read_response_line
        deadline = time.monotonic() + timeout
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
    result = owner_request(binding['source_home'], 'hosted-attest', {
        'selector': binding['selector'], 'operation': operation, 'params': params})
    if not isinstance(result, dict) or not isinstance(result.get('owner'), str) or not result['owner']:
        raise RuntimeStoreError('permission_denied')
    return result['owner']


def _principal(authority, binding):
    # Source-profile namespace prevents equal room owner strings on other owners
    # from aliasing a target's principal. Never widen Principal.profile_id.
    identity = json.dumps([binding['source_home'], binding['owner']], separators=(',', ':'))
    subject = 'hosted-owner:' + hashlib.sha256(identity.encode()).hexdigest()
    return Principal(subject, authority.profile_id, _CAPS, 'hosted-owner-transport')


def install_hosted_transport(server, authority, loop, *, attest):
    """Install private routing. attest(selector, operation, params) reads OWN state.

    The callback must validate current room authority/member and, for submit and
    execute, exact TaskIdentity, generation and prompt against durable task data.
    It returns {'owner': durable_room_owner_subject}, never a client actor.
    """
    def source(params, peer):
        if set(params) != {'selector', 'operation', 'params'}:
            raise RuntimeStoreError('invalid_params')
        if params['operation'] not in _OPERATIONS | {'execute'}:
            raise RuntimeStoreError('invalid_params')
        return attest(params['selector'], params['operation'], params['params'])

    def target(envelope, peer):
        if set(envelope) != {'source_home', 'selector', 'operation', 'params'}:
            raise RuntimeStoreError('invalid_params')
        operation, params = envelope['operation'], dict(envelope['params'])
        selector = envelope['selector']
        if operation not in _OPERATIONS or set(selector) != {'room_id', 'member_id', 'profile'}:
            raise RuntimeStoreError('invalid_params')
        home = Path(authority.profile_id)
        profile = home.name if home.parent.name == 'profiles' else 'default'
        if selector['profile'] != profile:
            raise RuntimeStoreError('profile_mismatch')
        binding = {'source_home': envelope['source_home'], 'selector': selector}
        binding['owner'] = _attest(binding, operation, params)
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
                or ref.profile_id != authority.profile_id):
            raise ValueError('binding mismatch')
        owner = _attest(binding, 'execute', {'task': identity,
            'execution_generation': generation, 'prompt': row['payload']['text']})
        if owner != binding['owner']:
            raise ValueError('owner changed')
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
